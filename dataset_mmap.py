import gc
import json
import random
import re
import pickle
import hashlib
import os
from collections import defaultdict
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
from sklearn.decomposition import PCA
from spectral import envi
from torch.utils.data import Dataset, DataLoader


def load_mapping(mapping_path):
    with open(mapping_path) as f:
        raw = json.load(f)
    extrude2final = {}
    for final_class, extrude_list in raw.items():
        for ext in extrude_list:
            extrude2final[ext.strip()] = final_class
    final_classes = sorted(raw.keys())
    class_names = final_classes + ["background"]
    label2idx = {name: i for i, name in enumerate(class_names)}
    return extrude2final, class_names, label2idx


def _load_envi_cube(hdr_path, source):
    img = envi.open(str(hdr_path))
    cube = img.load().astype(np.float32)
    wl = [float(w) for w in img.metadata.get("wavelength", [])]
    if cube.ndim == 3:
        if source == "real" and wl and cube.shape[2] == len(wl):
            cube = cube.transpose(2, 0, 1)
        elif source != "real" and wl and cube.shape[1] == len(wl):
            cube = cube.transpose(1, 0, 2)
    if cube.ndim != 3:
        raise ValueError(f"Cube invalide : {cube.shape}")
    return cube, wl


def _load_shapes(json_path):
    with open(json_path) as f:
        return json.load(f).get("shapes", [])


def collect_plots_meta(real_dir=None, aug_dir=None):
    meta_list = []
    if real_dir:
        real_dir = Path(real_dir)
        for hdr in sorted(real_dir.glob("plot_*_cube.hdr")):
            m = re.match(r"plot_(\d+)_cube", hdr.stem)
            if not m:
                continue
            jsn = real_dir / f"plot_{m.group(1)}_annotations.json"
            if jsn.exists():
                meta_list.append({"source": "real", "name": hdr.stem, "hdr": hdr, "jsn": jsn})
    if aug_dir:
        aug_dir = Path(aug_dir)
        for hdr in sorted(aug_dir.glob("*.hdr")):
            if any(hdr.stem.endswith(s) for s in ("_ratio", "_cube")):
                continue
            jsn = aug_dir / f"{hdr.stem}.json"
            if not jsn.exists():
                continue
            source = "circle" if "circle" in hdr.stem.lower() else "aug"
            meta_list.append({"source": source, "name": hdr.stem, "hdr": hdr, "jsn": jsn})
    if not meta_list:
        raise RuntimeError("Aucun plot trouvé.")
    return meta_list


def split_plots(plots_meta, split_mode="aug_real", val_ratio=0.15, test_ratio=0.15, random_state=345):
    rng = random.Random(random_state)
    if split_mode == "aug_real":
        aug = [p for p in plots_meta if p["source"] in ("aug", "circle")]
        real = [p for p in plots_meta if p["source"] == "real"]
        rng.shuffle(aug)
        n_val = max(1, int(len(aug) * val_ratio))
        train, val, test = aug[n_val:], aug[:n_val], real
    elif split_mode == "shuffle":
        all_p = plots_meta[:]
        rng.shuffle(all_p)
        n = len(all_p)
        n_test, n_val = max(1, int(n * test_ratio)), max(1, int(n * val_ratio))
        test = all_p[:n_test]
        val = all_p[n_test:n_test + n_val]
        train = all_p[n_test + n_val:]
    else:
        raise ValueError(f"split_mode inconnu : {split_mode}")
    return train, val, test


def _rasterize_cpu(points, H, W):
    pts = np.array(points, dtype=np.float32)
    n = len(pts)
    cols = np.arange(W, dtype=np.float32) + 0.5
    rows = np.arange(H, dtype=np.float32) + 0.5
    gc_, gr_ = np.meshgrid(cols, rows)
    px, py = gc_.ravel(), gr_.ravel()
    inside = np.zeros(len(px), dtype=bool)
    x1, y1 = pts[:, 0], pts[:, 1]
    x2, y2 = np.roll(x1, -1), np.roll(y1, -1)
    for i in range(n):
        dy = y2[i] - y1[i]
        if abs(dy) < 1e-10:
            continue
        cond = (y1[i] < py) != (y2[i] < py)
        xi = x1[i] + (py - y1[i]) * (x2[i] - x1[i]) / dy
        inside ^= cond & (px < xi)
    return inside.reshape(H, W)


def build_label_map(H, W, shapes, extrude2final, label2idx):
    bg = label2idx["background"]
    lm = np.full((H, W), bg, dtype=np.int32)
    for shape in shapes:
        raw = shape.get("label", "").strip()
        stype = shape.get("shape_type", "")
        if raw.lower() == "trou" or stype not in ("polygon", "rectangle"):
            continue
        final = extrude2final.get(raw)
        if final is None:
            continue
        pts = shape["points"]
        if len(pts) < 3:
            continue
        try:
            lm[_rasterize_cpu(pts, H, W)] = label2idx[final]
        except Exception as e:
            print(f"Warning rasterisation '{raw}': {e}")
    return lm


def fit_or_load_pca(real_meta_list, pca_components, cache_dir, config_hash, fit_max_pixels=500_000):
    path = Path(cache_dir) / f"pca_{config_hash}.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    pixels = []
    for meta in real_meta_list:
        try:
            cube, _ = _load_envi_cube(meta["hdr"], meta["source"])
            B, H, W = cube.shape
            n = min(20_000, H * W)
            idx = np.random.choice(H * W, n, replace=False)
            pixels.append(cube.reshape(B, -1)[:, idx].T)
            del cube
            gc.collect()
        except Exception as e:
            print(f"Warning {meta['name']}: {e}")
    pixels = np.concatenate(pixels)
    if len(pixels) > fit_max_pixels:
        pixels = pixels[np.random.choice(len(pixels), fit_max_pixels, replace=False)]
    pca = PCA(n_components=pca_components, whiten=True)
    pca.fit(pixels)
    del pixels
    gc.collect()
    with open(path, "wb") as f:
        pickle.dump(pca, f, protocol=pickle.HIGHEST_PROTOCOL)
    return pca


def apply_pca_chunked(cube, pca, chunk=30_000):
    B, H, W = cube.shape
    flat = cube.reshape(B, -1).T
    out = np.empty((H * W, pca.n_components_), dtype=np.float32)
    for s in range(0, len(flat), chunk):
        out[s:s + chunk] = pca.transform(flat[s:s + chunk])
    del flat
    return out.T.reshape(pca.n_components_, H, W).astype(np.float32)


def balance_background(labels, bg_idx, rng, bg_ratio=1.5):
    fg = labels != bg_idx
    keep = fg.copy()
    n_fg = int(fg.sum())
    if n_fg > 0:
        bg_pool = np.where(~fg)[0]
        n_keep = min(int(n_fg * bg_ratio), len(bg_pool))
        if n_keep > 0:
            keep[rng.choice(bg_pool, n_keep, replace=False)] = True
    return keep


def _extract_patches_cpu(cube, rows, cols, patch_size):
    B, H, W = cube.shape
    m = patch_size // 2
    padded = np.pad(cube, ((0, 0), (m, m), (m, m)), mode='constant')
    patches = np.empty((len(rows), B, patch_size, patch_size), dtype=np.float32)
    for i, (r, c) in enumerate(zip(rows, cols)):
        patches[i] = padded[:, r:r+patch_size, c:c+patch_size]
    return patches


def extract_split_to_mmap(meta_list, split_name, patch_size, extrude2final, label2idx,
                           bg_ratio, random_state, pca, base_dir):
    if not meta_list:
        return 0
    if pca is not None:
        num_bands = pca.n_components_
    else:
        tmp, _ = _load_envi_cube(meta_list[0]["hdr"], meta_list[0]["source"])
        num_bands = tmp.shape[0]
        del tmp
        gc.collect()
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    X_path = base_dir / f"{split_name}_X.dat"
    y_path = base_dir / f"{split_name}_y.dat"
    batch_size = 100000
    X_mmap = None
    y_mmap = None
    current_size = 0
    bg_idx = label2idx["background"]
    for i, meta in enumerate(meta_list):
        print(f"  [{split_name}] Cube {i+1}/{len(meta_list)}: {meta['name']}")
        try:
            cube_raw, _ = _load_envi_cube(meta["hdr"], meta["source"])
            if pca is not None:
                cube_pca = apply_pca_chunked(cube_raw, pca)
                del cube_raw
                gc.collect()
            else:
                cube_pca = cube_raw
            B, H, W = cube_pca.shape
            shapes = _load_shapes(meta["jsn"])
            label_map = build_label_map(H, W, shapes, extrude2final, label2idx)
            r_all, c_all = np.where(label_map >= 0)
            labels_all = label_map[r_all, c_all]
            del label_map
            rng = np.random.default_rng(random_state + hash(meta["name"]) % 10_000)
            keep = balance_background(labels_all, bg_idx, rng, bg_ratio)
            rows_k = r_all[keep].astype(np.int32)
            cols_k = c_all[keep].astype(np.int32)
            labels_k = labels_all[keep].astype(np.int32)
            del r_all, c_all, labels_all, keep
            gc.collect()
            if len(rows_k) > 0:
                patches = _extract_patches_cpu(cube_pca, rows_k, cols_k, patch_size)
                n_patches = len(patches)
                if X_mmap is None:
                    X_mmap = np.memmap(X_path, dtype=np.float32, mode='w+',
                                       shape=(n_patches, num_bands, patch_size, patch_size))
                    y_mmap = np.memmap(y_path, dtype=np.int64, mode='w+', shape=(n_patches,))
                else:
                    if current_size + n_patches > X_mmap.shape[0]:
                        new_shape = (current_size + n_patches + batch_size,
                                     num_bands, patch_size, patch_size)
                        X_mmap.flush()
                        y_mmap.flush()
                        del X_mmap, y_mmap
                        X_mmap = np.memmap(X_path, dtype=np.float32, mode='r+', shape=new_shape)
                        y_mmap = np.memmap(y_path, dtype=np.int64, mode='r+',
                                           shape=(new_shape[0],))
                X_mmap[current_size:current_size + n_patches] = patches
                y_mmap[current_size:current_size + n_patches] = labels_k
                current_size += n_patches
                X_mmap.flush()
                y_mmap.flush()
                print(f"    -> {n_patches} patches, total: {current_size}")
            del cube_pca
            gc.collect()
        except Exception as e:
            print(f"  Erreur {meta['name']}: {e}")
    if X_mmap is not None and current_size > 0:
        if current_size < X_mmap.shape[0]:
            X_mmap.flush()
            y_mmap.flush()
            del X_mmap, y_mmap
            X_mmap = np.memmap(X_path, dtype=np.float32, mode='r+',
                               shape=(current_size, num_bands, patch_size, patch_size))
            y_mmap = np.memmap(y_path, dtype=np.int64, mode='r+', shape=(current_size,))
            X_mmap.flush()
            y_mmap.flush()
        meta = {"shape": [current_size, num_bands, patch_size, patch_size], "dtype": "float32"}
        with open(base_dir / f"{split_name}_meta.json", 'w') as f:
            json.dump(meta, f)
        return current_size
    return 0


class MMapDataset(Dataset):
    def __init__(self, mmap_dir, split):
        mmap_dir = Path(mmap_dir)
        with open(mmap_dir / f"{split}_meta.json") as f:
            meta = json.load(f)
            shape = tuple(meta["shape"])
            dtype = meta["dtype"]
        self.X = np.memmap(mmap_dir / f"{split}_X.dat", dtype=dtype, mode='r', shape=shape)
        self.y = np.memmap(mmap_dir / f"{split}_y.dat", dtype=np.int64, mode='r',
                           shape=(shape[0],))
        self.length = len(self.y)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        X = torch.from_numpy(self.X[idx].copy()).float()
        y = torch.tensor(int(self.y[idx]), dtype=torch.long)
        return X.unsqueeze(0), y


def build_loaders(mmap_dir, batch_size=64, num_workers=4):
    mmap_dir = Path(mmap_dir)
    train_ds = MMapDataset(mmap_dir, 'train')
    val_ds = MMapDataset(mmap_dir, 'val')
    test_ds = MMapDataset(mmap_dir, 'test')
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=True if num_workers > 0 else False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True,
                            persistent_workers=True if num_workers > 0 else False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True,
                             persistent_workers=True if num_workers > 0 else False)
    with open(mmap_dir / "global_meta.json") as f:
        meta = json.load(f)
    return train_loader, val_loader, test_loader, meta


def build_mmap(real_dir=None, aug_dir=None, mapping_path="mapping.json",
               split_mode="aug_real", use_pca=True, pca_components=30,
               patch_size=13, test_ratio=0.15, val_ratio=0.15, batch_size=64,
               random_state=345, fit_max_pixels=500_000, num_workers=4,
               bg_ratio=1.5, mmap_dir="mmap_data", force_recompute=False):
    extrude2final, class_names, label2idx = load_mapping(mapping_path)
    mmap_dir = Path(mmap_dir)
    mmap_dir.mkdir(parents=True, exist_ok=True)
    config_str = (f"{real_dir}_{aug_dir}_{split_mode}_{patch_size}_{random_state}"
                  f"_{bg_ratio}_{pca_components if use_pca else 0}")
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:12]
    mmap_subdir = mmap_dir / config_hash
    mmap_subdir.mkdir(parents=True, exist_ok=True)
    meta_path = mmap_subdir / "global_meta.json"
    if not force_recompute and meta_path.exists():
        return build_loaders(str(mmap_subdir), batch_size, num_workers)
    plots_meta = collect_plots_meta(real_dir=real_dir, aug_dir=aug_dir)
    train_meta, val_meta, test_meta = split_plots(plots_meta, split_mode, val_ratio,
                                                   test_ratio, random_state)
    pca_model = None
    if use_pca:
        real_meta = [m for m in plots_meta if m["source"] == "real"]
        if real_meta:
            pca_model = fit_or_load_pca(real_meta, pca_components, mmap_subdir,
                                         config_hash, fit_max_pixels)
    for meta_list, sname in [(train_meta, "train"), (val_meta, "val"), (test_meta, "test")]:
        extract_split_to_mmap(meta_list, sname, patch_size, extrude2final, label2idx,
                               bg_ratio, random_state, pca_model, str(mmap_subdir))
    meta = {
        "class_names": class_names,
        "label2idx": label2idx,
        "num_bands": pca_components if use_pca else -1,
        "patch_size": patch_size,
        "created_at": datetime.now().isoformat(),
    }
    with open(meta_path, 'w') as f:
        json.dump(meta, f)
    return build_loaders(str(mmap_subdir), batch_size, num_workers)
