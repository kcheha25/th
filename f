import numpy as np
import random
from pathlib import Path
import spectral.io.envi as envi
import cv2
import json
from collections import defaultdict
import matplotlib.pyplot as plt

N_EXTRUDES_PER_PLOT = 12
MAX_PLOTS = 500
IGNORE_CLASSES = []
TARGET_PER_CLASS = 200

WAVELENGTH_1 = 996
WAVELENGTH_2 = 1197

def load_cube(hdr_path):
    img = envi.open(str(hdr_path))
    return np.array(img.load(), dtype=np.float32), img

def save_envi_cube(cube, path):
    envi.save_image(str(path) + ".hdr", cube, dtype=np.float32, interleave="bsq", force=True)

def get_band_index(img, wavelength):
    wl = np.array(img.metadata["wavelength"], dtype=float)
    return np.argmin(np.abs(wl - wavelength))

def compute_ratio_map(cube, b1, b2):
    band1 = cube[b1]
    band2 = cube[b2]
    ratio = np.zeros_like(band1)
    valid = band2 > 1e-6
    ratio[valid] = band1[valid] / band2[valid]
    return ratio

def save_ratio_map(ratio, polygons, out_path):
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(ratio, cmap="viridis")
    plt.colorbar(im, ax=ax, fraction=0.03)
    for poly in polygons:
        p = np.array(poly)
        p = np.vstack([p, p[0]])
        ax.plot(p[:,0], p[:,1], color="red", linewidth=1.5)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()

def get_valid_mask_fast(cube, threshold=1e-6):
    return cube.var(axis=0) > threshold

def crop_to_valid(cube, mask):
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return cube, mask
    r0, r1 = rows[0], rows[-1] + 1
    c0, c1 = cols[0], cols[-1] + 1
    return cube[:, r0:r1, c0:c1], mask[r0:r1, c0:c1]

def find_contour_polygons(mask):
    mask_u8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for cnt in contours:
        pts = cnt.reshape(-1, 2).tolist()
        if len(pts) >= 3:
            polys.append(pts)
    return polys

def polygon_to_mask(polygon, H, W):
    mask = np.zeros((H, W), dtype=np.uint8)
    pts = np.array(polygon, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)

def compute_valid_positions(zone_mask, extrude_mask):
    zone_u8 = zone_mask.astype(np.uint8) * 255
    kernel = extrude_mask.astype(np.uint8)
    valid = cv2.erode(zone_u8, kernel, iterations=1)
    return valid.astype(bool)

def can_place(mask_extrude, occ_mask, x, y):
    h, w = mask_extrude.shape
    region = occ_mask[y:y+h, x:x+w]
    return not (region & mask_extrude).any()

def update_occupancy(occ_mask, mask_extrude, x, y):
    h, w = mask_extrude.shape
    occ_mask[y:y+h, x:x+w] |= mask_extrude

def place_extrude(plot_cube, occ_mask, extrude_cube, extrude_mask, polygon, zone_polygon):
    _, H, W = plot_cube.shape
    h, w = extrude_mask.shape
    zone_mask = polygon_to_mask(zone_polygon, H, W)
    valid_positions = compute_valid_positions(zone_mask, extrude_mask)
    ys, xs = np.where(valid_positions)
    if len(xs) == 0:
        return False, plot_cube, occ_mask, None
    idxs = np.random.permutation(len(xs))
    for idx in idxs:
        y, x = ys[idx], xs[idx]
        if y + h > H or x + w > W:
            continue
        if not can_place(extrude_mask, occ_mask, x, y):
            continue
        sub_zone = zone_mask[y:y+h, x:x+w]
        if not np.all(sub_zone[extrude_mask]):
            continue
        for b in range(plot_cube.shape[0]):
            patch = plot_cube[b, y:y+h, x:x+w]
            patch[extrude_mask] = extrude_cube[b][extrude_mask]
            plot_cube[b, y:y+h, x:x+w] = patch
        update_occupancy(occ_mask, extrude_mask, x, y)
        poly = np.array(polygon)
        poly[:, 0] += x
        poly[:, 1] += y
        return True, plot_cube, occ_mask, poly.tolist()
    return False, plot_cube, occ_mask, None

def get_class_from_name(name):
    return name.split("_")[0]

def load_extrude_pool(root, ignore):
    files = list(Path(root).rglob("*.hdr"))
    pool = []
    for f in files:
        cls = get_class_from_name(f.stem)
        if cls in ignore:
            continue
        pool.append({"hdr": f, "class": cls})
    return pool

def load_plot(plot_hdr):
    cube, img = load_cube(plot_hdr)
    json_path = plot_hdr.with_suffix(".json")
    with open(json_path) as f:
        data = json.load(f)
    zones = [s["points"] for s in data["shapes"] if s["label"] == "zone_placement"]
    return cube, zones, img

def init_class_counter(pool):
    d = defaultdict(int)
    for e in pool:
        d[e["class"]] = 0
    return d

def select_balanced(pool, counter, n, target):
    valid = [e for e in pool if counter[e["class"]] < target]
    if len(valid) < n:
        return None
    selected = []
    while len(selected) < n:
        e = random.choice(valid)
        if e not in selected:
            selected.append(e)
    return selected

def save_labelme(shapes, name, H, W, path):
    data = {
        "version": "5.0.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": f"{name}.png",
        "imageData": None,
        "imageHeight": H,
        "imageWidth": W
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def generate(plot_root, extrude_root, out_dir):
    plot_files = list(Path(plot_root).rglob("*.hdr"))
    pool = load_extrude_pool(extrude_root, IGNORE_CLASSES)
    counter = init_class_counter(pool)
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    count = 0
    while count < MAX_PLOTS:
        selected = select_balanced(pool, counter, N_EXTRUDES_PER_PLOT, TARGET_PER_CLASS)
        if selected is None:
            break
        plot_hdr = random.choice(plot_files)
        plot_cube, zones, img = load_plot(plot_hdr)
        b1 = get_band_index(img, WAVELENGTH_1)
        b2 = get_band_index(img, WAVELENGTH_2)
        _, H, W = plot_cube.shape
        occ_mask = np.zeros((H, W), dtype=bool)
        shapes = []
        for e in selected:
            cube, _ = load_cube(e["hdr"])
            mask = get_valid_mask_fast(cube)
            cube, mask = crop_to_valid(cube, mask)
            polys = find_contour_polygons(mask)
            if not polys:
                continue
            placed = False
            for zone in zones:
                ok, plot_cube, occ_mask, poly = place_extrude(
                    plot_cube,
                    occ_mask,
                    cube,
                    mask,
                    polys[0],
                    zone
                )
                if ok:
                    shapes.append({
                        "label": e["class"],
                        "points": poly,
                        "shape_type": "polygon"
                    })
                    counter[e["class"]] += 1
                    placed = True
                    break
            if not placed:
                continue
        pool = [x for x in pool if x not in selected]
        name = f"aug_plot_{count}"
        save_envi_cube(plot_cube, out_dir / name)
        save_labelme(shapes, name, H, W, out_dir / f"{name}.json")
        ratio = compute_ratio_map(plot_cube, b1, b2)
        save_ratio_map(ratio, [s["points"] for s in shapes], out_dir / f"{name}_ratio.png")
        count += 1

if __name__ == "__main__":
    generate("plots_zone_to_add", "aug_after_rot_flip", "output")




################
import numpy as np
import random
from pathlib import Path
import spectral.io.envi as envi
import cv2
import json
from collections import defaultdict
import matplotlib.pyplot as plt

N_EXTRUDES_PER_PLOT = 12
MAX_PLOTS = 500
IGNORE_CLASSES = []
TARGET_PER_CLASS = 200
WAVELENGTH_1 = 996
WAVELENGTH_2 = 1197

def load_cube(hdr_path):
    img = envi.open(str(hdr_path))
    return np.array(img.load(), dtype=np.float32), img

def save_envi_cube(cube, path):
    envi.save_image(str(path)+".hdr", cube, dtype=np.float32, interleave="bsq", force=True)

def get_band_index(img, wavelength):
    wl = np.array(img.metadata["wavelength"], dtype=float)
    return np.argmin(np.abs(wl - wavelength))

def compute_ratio_map(cube, b1, b2):
    band1 = cube[b1]
    band2 = cube[b2]
    ratio = np.zeros_like(band1)
    valid = band2 > 1e-6
    ratio[valid] = band1[valid] / band2[valid]
    return ratio

def save_ratio_map(ratio, polygons, out_path):
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(ratio, cmap="viridis")
    plt.colorbar(im, ax=ax, fraction=0.03)
    if polygons:
        for poly in polygons:
            if poly is None or len(poly)<3:
                continue
            p = np.array(poly)
            p = np.vstack([p, p[0]])
            ax.plot(p[:,0], p[:,1], color="red", linewidth=1.5)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()

def get_valid_mask_fast(cube, threshold=1e-6):
    return cube.var(axis=0) > threshold

def crop_to_valid(cube, mask):
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if len(rows)==0 or len(cols)==0:
        return cube, mask
    r0, r1 = rows[0], rows[-1]+1
    c0, c1 = cols[0], cols[-1]+1
    return cube[:, r0:r1, c0:c1], mask[r0:r1, c0:c1]

def find_contour_polygons(mask):
    mask_u8 = mask.astype(np.uint8)*255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for cnt in contours:
        pts = cnt.reshape(-1,2).tolist()
        if len(pts)>=3:
            polys.append(pts)
    return polys

def polygon_to_mask(polygon, H, W):
    mask = np.zeros((H,W), dtype=np.uint8)
    pts = np.array(polygon, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)

def compute_valid_positions(zone_mask, extrude_mask):
    zone_u8 = zone_mask.astype(np.uint8) * 255
    kernel = extrude_mask.astype(np.uint8)
    valid = cv2.erode(zone_u8, kernel, iterations=1)
    return valid.astype(bool)

def crop_extrude_to_zone(extrude_cube, extrude_mask, zone_mask):
    extrude_h, extrude_w = extrude_mask.shape
    zone_coords = np.argwhere(zone_mask)
    if zone_coords.size==0:
        return None, None, None
    y0, x0 = zone_coords.min(axis=0)
    y1, x1 = zone_coords.max(axis=0)+1
    crop_h = min(extrude_h, y1-y0)
    crop_w = min(extrude_w, x1-x0)
    extrude_crop = extrude_cube[:, :crop_h, :crop_w]
    mask_crop = extrude_mask[:crop_h, :crop_w]
    mask_crop &= zone_mask[y0:y0+crop_h, x0:x0+crop_w]
    polys = find_contour_polygons(mask_crop)
    return extrude_crop, mask_crop, polys[0] if polys else None

def can_place(mask_extrude, occ_mask, x, y):
    h,w = mask_extrude.shape
    region = occ_mask[y:y+h, x:x+w]
    return not (region & mask_extrude).any()

def update_occupancy(occ_mask, mask_extrude, x, y):
    h,w = mask_extrude.shape
    occ_mask[y:y+h, x:x+w] |= mask_extrude

def place_single_strategy(plot_cube, occ_mask, extrude_cube, extrude_mask, zone_polygon):
    _, H, W = plot_cube.shape
    zone_mask = polygon_to_mask(zone_polygon, H, W)
    extrude_crop, mask_crop, poly = crop_extrude_to_zone(extrude_cube, extrude_mask, zone_mask)
    if extrude_crop is None or not mask_crop.any():
        return False, plot_cube, occ_mask, None
    h,w = mask_crop.shape
    for b in range(plot_cube.shape[0]):
        patch = plot_cube[b, 0:h, 0:w]
        patch[mask_crop] = extrude_crop[b][mask_crop]
        plot_cube[b, 0:h, 0:w] = patch
    update_occupancy(occ_mask, mask_crop, 0,0)
    poly = np.array(poly)
    poly[:,0]+=0
    poly[:,1]+=0
    return True, plot_cube, occ_mask, poly.tolist()

def place_extrude(plot_cube, occ_mask, extrude_cube, extrude_mask, polygon, zone_polygon, cls):
    _, H, W = plot_cube.shape
    zone_mask = polygon_to_mask(zone_polygon,H,W)
    extrude_crop, mask_crop, poly = crop_extrude_to_zone(extrude_cube, extrude_mask, zone_mask)
    if extrude_crop is None or not mask_crop.any():
        return False, plot_cube, occ_mask, None
    valid_positions = compute_valid_positions(zone_mask, mask_crop)
    ys,xs = np.where(valid_positions)
    if len(xs)==0:
        return False, plot_cube, occ_mask, None
    idxs = np.random.permutation(len(xs))
    for idx in idxs:
        y,x = ys[idx], xs[idx]
        h,w = mask_crop.shape
        if y+h>H or x+w>W:
            continue
        if not can_place(mask_crop, occ_mask, x, y):
            continue
        for b in range(plot_cube.shape[0]):
            patch = plot_cube[b, y:y+h, x:x+w]
            patch[mask_crop] = extrude_crop[b][mask_crop]
            plot_cube[b, y:y+h, x:x+w] = patch
        update_occupancy(occ_mask, mask_crop, x, y)
        poly = np.array(poly)
        poly[:,0]+=x
        poly[:,1]+=y
        return True, plot_cube, occ_mask, poly.tolist()
    return False, plot_cube, occ_mask, None

def get_class_from_name(name):
    return name.split("_")[0]

def load_extrude_pool(root):
    files = list(Path(root).rglob("*.hdr"))
    pool = []
    for f in files:
        cls = get_class_from_name(f.stem)
        pool.append({"hdr":f,"class":cls})
    return pool

def load_plot(plot_hdr):
    cube,img = load_cube(plot_hdr)
    json_path = plot_hdr.with_suffix(".json")
    with open(json_path) as f:
        data = json.load(f)
    zones = [s["points"] for s in data["shapes"] if s["label"]=="zone_placement"]
    return cube,zones,img

def init_class_counter(pool):
    d = defaultdict(int)
    for e in pool:
        d[e["class"]] = 0
    return d

def save_labelme(shapes,name,H,W,path):
    data = {
        "version":"5.0.1",
        "flags":{},
        "shapes":shapes,
        "imagePath":f"{name}.png",
        "imageData":None,
        "imageHeight":H,
        "imageWidth":W
    }
    with open(path,"w") as f:
        json.dump(data,f,indent=2)

def generate(plot_root, extrude_root, out_dir):
    plot_files = list(Path(plot_root).rglob("*.hdr"))
    pool = load_extrude_pool(extrude_root)
    counter = init_class_counter(pool)
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    count=0
    while count<MAX_PLOTS:
        classes = list(set([e["class"] for e in pool]))
        if not classes:
            break
        selected=[]
        for cls in classes:
            candidates = [e for e in pool if e["class"]==cls]
            if candidates:
                selected.append(random.choice(candidates))
        plot_hdr = random.choice(plot_files)
        plot_cube, zones, img = load_plot(plot_hdr)
        b1 = get_band_index(img,WAVELENGTH_1)
        b2 = get_band_index(img,WAVELENGTH_2)
        _,H,W = plot_cube.shape
        occ_mask = np.zeros((H,W),dtype=bool)
        shapes=[]
        single_used=False
        for e in selected:
            cube,_ = load_cube(e["hdr"])
            mask = get_valid_mask_fast(cube)
            cube, mask = crop_to_valid(cube, mask)
            polys = find_contour_polygons(mask)
            if not polys:
                continue
            cls = e["class"]
            placed=False
            for zone in zones:
                if cls in IGNORE_CLASSES:
                    placed, plot_cube, occ_mask, poly = place_single_strategy(plot_cube, occ_mask, cube, mask, zone)
                    if placed:
                        shapes.append({"label":cls,"points":poly,"shape_type":"polygon"})
                        counter[cls]+=1
                        single_used=True
                        break
                else:
                    placed, plot_cube, occ_mask, poly = place_extrude(plot_cube, occ_mask, cube, mask, polys[0], zone, cls)
                    if placed:
                        shapes.append({"label":cls,"points":poly,"shape_type":"polygon"})
                        counter[cls]+=1
                        break
            if single_used:
                selected=[e]
                break
        if not shapes:
            continue
        pool = [x for x in pool if x not in selected]
        name = f"aug_plot_{count}"
        save_envi_cube(plot_cube, out_dir/name)
        save_labelme(shapes,name,H,W,out_dir/f"{name}.json")
        ratio = compute_ratio_map(plot_cube,b1,b2)
        save_ratio_map(ratio,[s["points"] for s in shapes],out_dir/f"{name}_ratio.png")
        print(f"Plot {count}: compteur par classe {dict(counter)}")
        count+=1

if __name__=="__main__":
    generate("plots_zone_to_add","aug_after_rot_flip","output")