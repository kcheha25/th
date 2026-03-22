import numpy as np
import random
from pathlib import Path
import spectral.io.envi as envi
import cv2
import json
from collections import defaultdict

N_EXTRUDES_PER_PLOT = 12
MAX_PLOTS = 500
IGNORE_CLASSES = []
TARGET_PER_CLASS = 200

def load_cube(hdr_path):
    img = envi.open(str(hdr_path))
    return np.array(img.load(), dtype=np.float32)

def save_envi_cube(cube, path):
    envi.save_image(str(path) + ".hdr", cube, dtype=np.float32, interleave="bsq", force=True)

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
    valid = cv2.erode(zone_u8, kernel)
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
    indices = np.random.permutation(len(xs))
    for idx in indices[:100]:
        y, x = ys[idx], xs[idx]
        if y + h > H or x + w > W:
            continue
        if not can_place(extrude_mask, occ_mask, x, y):
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
    cube = load_cube(plot_hdr)
    json_path = plot_hdr.with_suffix(".json")
    with open(json_path) as f:
        data = json.load(f)
    zones = [s["points"] for s in data["shapes"] if s["label"] == "zone_placement"]
    return cube, zones

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
        plot_cube, zones = load_plot(plot_hdr)
        _, H, W = plot_cube.shape
        occ_mask = np.zeros((H, W), dtype=bool)
        shapes = []
        for e in selected:
            cube = load_cube(e["hdr"])
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
        count += 1

if __name__ == "__main__":
    generate("plots_zone_to_add", "aug_after_rot_flip", "output")