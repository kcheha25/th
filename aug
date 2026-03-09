import os
import json
import random
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import specarray as sa
import spectral.io.envi as envi

TARGET_N = 3000

def load_cube(folder):
    arr = sa.SpecArray.from_folder(folder)
    arr = arr.spectral_albedo()
    cube = arr.array  # (H, B, W)
    metadata = {"wavelength": arr.wavelengths}
    return cube, metadata

def find_band_index(metadata, wavelength):
    waves = np.array(metadata["wavelength"], dtype=float)
    return np.argmin(np.abs(waves - wavelength))

def spectral_ratio_map(cube, metadata):
    i1 = find_band_index(metadata, 996)
    i2 = find_band_index(metadata, 1197)
    ratio = cube[:, i1, :] / (cube[:, i2, :] + 1e-6)
    plt.figure()
    plt.imshow(ratio)
    plt.colorbar()
    plt.title("Ratio 996 / 1197 nm")
    plt.show()

def load_labelme_rectangles(json_path):
    with open(json_path) as f:
        data = json.load(f)
    rects = []
    for shape in data["shapes"]:
        if shape["shape_type"] != "rectangle":
            continue
        p1, p2 = shape["points"]
        x1, y1 = int(p1[0]), int(p1[1])
        x2, y2 = int(p2[0]), int(p2[1])
        rects.append((x1, y1, x2, y2))
    return rects

def load_labelme_polygons(json_path):
    with open(json_path) as f:
        data = json.load(f)
    polygons = []
    for shape in data["shapes"]:
        if shape["shape_type"] != "polygon":
            continue
        polygons.append(shape["points"])
    return polygons

def crop_plots_from_cube(cube, rectangles):
    plots = []
    for rect in rectangles:
        x1, y1, x2, y2 = rect
        xmin = min(x1, x2)
        xmax = max(x1, x2)
        ymin = min(y1, y2)
        ymax = max(y1, y2)
        plot_cube = cube[ymin:ymax, :, xmin:xmax]  # toutes les bandes
        plots.append(plot_cube)
    return plots

def polygon_bbox(poly):
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    x0, x1 = int(min(xs)), int(max(xs))
    y0, y1 = int(min(ys)), int(max(ys))
    return x0, y0, x1, y1

def contour_pixels(mask):
    H, W = mask.shape
    contour = []
    for y in range(H):
        for x in range(W):
            if not mask[y, x]:
                continue
            neighbors = [
                (y-1,x),(y+1,x),(y,x-1),(y,x+1),
                (y-1,x-1),(y-1,x+1),(y+1,x-1),(y+1,x+1)
            ]
            for ny, nx in neighbors:
                if ny < 0 or ny >= H or nx < 0 or nx >= W or not mask[ny, nx]:
                    contour.append([float(x), float(y)])
                    break
    return contour

def load_base_plots(annotation_folder):
    base_plots = []
    hsi_folder = os.path.join(annotation_folder, "hsi_cube")
    metadata = None
    for cube_name in os.listdir(hsi_folder):
        cube_folder = os.path.join(hsi_folder, cube_name)
        json_path = os.path.join(annotation_folder, cube_name + ".json")
        if not os.path.exists(json_path):
            continue
        cube, metadata = load_cube(cube_folder)
        rects = load_labelme_rectangles(json_path)
        plots = crop_plots_from_cube(cube, rects)
        base_plots.extend(plots)
    return base_plots, metadata

def load_augmented_extrudes(aug_folder):
    extrudes = []
    for class_name in os.listdir(aug_folder):
        class_folder = os.path.join(aug_folder, class_name)
        if not os.path.isdir(class_folder):
            continue
        for file in os.listdir(class_folder):
            if not file.endswith(".hdr"):
                continue
            hdr_path = os.path.join(class_folder, file)
            cube_envi = envi.open(hdr_path)
            cube = cube_envi.load().astype(np.float32)  # (H, B, W)
            mask = np.any(cube != 0, axis=1)
            polygon = contour_pixels(mask)
            H, B, W = cube.shape
            extrudes.append({
                "cube": cube,
                "polygon": polygon,
                "H": H,
                "W": W
            })
    random.shuffle(extrudes)
    return extrudes

def extract_patch(source_cube, h, w):
    H, B, W = source_cube.shape
    h = min(h, H)
    w = min(w, W)
    y = random.randint(0, H - h) if H > h else 0
    x = random.randint(0, W - w) if W > w else 0
    return source_cube[y:y+h, :, x:x+w]

def offset_polygon(poly, dx, dy):
    return [[p[0] + dx, p[1] + dy] for p in poly]

def save_labelme(file_path, shapes, image_shape):
    data = {
        "version": "5.0.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": Path(file_path).name,
        "imageData": None,
        "imageHeight": image_shape[0],
        "imageWidth": image_shape[2]
    }
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)

def save_envi_cube(file_path, cube, metadata):
    envi.save_image(
        file_path + ".hdr",
        cube.astype(np.float32),
        dtype=np.float32,
        interleave="bil",
        metadata={
            "wavelength": metadata["wavelength"],
            "wavelength units": "nm"
        },
        force=True
    )

def generate_augmented_dataset(annotation_folder, aug_folder, output_folder):
    base_plots, metadata = load_base_plots(annotation_folder)
    extrudes = load_augmented_extrudes(aug_folder)
    hsi_folder = os.path.join(annotation_folder, "hsi_cube")
    os.makedirs(output_folder, exist_ok=True)
    cube_id = 0

    for cube_name in os.listdir(hsi_folder):
        cube_path = os.path.join(hsi_folder, cube_name)
        cube_envi = envi.open(cube_path)
        cube = cube_envi.load().astype(np.float32)
        json_path = os.path.join(annotation_folder, cube_name + ".json")
        polygons = load_labelme_polygons(json_path)
        H, B, W = cube.shape
        shapes = []

        for poly in polygons:
            x0, y0, x1, y1 = polygon_bbox(poly)
            real_H, real_W = y1 - y0, x1 - x0

            candidates = [e for e in extrudes if e["H"] >= real_H and e["W"] >= real_W]
            if candidates:
                ext = random.choice(candidates)
                patch = extract_patch(ext["cube"], real_H, real_W)
                cube[y0:y1, :, x0:x1] = patch
                mask = np.any(patch != 0, axis=1)
                new_poly = contour_pixels(mask)
                new_poly_offset = offset_polygon(new_poly, x0, y0)
                shapes.append({
                    "label": "augmented",
                    "points": new_poly_offset,
                    "group_id": None,
                    "shape_type": "polygon",
                    "flags": {}
                })
            else:
                shapes.append({
                    "label": "original",
                    "points": poly,
                    "group_id": None,
                    "shape_type": "polygon",
                    "flags": {}
                })

        cube_out_path = os.path.join(output_folder, f"cube_{cube_id}")
        save_envi_cube(cube_out_path, cube, metadata)
        save_labelme(os.path.join(output_folder, f"cube_{cube_id}.json"), shapes, cube.shape)
        spectral_ratio_map(cube, metadata)
        cube_id += 1