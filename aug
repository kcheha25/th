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
    cube = arr.array
    metadata = {"wavelength": arr.wavelengths}
    return cube, metadata

def find_band_index(metadata, wavelength):
    waves = np.array(metadata["wavelength"], dtype=float)
    return np.argmin(np.abs(waves - wavelength))

def spectral_ratio_map(cube, metadata):
    i1 = find_band_index(metadata, 996)
    i2 = find_band_index(metadata, 1197)
    ratio = cube[:, :, i1] / (cube[:, :, i2] + 1e-6)
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

def crop_plots_from_cube(cube, rectangles):
    plots = []
    for rect in rectangles:
        x1, y1, x2, y2 = rect
        xmin = min(x1, x2)
        xmax = max(x1, x2)
        ymin = min(y1, y2)
        ymax = max(y1, y2)
        plot_cube = cube[ymin:ymax, xmin:xmax]
        plots.append(plot_cube)
    return plots

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

def trim_null_borders(cube):
    mask = np.any(cube != 0, axis=2)
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return cube
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    return cube[y0:y1+1, x0:x1+1]

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

import spectral.io.envi as envi
import numpy as np
import random

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
            cube = cube_envi.load().astype(np.float32)  # cube shape (lines, bands, samples)
            mask = np.any(cube != 0, axis=1)  # bandes = axis 1
            polygon = contour_pixels(mask)
            extrudes.append({
                "cube": cube,
                "class": class_name,
                "polygon": polygon
            })
    random.shuffle(extrudes)
    return extrudes

def extract_patch(source_cube, h, w):
    H, W, _ = source_cube.shape
    if H < h or W < w:
        return None
    y = random.randint(0, H-h)
    x = random.randint(0, W-w)
    return source_cube[y:y+h, x:x+w]

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
        "imageWidth": image_shape[1]
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

def generate_dataset(base_plots, extrude_folder, output_folder, metadata):
    extrudes = load_augmented_extrudes(extrude_folder)
    os.makedirs(output_folder, exist_ok=True)
    extrude_id = 0
    cube_id = 0

    while extrude_id < len(extrudes) and cube_id < TARGET_N:
        base_cube = random.choice(base_plots).copy()
        H, W, B = base_cube.shape
        shapes = []
        n_insert = random.randint(1, 6)

        for _ in range(n_insert):
            if extrude_id >= len(extrudes):
                break
            ext = extrudes[extrude_id]
            ext_cube = ext["cube"]
            polygon = ext["polygon"]
            cls = ext["class"]
            eh, ew, _ = ext_cube.shape
            h = random.randint(int(eh*0.7), eh)
            w = random.randint(int(ew*0.7), ew)
            patch = extract_patch(ext_cube, h, w)
            if patch is None:
                extrude_id += 1
                continue
            y = random.randint(0, H-h)
            x = random.randint(0, W-w)
            base_cube[y:y+h, x:x+w] = patch
            poly = offset_polygon(polygon, x, y)
            shapes.append({
                "label": cls,
                "points": poly,
                "group_id": None,
                "shape_type": "polygon",
                "flags": {}
            })
            extrude_id += 1

        cube_path = os.path.join(output_folder, f"cube_{cube_id}")
        save_envi_cube(cube_path, base_cube, metadata)

        label_file = os.path.join(output_folder, f"cube_{cube_id}.json")
        save_labelme(label_file, shapes, base_cube.shape)

        spectral_ratio_map(base_cube, metadata)

        cube_id += 1

def main():
    annotation_folder = "ann"
    extrudes_folder = "extrudes_augmented"
    output_folder = "generated_dataset"
    base_plots, metadata = load_base_plots(annotation_folder)
    generate_dataset(base_plots, extrudes_folder, output_folder, metadata)

if __name__ == "__main__":
    main()