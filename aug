import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
from specarray import SpecArray
from ultralytics import YOLO
import cv2

# ============================
# CONFIG
# ============================

DATA_ROOT = Path("data")      # dossier contenant les cubes
MODEL_PATH = "best.pt"        # ton modèle YOLO
OUT_DIR = Path("output_plots")

WL1, WL2 = 900, 1100

OUT_DIR.mkdir(exist_ok=True)


# ============================
# UTILS
# ============================

def find_nearest_idx(target_wl, wavelengths):
    return np.argmin(np.abs(wavelengths - target_wl))


def compute_ratio(cube, wavelengths, wl1, wl2):

    i1 = find_nearest_idx(wl1, wavelengths)
    i2 = find_nearest_idx(wl2, wavelengths)

    band1 = cube[:, i1, :]
    band2 = cube[:, i2, :]

    ratio = band1 / (band2 + 1e-8)

    ratio_norm = (ratio - ratio.min()) / (ratio.max() - ratio.min())

    return ratio_norm


def prepare_yolo_image(ratio):

    img = (ratio * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    return img


# ============================
# LOAD YOLO
# ============================

model = YOLO(MODEL_PATH)


# ============================
# MAIN LOOP
# ============================

plot_id = 0

for cube_dir in DATA_ROOT.iterdir():

    if not cube_dir.is_dir():
        continue

    print(f"Processing: {cube_dir.name}")

    # ----------------------------
    # Load HSI cube
    # ----------------------------

    spec = SpecArray.from_folder(cube_dir)

    cube = np.array(spec.spectral_albedo)      # (y, wl, x)
    wavelengths = np.array(spec.wavelengths)

    H, B, W = cube.shape

    print("Cube shape:", cube.shape)

    # ----------------------------
    # Compute ratio map
    # ----------------------------

    ratio_map = compute_ratio(cube, wavelengths, WL1, WL2)

    # ----------------------------
    # YOLO inference
    # ----------------------------

    yolo_img = prepare_yolo_image(ratio_map)

    results = model.predict(
        source=yolo_img,
        conf=0.3,
        device=0,
        verbose=False
    )

    boxes = results[0].boxes

    if boxes is None:
        print("No plots detected.")
        continue

    # ----------------------------
    # Crop HSI cubes
    # ----------------------------

    for i, box in enumerate(boxes):

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

        # Clip to image size
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(W, x2)
        y2 = min(H, y2)

        plot_cube = cube[y1:y2, :, x1:x2]

        print(f"  Plot {i}: shape {plot_cube.shape}")

        # ----------------------------
        # Save plot
        # ----------------------------

        out_path = OUT_DIR / f"{cube_dir.name}_plot_{plot_id}.npy"

        np.save(out_path, plot_cube)

        plot_id += 1


print("Done.")
