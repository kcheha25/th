import numpy as np
import cv2
import os
from pathlib import Path
from specarray import SpecArray
from ultralytics import YOLO

DATA_ROOT = Path("data")
MODEL_PATH = "best.pt"
OUT_DIR = Path("output_plots")

WL1 = 900
WL2 = 1100

YOLO_H = 640

OUT_DIR.mkdir(exist_ok=True)


def find_nearest_idx(target, arr):
    return np.argmin(np.abs(arr - target))


def compute_ratio(cube, wavelengths):
    i1 = find_nearest_idx(WL1, wavelengths)
    i2 = find_nearest_idx(WL2, wavelengths)
    b1 = cube[:, i1, :]
    b2 = cube[:, i2, :]
    r = b1 / (b2 + 1e-8)
    r = (r - r.min()) / (r.max() - r.min())
    return r


def prepare_yolo_image(ratio, target_h):
    img = (ratio * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    H, W = ratio.shape
    img = cv2.resize(img, (W, target_h))
    return img


model = YOLO(MODEL_PATH)
plot_id = 0

for cube_dir in DATA_ROOT.iterdir():
    if not cube_dir.is_dir():
        continue

    spec = SpecArray.from_folder(cube_dir)
    cube = np.array(spec.spectral_albedo)
    wavelengths = np.array(spec.wavelengths)
    H, B, W = cube.shape

    ratio = compute_ratio(cube, wavelengths)
    yolo_img = prepare_yolo_image(ratio, YOLO_H)
    results = model.predict(source=yolo_img, imgsz=(YOLO_H, W), conf=0.3, device=0, verbose=False)
    boxes = results[0].boxes
    if boxes is None:
        continue

    scale_y = H / YOLO_H

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        x1 = int(x1)
        x2 = int(x2)
        y1 = int(y1 * scale_y)
        y2 = int(y2 * scale_y)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(W, x2)
        y2 = min(H, y2)
        plot_cube = cube[y1:y2, :, x1:x2]
        out_path = OUT_DIR / f"{cube_dir.name}_plot_{plot_id}.npy"
        np.save(out_path, plot_cube)
        plot_id += 1

print("Done")

import numpy as np
import cv2
from pathlib import Path
from specarray import SpecArray
from ultralytics import YOLO
import spectral.io.envi as envi

DATA_ROOT = Path("data")
MODEL_PATH = "best.pt"
OUT_DIR = Path("output_plots")
WL1 = 900
WL2 = 1100
YOLO_H = 640

OUT_DIR.mkdir(exist_ok=True)

def find_nearest_idx(target, arr):
    return np.argmin(np.abs(arr - target))

def compute_ratio(cube, wavelengths):
    i1 = find_nearest_idx(WL1, wavelengths)
    i2 = find_nearest_idx(WL2, wavelengths)
    b1 = cube[:, i1, :]
    b2 = cube[:, i2, :]
    r = b1 / (b2 + 1e-8)
    r = (r - r.min()) / (r.max() - r.min())
    return r

def prepare_yolo_image(ratio, target_h):
    img = (ratio * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    H, W = ratio.shape
    img = cv2.resize(img, (W, target_h))
    return img

model = YOLO(MODEL_PATH)
plot_id = 0

for cube_dir in DATA_ROOT.iterdir():
    if not cube_dir.is_dir():
        continue

    spec = SpecArray.from_folder(cube_dir)
    cube = np.array(spec.spectral_albedo)
    wavelengths = np.array(spec.wavelengths)
    H, B, W = cube.shape

    ratio = compute_ratio(cube, wavelengths)
    yolo_img = prepare_yolo_image(ratio, YOLO_H)
    results = model.predict(source=yolo_img, imgsz=(YOLO_H, W), conf=0.3, device=0, verbose=False)
    boxes = results[0].boxes
    if boxes is None:
        continue

    scale_y = H / YOLO_H

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        x1 = int(x1)
        x2 = int(x2)
        y1 = int(y1 * scale_y)
        y2 = int(y2 * scale_y)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(W, x2)
        y2 = min(H, y2)

        plot_cube = cube[y1:y2, :, x1:x2]

        out_plot_dir = OUT_DIR / f"{cube_dir.name}_plot_{plot_id}"
        out_plot_dir.mkdir(exist_ok=True)

        hdr_path = out_plot_dir / f"plot_{plot_id}.hdr"
        raw_path = out_plot_dir / f"plot_{plot_id}.raw"

        envi.save_image(
            str(hdr_path),
            plot_cube.transpose(1,0,2),  # ENVI attends (bands, rows, cols)
            interleave='bil',
            dtype=np.float32,
            metadata={'wavelength': list(wavelengths)}
        )
        plot_id += 1


import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

PLOTS_DIR = Path("output_plots")
WL1 = 900
WL2 = 1100
MAX_PLOTS = 10

def find_nearest_idx(target, arr):
    return np.argmin(np.abs(arr - target))

def compute_ratio(cube, wavelengths):
    i1 = find_nearest_idx(WL1, wavelengths)
    i2 = find_nearest_idx(WL2, wavelengths)
    band1 = cube[:, i1, :]
    band2 = cube[:, i2, :]
    ratio = band1 / (band2 + 1e-8)
    ratio_norm = (ratio - ratio.min()) / (ratio.max() - ratio.min())
    return ratio_norm

plot_files = list(PLOTS_DIR.glob("*.npy"))[:MAX_PLOTS]

for plot_path in plot_files:
    cube = np.load(plot_path)
    bands = cube.shape[1]
    wavelengths = np.linspace(400, 1100, bands)
    ratio_map = compute_ratio(cube, wavelengths)
    plt.figure(figsize=(6, 4))
    plt.imshow(ratio_map, cmap='gray', aspect='auto')
    plt.title(f"Ratio {WL1}nm/{WL2}nm - {plot_path.name}")
    plt.axis('off')
    plt.show()
