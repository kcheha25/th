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

plot_files = list(PLOTS_DIR.glob("*/*.hdr"))[:MAX_PLOTS]

for hdr_file in plot_files:
    img = envi.open(str(hdr_file))
    cube = np.array(img.load())  # shape (bands, rows, cols)
    cube = np.transpose(cube, (1,0,2))  # (rows, bands, cols)
    
    if 'wavelength' in img.metadata:
        wavelengths = np.array([float(w) for w in img.metadata['wavelength']])
    else:
        bands = cube.shape[1]
        wavelengths = np.linspace(400, 1100, bands)
    
    ratio_map = compute_ratio(cube, wavelengths)
    plt.figure(figsize=(6,4))
    plt.imshow(ratio_map, cmap='gray', aspect='auto')
    plt.title(f"Ratio {WL1}nm/{WL2}nm - {hdr_file.stem}")
    plt.axis('off')
    plt.show()


import numpy as np
import spectral.io.envi as envi
from pathlib import Path
import cv2
import torch
from torch.utils.data import Dataset, DataLoader

PLOTS_DIR = Path("output_plots")
OUTPUT_ENVI = Path("wgan_input")
OUTPUT_ENVI.mkdir(exist_ok=True)

hdr_files = list(PLOTS_DIR.glob("*/*.hdr"))

H_min, W_min = float('inf'), float('inf')
for hdr_file in hdr_files:
    img = envi.open(str(hdr_file))
    cube = np.array(img.load())
    H_min = min(H_min, cube.shape[1])
    W_min = min(W_min, cube.shape[2])

H_min = int(H_min)
W_min = int(W_min)

def load_and_preprocess_cube(hdr_file, H_target, W_target):
    img = envi.open(str(hdr_file))
    cube = np.array(img.load())
    cube = np.transpose(cube, (1,2,0))
    cube_resized = np.zeros((H_target, W_target, cube.shape[2]), dtype=np.float32)
    for b in range(cube.shape[2]):
        cube_resized[:,:,b] = cv2.resize(cube[:,:,b], (W_target, H_target), interpolation=cv2.INTER_LINEAR)
    cube_resized -= cube_resized.min()
    if cube_resized.max() > 0:
        cube_resized /= cube_resized.max()
    return cube_resized, img.metadata

for i, hdr_file in enumerate(hdr_files):
    cube_proc, metadata = load_and_preprocess_cube(hdr_file, H_min, W_min)
    out_dir = OUTPUT_ENVI / f"plot_{i}"
    out_dir.mkdir(exist_ok=True)
    hdr_path = out_dir / f"plot_{i}.hdr"
    envi.save_image(
        str(hdr_path),
        cube_proc.transpose(2,0,1),
        interleave='bil',
        dtype=np.float32,
        metadata={'wavelength': [float(w) for w in metadata['wavelength']]}
    )

class HSIDataset(Dataset):
    def __init__(self, envi_dir):
        self.dirs = list(Path(envi_dir).glob("plot_*"))
    def __len__(self):
        return len(self.dirs)
    def __getitem__(self, idx):
        hdr_file = list(self.dirs[idx].glob("*.hdr"))[0]
        img = envi.open(str(hdr_file))
        cube = np.array(img.load()).transpose(1,2,0)
        cube = np.transpose(cube, (2,0,1))
        return torch.tensor(cube, dtype=torch.float32)

dataset = HSIDataset(OUTPUT_ENVI)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

import json
import numpy as np
import cv2
from pathlib import Path
from spectral import envi
from specarray import SpecArray

DATA_ROOT = Path("data_cubes")
LABELME_DIR = Path("labelme")
OUTPUT_DIR = Path("extrudes_eroded")
OUTPUT_DIR.mkdir(exist_ok=True)

CLASSES_A_IGNORER = ["trou", "plots"]
json_files = list(LABELME_DIR.glob("*.json"))
class_counter = {}
kernel = np.ones((3,3), np.uint8)

for json_file in json_files:
    with open(json_file) as f:
        data = json.load(f)

    base = json_file.stem
    cube_dir = DATA_ROOT / base
    if not cube_dir.exists():
        continue

    spec = SpecArray.from_folder(cube_dir)
    cube = np.array(spec.spectral_albedo)
    wavelengths = np.array(spec.wavelengths)
    H, B, W = cube.shape

    for shape in data["shapes"]:
        label = shape["label"]
        if label in CLASSES_A_IGNORER:
            continue

        pts = np.array(shape["points"], dtype=np.int32)
        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 1)
        mask_eroded = cv2.erode(mask, kernel, iterations=1)
        if mask_eroded.sum() == 0:
            continue

        ys, xs = np.where(mask_eroded > 0)
        y_min, y_max = ys.min(), ys.max()
        x_min, x_max = xs.min(), xs.max()
        crop = cube[y_min:y_max+1, :, x_min:x_max+1]
        crop_mask = mask_eroded[y_min:y_max+1, x_min:x_max+1]
        crop = crop * crop_mask[:, None, :]
        h_c, b_c, w_c = crop.shape
        print(f"{label}: H={h_c}, B={b_c}, W={w_c}")

        if label not in class_counter:
            class_counter[label] = 0
        idx = class_counter[label]
        class_counter[label] += 1

        out_dir = OUTPUT_DIR / f"{label}_{idx}"
        out_dir.mkdir(exist_ok=True)
        hdr_out = out_dir / f"{label}_{idx}.hdr"
        envi.save_image(
            str(hdr_out),
            crop.transpose(1,0,2),
            interleave="bil",
            dtype=crop.dtype,
            metadata={"wavelength": [float(w) for w in wavelengths]}
        )

print("Extraction terminée (sans trou/plots).")
