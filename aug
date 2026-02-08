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

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from spectral import envi
import random

EXTRUDES_DIR = Path("extrudes_eroded")
cube_dirs = list(EXTRUDES_DIR.iterdir())
cube_dirs = [d for d in cube_dirs if d.is_dir()]
random.shuffle(cube_dirs)

for cube_dir in cube_dirs[:10]:
    hdr_file = list(cube_dir.glob("*.hdr"))[0]
    img = envi.open(str(hdr_file))
    cube = np.array(img.load())          # (Bands, H, W)
    cube = np.transpose(cube, (1,2,0))   # (H, W, Bands)
    
    mid_band = cube.shape[2] // 2
    slice_img = cube[:, :, mid_band]
    
    plt.figure(figsize=(4,4))
    plt.imshow(slice_img, cmap='gray')
    plt.title(f"{cube_dir.name} - band {mid_band}")
    plt.axis('off')
    plt.show()

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from spectral import envi
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from spectral import envi
import numpy as np
import cv2

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from spectral import envi
import numpy as np
import cv2

import torch
from torch.utils.data import Dataset
from pathlib import Path
from spectral import envi
import numpy as np
import cv2

class HyperspectralDataset(Dataset):
    def __init__(self, root_dir, target_size=32):
        self.dirs = [d for d in Path(root_dir).iterdir() if d.is_dir()]
        self.H_target = target_size
        self.W_target = target_size

    def __len__(self):
        return len(self.dirs)

    def __getitem__(self, idx):
        hdr_file = list(self.dirs[idx].glob("*.hdr"))[0]
        img = envi.open(str(hdr_file))
        cube = np.array(img.load())          # (Bands, H, W)
        B, H, W = cube.shape
        cube_resized = np.zeros((B, self.H_target, self.W_target), dtype=np.float32)
        for b in range(B):
            cube_resized[b] = cv2.resize(cube[b], (self.W_target, self.H_target), interpolation=cv2.INTER_LINEAR)
        cube_resized = torch.tensor(cube_resized, dtype=torch.float32)
        cube_resized = (cube_resized - cube_resized.min()) / (cube_resized.max() - cube_resized.min() + 1e-8)
        return cube_resized, 0  # dummy label pour DataLoader



BATCH_SIZE = 16
dataset = HyperspectralDataset("extrudes_eroded")
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
channels = next(iter(dataloader))[0].shape[0]  # nombre de bandes

import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable, grad as autograd_grad

class Generator(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.main_module = nn.Sequential(
            nn.ConvTranspose2d(100, 1024, 4, 1, 0),
            nn.BatchNorm2d(1024), nn.ReLU(True),
            nn.ConvTranspose2d(1024, 512, 4, 2, 1),
            nn.BatchNorm2d(512), nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, channels, 4, 2, 1)
        )
        self.output = nn.Tanh()
    def forward(self, x):
        return self.output(self.main_module(x))

class Discriminator(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.main_module = nn.Sequential(
            nn.Conv2d(channels, 256, 4, 2, 1),
            nn.InstanceNorm2d(256, affine=True), nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.InstanceNorm2d(512, affine=True), nn.LeakyReLU(0.2, True),
            nn.Conv2d(512, 1024, 4, 2, 1),
            nn.InstanceNorm2d(1024, affine=True), nn.LeakyReLU(0.2, True)
        )
        self.output = nn.Conv2d(1024, 1, 4, 1, 0)
    def forward(self, x):
        return self.output(self.main_module(x))
    def feature_extraction(self, x):
        return self.main_module(x).view(-1, 1024*4*4)

class WGAN_GP:
    def __init__(self, channels, cuda=False, batch_size=16, generator_iters=1000, critic_iter=5, lambda_term=10):
        self.G = Generator(channels)
        self.D = Discriminator(channels)
        self.C = channels
        self.cuda = cuda
        self.batch_size = batch_size
        self.generator_iters = generator_iters
        self.critic_iter = critic_iter
        self.lambda_term = lambda_term
        self.d_optimizer = optim.Adam(self.D.parameters(), lr=1e-4, betas=(0.5,0.999))
        self.g_optimizer = optim.Adam(self.G.parameters(), lr=1e-4, betas=(0.5,0.999))
        if cuda:
            self.G.cuda(); self.D.cuda()
    def get_var(self, x):
        return x.cuda() if self.cuda else x
    def gradient_penalty(self, real, fake):
        eta = torch.rand(real.size(0),1,1,1)
        eta = eta.expand(real.size())
        eta = self.get_var(eta)
        interpolated = eta * real + (1-eta)*fake
        interpolated = Variable(interpolated, requires_grad=True)
        prob_interpolated = self.D(interpolated)
        gradients = autograd_grad(outputs=prob_interpolated, inputs=interpolated,
                                  grad_outputs=torch.ones_like(prob_interpolated),
                                  create_graph=True, retain_graph=True)[0]
        gradients = gradients.view(gradients.size(0), -1)
        return ((gradients.norm(2,dim=1)-1)**2).mean()*self.lambda_term
    def train(self, loader):
        data_iter = iter(loader)
        one = self.get_var(torch.tensor(1., dtype=torch.float))
        mone = -one
        for g_iter in range(self.generator_iters):
            for p in self.D.parameters(): p.requires_grad=True
            for d_iter in range(self.critic_iter):
                self.D.zero_grad()
                try: real_imgs = next(data_iter)[0]
                except StopIteration: data_iter = iter(loader); real_imgs = next(data_iter)[0]
                if real_imgs.size(0)!=self.batch_size: continue
                z = self.get_var(torch.randn(self.batch_size,100,1,1))
                real_imgs = self.get_var(real_imgs)
                d_loss_real = self.D(real_imgs).mean(); d_loss_real.backward(mone)
                fake_imgs = self.G(z); d_loss_fake = self.D(fake_imgs).mean(); d_loss_fake.backward(one)
                gp = self.gradient_penalty(real_imgs.data, fake_imgs.data); gp.backward()
                self.d_optimizer.step()
            for p in self.D.parameters(): p.requires_grad=False
            self.G.zero_grad()
            z = self.get_var(torch.randn(self.batch_size,100,1,1))
            fake_imgs = self.G(z)
            g_loss = self.D(fake_imgs).mean(); g_loss.backward(mone); self.g_optimizer.step()
            print(f'Iter {g_iter}: D loss real {d_loss_real.item():.4f} fake {d_loss_fake.item():.4f}, G loss {g_loss.item():.4f}')

wgan = WGAN_GP(
    channels=channels,
    cuda=torch.cuda.is_available(),
    batch_size=16,
    generator_iters=1000,  # nombre total d’itérations du générateur
    critic_iter=5,
    lambda_term=10
)
wgan.train(dataloader)


import os
from torchvision import utils

class WGAN_GP(WGAN_GP):  # hérite de ta classe existante
    def save_model(self, g_path='./generator.pkl', d_path='./discriminator.pkl'):
        torch.save(self.G.state_dict(), g_path)
        torch.save(self.D.state_dict(), d_path)
        print(f'Models saved: {g_path}, {d_path}')

    def generate_and_save_images(self, n_samples=64, save_dir='generated_images', iter_num=0):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        z = self.get_var(torch.randn(n_samples, 100, 1, 1))
        samples = self.G(z).detach().cpu()
        samples = (samples + 1)/2  # rescale [-1,1] -> [0,1]
        grid = utils.make_grid(samples, nrow=8)
        utils.save_image(grid, os.path.join(save_dir, f'iter_{iter_num:04d}.png'))
        print(f'Saved generated images at iteration {iter_num}')

if g_iter % 50 == 0:  # toutes les 50 itérations
    self.generate_and_save_images(n_samples=64, iter_num=g_iter)
    self.save_model()


import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, channels, z_dim=256):
        super().__init__()
        self.main_module = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 2048, 4, 1, 0),
            nn.BatchNorm2d(2048), nn.ReLU(True),
            nn.ConvTranspose2d(2048, 1024, 4, 2, 1),
            nn.BatchNorm2d(1024), nn.ReLU(True),
            nn.ConvTranspose2d(1024, 512, 4, 2, 1),
            nn.BatchNorm2d(512), nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, channels, 4, 2, 1)
        )
        self.output = nn.Tanh()

    def forward(self, x):
        return self.output(self.main_module(x))


class Discriminator(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.main_module = nn.Sequential(
            nn.Conv2d(channels, 256, 4, 2, 1),
            nn.InstanceNorm2d(256, affine=True), nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.InstanceNorm2d(512, affine=True), nn.LeakyReLU(0.2, True),
            nn.Conv2d(512, 1024, 4, 2, 1),
            nn.InstanceNorm2d(1024, affine=True), nn.LeakyReLU(0.2, True),
            nn.Conv2d(1024, 2048, 4, 2, 1),
            nn.InstanceNorm2d(2048, affine=True), nn.LeakyReLU(0.2, True)
        )
        self.output = nn.Conv2d(2048, 1, 4, 1, 0)

    def forward(self, x):
        return self.output(self.main_module(x))

    def feature_extraction(self, x):
        return self.main_module(x).view(-1, 2048*2*2)  # flatten selon nouvelle couche


import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr

# 1️⃣ Générer des cubes avec le WGAN
def generate_augmented_cubes(wgan_model, num_samples=10, z_dim=256):
    wgan_model.G.eval()
    z = torch.randn(num_samples, z_dim, 1, 1)
    z = z.cuda() if wgan_model.cuda else z
    with torch.no_grad():
        cubes = wgan_model.G(z).cpu().numpy()  # (N, C, H, W)
    return cubes

# 2️⃣ Spectral Angle Mapper (SAM)
def compute_sam(real_cube, fake_cube):
    bands, H, W = real_cube.shape
    sam_map = np.zeros((H,W))
    for i in range(H):
        for j in range(W):
            r = real_cube[:,i,j]
            f = fake_cube[:,i,j]
            cos_angle = np.dot(r,f) / (np.linalg.norm(r)*np.linalg.norm(f)+1e-8)
            cos_angle = np.clip(cos_angle, -1, 1)
            sam_map[i,j] = np.arccos(cos_angle)
    return sam_map.mean()

# 3️⃣ RMSE spectral
def compute_rmse(real_cube, fake_cube):
    return np.sqrt(np.mean((real_cube - fake_cube)**2))

# 4️⃣ Corrélation spectrale
def compute_spectral_corr(real_cube, fake_cube):
    bands, H, W = real_cube.shape
    corr_list = []
    for i in range(H):
        for j in range(W):
            corr, _ = pearsonr(real_cube[:,i,j], fake_cube[:,i,j])
            corr_list.append(corr)
    return np.mean(corr_list)

# 5️⃣ PCA / t-SNE visualization
def visualize_pca_tsne(real_cubes, fake_cubes, n_components=3):
    # Flatten HxW into samples
    N_real = len(real_cubes)
    N_fake = len(fake_cubes)
    bands, H, W = real_cubes[0].shape
    X_real = np.array([cube.reshape(bands, -1).T for cube in real_cubes])
    X_real = X_real.reshape(N_real*H*W, bands)
    X_fake = np.array([cube.reshape(bands, -1).T for cube in fake_cubes])
    X_fake = X_fake.reshape(N_fake*H*W, bands)
    
    X = np.vstack([X_real, X_fake])
    y = np.array([0]*(N_real*H*W) + [1]*(N_fake*H*W))

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_tsne = tsne.fit_transform(X)

    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.scatter(X_pca[:,0], X_pca[:,1], c=y, s=1, cmap='coolwarm')
    plt.title("PCA Visualization")
    plt.subplot(1,2,2)
    plt.scatter(X_tsne[:,0], X_tsne[:,1], c=y, s=1, cmap='coolwarm')
    plt.title("t-SNE Visualization")
    plt.show()

# 6️⃣ Pipeline pour un plot
def evaluate_plot(wgan_model, real_cubes):
    fake_cubes = generate_augmented_cubes(wgan_model, num_samples=len(real_cubes))
    sam_scores = [compute_sam(r,f) for r,f in zip(real_cubes, fake_cubes)]
    rmse_scores = [compute_rmse(r,f) for r,f in zip(real_cubes, fake_cubes)]
    corr_scores = [compute_spectral_corr(r,f) for r,f in zip(real_cubes, fake_cubes)]
    print(f"SAM mean: {np.mean(sam_scores):.4f} rad")
    print(f"RMSE mean: {np.mean(rmse_scores):.4f}")
    print(f"Spectral correlation mean: {np.mean(corr_scores):.4f}")
    visualize_pca_tsne(real_cubes, fake_cubes)
    return sam_scores, rmse_scores, corr_scores

# real_cubes : liste des cubes réels pour un plot (shape B,H,W)
# wgan : ton modèle WGAN_GP déjà entraîné

sam_scores, rmse_scores, corr_scores = evaluate_plot(wgan, real_cubes)


import torch
import torch.nn as nn
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Generator et Discriminator identiques à l'entraînement
class Generator(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.main_module = nn.Sequential(
            nn.ConvTranspose2d(100, 1024, 4, 1, 0),
            nn.BatchNorm2d(1024), nn.ReLU(True),
            nn.ConvTranspose2d(1024, 512, 4, 2, 1),
            nn.BatchNorm2d(512), nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, channels, 4, 2, 1)
        )
        self.output = nn.Tanh()
    def forward(self, x):
        return self.output(self.main_module(x))

class Discriminator(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.main_module = nn.Sequential(
            nn.Conv2d(channels, 256, 4, 2, 1),
            nn.InstanceNorm2d(256, affine=True), nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.InstanceNorm2d(512, affine=True), nn.LeakyReLU(0.2, True),
            nn.Conv2d(512, 1024, 4, 2, 1),
            nn.InstanceNorm2d(1024, affine=True), nn.LeakyReLU(0.2, True)
        )
        self.output = nn.Conv2d(1024, 1, 4, 1, 0)
    def forward(self, x):
        return self.output(self.main_module(x))

# WGAN wrapper pour charger et générer
class WGAN_GP:
    def __init__(self, channels, cuda=False):
        self.G = Generator(channels)
        self.D = Discriminator(channels)
        self.C = channels
        self.cuda = cuda
        if cuda:
            self.G.cuda()
            self.D.cuda()
    def load_weights(self, g_path="generator.pkl", d_path="discriminator.pkl"):
        self.G.load_state_dict(torch.load(g_path, map_location='cuda' if self.cuda else 'cpu'))
        self.D.load_state_dict(torch.load(d_path, map_location='cuda' if self.cuda else 'cpu'))
        self.G.eval()
        self.D.eval()
    def generate(self, num_samples=10, z_dim=100):
        z = torch.randn(num_samples, z_dim, 1, 1)
        z = z.cuda() if self.cuda else z
        with torch.no_grad():
            cubes = self.G(z).cpu().numpy()
        return cubes

# Fonctions d’évaluation
def compute_sam(real_cube, fake_cube):
    bands, H, W = real_cube.shape
    sam_map = np.zeros((H,W))
    for i in range(H):
        for j in range(W):
            r = real_cube[:,i,j]
            f = fake_cube[:,i,j]
            cos_angle = np.dot(r,f)/(np.linalg.norm(r)*np.linalg.norm(f)+1e-8)
            cos_angle = np.clip(cos_angle, -1, 1)
            sam_map[i,j] = np.arccos(cos_angle)
    return sam_map.mean()

def compute_rmse(real_cube, fake_cube):
    return np.sqrt(np.mean((real_cube-fake_cube)**2))

def compute_spectral_corr(real_cube, fake_cube):
    bands, H, W = real_cube.shape
    corr_list = []
    for i in range(H):
        for j in range(W):
            corr,_ = pearsonr(real_cube[:,i,j], fake_cube[:,i,j])
            corr_list.append(corr)
    return np.mean(corr_list)

def visualize_pca_tsne(real_cubes, fake_cubes, n_components=3):
    N_real = len(real_cubes)
    N_fake = len(fake_cubes)
    bands, H, W = real_cubes[0].shape
    X_real = np.array([cube.reshape(bands,-1).T for cube in real_cubes]).reshape(N_real*H*W,bands)
    X_fake = np.array([cube.reshape(bands,-1).T for cube in fake_cubes]).reshape(N_fake*H*W,bands)
    X = np.vstack([X_real, X_fake])
    y = np.array([0]*(N_real*H*W)+[1]*(N_fake*H*W))
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_tsne = tsne.fit_transform(X)
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.scatter(X_pca[:,0], X_pca[:,1], c=y, s=1, cmap='coolwarm')
    plt.title("PCA Visualization")
    plt.subplot(1,2,2)
    plt.scatter(X_tsne[:,0], X_tsne[:,1], c=y, s=1, cmap='coolwarm')
    plt.title("t-SNE Visualization")
    plt.show()

def evaluate_plot(wgan_model, real_cubes):
    fake_cubes = wgan_model.generate(num_samples=len(real_cubes))
    sam_scores = [compute_sam(r,f) for r,f in zip(real_cubes, fake_cubes)]
    rmse_scores = [compute_rmse(r,f) for r,f in zip(real_cubes, fake_cubes)]
    corr_scores = [compute_spectral_corr(r,f) for r,f in zip(real_cubes, fake_cubes)]
    print(f"SAM mean: {np.mean(sam_scores):.4f} rad")
    print(f"RMSE mean: {np.mean(rmse_scores):.4f}")
    print(f"Spectral correlation mean: {np.mean(corr_scores):.4f}")
    visualize_pca_tsne(real_cubes, fake_cubes)
    return sam_scores, rmse_scores, corr_scores

# ----------------------------
# Exemple d’utilisation
channels = 128  # nombre de bandes
wgan = WGAN_GP(channels=channels, cuda=torch.cuda.is_available())
wgan.load_weights("generator.pkl", "discriminator.pkl")

# real_cubes = liste de cubes hyperspectraux à comparer (N, C, H, W)
# sam_scores, rmse_scores, corr_scores = evaluate_plot(wgan, real_cubes)

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from pathlib import Path
import numpy as np
import cv2
import spectral.io.envi as envi
from collections import defaultdict


class SpectralPixelDataset(Dataset):

    def __init__(self, root_dir, target_size=32):

        self.root_dir = Path(root_dir)
        self.target_size = target_size

        self.samples = []

        self.class_map = {}
        self.class_names = []

        class_counter = 0

        hdr_files = list(self.root_dir.rglob("*.hdr"))

        for hdr_file in sorted(hdr_files):

            name = hdr_file.stem
            prefix = name.split("_")[0]

            if prefix not in self.class_map:
                self.class_map[prefix] = class_counter
                self.class_names.append(prefix)
                class_counter += 1

            class_id = self.class_map[prefix]

            img = envi.open(str(hdr_file))
            cube = np.array(img.load())

            B, H, W = cube.shape

            cube_resized = np.zeros(
                (B, self.target_size, self.target_size),
                dtype=np.float32
            )

            for b in range(B):
                cube_resized[b] = cv2.resize(
                    cube[b],
                    (self.target_size, self.target_size),
                    interpolation=cv2.INTER_LINEAR
                )

            cube_resized = (cube_resized - cube_resized.min()) / \
                           (cube_resized.max() - cube_resized.min() + 1e-8)

            spectra = cube_resized.reshape(B, -1).T

            for s in spectra:
                self.samples.append((
                    torch.tensor(s, dtype=torch.float32),
                    class_id
                ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def create_class_dataloaders(dataset, batch_size=128):

    class_indices = defaultdict(list)

    for i, (_, label) in enumerate(dataset):
        class_indices[label].append(i)

    loaders = {}

    for label, indices in class_indices.items():

        subset = Subset(dataset, indices)

        loaders[label] = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True
        )

    return loaders


if __name__ == "__main__":

    dataset = SpectralPixelDataset("extrudes_eroded", target_size=32)

    loaders = create_class_dataloaders(dataset, batch_size=128)

    print(dataset.class_map)
    print(dataset.class_names)

    class_id = dataset.class_map[dataset.class_names[0]]

    for spectra, labels in loaders[class_id]:
        print(spectra.shape)
        print(labels.shape)
        break


import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import matplotlib.pyplot as plt
import numpy as np

# -------------------------------
# Hyperparameters
# -------------------------------
nz = 100
n_critic = 5
p_coeff = 10
lr = 1e-4
batch_size = 16
epoch_num = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_bands = 272

# -------------------------------
# Modèles (Generator / Discriminator)
# -------------------------------
class Discriminator(nn.Module):
    def __init__(self, n_bands):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv1d(1, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(512, 1, kernel_size=17, stride=1, padding=0, bias=False)
        )

    def forward(self, x):
        return self.main(x).view(-1)


class Generator(nn.Module):
    def __init__(self, nz, n_bands):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose1d(nz, 512, 17, 1, 0, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.ConvTranspose1d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.ConvTranspose1d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.ConvTranspose1d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.ConvTranspose1d(64, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# -------------------------------
# Métriques de cohérence
# -------------------------------
def compute_metrics(real, fake):
    """
    real, fake : (batch, 1, n_bands)
    Retourne dictionnaire : {'MSE':..., 'Pearson':..., 'SAM':...}
    """
    real_np = real.detach().cpu().numpy().reshape(real.size(0), -1)
    fake_np = fake.detach().cpu().numpy().reshape(fake.size(0), -1)

    # MSE
    mse = np.mean((real_np - fake_np) ** 2)

    # Pearson correlation moyenne
    pearson = np.mean([np.corrcoef(real_np[i], fake_np[i])[0,1] for i in range(real_np.shape[0])])

    # SAM : angle spectral moyenne
    sam = np.mean([
        np.arccos(np.clip(np.dot(real_np[i], fake_np[i]) /
                          (np.linalg.norm(real_np[i]) * np.linalg.norm(fake_np[i]) + 1e-8), -1, 1))
        for i in range(real_np.shape[0])
    ])

    return {'MSE': mse, 'Pearson': pearson, 'SAM': sam}


# -------------------------------
# Entraînement pour une seule classe
# -------------------------------
def train_single_class(loader):
    netD = Discriminator(n_bands).to(device)
    netG = Generator(nz, n_bands).to(device)
    netD.apply(weights_init)
    netG.apply(weights_init)

    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0, 0.9))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0, 0.9))

    fixed_noise = torch.randn(16, nz, 1, device=device)

    for epoch in range(epoch_num):
        epoch_metrics = {'MSE': [], 'Pearson': [], 'SAM': []}

        for i, data in enumerate(loader):
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)

            # --------------------
            # Discriminator update
            # --------------------
            for _ in range(n_critic):
                netD.zero_grad()
                noise = torch.randn(b_size, nz, 1, device=device)
                fake = netG(noise)

                # Gradient penalty
                eps = torch.rand(b_size, 1, 1, device=device)
                x_p = eps * real_cpu + (1 - eps) * fake
                x_p.requires_grad_(True)
                grad = autograd.grad(
                    outputs=netD(x_p).sum(),
                    inputs=x_p,
                    create_graph=True,
                    retain_graph=True
                )[0]
                grad_norm = grad.view(b_size, -1).norm(2, dim=1)
                grad_penalty = p_coeff * ((grad_norm - 1) ** 2).mean()

                loss_D = torch.mean(netD(fake)) - torch.mean(netD(real_cpu)) + grad_penalty
                loss_D.backward()
                optimizerD.step()

            # --------------------
            # Generator update
            # --------------------
            netG.zero_grad()
            noise = torch.randn(b_size, nz, 1, device=device)
            fake = netG(noise)
            loss_G = -torch.mean(netD(fake))
            loss_G.backward()
            optimizerG.step()

            # --------------------
            # Calcul des métriques pour ce batch
            # --------------------
            metrics = compute_metrics(real_cpu, fake)
            for k in epoch_metrics:
                epoch_metrics[k].append(metrics[k])

        # --------------------
        # Moyenne des métriques de l'epoch
        # --------------------
        epoch_avg_metrics = {k: np.mean(v) for k,v in epoch_metrics.items()}
        print(f"Epoch {epoch} | Loss_D: {loss_D.item():.4f} | Loss_G: {loss_G.item():.4f} | "
              f"MSE: {epoch_avg_metrics['MSE']:.4f} | Pearson: {epoch_avg_metrics['Pearson']:.4f} | SAM: {epoch_avg_metrics['SAM']:.4f}")

        # Visualisation simple
        with torch.no_grad():
            fake = netG(fixed_noise).cpu()
            plt.figure(figsize=(8,4))
            for k in range(16):
                plt.plot(fake[k,0])
            plt.title(f"Epoch {epoch}")
            plt.show()

    return netG, netD


if __name__ == "__main__":
    from preprocessing import create_class_dataloaders, SpectralPixelDataset

    dataset = SpectralPixelDataset("extrudes_eroded", target_size=32)
    loaders = create_class_dataloaders(dataset, batch_size=batch_size)

    # Choisir une classe, par exemple la première
    class_id = 0
    loader = loaders[class_id]

    netG, netD = train_single_class(loader)


with torch.no_grad():
    fake = netG(fixed_noise).cpu()  # shape (16,1,n_bands)
    real_batch = next(iter(loader))[0][:16].cpu()  # récupère les 16 premiers spectres réels du loader
    real_batch = real_batch.unsqueeze(1) if real_batch.dim() == 2 else real_batch  # (16,1,n_bands)

    plt.figure(figsize=(12,5))
    for k in range(16):
        plt.plot(real_batch[k,0], color='blue', alpha=0.5, label='Real' if k==0 else "")
        plt.plot(fake[k,0], color='red', alpha=0.5, label='Fake' if k==0 else "")
    plt.title(f"Epoch {epoch}")
    plt.xlabel("Band")
    plt.ylabel("Reflectance")
    plt.legend()
    plt.show()

pearson = np.mean([
    np.corrcoef(real_np[i], fake_np[i])[0,1] 
    if np.std(real_np[i])>1e-8 and np.std(fake_np[i])>1e-8 
    else 0.0
    for i in range(real_np.shape[0])
])

# Vérifier les spectres nuls ou quasi-constants par classe
threshold_variance = 1e-6  # seuil pour considérer un spectre comme nul

for class_id, loader in loaders.items():
    null_count = 0
    total_count = 0
    
    for spectra, labels in loader:
        # spectra shape: (batch_size, n_bands)
        batch_size = spectra.size(0)
        for i in range(batch_size):
            spectrum = spectra[i]
            if torch.var(spectrum) < threshold_variance:
                null_count += 1
            total_count += 1
    
    print(f"Classe {dataset.class_names[class_id]} : {null_count}/{total_count} spectres nuls ou quasi-constants")

import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import spectral.io.envi as envi
from collections import defaultdict

class SpectralPixelDatasetOriginal(Dataset):
    """
    Dataset qui conserve pour chaque spectre ses coordonnées spatiales (h, w) dans le cube original, sans resize.
    """
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.samples = []  # list of tuples: (spectrum tensor, class_id, (h, w))
        self.class_map = {}
        self.class_names = []

        class_counter = 0
        hdr_files = list(self.root_dir.rglob("*.hdr"))

        for hdr_file in sorted(hdr_files):
            name = hdr_file.stem
            prefix = name.split("_")[0]

            if prefix not in self.class_map:
                self.class_map[prefix] = class_counter
                self.class_names.append(prefix)
                class_counter += 1

            class_id = self.class_map[prefix]

            img = envi.open(str(hdr_file))
            cube = np.array(img.load())  # shape (B, H, W)
            B, H, W = cube.shape

            # Normalisation par cube
            cube = (cube - cube.min()) / (cube.max() - cube.min() + 1e-8)

            for h in range(H):
                for w in range(W):
                    spectrum = cube[:, h, w]
                    self.samples.append((
                        torch.tensor(spectrum, dtype=torch.float32),
                        class_id,
                        (h, w)
                    ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]  # returns (spectrum, class_id, (h, w))


# --------------------------
# Vérification des spectres nuls
# --------------------------
def check_null_spectra(dataset, threshold=1e-6):
    """
    Retourne un dictionnaire par classe avec les indices et positions des spectres quasi nuls.
    """
    null_positions = defaultdict(list)

    for idx, (spectrum, class_id, (h, w)) in enumerate(dataset):
        if torch.var(spectrum) < threshold:
            null_positions[class_id].append((idx, h, w))

    # Affichage
    for class_id, positions in null_positions.items():
        print(f"Classe {dataset.class_names[class_id]} : {len(positions)} spectres nuls")
        if len(positions) > 0:
            print("Exemples (index, h, w) :", positions[:10])  # montre les 10 premiers

    return null_positions


if __name__ == "__main__":
    dataset = SpectralPixelDatasetOriginal("extrudes_eroded")
    null_positions = check_null_spectra(dataset)



import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from torch.utils.data import Dataset, DataLoader, Subset
from pathlib import Path
import numpy as np
import spectral.io.envi as envi
from collections import defaultdict
import matplotlib.pyplot as plt

# -------------------------------
# Dataset
# -------------------------------
class SpectralPixelDataset(Dataset):
    def __init__(self, root_dir, null_threshold=1e-6):
        self.root_dir = Path(root_dir)
        self.null_threshold = null_threshold
        self.samples = []
        self.class_map = {}
        self.class_names = []
        class_counter = 0
        hdr_files = list(self.root_dir.rglob("*.hdr"))

        for hdr_file in sorted(hdr_files):
            name = hdr_file.stem
            prefix = name.split("_")[0]
            if prefix not in self.class_map:
                self.class_map[prefix] = class_counter
                self.class_names.append(prefix)
                class_counter += 1
            class_id = self.class_map[prefix]
            img = envi.open(str(hdr_file))
            cube = np.array(img.load(), dtype=np.float32)
            B, H, W = cube.shape
            spectra = cube.reshape(B, -1).T
            for s in spectra:
                if np.var(s) > self.null_threshold:
                    s_norm = 2 * (s - s.min()) / (s.max() - s.min() + 1e-8) - 1
                    self.samples.append((torch.tensor(s_norm, dtype=torch.float32), class_id))

        print(f"Dataset chargé : {len(self.samples)} spectres non nuls")
        print(f"Classes : {self.class_names}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def create_class_dataloaders(dataset, batch_size=128):
    class_indices = defaultdict(list)
    for i, (_, label) in enumerate(dataset):
        class_indices[label].append(i)
    loaders = {}
    for label, indices in class_indices.items():
        subset = Subset(dataset, indices)
        loaders[label] = DataLoader(subset, batch_size=batch_size, shuffle=True, drop_last=True)
    return loaders


import os

def train_single_class(loader, class_name="class0"):
    netD = Discriminator(n_bands).to(device)
    netG = Generator(nz, n_bands).to(device)
    netD.apply(weights_init)
    netG.apply(weights_init)

    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0, 0.9))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0, 0.9))

    # fixed noise pour générer toujours les mêmes fakes
    fixed_noise = torch.randn(16, nz, 1, device=device)

    # selectionner les 16 premiers spectres réels fixes du loader
    real_batch_iter = iter(loader)
    real_fixed = next(real_batch_iter)[0][:16].to(device)
    if real_fixed.dim() == 2:
        real_fixed = real_fixed.unsqueeze(1)  # (16,1,n_bands)

    os.makedirs(f"./results/{class_name}", exist_ok=True)
    best_sam = float('inf')
    best_netG = None
    best_netD = None

    for epoch in range(epoch_num):
        epoch_metrics = {'MSE': [], 'Pearson': [], 'SAM': []}

        for i, data in enumerate(loader):
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)

            for _ in range(n_critic):
                netD.zero_grad()
                noise = torch.randn(b_size, nz, 1, device=device)
                fake = netG(noise)

                eps = torch.rand(b_size, 1, 1, device=device)
                x_p = eps * real_cpu + (1 - eps) * fake
                x_p.requires_grad_(True)
                grad = autograd.grad(
                    outputs=netD(x_p).sum(),
                    inputs=x_p,
                    create_graph=True,
                    retain_graph=True
                )[0]
                grad_norm = grad.view(b_size, -1).norm(2, dim=1)
                grad_penalty = p_coeff * ((grad_norm - 1) ** 2).mean()

                loss_D = torch.mean(netD(fake)) - torch.mean(netD(real_cpu)) + grad_penalty
                loss_D.backward()
                optimizerD.step()

            netG.zero_grad()
            noise = torch.randn(b_size, nz, 1, device=device)
            fake = netG(noise)
            loss_G = -torch.mean(netD(fake))
            loss_G.backward()
            optimizerG.step()

            metrics = compute_metrics(real_cpu, fake)
            for k in epoch_metrics:
                epoch_metrics[k].append(metrics[k])

        epoch_avg_metrics = {k: np.mean(v) for k,v in epoch_metrics.items()}
        print(f"Epoch {epoch} | Loss_D: {loss_D.item():.4f} | Loss_G: {loss_G.item():.4f} | "
              f"MSE: {epoch_avg_metrics['MSE']:.4f} | Pearson: {epoch_avg_metrics['Pearson']:.4f} | SAM: {epoch_avg_metrics['SAM']:.4f}")

        # -------------------- Visualisation --------------------
        with torch.no_grad():
            fake_fixed = netG(fixed_noise).cpu()
            plt.figure(figsize=(12,5))
            for k in range(16):
                plt.plot(real_fixed[k,0].cpu(), color='blue', alpha=0.5, label='Real' if k==0 else "")
                plt.plot(fake_fixed[k,0].cpu(), color='red', alpha=0.5, label='Fake' if k==0 else "")
            plt.title(f"Epoch {epoch}")
            plt.xlabel("Band")
            plt.ylabel("Reflectance")
            plt.legend()
            plt.savefig(f"./results/{class_name}/epoch_{epoch:03d}.png")
            plt.close()

        # -------------------- Sauvegarde modèles tous les 50 epochs --------------------
        if (epoch+1) % 50 == 0:
            torch.save(netG.state_dict(), f"./results/{class_name}/netG_epoch_{epoch:03d}.pth")
            torch.save(netD.state_dict(), f"./results/{class_name}/netD_epoch_{epoch:03d}.pth")

        # -------------------- Sauvegarde du meilleur modèle (SAM minimal) --------------------
        if epoch_avg_metrics['SAM'] < best_sam:
            best_sam = epoch_avg_metrics['SAM']
            best_netG = netG.state_dict()
            best_netD = netD.state_dict()

    # sauvegarde final du meilleur modèle
    torch.save(best_netG, f"./results/{class_name}/netG_best.pth")
    torch.save(best_netD, f"./results/{class_name}/netD_best.pth")

    return netG, netD

class SpectralLoss1D(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.l1 = nn.L1Loss()

    def forward(self, real, fake):
        # real, fake : (B,1,N)

        real_f = torch.fft.rfft(real, dim=2)
        fake_f = torch.fft.rfft(fake, dim=2)

        real_mag = torch.abs(real_f) + self.eps
        fake_mag = torch.abs(fake_f) + self.eps

        real_log = torch.log(real_mag)
        fake_log = torch.log(fake_mag)

        return self.l1(fake_log, real_log)

def train_single_class(loader, class_name="class0"):

    netD = Discriminator(n_bands).to(device)
    netG = Generator(nz, n_bands).to(device)

    netD.apply(weights_init)
    netG.apply(weights_init)

    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0, 0.9))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0, 0.9))

    spectral_loss = SpectralLoss1D().to(device)
    mse_loss = nn.MSELoss()

adv_loss = -torch.mean(netD(fake))
mse = mse_loss(fake, real_cpu)
spec = spectral_loss(real_cpu, fake)

loss_G = adv_loss + 1.0*mse + 0.3*spec

def smooth_loss(x):
    return torch.mean(torch.abs(x[:,:,1:] - x[:,:,:-1]))

def deriv_loss(real, fake):
    d_real = real[:,:,1:] - real[:,:,:-1]
    d_fake = fake[:,:,1:] - fake[:,:,:-1]
    return torch.mean(torch.abs(d_real - d_fake))

def freq_smooth_loss(x, cutoff=0.3):
    fft = torch.fft.rfft(x, dim=2)
    freqs = torch.linspace(0,1,fft.shape[2],device=x.device)

    mask = (freqs > cutoff).float()
    return torch.mean(torch.abs(fft)*mask)

def curvature_loss(x):
    return torch.mean(
        torch.abs(x[:,:,2:] - 2*x[:,:,1:-1] + x[:,:,:-2])
    )

smooth = tv_loss_1d(fake)
curve = curvature_loss(fake)
spec = spectral_loss(real, fake)
mse = F.mse_loss(fake, real)

loss_G = (
    adv
    + 40*mse
    + 10*spec
    + 5*smooth
    + 2*curve
)

from collections import defaultdict
from imblearn.over_sampling import SMOTE

class SpectralPixelDataset(Dataset):
    def __init__(self, root_dir, null_threshold=1e-6, min_spectra=400, smote_k=5, shift_level=0.01):
        self.root_dir = Path(root_dir)
        self.null_threshold = null_threshold
        self.samples = []
        self.class_map = {}
        self.class_names = []
        class_counter = 0
        hdr_files = list(self.root_dir.rglob("*.hdr"))

        class_spectra = defaultdict(list)

        # Chargement des spectres
        for hdr_file in sorted(hdr_files):
            name = hdr_file.stem
            prefix = name.split("_")[0]
            if prefix not in self.class_map:
                self.class_map[prefix] = class_counter
                self.class_names.append(prefix)
                class_counter += 1
            class_id = self.class_map[prefix]

            img = envi.open(str(hdr_file))
            cube = np.array(img.load(), dtype=np.float32)
            B, H, W = cube.shape
            spectra = cube.reshape(B, -1).T
            for s in spectra:
                if np.var(s) > self.null_threshold:
                    s_norm = 2 * (s - s.min()) / (s.max() - s.min() + 1e-8) - 1
                    class_spectra[class_id].append(s_norm)

        # Comptage des spectres par classe
        counts = {c: len(class_spectra[c]) for c in class_spectra}

        # Pour chaque classe
        for class_id, spect_list in class_spectra.items():
            X_class = np.stack(spect_list)
            y_class = np.full(len(spect_list), class_id)

            # Augmenter seulement les classes < min_spectra
            if counts[class_id] < min_spectra:
                n_needed = min_spectra - counts[class_id]
                smote = SMOTE(sampling_strategy={class_id: min_spectra},
                              k_neighbors=min(smote_k, counts[class_id]-1),
                              random_state=42)
                X_res, y_res = smote.fit_resample(X_class, y_class)

                # Appliquer un petit décalage uniquement sur les spectres générés par SMOTE
                X_new = X_res[len(X_class):]  # spectres créés par SMOTE
                X_new += np.random.uniform(-shift_level, shift_level, X_new.shape)
                X_res[len(X_class):] = np.clip(X_new, -1, 1)  # clip [-1,1]
            else:
                X_res, y_res = X_class, y_class  # classes avec >= min_spectra inchangées

            # Ajouter les spectres finaux au dataset
            for i in range(len(X_res)):
                self.samples.append((torch.tensor(X_res[i], dtype=torch.float32),
                                     int(y_res[i])))

        print(f"Dataset chargé : {len(self.samples)} spectres après augmentation")
        print(f"Classes : {self.class_names}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
    

    scp mon_env.tar.gz user@serveur:/chemin/de/destination

cd ~/miniforge3/envs/mon_env
mkdir test2
cd test2
tar -xzf ~/kc_env.tar.gz -C ~/miniconda3/envs/test2
~/miniconda3/envs/test2/bin/conda-unpack


import glob

# Liste des morceaux triés par nom
parts = sorted(glob.glob("kc_env_part_*"))
output_file = "kc_env_recombine.tar.gz"

with open(output_file, "wb") as outfile:
    for part in parts:
        with open(part, "rb") as infile:
            outfile.write(infile.read())
        print(f"Ajouté : {part}")

print(f"Fichier recombiné : {output_file}")



###################################
import numpy as np
import torch
from pathlib import Path
from spectral import envi
import os

def is_null_spectrum(s, threshold=1e-6):
    return np.var(s) < threshold

def normalize_spectrum(s):
    s_min = s.min()
    s_max = s.max()
    s_norm = 2 * (s - s_min) / (s_max - s_min + 1e-8) - 1
    return s_norm, s_min, s_max

def denormalize_spectrum(s_norm, s_min, s_max):
    return 0.5 * (s_norm + 1) * (s_max - s_min) + s_min

def load_generators(class_names, model_dir, Generator, nz, n_bands, device):
    generators = {}

    for cname in class_names:
        netG = Generator(nz, n_bands).to(device)
        path = Path(model_dir) / cname / "netG_best.pth"
        netG.load_state_dict(torch.load(path, map_location=device))
        netG.eval()
        generators[cname] = netG

    return generators


@torch.no_grad()
def generate_spectrum(netG, nz, device):
    z = torch.randn(1, nz, 1, device=device)
    fake = netG(z).squeeze().cpu().numpy()
    return fake  # ∈ [-1,1]


def augment_extrude(
    hdr_path,
    netG,
    nz,
    device,
    null_threshold=1e-6
):
    img = envi.open(str(hdr_path))
    cube = np.array(img.load(), dtype=np.float32)  # (B,H,W)

    B, H, W = cube.shape
    cube_aug = np.zeros_like(cube)

    for i in range(H):
        for j in range(W):
            s = cube[:, i, j]

            if is_null_spectrum(s, null_threshold):
                cube_aug[:, i, j] = s
            else:
                fake_norm = generate_spectrum(netG, nz, device)

                s_min = s.min()
                s_max = s.max()
                fake_phys = denormalize_spectrum(fake_norm, s_min, s_max)

                cube_aug[:, i, j] = fake_phys

    return cube_aug


def save_envi_cube(cube, ref_hdr_path, out_path):
    img = envi.open(str(ref_hdr_path))
    metadata = img.metadata.copy()

    out_path = Path(out_path)
    envi.save_image(
        str(out_path.with_suffix(".hdr")),
        cube,
        metadata=metadata,
        dtype=np.float32,
        force=True
    )


def augment_all_extrudes(
    root_dir,
    output_dir,
    generators,
    nz,
    device
):
    root_dir = Path(root_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    hdr_files = list(root_dir.rglob("*.hdr"))

    for hdr in hdr_files:
        name = hdr.stem
        class_name = name.split("_")[0]

        if class_name not in generators:
            continue

        print(f"Augmentation : {hdr.name}")

        cube_aug = augment_extrude(
            hdr,
            generators[class_name],
            nz,
            device
        )

        class_out = output_dir / class_name
        class_out.mkdir(exist_ok=True)

        out_path = class_out / f"{name}_aug"
        save_envi_cube(cube_aug, hdr, out_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
nz = 100
n_bands = 224

generators = load_generators(
    class_names=dataset.class_names,
    model_dir="./results",
    Generator=Generator,
    nz=nz,
    n_bands=n_bands,
    device=device
)

augment_all_extrudes(
    root_dir="./extrudes",
    output_dir="./extrudes_augmented",
    generators=generators,
    nz=nz,
    device=device
)
#################################3
# nombre d'extrudes augmentés à générer PAR extrude original
AUG_PER_CLASS = {
    "class0": 5,
    "class1": 2,
    "class2": 10,
}

def augment_extrude(
    hdr_path,
    netG,
    nz,
    device,
    null_threshold=1e-6
):
    img = envi.open(str(hdr_path))
    cube = np.array(img.load(), dtype=np.float32)

    B, H, W = cube.shape
    cube_aug = np.zeros_like(cube)

    for i in range(H):
        for j in range(W):
            s = cube[:, i, j]

            if np.var(s) < null_threshold:
                cube_aug[:, i, j] = s
            else:
                fake_norm = generate_spectrum(netG, nz, device)
                fake_phys = denormalize_spectrum(
                    fake_norm, s.min(), s.max()
                )
                cube_aug[:, i, j] = fake_phys

    return cube_aug

def augment_all_extrudes(
    root_dir,
    output_dir,
    generators,
    aug_per_class,
    nz,
    device
):
    root_dir = Path(root_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    hdr_files = list(root_dir.rglob("*.hdr"))

    for hdr in hdr_files:
        name = hdr.stem
        class_name = name.split("_")[0]

        if class_name not in generators:
            continue

        n_aug = aug_per_class.get(class_name, 1)

        print(f"{hdr.name} | classe={class_name} | N_aug={n_aug}")

        class_out = output_dir / class_name
        class_out.mkdir(exist_ok=True)

        for k in range(n_aug):
            cube_aug = augment_extrude(
                hdr,
                generators[class_name],
                nz,
                device
            )

            out_path = class_out / f"{name}_aug{k+1}"
            save_envi_cube(cube_aug, hdr, out_path)
#################################
def sam_np(x, y, eps=1e-8):
    num = np.dot(x, y)
    den = np.linalg.norm(x) * np.linalg.norm(y) + eps
    cos = np.clip(num / den, -1, 1)
    return np.arccos(cos)


def generate_best_spectrum(
    netG,
    nz,
    device,
    real_phys,
    K=5
):
    s_min = real_phys.min()
    s_max = real_phys.max()

    real_norm = 2 * (real_phys - s_min) / (s_max - s_min + 1e-8) - 1
    real_norm = real_norm.astype(np.float32)

    best_sam = np.inf
    best_fake = None

    for _ in range(K):
        z = torch.randn(1, nz, 1, device=device)
        with torch.no_grad():
            fake_norm = netG(z).cpu().numpy().squeeze()

        sam_val = sam_np(real_norm, fake_norm)

        if sam_val < best_sam:
            best_sam = sam_val
            best_fake = fake_norm

    best_phys = (best_fake + 1) / 2 * (s_max - s_min) + s_min
    return best_phys

def augment_extrude(
    hdr_path,
    netG,
    nz,
    device,
    K=5,
    null_threshold=1e-6
):
    img = envi.open(str(hdr_path))
    cube = np.array(img.load(), dtype=np.float32)

    B, H, W = cube.shape
    cube_aug = np.zeros_like(cube)

    for i in range(H):
        for j in range(W):
            s = cube[:, i, j]

            if np.var(s) < null_threshold:
                cube_aug[:, i, j] = s
            else:
                cube_aug[:, i, j] = generate_best_spectrum(
                    netG,
                    nz,
                    device,
                    s,
                    K
                )

    return cube_aug

def augment_all_extrudes(
    root_dir,
    output_dir,
    generators,
    aug_per_class,
    K_per_class,
    nz,
    device
):
    root_dir = Path(root_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    hdr_files = list(root_dir.rglob("*.hdr"))

    for hdr in hdr_files:
        name = hdr.stem
        class_name = name.split("_")[0]

        if class_name not in generators:
            continue

        n_aug = aug_per_class.get(class_name, 1)
        K = K_per_class.get(class_name, 5)

        print(f"{hdr.name} | classe={class_name} | N_aug={n_aug} | K={K}")

        class_out = output_dir / class_name
        class_out.mkdir(exist_ok=True)

        for k in range(n_aug):
            cube_aug = augment_extrude(
                hdr,
                generators[class_name],
                nz,
                device,
                K=K
            )

            out_path = class_out / f"{name}_aug{k+1}"
            save_envi_cube(cube_aug, hdr, out_path)

AUG_PER_CLASS = {
    "class0": 5,
    "class1": 2,
}

K_PER_CLASS = {
    "class0": 10,
    "class1": 5,
}
