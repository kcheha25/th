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


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import spectral.io.envi as envi
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr

# Hyperparamètres
BATCH_SIZE = 8
LATENT_DIM = 100
CRITIC_ITER = 5  # Nombre d'itérations du critique par itération du générateur
LAMBDA_GP = 10   # Coefficient pour la pénalité de gradient
LEARNING_RATE = 1e-4
NUM_EPOCHS = 1000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Chargement des données (votre code existant)
PLOTS_DIR = Path("output_plots")
OUTPUT_ENVI = Path("wgan_input")
OUTPUT_ENVI.mkdir(exist_ok=True)

hdr_files = list(PLOTS_DIR.glob("*/*.hdr"))

# Trouver les dimensions minimales
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

# Sauvegarder les cubes prétraités
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
        self.wavelengths = None
        if self.dirs:
            hdr_file = list(self.dirs[0].glob("*.hdr"))[0]
            img = envi.open(str(hdr_file))
            self.wavelengths = img.metadata.get('wavelength', [])
            
    def __len__(self):
        return len(self.dirs)
    
    def __getitem__(self, idx):
        hdr_file = list(self.dirs[idx].glob("*.hdr"))[0]
        img = envi.open(str(hdr_file))
        cube = np.array(img.load()).transpose(1,2,0)  # Shape: (H, W, Bands)
        # Normalisation
        cube_min = cube.min(axis=(0,1), keepdims=True)
        cube_max = cube.max(axis=(0,1), keepdims=True)
        cube = (cube - cube_min) / (cube_max - cube_min + 1e-8)
        cube = np.transpose(cube, (2,0,1))  # Shape: (Bands, H, W)
        return torch.tensor(cube, dtype=torch.float32)

dataset = HSIDataset(OUTPUT_ENVI)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Dimensions
BANDS = dataset[0].shape[0]
HEIGHT = dataset[0].shape[1]
WIDTH = dataset[0].shape[2]

# === MODÈLES WGAN-GP ===

class Generator(nn.Module):
    def __init__(self, latent_dim, bands, height, width):
        super().__init__()
        self.init_size = height // 8
        self.l1 = nn.Linear(latent_dim, 128 * self.init_size**2)
        
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, bands, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

class Critic(nn.Module):
    def __init__(self, bands, height, width):
        super().__init__()
        
        def critic_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1)]
            if bn:
                block.append(nn.InstanceNorm2d(out_filters))
            block.append(nn.LeakyReLU(0.2, inplace=True))
            return block

        self.model = nn.Sequential(
            *critic_block(bands, 32, bn=False),
            *critic_block(32, 64),
            *critic_block(64, 128),
            *critic_block(128, 256),
        )

        # Calcul de la taille de sortie
        ds_size = height // 2**4
        self.adv_layer = nn.Sequential(
            nn.Linear(256 * ds_size**2, 1),
            nn.Identity()  # Pas de sigmoid pour WGAN
        )

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity

# === FONCTION DE PÉNALITÉ DE GRADIENT ===

def compute_gradient_penalty(critic, real_samples, fake_samples):
    """Calcule la pénalité de gradient pour WGAN-GP"""
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=DEVICE)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    critic_interpolates = critic(interpolates)
    
    gradients = torch.autograd.grad(
        outputs=critic_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones(critic_interpolates.size(), device=DEVICE),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

# === INITIALISATION ===

generator = Generator(LATENT_DIM, BANDS, HEIGHT, WIDTH).to(DEVICE)
critic = Critic(BANDS, HEIGHT, WIDTH).to(DEVICE)

optimizer_G = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
optimizer_C = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

# === ENTRAÎNEMENT ===

print(f"Début de l'entraînement WGAN-GP sur {DEVICE}")
print(f"Dimensions: {BANDS} bandes, {HEIGHT}x{WIDTH} pixels")

for epoch in range(NUM_EPOCHS):
    for i, real_imgs in enumerate(dataloader):
        real_imgs = real_imgs.to(DEVICE)
        batch_size = real_imgs.size(0)
        
        # === Entraînement du Critique ===
        optimizer_C.zero_grad()
        
        # Bruit pour le générateur
        z = torch.randn(batch_size, LATENT_DIM, device=DEVICE)
        
        # Générer des fakes
        fake_imgs = generator(z)
        
        # Scores des vrais et faux
        real_validity = critic(real_imgs)
        fake_validity = critic(fake_imgs.detach())
        
        # Pénalité de gradient
        gradient_penalty = compute_gradient_penalty(critic, real_imgs.data, fake_imgs.data)
        
        # Perte du critique (Wasserstein + pénalité)
        loss_C = -torch.mean(real_validity) + torch.mean(fake_validity) + LAMBDA_GP * gradient_penalty
        
        loss_C.backward()
        optimizer_C.step()
        
        # === Entraînement du Générateur (tous les CRITIC_ITER steps) ===
        if i % CRITIC_ITER == 0:
            optimizer_G.zero_grad()
            
            # Générer de nouvelles images
            gen_imgs = generator(z)
            
            # Perte du générateur (on veut tromper le critique)
            loss_G = -torch.mean(critic(gen_imgs))
            
            loss_G.backward()
            optimizer_G.step()
    
    # === Logging ===
    if epoch % 100 == 0:
        print(f"[Epoch {epoch}/{NUM_EPOCHS}] [D loss: {loss_C.item():.4f}] [G loss: {loss_G.item():.4f}]")
        
        # Sauvegarder quelques échantillons
        with torch.no_grad():
            sample_z = torch.randn(4, LATENT_DIM, device=DEVICE)
            gen_samples = generator(sample_z).cpu().numpy()
            
            # Visualiser une bande
            fig, axes = plt.subplots(2, 2, figsize=(10, 10))
            for idx, ax in enumerate(axes.flat):
                ax.imshow(gen_samples[idx, BANDS//2, :, :], cmap='viridis')
                ax.axis('off')
                ax.set_title(f"Échantillon {idx+1}")
            plt.suptitle(f"Échantillons générés - Époque {epoch}")
            plt.tight_layout()
            plt.savefig(f"generated_samples_epoch_{epoch}.png")
            plt.close()
            
        # Sauvegarder les modèles
        torch.save(generator.state_dict(), f"generator_epoch_{epoch}.pth")
        torch.save(critic.state_dict(), f"critic_epoch_{epoch}.pth")

print("Entraînement terminé !")


# === GÉNÉRATION DES DONNÉES AUGMENTÉES ===

def generate_augmented_samples(generator, num_samples, save_dir="augmented_plots"):
    """Génère et sauvegarde des échantillons augmentés"""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    generator.eval()
    all_generated = []
    
    with torch.no_grad():
        for i in range(0, num_samples, BATCH_SIZE):
            current_batch = min(BATCH_SIZE, num_samples - i)
            z = torch.randn(current_batch, LATENT_DIM, device=DEVICE)
            gen_imgs = generator(z).cpu().numpy()
            
            # Sauvegarde au format ENVI
            for j in range(current_batch):
                sample_idx = i + j
                cube = gen_imgs[j].transpose(1, 2, 0)  # (H, W, Bands)
                
                # Dénormalisation
                cube = (cube + 1) / 2  # De [-1,1] à [0,1]
                
                # Sauvegarde
                sample_dir = save_dir / f"aug_plot_{sample_idx}"
                sample_dir.mkdir(exist_ok=True)
                
                hdr_path = sample_dir / f"aug_plot_{sample_idx}.hdr"
                envi.save_image(
                    str(hdr_path),
                    cube.transpose(2, 0, 1),
                    interleave='bil',
                    dtype=np.float32,
                    metadata={'wavelength': dataset.wavelengths}
                )
                
                # Sauvegarde PNG pour visualisation
                rgb_idx = [np.argmin(np.abs(np.array(dataset.wavelengths) - wl)) 
                          for wl in [650, 550, 450]] if dataset.wavelengths else [0, 1, 2]
                rgb_img = cube[:, :, rgb_idx]
                rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min())
                plt.imsave(sample_dir / f"aug_plot_{sample_idx}_rgb.png", rgb_img)
            
            all_generated.append(gen_imgs)
    
    generator.train()
    return np.vstack(all_generated)

# Générer N nouveaux échantillons
NUM_AUGMENTED = 100
augmented_samples = generate_augmented_samples(generator, NUM_AUGMENTED)

# === CALCUL DES MÉTRIQUES ===

def extract_spectral_signatures(cubes, num_pixels=100):
    """Extrait des signatures spectrales aléatoires des cubes"""
    all_signatures = []
    for cube in cubes:
        H, W, B = cube.shape if len(cube.shape) == 3 else cube.shape[1:]
        cube_2d = cube.reshape(-1, B) if len(cube.shape) == 3 else cube
        
        # Échantillonner aléatoirement
        idx = np.random.choice(cube_2d.shape[0], min(num_pixels, cube_2d.shape[0]), replace=False)
        signatures = cube_2d[idx]
        all_signatures.append(signatures)
    
    return np.vstack(all_signatures)

# Charger les échantillons originaux
original_samples = []
for i in range(len(dataset)):
    original_samples.append(dataset[i].numpy().transpose(1, 2, 0))
original_samples = np.array(original_samples)

# Extraire les signatures spectrales
print("Extraction des signatures spectrales...")
orig_signatures = extract_spectral_signatures(original_samples, num_pixels=500)
aug_signatures = extract_spectral_signatures(augmented_samples, num_pixels=500)

# 1️⃣ Spectral Angle Mapper (SAM)
def compute_sam(spectra1, spectra2):
    """Calcule l'angle spectral moyen entre deux ensembles de spectres"""
    sam_values = []
    for s1 in spectra1:
        sam_for_s1 = []
        for s2 in spectra2:
            # Éviter la division par zéro
            s1_norm = s1 / (np.linalg.norm(s1) + 1e-10)
            s2_norm = s2 / (np.linalg.norm(s2) + 1e-10)
            dot_product = np.dot(s1_norm, s2_norm)
            dot_product = np.clip(dot_product, -1.0, 1.0)
            angle = np.arccos(dot_product)
            sam_for_s1.append(np.degrees(angle))
        sam_values.append(np.mean(sam_for_s1))
    
    return np.mean(sam_values), np.std(sam_values)

sam_mean, sam_std = compute_sam(orig_signatures, aug_signatures)
print(f"✅ 1️⃣ Spectral Angle Mapper (SAM): {sam_mean:.2f}° ± {sam_std:.2f}°")

# 2️⃣ Distance spectrale globale (RMSE spectral)
def compute_spectral_rmse(spectra1, spectra2):
    """Calcule le RMSE spectral moyen"""
    rmse_values = []
    for s1 in spectra1:
        for s2 in spectra2:
            rmse = np.sqrt(np.mean((s1 - s2) ** 2))
            rmse_values.append(rmse)
    
    return np.mean(rmse_values), np.std(rmse_values)

rmse_mean, rmse_std = compute_spectral_rmse(orig_signatures, aug_signatures)
print(f"✅ 2️⃣ RMSE Spectral: {rmse_mean:.4f} ± {rmse_std:.4f}")

# 3️⃣ Corrélation spectrale moyenne
def compute_spectral_correlation(spectra1, spectra2):
    """Calcule la corrélation spectrale moyenne"""
    corr_values = []
    for s1 in spectra1:
        for s2 in spectra2:
            corr, _ = pearsonr(s1, s2)
            corr_values.append(corr)
    
    return np.mean(corr_values), np.std(corr_values)

corr_mean, corr_std = compute_spectral_correlation(orig_signatures, aug_signatures)
print(f"✅ 3️⃣ Corrélation Spectrale: {corr_mean:.4f} ± {corr_std:.4f}")

# 4️⃣ Analyse de la distribution (PCA, t-SNE, UMAP)
print("Calcul des projections pour l'analyse de distribution...")

# Combiner les données
all_signatures = np.vstack([orig_signatures, aug_signatures])
labels = ['Original'] * len(orig_signatures) + ['Augmenté'] * len(aug_signatures)

# 4A - PCA
print("  → Calcul PCA...")
pca = PCA(n_components=2)
pca_result = pca.fit_transform(all_signatures)

plt.figure(figsize=(15, 5))

plt.subplot(131)
scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=[0 if l == 'Original' else 1 for l in labels], 
                     alpha=0.6, cmap='coolwarm')
plt.xlabel('PC1 ({:.1f}%)'.format(pca.explained_variance_ratio_[0]*100))
plt.ylabel('PC2 ({:.1f}%)'.format(pca.explained_variance_ratio_[1]*100))
plt.title('PCA des Signatures Spectrales')
plt.colorbar(scatter, ticks=[0, 1], label='Type')
plt.clim(-0.5, 1.5)
plt.grid(True, alpha=0.3)

# 4B - t-SNE
print("  → Calcul t-SNE...")
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
tsne_result = tsne.fit_transform(all_signatures)

plt.subplot(132)
plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=[0 if l == 'Original' else 1 for l in labels], 
           alpha=0.6, cmap='coolwarm')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.title('t-SNE des Signatures Spectrales')
plt.grid(True, alpha=0.3)

# 4C - UMAP
print("  → Calcul UMAP...")
umap_reducer = umap.UMAP(n_components=2, random_state=42)
umap_result = umap_reducer.fit_transform(all_signatures)

plt.subplot(133)
scatter = plt.scatter(umap_result[:, 0], umap_result[:, 1], c=[0 if l == 'Original' else 1 for l in labels], 
                     alpha=0.6, cmap='coolwarm')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.title('UMAP des Signatures Spectrales')
plt.colorbar(scatter, ticks=[0, 1], label='Type')
plt.clim(-0.5, 1.5)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('distribution_analysis.png', dpi=150)
plt.show()

# === RAPPORT DE SYNTHÈSE ===

print("\n" + "="*60)
print("RAPPORT D'ANALYSE DES DONNÉES AUGMENTÉES")
print("="*60)
print(f"Échantillons originaux: {len(original_samples)}")
print(f"Échantillons augmentés générés: {NUM_AUGMENTED}")
print(f"Signatures analysées: {len(orig_signatures)} originales, {len(aug_signatures)} augmentées")
print("\nRÉSULTATS DES MÉTRIQUES SPECTRALES:")
print(f"  • SAM moyen: {sam_mean:.2f}° (idéal: < 10°)")
print(f"  • RMSE spectral: {rmse_mean:.4f} (idéal: < 0.1)")
print(f"  • Corrélation: {corr_mean:.4f} (idéal: > 0.8)")
print("\nINTERPRÉTATION:")
if sam_mean < 10 and rmse_mean < 0.15 and corr_mean > 0.8:
    print("✅ Les données augmentées sont de haute qualité et similaires aux originales.")
elif sam_mean < 20 and rmse_mean < 0.3 and corr_mean > 0.6:
    print("⚠️ Qualité acceptable, mais certaines différences spectrales existent.")
else:
    print("❌ Les données générées sont trop différentes des originales.")
print("="*60)

# === VISUALISATION SUPPLÉMENTAIRE: SPECTRES MOYENS ===

print("\nVisualisation des spectres moyens...")

# Calcul des spectres moyens
original_mean_spectrum = np.mean(orig_signatures, axis=0)
augmented_mean_spectrum = np.mean(aug_signatures, axis=0)
original_std = np.std(orig_signatures, axis=0)
augmented_std = np.std(aug_signatures, axis=0)

plt.figure(figsize=(12, 6))
if dataset.wavelengths:
    wavelengths = dataset.wavelengths[:len(original_mean_spectrum)]
    plt.plot(wavelengths, original_mean_spectrum, 'b-', label='Original (moyenne)', linewidth=2)
    plt.fill_between(wavelengths, 
                     original_mean_spectrum - original_std, 
                     original_mean_spectrum + original_std, 
                     alpha=0.2, color='blue', label='Original (±1σ)')
    
    plt.plot(wavelengths, augmented_mean_spectrum, 'r-', label='Augmenté (moyenne)', linewidth=2)
    plt.fill_between(wavelengths, 
                     augmented_mean_spectrum - augmented_std, 
                     augmented_mean_spectrum + augmented_std, 
                     alpha=0.2, color='red', label='Augmenté (±1σ)')
    plt.xlabel('Longueur d\'onde (nm)')
else:
    plt.plot(original_mean_spectrum, 'b-', label='Original (moyenne)', linewidth=2)
    plt.plot(augmented_mean_spectrum, 'r-', label='Augmenté (moyenne)', linewidth=2)
    plt.xlabel('Index de bande')

plt.ylabel('Réflectance normalisée')
plt.title('Comparaison des Spectres Moyens')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('mean_spectra_comparison.png', dpi=150)
plt.show()

# Sauvegarde des métriques dans un fichier
with open('augmentation_metrics_report.txt', 'w') as f:
    f.write("=== RAPPORT DES MÉTRIQUES D'AUGMENTATION ===\n\n")
    f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Nombre d'échantillons originaux: {len(original_samples)}\n")
    f.write(f"Nombre d'échantillons augmentés: {NUM_AUGMENTED}\n\n")
    f.write("MÉTRIQUES SPECTRALES:\n")
    f.write(f"1. Spectral Angle Mapper (SAM): {sam_mean:.2f}° ± {sam_std:.2f}°\n")
    f.write(f"2. RMSE Spectral: {rmse_mean:.4f} ± {rmse_std:.4f}\n")
    f.write(f"3. Corrélation Spectrale: {corr_mean:.4f} ± {corr_std:.4f}\n\n")
    f.write("INTERPRÉTATION:\n")
    if sam_mean < 10 and rmse_mean < 0.15 and corr_mean > 0.8:
        f.write("Qualité EXCELLENTE - Les données augmentées préservent fidèlement les caractéristiques spectrales.\n")
    elif sam_mean < 20 and rmse_mean < 0.3 and corr_mean > 0.6:
        f.write("Qualité ACCEPTABLE - Bonne similarité globale avec quelques variations.\n")
    else:
        f.write("ATTENTION - Différences importantes détectées. Revoir l'entraînement du WGAN-GP.\n")

print("\n✅ Analyse terminée !")
print(f"• Graphiques sauvegardés: 'distribution_analysis.png', 'mean_spectra_comparison.png'")
print(f"• Rapport sauvegardé: 'augmentation_metrics_report.txt'")
print(f"• Données augmentées sauvegardées dans: 'augmented_plots/'")