import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import spectral.io.envi as envi
from pathlib import Path
import numpy as np
from einops import rearrange, repeat
from sklearn.metrics import confusion_matrix
import time
import torch.nn.functional as F
import json
import random

device = "cuda" if torch.cuda.is_available() else "cpu"

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x, mask = None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max
        if mask is not None:
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
        attn = dots.softmax(dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_head, dropout, num_channel, mode):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_head, dropout = dropout)))
            ]))
        self.mode = mode
        self.skipcat = nn.ModuleList([])
        for _ in range(depth-2):
            self.skipcat.append(nn.Conv2d(num_channel+1, num_channel+1, [1, 2], 1, 0))
    def forward(self, x, mask = None):
        if self.mode == 'ViT':
            for attn, ff in self.layers:
                x = attn(x, mask = mask)
                x = ff(x)
        elif self.mode == 'CAF':
            last_output = []
            nl = 0
            for attn, ff in self.layers:           
                last_output.append(x)
                if nl > 1:             
                    x = self.skipcat[nl-2](torch.cat([x.unsqueeze(3), last_output[nl-2].unsqueeze(3)], dim=3)).squeeze(3)
                x = attn(x, mask = mask)
                x = ff(x)
                nl += 1
        return x

class SpectralFormer(nn.Module):
    def __init__(self, num_bands, num_patches, num_classes, dim=64, depth=5, heads=4, 
                 mlp_dim=8, dim_head=16, dropout=0.1, emb_dropout=0.1, mode='CAF'):
        super().__init__()
        patch_dim = num_patches * num_bands
        self.pos_embedding = nn.Parameter(torch.randn(1, num_bands + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, num_bands, mode)
        self.pool = 'cls'
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
    def forward(self, x, mask=None):
        x = self.patch_to_embedding(x)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x, mask)
        x = self.to_latent(x[:, 0])
        return self.mlp_head(x)

class SpectralFormerDataset(Dataset):
    def __init__(self, root_dir, patch_size=5, band_patch=3, stride=5, 
                 class_mapping_file=None, file_list=None, mode='train'):
        self.root_dir = Path(root_dir)
        self.patch_size = patch_size
        self.band_patch = band_patch
        self.stride = stride
        self.mode = mode  # 'train' ou 'test'
        self.samples = []
        self.labels = []
        self.original_class_names = []
        self.mapped_class_names = []
        self.file_patches = {}
        self.single_file_classes = {}  # Stocke les infos pour les classes avec un seul fichier
        
        # Charger le mapping des classes
        self.class_mapping = {}
        if class_mapping_file is not None:
            with open(class_mapping_file, 'r') as f:
                self.class_mapping = json.load(f)
            print(f"Mapping des classes chargé: {self.class_mapping}")
        
        # Obtenir la liste des fichiers
        if file_list is not None:
            hdr_files = [f for f in sorted(self.root_dir.rglob("*.hdr")) if f.name in file_list]
        else:
            hdr_files = sorted(self.root_dir.rglob("*.hdr"))
        
        # Premier passage pour identifier toutes les classes originales
        original_class_set = set()
        for hdr_file in hdr_files:
            name = hdr_file.stem
            prefix = name.split("_")[0]
            original_class_set.add(prefix)
        
        self.original_class_names = sorted(list(original_class_set))
        print(f"Classes originales trouvées: {self.original_class_names}")
        
        # Créer le mapping des classes originales vers les nouvelles classes
        self.original_to_mapped = {}
        self.mapped_class_names = []
        
        if self.class_mapping:
            for old_class in self.original_class_names:
                found = False
                for new_class, old_classes in self.class_mapping.items():
                    if old_class in old_classes:
                        self.original_to_mapped[old_class] = new_class
                        if new_class not in self.mapped_class_names:
                            self.mapped_class_names.append(new_class)
                        found = True
                        break
                if not found:
                    print(f"Attention: La classe {old_class} n'est pas mappée, elle sera ignorée")
        else:
            for old_class in self.original_class_names:
                self.original_to_mapped[old_class] = old_class
            self.mapped_class_names = self.original_class_names.copy()
        
        self.mapped_class_names = sorted(self.mapped_class_names)
        self.class_to_idx = {name: idx for idx, name in enumerate(self.mapped_class_names)}
        self.num_classes = len(self.mapped_class_names)
        
        print(f"Classes après mapping: {self.mapped_class_names}")
        print(f"Nombre de classes finales: {self.num_classes}")
        
        # Compter le nombre de fichiers par classe mappée
        class_file_count = {}
        for hdr_file in hdr_files:
            name = hdr_file.stem
            prefix = name.split("_")[0]
            if prefix in self.original_to_mapped:
                mapped_class = self.original_to_mapped[prefix]
                class_file_count[mapped_class] = class_file_count.get(mapped_class, 0) + 1
        
        # Identifier les classes avec un seul fichier
        single_file_classes = {cls for cls, count in class_file_count.items() if count == 1}
        print(f"Classes avec un seul fichier: {single_file_classes}")
        
        # Charger et préparer les données
        all_data = []
        all_labels = []
        
        for hdr_file in hdr_files:
            name = hdr_file.stem
            prefix = name.split("_")[0]
            
            if prefix not in self.original_to_mapped:
                continue
            
            mapped_class_name = self.original_to_mapped[prefix]
            class_id = self.class_to_idx[mapped_class_name]
            
            img = envi.open(str(hdr_file))
            cube = np.array(img.load(), dtype=np.float32)
            
            padded_cube = self._mirror_hsi(cube, patch_size)
            cube_tensor = torch.from_numpy(padded_cube).float()
            
            # Vérifier si c'est une classe avec un seul fichier
            if mapped_class_name in single_file_classes:
                # Créer un masque checkerboard pour ce fichier
                H, W = cube.shape[1], cube.shape[2]  # Dimensions originales
                mask = self._create_checkerboard_mask(H, W)
                
                # Sauvegarder le masque pour référence
                self.single_file_classes[hdr_file.name] = {
                    'class': mapped_class_name,
                    'class_id': class_id,
                    'mask': mask
                }
                
                # Extraire les patches avec le masque selon le mode
                patches = self._extract_patches_with_mask(cube_tensor, patch_size, stride, mask)
            else:
                # Extraction normale sans masque
                patches = self._extract_patches_unfold(cube_tensor, patch_size, stride)
            
            self.file_patches[hdr_file.name] = {
                'patches': patches,
                'label': class_id,
                'n_patches': len(patches)
            }
            
            for patch in patches:
                all_data.append(patch)
                all_labels.append(class_id)
        
        self.data = np.array(all_data)
        self.labels = np.array(all_labels)
        
        print(f"Données chargées: {self.data.shape}")
        print(f"Labels: {self.labels.shape}")
        print(f"Mode {self.mode}: {len(self.data)} patches")
    
    def _create_checkerboard_mask(self, height, width, cell_size=4):
        """
        Crée un masque en damier pour diviser l'image en train/test
        Retourne: mask (bool) avec True pour train, False pour test
        """
        mask = np.zeros((height, width), dtype=bool)
        for i in range(height):
            for j in range(width):
                # Alterne comme un damier
                if ((i // cell_size) + (j // cell_size)) % 2 == 0:
                    mask[i, j] = True  # Train
                else:
                    mask[i, j] = False  # Test
        return mask
    
    def _extract_patches_with_mask(self, cube_tensor, patch_size, stride, mask):
        """
        Extrait les patches selon le masque et le mode (train/test)
        """
        B, H, W = cube_tensor.shape
        padding = patch_size // 2
        patch_list = []
        
        for i in range(0, H - patch_size + 1, stride):
            for j in range(0, W - patch_size + 1, stride):
                # Position du centre du patch dans l'image originale
                center_i = i + patch_size//2 - padding
                center_j = j + patch_size//2 - padding
                
                # Vérifier si le centre est dans les limites
                if 0 <= center_i < mask.shape[0] and 0 <= center_j < mask.shape[1]:
                    # Selon le mode, on prend les patches des zones différentes
                    if self.mode == 'train' and mask[center_i, center_j]:
                        # Train: zones où mask = True
                        patch = cube_tensor[:, i:i+patch_size, j:j+patch_size]
                        patch_np = patch.numpy()
                        band_patches = self._add_spectral_neighborhood(patch_np)
                        patch_list.append(band_patches)
                    elif self.mode == 'test' and not mask[center_i, center_j]:
                        # Test: zones où mask = False
                        patch = cube_tensor[:, i:i+patch_size, j:j+patch_size]
                        patch_np = patch.numpy()
                        band_patches = self._add_spectral_neighborhood(patch_np)
                        patch_list.append(band_patches)
        
        return patch_list
    
    def _mirror_hsi(self, cube, patch):
        B, H, W = cube.shape
        padding = patch // 2
        mirror_hsi = np.zeros((B, H + 2*padding, W + 2*padding), dtype=float)
        mirror_hsi[:, padding:padding+H, padding:padding+W] = cube
        for i in range(padding):
            mirror_hsi[:, padding:padding+H, i] = cube[:, :, padding-i-1]
        for i in range(padding):
            mirror_hsi[:, padding:padding+H, W+padding+i] = cube[:, :, W-1-i]
        for i in range(padding):
            mirror_hsi[:, i, :] = mirror_hsi[:, padding*2-i-1, :]
        for i in range(padding):
            mirror_hsi[:, H+padding+i, :] = mirror_hsi[:, H+padding-1-i, :]
        return mirror_hsi
    
    def _extract_patches_unfold(self, cube_tensor, patch_size, stride):
        B, H, W = cube_tensor.shape
        cube_tensor = cube_tensor.unsqueeze(0)
        patches = F.unfold(cube_tensor, kernel_size=(patch_size, patch_size), dilation=1, stride=stride)
        L = patches.shape[-1]
        patches = patches.squeeze(0).reshape(B, patch_size * patch_size, L).permute(2, 0, 1)
        patches = patches.reshape(L, B, patch_size, patch_size)
        patch_list = []
        for i in range(L):
            patch = patches[i].numpy()
            band_patches = self._add_spectral_neighborhood(patch)
            patch_list.append(band_patches)
        return patch_list
    
    def _add_spectral_neighborhood(self, spatial_patch):
        B, H, W = spatial_patch.shape
        nn = self.band_patch // 2
        x_train_reshape = spatial_patch.reshape(B, H*W)
        x_train_band = np.zeros((B, H*W*self.band_patch), dtype=float)
        x_train_band[:, nn*H*W:(nn+1)*H*W] = x_train_reshape
        for i in range(nn):
            x_train_band[:, i*H*W:(i+1)*H*W] = x_train_reshape[:, :, (self.band_patch - nn + i):]
        for i in range(nn):
            x_train_band[:, (nn+i+1)*H*W:(nn+i+2)*H*W] = x_train_reshape[:, :, (i+1):]
        return x_train_band.flatten()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        x = x.reshape(1, -1).squeeze(0)
        return torch.FloatTensor(x), torch.LongTensor([y])[0]

def split_files_by_class(hdr_files, class_mapping, train_ratio=0.7, random_seed=42):
    """
    Divise les fichiers en train/test en préservant la proportion par classe
    après application du mapping
    """
    random.seed(random_seed)
    
    # Grouper les fichiers par classe originale
    files_by_original_class = {}
    for f in hdr_files:
        prefix = f.stem.split("_")[0]
        if prefix not in files_by_original_class:
            files_by_original_class[prefix] = []
        files_by_original_class[prefix].append(f)
    
    # Créer le mapping inverse pour savoir quelle classe originale va dans quelle classe mappée
    original_to_mapped = {}
    for new_class, old_classes in class_mapping.items():
        for old_class in old_classes:
            original_to_mapped[old_class] = new_class
    
    # Grouper les fichiers par classe mappée
    files_by_mapped_class = {}
    for old_class, files in files_by_original_class.items():
        if old_class in original_to_mapped:
            mapped_class = original_to_mapped[old_class]
            if mapped_class not in files_by_mapped_class:
                files_by_mapped_class[mapped_class] = []
            files_by_mapped_class[mapped_class].extend(files)
    
    train_files = []
    test_files = []
    
    # Pour chaque classe mappée, diviser ses fichiers
    for mapped_class, files in files_by_mapped_class.items():
        random.shuffle(files)
        n_files = len(files)
        n_train = max(1, int(train_ratio * n_files))  # Au moins 1 fichier en train
        n_test = n_files - n_train
        
        train_files.extend(files[:n_train])
        test_files.extend(files[n_train:])
        
        print(f"Classe {mapped_class}: {n_files} fichiers → {n_train} train, {n_test} test")
    
    return train_files, test_files

class AvgrageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0
    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0/batch_size))
    return res, target, pred.squeeze()

def cal_results(matrix):
    shape = np.shape(matrix)
    number = 0
    sum_val = 0
    AA = np.zeros([shape[0]], dtype=np.float)
    for i in range(shape[0]):
        number += matrix[i, i]
        AA[i] = matrix[i, i] / np.sum(matrix[i, :])
        sum_val += np.sum(matrix[i, :]) * np.sum(matrix[:, i])
    OA = number / np.sum(matrix)
    AA_mean = np.mean(AA)
    pe = sum_val / (np.sum(matrix) ** 2)
    Kappa = (OA - pe) / (1 - pe)
    return OA, AA_mean, Kappa, AA

def output_metric(tar, pre):
    matrix = confusion_matrix(tar, pre)
    OA, AA_mean, Kappa, AA = cal_results(matrix)
    return OA, AA_mean, Kappa, AA

def train_epoch(model, train_loader, criterion, optimizer):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    model.train()
    for batch_idx, (batch_data, batch_target) in enumerate(train_loader):
        batch_data = batch_data.to(device)
        batch_target = batch_target.to(device)
        optimizer.zero_grad()
        batch_pred = model(batch_data)
        loss = criterion(batch_pred, batch_target)
        loss.backward()
        optimizer.step()
        prec1, t, p = accuracy(batch_pred, batch_target, topk=(1,))
        n = batch_data.shape[0]
        objs.update(loss.item(), n)
        top1.update(prec1[0].item(), n)
        tar = np.append(tar, t.cpu().numpy())
        pre = np.append(pre, p.cpu().numpy())
    return top1.avg, objs.avg, tar, pre

def test_epoch(model, test_loader, criterion):
    model.eval()
    tar = np.array([])
    pre = np.array([])
    with torch.no_grad():
        for batch_idx, (batch_data, batch_target) in enumerate(test_loader):
            batch_data = batch_data.to(device)
            batch_target = batch_target.to(device)
            batch_pred = model(batch_data)
            _, t, p = accuracy(batch_pred, batch_target, topk=(1,))
            tar = np.append(tar, t.cpu().numpy())
            pre = np.append(pre, p.cpu().numpy())
    return tar, pre

def main():
    # Paramètres en dur
    data_dir = "/chemin/vers/vos/donnees"
    class_mapping_file = "mapping.json"
    mode = 'CAF'  # 'ViT' ou 'CAF'
    batch_size = 64
    patch_size = 5
    band_patch = 3
    stride = 5
    epochs = 100
    learning_rate = 5e-4
    dim = 64
    depth = 5
    heads = 4
    test_freq = 5
    train_ratio = 0.7
    random_seed = 42
    
    print("="*50)
    print("CONFIGURATION")
    print("="*50)
    print(f"Data directory: {data_dir}")
    print(f"Mapping file: {class_mapping_file}")
    print(f"Mode: {mode}")
    print(f"Batch size: {batch_size}")
    print(f"Patch size: {patch_size}")
    print(f"Band patch: {band_patch}")
    print(f"Stride: {stride}")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Dimension: {dim}")
    print(f"Depth: {depth}")
    print(f"Heads: {heads}")
    print(f"Train ratio: {train_ratio}")
    print("="*50)
    
    # Charger le mapping des classes
    with open(class_mapping_file, 'r') as f:
        class_mapping = json.load(f)
    
    # Récupérer tous les fichiers .hdr
    all_hdr_files = sorted(Path(data_dir).rglob("*.hdr"))
    print(f"\nTotal fichiers trouvés: {len(all_hdr_files)}")
    
    # Diviser les fichiers en train/test par classe
    train_files, test_files = split_files_by_class(
        all_hdr_files, 
        class_mapping, 
        train_ratio=train_ratio,
        random_seed=random_seed
    )
    
    print(f"\nFichiers train: {len(train_files)}")
    print(f"Fichiers test: {len(test_files)}")
    
    # Créer les datasets
    print("\nChargement du dataset train...")
    train_dataset = SpectralFormerDataset(
        root_dir=data_dir,
        patch_size=patch_size,
        band_patch=band_patch,
        stride=stride,
        class_mapping_file=class_mapping_file,
        file_list=[f.name for f in train_files]
    )
    
    print("\nChargement du dataset test...")
    test_dataset = SpectralFormerDataset(
        root_dir=data_dir,
        patch_size=patch_size,
        band_patch=band_patch,
        stride=stride,
        class_mapping_file=class_mapping_file,
        file_list=[f.name for f in test_files]
    )
    
    print(f"\nPatches train: {len(train_dataset)}")
    print(f"Patches test: {len(test_dataset)}")
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Créer le modèle
    sample_data, _ = train_dataset[0]
    num_bands = sample_data.shape[0]
    num_patches = patch_size * patch_size
    
    print(f"\nDimensions: num_bands={num_bands}, num_patches={num_patches}")
    
    model = SpectralFormer(
        num_bands=num_bands,
        num_patches=num_patches,
        num_classes=train_dataset.num_classes,
        dim=dim,
        depth=depth,
        heads=heads,
        mode=mode
    ).to(device)
    
    print(f"Modèle créé avec {sum(p.numel() for p in model.parameters())} paramètres")
    print(f"Nombre de classes: {train_dataset.num_classes}")
    
    # Critère et optimiseur
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=epochs//10, gamma=0.9)
    
    # Entraînement
    print("\nDébut de l'entraînement...")
    tic = time.time()
    
    best_acc = 0
    for epoch in range(epochs):
        scheduler.step()
        
        train_acc, train_loss, tar_t, pre_t = train_epoch(
            model, train_loader, criterion, optimizer
        )
        
        print(f"Epoch: {epoch+1:03d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        
        # Test à intervalles réguliers
        if (epoch % test_freq == 0) or (epoch == epochs - 1):
            tar_test, pre_test = test_epoch(model, test_loader, criterion)
            OA, AA_mean, Kappa, AA = output_metric(tar_test, pre_test)
            print(f"Test - OA: {OA:.4f} | AA: {AA_mean:.4f} | Kappa: {Kappa:.4f}")
            
            if OA > best_acc:
                best_acc = OA
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_acc': best_acc,
                    'class_names': train_dataset.mapped_class_names
                }, 'best_model.pt')
                print(f"  → Meilleur modèle sauvegardé (OA: {best_acc:.4f})")
    
    toc = time.time()
    print(f"\nTemps d'entraînement: {toc-tic:.2f} secondes")
    
    # Test final avec le meilleur modèle
    print("\n" + "="*50)
    print("ÉVALUATION FINALE SUR LE TEST SET")
    print("="*50)
    
    checkpoint = torch.load('best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    tar_test, pre_test = test_epoch(model, test_loader, criterion)
    OA, AA_mean, Kappa, AA = output_metric(tar_test, pre_test)
    
    print(f"Overall Accuracy (OA): {OA:.4f}")
    print(f"Average Accuracy (AA): {AA_mean:.4f}")
    print(f"Kappa Coefficient: {Kappa:.4f}")
    print("\nAccuracy par classe:")
    for i, acc in enumerate(AA):
        class_name = train_dataset.mapped_class_names[i]
        print(f"  Classe {class_name}: {acc:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()



def split_files_by_class(hdr_files, class_mapping, train_ratio=0.7, random_seed=42):
    """
    Divise les fichiers en train/test en préservant la proportion par classe
    après application du mapping
    """
    random.seed(random_seed)
    
    # Grouper les fichiers par classe originale
    files_by_original_class = {}
    for f in hdr_files:
        prefix = f.stem.split("_")[0]
        if prefix not in files_by_original_class:
            files_by_original_class[prefix] = []
        files_by_original_class[prefix].append(f)
    
    # Créer le mapping inverse pour savoir quelle classe originale va dans quelle classe mappée
    original_to_mapped = {}
    for new_class, old_classes in class_mapping.items():
        for old_class in old_classes:
            original_to_mapped[old_class] = new_class
    
    # Grouper les fichiers par classe mappée
    files_by_mapped_class = {}
    for old_class, files in files_by_original_class.items():
        if old_class in original_to_mapped:
            mapped_class = original_to_mapped[old_class]
            if mapped_class not in files_by_mapped_class:
                files_by_mapped_class[mapped_class] = []
            files_by_mapped_class[mapped_class].extend(files)
    
    train_files = []
    test_files = []
    
    # Pour chaque classe mappée, diviser ses fichiers
    for mapped_class, files in files_by_mapped_class.items():
        random.shuffle(files)
        n_files = len(files)
        
        if n_files == 1:
            # CAS SPÉCIAL: Un seul fichier
            print(f"\nClasse {mapped_class}: 1 seul fichier - {files[0].name}")
            print(f"   → Utilisation d'un masque spatial (checkerboard) pour diviser en train/test")
            print(f"   → Le même fichier sera utilisé pour train ET test avec des masques inversés")
            
            # Le même fichier va dans train ET test
            train_files.append(files[0])
            test_files.append(files[0])
            
        else:
            # Cas normal: plusieurs fichiers
            n_train = max(1, int(train_ratio * n_files))
            n_test = n_files - n_train
            
            # S'assurer d'avoir au moins 1 fichier en test si possible
            if n_test == 0 and n_files > 1:
                n_train = n_files - 1
                n_test = 1
                
            train_files.extend(files[:n_train])
            test_files.extend(files[n_train:])
            
            print(f"classe {mapped_class}: {n_files} fichiers → {n_train} train, {n_test} test")
    
    return train_files, test_files
