# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 07:59:48 2026

@author: karim

Version modifiee (3 ameliorations pour reduire l'ecart CV->test):
  1. AUGMENTATION EN LIGNE : la rotation (facon Colab, +/- 30 deg) est
     appliquee a la volee dans le Dataset, DIFFERENTE a chaque epoch. On garde
     la notion de `target_count` via un re-echantillonnage d'indices (equilibrage
     des classes) au lieu de creer des copies figees. -> moins d'overfit.
  2. TTA ELAGUEE : suppression du scale 1.05/0.95 (inutile car les features sont
     deja invariantes a l'echelle apres scale_normalize). On garde les rotations.
  3. PSEUDO-LABELING : round 2. L'ensemble predit le test, les predictions
     confiantes (> PSEUDO_THRESHOLD) sont reinjectees dans le train et on
     re-entraine -> le modele s'adapte a la distribution du test.

Workflow recommande:
  A) Round 1  : INFERENCE_ONLY=False, USE_PSEUDO_LABELS=False   (CV classique)
  B) Round 2  : INFERENCE_ONLY=False, USE_PSEUDO_LABELS=True    (CV + pseudo-labels)
  C) Inference: INFERENCE_ONLY=True                             (soumission finale)
"""

import os
import copy
import shutil
import pickle
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import wandb
import random

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import seaborn as sns

sns.set_style("whitegrid")

INFERENCE_ONLY = True
USE_KFOLD = True
N_FOLDS = 5
MAKE_PLOTS = True          # active tous les plots facon Colab
PLOTS_DIR = 'plots'        # les figures sont sauvegardees ici
NUM_WORKERS = 4            # workers DataLoader (augmentation en ligne = plus lourd)

# --- Pseudo-labeling (round 2) ---
USE_PSEUDO_LABELS = False  # True = charge l'ensemble round-1, genere des pseudo-labels
PSEUDO_THRESHOLD = 0.9     # ne garde que les predictions test de proba max > seuil

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

NUM_CLASSES = 13
os.environ.setdefault("WANDB_MODE", "offline")
wandb.init(project="hand_gesture_multistream_stgcn_moe_fixed", config={"num_classes": NUM_CLASSES})
data_path = 'hand-gesture-recognition-challenge/'


# ---------------------------------------------------------------------------
# TTA (elaguee) : rotations seulement, PAS de scale apres scale_normalize.
# ---------------------------------------------------------------------------
TTA_CONFIGS = [
    {"angle": 0.0,        "axis": 2},
    {"angle": np.pi / 18, "axis": 2},
    {"angle": -np.pi / 18, "axis": 2},
    {"angle": np.pi / 36, "axis": 2},
    {"angle": -np.pi / 36, "axis": 2},
]


def save_and_show(name):
    if MAKE_PLOTS:
        os.makedirs(PLOTS_DIR, exist_ok=True)
        path = os.path.join(PLOTS_DIR, f"{name}.png")
        plt.savefig(path, dpi=120, bbox_inches='tight')
        print(f"Plot sauvegarde: {path}")
    try:
        plt.show()
    except Exception:
        pass
    plt.close()


def linear_interpolation(landmarks, t_from, t_to):
    if len(landmarks) == 0:
        return np.zeros((len(t_to), landmarks.shape[1]))
    interpolated = np.zeros((len(t_to), landmarks.shape[1]))
    for i in range(landmarks.shape[1]):
        f = interp1d(t_from, landmarks[:, i], kind='linear', bounds_error=False, fill_value='extrapolate')
        interpolated[:, i] = f(t_to)
    return interpolated


def load_train_data():
    data_dir = os.path.join(data_path, 'data_train/')
    data_txt_file = os.path.join(data_path, 'data_train/train_annotations.txt')
    max_len = 80
    all_landmarks = []
    labels = []
    to_discard = []
    with open(data_txt_file, 'r') as f:
        files = f.readlines()
    for i, file in tqdm(enumerate(files), desc="Loading training data"):
        try:
            parts = file.strip().split(', ')
            if len(parts) >= 4:
                video_path, landmarks_path, label, nframes = parts
            else:
                continue
            labels.append(label)
            landmark_file = os.path.join(data_dir, landmarks_path)
            if not os.path.exists(landmark_file):
                to_discard.append(i)
                continue
            with open(landmark_file) as f:
                lines = f.readlines()
            landmarks = []
            is_available = []
            for line in lines:
                lands = list(filter(lambda x: len(x), line.strip('\n').strip(' ').split(';')))
                if len(lands):
                    is_available.append(True)
                    landmarks.append([float(x) for x in lands])
                else:
                    landmarks.append(np.zeros(63))
                    is_available.append(False)
            if sum(is_available) > 1:
                landmarks = np.array(landmarks).astype('float32')
                is_available = np.array(is_available)
                t_from = np.arange(len(landmarks))[is_available]
                landmarks = linear_interpolation(landmarks[is_available], t_from=t_from,
                                                  t_to=np.linspace(min(t_from), max(t_from), max_len + 1))
                all_landmarks.append(landmarks)
            else:
                to_discard.append(i)
        except Exception as e:
            print(f"Error processing file {i}: {e}")
            to_discard.append(i)
    train_landmarks = np.array(all_landmarks).reshape(-1, 81, 21, 3)
    y_true = np.array([labels[i] for i in range(len(labels)) if i not in to_discard])
    return train_landmarks, y_true


def load_test_data():
    data_dir = os.path.join(data_path, 'data_test/')
    data_txt_file = os.path.join(data_path, 'data_test/test_annotations.txt')
    max_len = 80
    all_landmarks = []
    with open(data_txt_file, 'r') as f:
        files = f.readlines()
    for i, file in tqdm(enumerate(files), desc="Loading test data"):
        try:
            parts = file.strip().split(', ')
            if len(parts) >= 3:
                video_path, landmarks_path, nframes = parts
            else:
                continue
            landmark_file = os.path.join(data_dir, landmarks_path)
            if not os.path.exists(landmark_file):
                all_landmarks.append(np.zeros((81, 63)))
                continue
            with open(landmark_file) as f:
                lines = f.readlines()
            landmarks = []
            is_available = []
            for line in lines:
                lands = list(filter(lambda x: len(x), line.strip('\n').strip(' ').split(';')))
                if len(lands):
                    is_available.append(True)
                    landmarks.append([float(x) for x in lands])
                else:
                    landmarks.append(np.zeros(63))
                    is_available.append(False)
            if sum(is_available) > 1:
                landmarks = np.array(landmarks).astype('float32')
                is_available = np.array(is_available)
                t_from = np.arange(len(landmarks))[is_available]
                landmarks = linear_interpolation(landmarks[is_available], t_from=t_from,
                                                  t_to=np.linspace(min(t_from), max(t_from), max_len + 1))
                all_landmarks.append(landmarks)
            else:
                all_landmarks.append(np.zeros((81, 63)))
        except Exception as e:
            print(f"Error processing test file {i}: {e}")
            all_landmarks.append(np.zeros((81, 63)))
    test_landmarks = np.array(all_landmarks).reshape(-1, 81, 21, 3)
    return test_landmarks


AREAS = {0: [[0, 4, 5], [4, 5, 8]], 1: [[5, 8, 9], [8, 9, 12]],
         2: [[9, 12, 13], [12, 13, 16]], 3: [[13, 16, 17], [16, 17, 20]]}


def areas_landmarks(vids):
    areas = np.zeros((vids.shape[0], vids.shape[1], 4))
    for i, v in AREAS.items():
        vec_1_1 = vids[:, :, v[0][1]] - vids[:, :, v[0][0]]
        vec_1_2 = vids[:, :, v[0][2]] - vids[:, :, v[0][0]]
        vec_2_1 = vids[:, :, v[1][1]] - vids[:, :, v[1][0]]
        vec_2_2 = vids[:, :, v[1][2]] - vids[:, :, v[1][0]]
        areas[:, :, i] = (np.linalg.norm(np.cross(vec_1_1, vec_1_2, axis=2), axis=2) / 2 +
                          np.linalg.norm(np.cross(vec_2_1, vec_2_2, axis=2), axis=2) / 2)
    return areas


FINGERS = [[0, 1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]]


def fingers_deployment(vids):
    features = np.zeros((vids.shape[0], vids.shape[1], len(FINGERS)))
    for i, f in enumerate(FINGERS):
        features[:, :, i] = np.linalg.norm(vids[:, :, f[-1]] - vids[:, :, f[0]], axis=2)
    return features


HARDCODED = [[0, 12], [0, 16], [4, 12], [4, 16], [4, 20], [4, 8], [8, 12], [12, 16], [16, 20], [2, 17]]


def remarkable_distances(vids):
    features = np.zeros((vids.shape[0], vids.shape[1], len(HARDCODED)))
    for i, (p1, p2) in enumerate(HARDCODED):
        features[:, :, i] = np.linalg.norm(vids[:, :, p1] - vids[:, :, p2], axis=2)
    return features


def diff(vids):
    return vids[:, 1:] - vids[:, :-1]


def angle(vids):
    center = np.sum(vids[:, :, [0, 17, 13, 9, 5]], axis=2) / 5
    center = np.repeat(center[:, :, np.newaxis, :], 5, axis=-2)
    vectors = vids[:, :, [4, 8, 12, 16, 20]] - center
    dot_product = np.sum(vectors[:, :, :-1] * vectors[:, :, 1:], axis=-1)
    norm_product = (np.linalg.norm(vectors[:, :, :-1], axis=-1) * np.linalg.norm(vectors[:, :, 1:], axis=-1))
    norm_product = np.where(norm_product == 0, 1e-8, norm_product)
    dot_product = np.clip(dot_product / norm_product, -1, 1)
    return np.arccos(dot_product)


def normalize(vids):
    bases = vids[:, :1]
    normalized = vids.copy()
    normalized[:, 1:] -= bases
    return normalized


def direction(landmarks):
    mean_land_9_13 = np.mean(landmarks[:, :, [9, 13]], axis=-2)
    hand_direction = mean_land_9_13 - landmarks[:, :, 0]
    phi = np.arctan2(hand_direction[:, :, 0], hand_direction[:, :, 2])
    theta = np.arccos(np.clip(hand_direction[:, :, 1] / np.linalg.norm(hand_direction, axis=-1), -1, 1))
    return np.concatenate((hand_direction, np.expand_dims(phi, axis=-1), np.expand_dims(theta, axis=-1)), axis=-1)


def rotate(arr, theta, axis):
    if axis == 0:
        rot_mat = np.array([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])
    elif axis == 1:
        rot_mat = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
    else:
        rot_mat = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    return (rot_mat @ arr.T).T


def scale_normalize(vids):
    ref = np.linalg.norm(vids[:, :, 9] - vids[:, :, 0], axis=-1, keepdims=True)
    ref = np.where(ref < 1e-6, 1e-6, ref)
    ref = ref[:, :, np.newaxis]
    return vids / ref


def create_features(landmarks, with_normalized=True):
    differences = diff(landmarks)
    if with_normalized:
        normalized = normalize(landmarks)
        features = np.zeros((landmarks.shape[0], landmarks.shape[1], 31))
    else:
        features = np.zeros((landmarks.shape[0], landmarks.shape[1], 28))
    features[:, :, :4] = areas_landmarks(landmarks)
    features[:, :, 4:9] = fingers_deployment(landmarks)
    features[:, :, 9:13] = angle(landmarks)
    features[:, :, 13:23] = remarkable_distances(landmarks)
    features[:, :, 23:28] = direction(landmarks)
    if with_normalized:
        features[:, :, 28:31] = normalized[:, :, 0]
        return [landmarks, normalized, differences, features]
    return [landmarks, differences, features]


# ---------------------------------------------------------------------------
# AUGMENTATION : rotation seule (facon Colab, +/- 30 deg autour d'un axe aleatoire)
# ---------------------------------------------------------------------------
def augment_gesture_colab(seq):
    """Une seule rotation aleatoire appliquee a toute la sequence (comme le Colab)."""
    angles = np.arange(-np.pi / 6, np.pi / 6, 0.05)   # +/- 30 deg
    ang = np.random.choice(angles)
    axis = np.random.choice([0, 1, 2])
    rotated = np.empty_like(seq)
    for f in range(seq.shape[0]):
        rotated[f] = rotate(seq[f], ang, axis)
    return rotated.astype('float32')


def augment_data_balanced(landmarks, labels, target_count=450):
    """Version OFFLINE (utilisee seulement pour les plots d'augmentation)."""
    unique_labels, counts = np.unique(labels, return_counts=True)
    print(f"Classes avant augmentation: {dict(zip(unique_labels, counts))}")
    print(f"Target: {target_count} echantillons par classe")
    X_added, y_added = [], []
    for lab in unique_labels:
        lab_indices = np.where(labels == lab)[0]
        current = len(lab_indices)
        if current < target_count:
            n_to_add = target_count - current
            print(f"Classe {lab}: {current} -> {target_count} (+{n_to_add})")
            for _ in range(n_to_add):
                idx = np.random.choice(lab_indices)
                aug = augment_gesture_colab(landmarks[idx])
                X_added.append(aug)
                y_added.append(lab)
        else:
            print(f"Classe {lab}: {current} (aucun ajout)")
    if X_added:
        landmarks_aug = np.vstack((landmarks, np.array(X_added)))
        labels_aug = np.concatenate((labels, np.array(y_added)))
        unique_after, counts_after = np.unique(labels_aug, return_counts=True)
        print(f"Classes apres augmentation: {dict(zip(unique_after, counts_after))}")
        return landmarks_aug, labels_aug
    return landmarks, labels


# ---------------------------------------------------------------------------
# Plots d'exploration / augmentation (facon Colab)
# ---------------------------------------------------------------------------
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def plot_hand_skeleton(ax, landmarks_frame, color="#1f77b4", title=None):
    xs, ys = landmarks_frame[:, 0], landmarks_frame[:, 1]
    for a, b in HAND_CONNECTIONS:
        ax.plot([xs[a], xs[b]], [ys[a], ys[b]], color=color, linewidth=1.5, zorder=1)
    ax.scatter(xs, ys, color=color, s=25, zorder=2)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=10)


def plot_data_exploration(train_landmarks, train_labels):
    if not MAKE_PLOTS:
        return
    unique, counts = np.unique(train_labels.astype(int), return_counts=True)

    plt.figure(figsize=(9, 4))
    bars = plt.bar(unique, counts, color="#1f77b4", edgecolor="white")
    plt.xlabel("Gesture class"); plt.ylabel("Number of sequences")
    plt.title("Class distribution - training set (before augmentation)")
    plt.xticks(unique)
    for b, c in zip(bars, counts):
        plt.text(b.get_x() + b.get_width() / 2, c + 0.5, str(c), ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    save_and_show("01_class_distribution_before_aug")

    fig, axes = plt.subplots(1, 5, figsize=(16, 4))
    for ax, cls in zip(axes, unique[:5]):
        idx = np.where(train_labels.astype(int) == cls)[0][0]
        mid_frame = train_landmarks[idx, train_landmarks.shape[1] // 2]
        plot_hand_skeleton(ax, mid_frame, title=f"Class {cls}")
    plt.suptitle("Example hand skeleton (middle frame) for several gesture classes")
    plt.tight_layout()
    save_and_show("02_hand_skeletons")

    traj = train_landmarks[0, :, 0, :]
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(traj[:, 0], traj[:, 1], c=np.arange(len(traj)), cmap="viridis", s=15)
    ax.plot(traj[:, 0], traj[:, 1], color="gray", alpha=0.3, linewidth=1)
    ax.invert_yaxis()
    ax.set_title(f"Wrist trajectory over 81 frames - sample #0 (class {int(train_labels[0])})")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    plt.colorbar(sc, ax=ax, label="frame index")
    plt.tight_layout()
    save_and_show("03_wrist_trajectory")


def plot_augmentation_effect(train_landmarks, train_labels, target_count):
    if not MAKE_PLOTS:
        return
    print("\n--- Demonstration de l'augmentation (pour les plots) ---")
    train_landmarks_aug, train_labels_aug = augment_data_balanced(
        train_landmarks, train_labels, target_count=target_count
    )
    print(f"Before augmentation: {train_landmarks.shape[0]} sequences")
    print(f"After augmentation:  {train_landmarks_aug.shape[0]} sequences")

    unique_before, counts_before = np.unique(train_labels.astype(int), return_counts=True)
    unique_after, counts_after = np.unique(train_labels_aug.astype(int), return_counts=True)
    all_classes = unique_after
    cb = np.array([counts_before[list(unique_before).index(c)] if c in unique_before else 0
                   for c in all_classes])
    ca = counts_after
    x = np.arange(len(all_classes)); width = 0.35
    plt.figure(figsize=(10, 4))
    plt.bar(x - width / 2, cb, width, label="Before augmentation", color="#1f77b4")
    plt.bar(x + width / 2, ca, width, label="After augmentation", color="#ff7f0e")
    plt.xticks(x, all_classes)
    plt.xlabel("Gesture class"); plt.ylabel("Number of sequences")
    plt.title(f"Class distribution before vs. after augmentation (target_count={target_count})")
    plt.legend()
    plt.tight_layout()
    save_and_show("04_class_distribution_before_vs_after")

    orig_idx = 1
    orig_frame = train_landmarks[orig_idx, train_landmarks.shape[1] // 2]
    aug_seq = augment_gesture_colab(train_landmarks[orig_idx])
    aug_frame = aug_seq[aug_seq.shape[0] // 2]
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    plot_hand_skeleton(axes[0], orig_frame, color="#1f77b4", title="Original")
    plot_hand_skeleton(axes[1], aug_frame, color="#ff7f0e", title="Rotated (augmented)")
    plt.suptitle("Rotation augmentation - before vs. after")
    plt.tight_layout()
    save_and_show("05_original_vs_augmented_skeleton")


EDGES = [(0, 1), (1, 2), (2, 3), (3, 4),
         (0, 5), (5, 6), (6, 7), (7, 8),
         (0, 9), (9, 10), (10, 11), (11, 12),
         (0, 13), (13, 14), (14, 15), (15, 16),
         (0, 17), (17, 18), (18, 19), (19, 20),
         (5, 9), (9, 13), (13, 17)]


def build_adjacency(num_nodes=21):
    A = np.eye(num_nodes)
    for i, j in EDGES:
        A[i, j] = 1
        A[j, i] = 1
    Deg = np.sum(A, axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(Deg))
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt
    return torch.FloatTensor(A_norm)


def compute_velocity(x):
    vel = np.zeros_like(x)
    vel[:, 1:] = x[:, 1:] - x[:, :-1]
    return vel


def prepare_graph_stream(x):
    vel = compute_velocity(x)
    combined = np.concatenate([x, vel], axis=-1)
    combined = combined.transpose(0, 3, 1, 2)
    return combined.astype('float32')


def build_streams_single(seq, scaler):
    """A partir d'UNE sequence brute (81,21,3), calcule les 4 streams (tenseurs)."""
    lm, norm, diff_, feat = create_features(seq[None], with_normalized=True)
    feat = scaler.transform(feat.reshape(-1, 31)).reshape(feat.shape)
    lm_in = prepare_graph_stream(lm)[0]
    norm_in = prepare_graph_stream(norm)[0]
    diff_in = prepare_graph_stream(diff_)[0]
    feat_in = feat[0].astype('float32')
    return (torch.from_numpy(lm_in), torch.from_numpy(norm_in),
            torch.from_numpy(diff_in), torch.from_numpy(feat_in))


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.2, label_smoothing=0.08):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        num_classes = inputs.size(-1)
        log_probs = F.log_softmax(inputs, dim=-1)
        probs = log_probs.exp()
        if self.label_smoothing > 0:
            with torch.no_grad():
                true_dist = torch.zeros_like(log_probs)
                true_dist.fill_(self.label_smoothing / (num_classes - 1))
                true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
            ce = -(true_dist * log_probs).sum(dim=-1)
            pt = (true_dist * probs).sum(dim=-1)
        else:
            ce = F.nll_loss(log_probs, targets, reduction='none')
            pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_term = (1 - pt).clamp(min=1e-6) ** self.gamma
        loss = focal_term * ce
        if self.alpha is not None:
            at = self.alpha.gather(0, targets)
            loss = at * loss
        return loss.mean()


def compute_focal_alpha(labels_int, num_classes, device):
    counts = np.array([np.sum(labels_int == c) for c in range(1, num_classes + 1)], dtype=np.float32)
    counts = np.where(counts == 0, 1, counts)
    alpha = (counts.sum() / (num_classes * counts)) ** 1.5
    alpha = alpha / alpha.mean()
    return torch.FloatTensor(alpha).to(device)


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model.state_dict())

    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k] = v.clone()

    def state_dict(self):
        return self.shadow


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = keep_prob + torch.rand(shape, device=x.device, dtype=x.dtype)
        mask.floor_()
        return x.div(keep_prob) * mask


class UnitGCN(nn.Module):
    def __init__(self, in_channels, out_channels, A, dropout=0.35):
        super().__init__()
        self.register_buffer('A', A)
        self.A_res = nn.Parameter(torch.zeros_like(A))
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = None

    def forward(self, x):
        A = self.A + self.A_res
        x_res = x if self.down is None else self.down(x)
        x = self.conv(x)
        x = torch.einsum('nctv,vw->nctw', x, A)
        x = self.bn(x)
        x = self.relu(x + x_res)
        x = self.dropout(x)
        return x


class UnitTCN(nn.Module):
    def __init__(self, channels, kernel_size=9, stride=1, dropout=0.35):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            channels, channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1)
        )
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, dropout=0.35, drop_path=0.2):
        super().__init__()
        self.gcn = UnitGCN(in_channels, out_channels, A, dropout=dropout)
        self.tcn = UnitTCN(out_channels, stride=stride, dropout=dropout)
        if in_channels != out_channels or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.residual = nn.Identity()
        self.drop_path = DropPath(drop_path)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = self.residual(x)
        x = self.gcn(x)
        x = self.tcn(x)
        x = self.relu(self.drop_path(x) + res)
        return x


class STGCNStream(nn.Module):
    def __init__(self, in_channels, A, num_blocks, drop_path_rate=0.2, dropout=0.35):
        super().__init__()
        num_nodes = A.shape[0]
        self.data_bn = nn.BatchNorm1d(in_channels * num_nodes)
        channels = [in_channels] + num_blocks
        strides = [1, 1, 2, 1, 2, 1][:len(num_blocks)]
        dpr = np.linspace(0, drop_path_rate, len(num_blocks))
        blocks = []
        for i in range(len(num_blocks)):
            blocks.append(
                STGCNBlock(
                    channels[i], channels[i + 1], A,
                    stride=strides[i],
                    dropout=dropout,
                    drop_path=float(dpr[i])
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.out_channels = num_blocks[-1]

    def forward(self, x):
        N, C, T, V = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, V, C, T).permute(0, 2, 3, 1).contiguous()
        for block in self.blocks:
            x = block(x)
        x = x.mean(dim=[2, 3])
        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio, dropout, drop_path):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.drop_path1 = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout)
        )
        self.drop_path2 = DropPath(drop_path)

    def forward(self, x):
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + self.drop_path1(attn_out)
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


class TransformerBranch(nn.Module):
    def __init__(self, patch_dim, seq_len, embed_dim=80, num_heads=5, num_layers=3,
                 mlp_ratio=4.0, dropout=0.1, drop_path_rate=0.1):
        super().__init__()
        self.patch_embed = nn.Linear(patch_dim, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        dpr = np.linspace(0, drop_path_rate, num_layers)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim, num_heads, mlp_ratio, dropout, float(dpr[i])
            ) for i in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch = x.size(0)
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(batch, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x[:, 0]


class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.35):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        return self.net(x)


class Router(nn.Module):
    def __init__(self, input_dim, num_experts):
        super().__init__()
        self.router = nn.Linear(input_dim, num_experts)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.router(x))


class MixtureOfExpertsLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_experts, k, dropout_rate=0.35):
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        self.experts = nn.ModuleList([
            Expert(input_dim, hidden_dim, input_dim, dropout_rate)
            for _ in range(num_experts)
        ])
        self.router = Router(input_dim, num_experts)
        self.noise_std = 0.05

    def forward(self, x):
        batch_size, seq_len, input_dim = x.shape
        x_flat = x.view(-1, input_dim)
        router_probs = self.router(x_flat)
        if self.training:
            noise = torch.randn_like(router_probs) * self.noise_std
            router_probs = torch.softmax(router_probs + noise, dim=-1)
        top_k_probs, top_k_indices = torch.topk(router_probs, self.k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        all_expert_outputs = torch.stack([expert(x_flat) for expert in self.experts], dim=1)
        gathered = torch.gather(
            all_expert_outputs, 1,
            top_k_indices.unsqueeze(-1).expand(-1, -1, input_dim)
        )
        output = (gathered * top_k_probs.unsqueeze(-1)).sum(dim=1)
        output = output.view(batch_size, seq_len, input_dim)
        aux_loss = self._calculate_aux_loss(router_probs)
        return output, {'aux_loss': aux_loss}

    def _calculate_aux_loss(self, router_probs):
        expert_usage = router_probs.mean(dim=0)
        expert_weights = router_probs.sum(dim=0) / router_probs.sum()
        return (expert_usage * expert_weights).sum() * self.num_experts


class HandGestureMultiStreamBackbone(nn.Module):
    def __init__(self, A, num_blocks, transformer_embed_dim=80, transformer_heads=5,
                 transformer_layers=3, transformer_mlp_ratio=4.0, dropout=0.35,
                 drop_path_rate=0.2, output_dim=384):
        super().__init__()
        self.landmark_branch = STGCNStream(
            in_channels=6, A=A, num_blocks=num_blocks,
            drop_path_rate=drop_path_rate, dropout=dropout
        )
        self.norm_branch = STGCNStream(
            in_channels=6, A=A, num_blocks=num_blocks,
            drop_path_rate=drop_path_rate, dropout=dropout
        )
        self.diff_branch = STGCNStream(
            in_channels=6, A=A, num_blocks=num_blocks,
            drop_path_rate=drop_path_rate, dropout=dropout
        )
        self.feature_branch = TransformerBranch(
            patch_dim=31, seq_len=81, embed_dim=transformer_embed_dim,
            num_heads=transformer_heads, num_layers=transformer_layers,
            mlp_ratio=transformer_mlp_ratio, dropout=dropout,
            drop_path_rate=drop_path_rate
        )
        fusion_dim = num_blocks[-1] * 3 + transformer_embed_dim
        self.combiner = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

    def forward(self, landmark_in, norm_in, diff_in, feat_in):
        l = self.landmark_branch(landmark_in)
        n = self.norm_branch(norm_in)
        d = self.diff_branch(diff_in)
        f = self.feature_branch(feat_in)
        combined = torch.cat([l, n, d, f], dim=1)
        return self.combiner(combined)


class GestureMultiStreamMoEModel(nn.Module):
    def __init__(self, num_classes, A, num_blocks, transformer_embed_dim=80, transformer_heads=5,
                 transformer_layers=3, transformer_mlp_ratio=4.0, hidden_size=384, num_experts=10,
                 k=4, num_moe_layers=2, dropout=0.35, drop_path_rate=0.2):
        super().__init__()
        self.backbone = HandGestureMultiStreamBackbone(
            A=A, num_blocks=num_blocks,
            transformer_embed_dim=transformer_embed_dim,
            transformer_heads=transformer_heads,
            transformer_layers=transformer_layers,
            transformer_mlp_ratio=transformer_mlp_ratio,
            dropout=dropout,
            drop_path_rate=drop_path_rate,
            output_dim=hidden_size
        )
        self.moe_layers = nn.ModuleList([
            MixtureOfExpertsLayer(hidden_size, hidden_size * 2, num_experts, k, dropout)
            for _ in range(num_moe_layers)
        ])
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, landmark_in, norm_in, diff_in, feat_in):
        features = self.backbone(landmark_in, norm_in, diff_in, feat_in).unsqueeze(1)
        total_aux_loss = 0
        for moe_layer in self.moe_layers:
            features, info = moe_layer(features)
            features = self.layer_norm(features)
            features = self.dropout(features)
            total_aux_loss += info['aux_loss']
        features = features.squeeze(1)
        output = self.classifier(features)
        return output, {'aux_loss': total_aux_loss}


def build_model(cfg):
    A = build_adjacency(21).to(device)
    return GestureMultiStreamMoEModel(
        num_classes=NUM_CLASSES, A=A,
        num_blocks=[int(x) for x in cfg['num_blocks']],
        transformer_embed_dim=int(cfg['transformer_embed_dim']),
        transformer_heads=int(cfg['transformer_heads']),
        transformer_layers=int(cfg['transformer_layers']),
        transformer_mlp_ratio=float(cfg['transformer_mlp_ratio']),
        hidden_size=int(cfg['hidden_size']),
        num_experts=int(cfg['num_experts']),
        k=int(cfg['k']),
        num_moe_layers=int(cfg['num_moe_layers']),
        dropout=float(cfg['dropout']),
        drop_path_rate=float(cfg['drop_path_rate'])
    ).to(device)


# ---------------------------------------------------------------------------
# Plots de complexite du modele (facon Colab)
# ---------------------------------------------------------------------------
def count_parameters(module):
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def plot_model_complexity(cfg):
    if not MAKE_PLOTS:
        return

    fig, ax = plt.subplots(figsize=(13, 7.5))
    ax.axis("off")

    def box(x, y, w, h, text, color, fontsize=9):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                              linewidth=1.2, edgecolor="#333333", facecolor=color)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, wrap=True)
        return (x + w / 2, y, x + w / 2, y + h)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=12, color="#555555", linewidth=1.2))

    inputs = [("landmarks\n(6,81,21)", "#cfe8ff"), ("normalized\n(6,81,21)", "#cfe8ff"),
              ("differences\n(6,80,21)", "#ffe1c4"), ("features\n(81,31)", "#d9f2d9")]
    in_x = [0.3, 3.3, 6.3, 9.3]
    in_centers = []
    for x, (label, color) in zip(in_x, inputs):
        cx, by, cx2, ty = box(x, 6.4, 2.6, 0.9, label, color)
        in_centers.append((cx, by))

    proc_labels = ["ST-GCN stream", "ST-GCN stream", "ST-GCN stream", "Transformer branch"]
    proc_colors = ["#cfe8ff", "#cfe8ff", "#ffe1c4", "#d9f2d9"]
    proc_centers = []
    for x, label, color, (icx, iby) in zip(in_x, proc_labels, proc_colors, in_centers):
        cx, by, cx2, ty = box(x, 4.9, 2.6, 0.9, label, color, fontsize=8)
        arrow(icx, iby, cx, ty)
        proc_centers.append((cx, by))

    ccx, cby, ccx2, cty = box(4.5, 3.4, 4.0, 0.9,
                              "Combiner (concat -> MLP)\n-> hidden_size embedding", "#f5d0d0")
    for cx, by in proc_centers:
        arrow(cx, by, ccx, cty)

    m1x, m1y, m1x2, m1y2 = box(4.5, 2.1, 4.0, 0.9,
                               f"MoE layer 1\n(router + top-{cfg['k']} experts)", "#e6d0f5")
    arrow(ccx, cby, m1x, m1y2)
    box(4.5, 1.1, 4.0, 0.9, f"MoE layer 2\n(router + top-{cfg['k']} experts)", "#e6d0f5")
    box(4.5, 0.1, 4.0, 0.7, "Linear classifier -> 13 classes", "#ffe8a3")

    ax.set_xlim(0, 13); ax.set_ylim(-0.1, 7.6)
    plt.title("GestureMultiStreamMoEModel - multi-stream ST-GCN/Transformer + MoE head", fontsize=12)
    plt.tight_layout()
    save_and_show("06_architecture_diagram")

    model_probe = build_model(cfg)
    blocks = {
        "backbone": model_probe.backbone,
        "moe_layers": model_probe.moe_layers,
        "layer_norm": model_probe.layer_norm,
        "classifier": model_probe.classifier,
    }
    names, counts = [], []
    print(f"{'Block':<15}{'Parameters':>15}")
    print("-" * 30)
    for name, block in blocks.items():
        total, _ = count_parameters(block)
        print(f"{name:<15}{total:>15,}")
        names.append(name)
        counts.append(total)
    total_params, trainable_params = count_parameters(model_probe)
    print("-" * 30)
    print(f"{'TOTAL':<15}{total_params:>15,}")
    print(f"Trainable parameters: {trainable_params:,}")

    plt.figure(figsize=(7, 4))
    plt.bar(names, counts, color=["#cfe8ff", "#e6d0f5", "#d9f2d9", "#ffe8a3"])
    plt.ylabel("Number of parameters")
    plt.title("Parameter count per model block")
    for i, c in enumerate(counts):
        plt.text(i, c + max(counts) * 0.01, f"{c:,}", ha="center", fontsize=8)
    plt.tight_layout()
    save_and_show("07_parameter_count")

    del model_probe
    if device.type == "cuda":
        torch.cuda.empty_cache()


def plot_training_curves(history, run_id):
    if not MAKE_PLOTS or len(history["train_loss"]) == 0:
        return
    has_val = len(history["val_loss"]) > 0
    epochs_range = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))

    axes[0].plot(epochs_range, history["train_loss"], label="Train", marker="o", markersize=3)
    if has_val:
        axes[0].plot(epochs_range, history["val_loss"], label="Validation", marker="o", markersize=3)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Focal loss")
    axes[0].set_title("Main classification loss"); axes[0].legend()

    axes[1].plot(epochs_range, history["train_aux_loss"], label="Train", marker="o", markersize=3)
    if has_val:
        axes[1].plot(epochs_range, history["val_aux_loss"], label="Validation", marker="o", markersize=3)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Aux loss")
    axes[1].set_title("MoE load-balancing auxiliary loss"); axes[1].legend()

    axes[2].plot(epochs_range, history["train_acc"], label="Train", marker="o", markersize=3)
    if has_val:
        axes[2].plot(epochs_range, history["val_acc"], label="Validation", marker="o", markersize=3)
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Accuracy (%)")
    axes[2].set_title("Classification accuracy"); axes[2].legend()

    plt.tight_layout()
    save_and_show(f"08_training_curves_run{run_id}")


def plot_val_evaluation(val_preds, val_targets, run_id):
    if not MAKE_PLOTS:
        return
    val_preds = np.array(val_preds) + 1
    val_targets = np.array(val_targets) + 1
    class_labels = sorted(np.unique(val_targets))
    print(f"[Run {run_id}] Validation accuracy: {(val_preds == val_targets).mean() * 100:.2f}%")

    cm = confusion_matrix(val_targets, val_preds, labels=class_labels)
    plt.figure(figsize=(8, 6.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_labels, yticklabels=class_labels, cbar=True)
    plt.xlabel("Predicted class"); plt.ylabel("True class")
    plt.title(f"Confusion matrix - validation set (run {run_id})")
    plt.tight_layout()
    save_and_show(f"09_confusion_matrix_run{run_id}")

    precision, recall, f1, support = precision_recall_fscore_support(
        val_targets, val_preds, labels=class_labels, zero_division=0
    )
    metrics_df = pd.DataFrame({
        "class": class_labels, "precision": precision,
        "recall": recall, "f1_score": f1, "support": support,
    })
    print(metrics_df.to_string(index=False))
    print("\nFull classification report:")
    print(classification_report(val_targets, val_preds, labels=class_labels, zero_division=0))

    x = np.arange(len(class_labels)); width = 0.25
    plt.figure(figsize=(11, 4.5))
    plt.bar(x - width, precision, width, label="Precision", color="#1f77b4")
    plt.bar(x, recall, width, label="Recall", color="#ff7f0e")
    plt.bar(x + width, f1, width, label="F1-score", color="#2ca02c")
    plt.xticks(x, class_labels)
    plt.xlabel("Gesture class"); plt.ylabel("Score"); plt.ylim(0, 1.05)
    plt.title(f"Per-class precision / recall / F1-score - validation set (run {run_id})")
    plt.legend()
    plt.tight_layout()
    save_and_show(f"10_prf_per_class_run{run_id}")

    macro_f1 = f1.mean()
    weighted_f1 = np.average(f1, weights=support)
    print(f"Macro-average F1: {macro_f1:.4f}")
    print(f"Weighted-average F1: {weighted_f1:.4f}")


# ---------------------------------------------------------------------------
# Dataset avec AUGMENTATION EN LIGNE (rotation fraiche a chaque epoch)
# + equilibrage par target_count via re-echantillonnage d'indices.
# ---------------------------------------------------------------------------
class HandGestureOnlineDataset(Dataset):
    def __init__(self, landmarks_raw, labels, scaler, target_count=None, augment=False):
        """
        landmarks_raw : (N, 81, 21, 3) landmarks bruts (deja scale_normalized)
        labels        : (N,) labels 1-indexes (ou None pour l'inference)
        scaler        : StandardScaler deja fit
        target_count  : si defini + labels, construit une liste d'indices equilibree
                        (chaque classe apparait target_count fois) -> remplace les
                        copies offline. augment=True => rotation aleatoire par item.
        """
        self.landmarks = landmarks_raw.astype('float32')
        self.scaler = scaler
        self.augment = augment
        self.labels = labels.astype(int) if labels is not None else None

        if target_count is not None and self.labels is not None:
            idx_list = []
            for lab in np.unique(self.labels):
                lab_idx = np.where(self.labels == lab)[0]
                if len(lab_idx) < target_count:
                    extra = np.random.choice(lab_idx, size=target_count - len(lab_idx), replace=True)
                    idx_list.append(np.concatenate([lab_idx, extra]))
                else:
                    idx_list.append(lab_idx)
            self.sample_indices = np.concatenate(idx_list)
        else:
            self.sample_indices = np.arange(len(self.landmarks))

    @property
    def balanced_labels(self):
        return self.labels[self.sample_indices] if self.labels is not None else None

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, i):
        idx = self.sample_indices[i]
        seq = self.landmarks[idx]
        if self.augment:
            seq = augment_gesture_colab(seq)   # rotation fraiche a chaque acces
        streams = build_streams_single(seq, self.scaler)
        if self.labels is not None:
            return streams, int(self.labels[idx]) - 1
        return streams


def make_loader(dataset, batch_size, shuffle):
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == 'cuda'),
        persistent_workers=(NUM_WORKERS > 0),
    )


def evaluate(model, loader, criterion, aux_loss_weight):
    model.eval()
    correct, total, running_loss, running_aux = 0, 0, 0.0, 0.0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for inputs, targets in loader:
            landmark_in, norm_in, diff_in, feat_in = [inp.to(device) for inp in inputs]
            targets = targets.to(device)
            outputs, moe_info = model(landmark_in, norm_in, diff_in, feat_in)
            main_loss = criterion(outputs, targets)
            running_loss += main_loss.item()
            running_aux += moe_info['aux_loss'].item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    return (100. * correct / total, running_loss / len(loader),
            running_aux / len(loader), np.array(all_preds), np.array(all_targets))


def rotate_sequence(seq, ang, axis):
    out = seq.copy()
    if axis == 0:
        R = np.array([[1, 0, 0], [0, np.cos(ang), -np.sin(ang)], [0, np.sin(ang), np.cos(ang)]], dtype=np.float32)
    elif axis == 1:
        R = np.array([[np.cos(ang), 0, np.sin(ang)], [0, 1, 0], [-np.sin(ang), 0, np.cos(ang)]], dtype=np.float32)
    else:
        R = np.array([[np.cos(ang), -np.sin(ang), 0], [np.sin(ang), np.cos(ang), 0], [0, 0, 1]], dtype=np.float32)
    out = out.reshape(-1, 3)
    out = out @ R.T
    return out.reshape(seq.shape)


# ---------------------------------------------------------------------------
# Ensemble : chargement + prediction test (TTA elaguee, rotations seules)
# ---------------------------------------------------------------------------
def load_ensemble(cfg, n_models=N_FOLDS, suffix=''):
    models, scalers = [], []
    for run_id in range(1, n_models + 1):
        model_path = f"best_model_run{run_id}{suffix}.pth"
        scaler_path = f"scaler_best_run{run_id}{suffix}.pkl"
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            print(f"Attention: modele/scaler du run {run_id} manquant, skip.")
            continue
        with open(scaler_path, 'rb') as f:
            scalers.append(pickle.load(f))
        model = build_model(cfg)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        models.append(model)
    print(f"Charge {len(models)} modeles.")
    return models, scalers


def backup_round1_models(n_models=N_FOLDS, suffix='_r1'):
    """Sauvegarde les modeles/scalers round-1 (suffixe _r1) AVANT que le round 2
    ne les ecrase. Ne remplace jamais un backup existant -> le vrai round-1 est
    preserve meme si on relance le round 2 plusieurs fois."""
    print("\n--- Sauvegarde des modeles round-1 (avant ecrasement par le round 2) ---")
    n_saved = 0
    for run_id in range(1, n_models + 1):
        pairs = [
            (f"best_model_run{run_id}.pth",  f"best_model_run{run_id}{suffix}.pth"),
            (f"scaler_best_run{run_id}.pkl", f"scaler_best_run{run_id}{suffix}.pkl"),
        ]
        for src, dst in pairs:
            if os.path.exists(src) and not os.path.exists(dst):
                shutil.copy2(src, dst)
                n_saved += 1
                print(f"  {src} -> {dst}")
            elif os.path.exists(dst):
                print(f"  {dst} existe deja (backup round-1 preserve, non ecrase)")
    if n_saved == 0:
        print("  Aucun nouveau backup (deja fait ou modeles round-1 absents).")


def predict_test_probs(models, scalers, test_landmarks, batch_size=32):
    """Retourne les probabilites moyennes (ensemble + TTA rotations) : (N, NUM_CLASSES)."""
    out_probs = []
    for start in tqdm(range(0, len(test_landmarks), batch_size), desc="TTA + Ensemble"):
        batch = test_landmarks[start:start + batch_size]
        model_probs = []
        for m_idx, model in enumerate(models):
            scaler = scalers[m_idx]
            tta_probs = []
            for cfg_tta in TTA_CONFIGS:
                aug = batch.copy()
                for i in range(len(aug)):
                    aug[i] = rotate_sequence(aug[i], cfg_tta["angle"], cfg_tta["axis"])
                lm, norm, diff_, feat = create_features(aug, with_normalized=True)
                feat = scaler.transform(feat.reshape(-1, 31)).reshape(feat.shape)
                landmark_in = torch.FloatTensor(prepare_graph_stream(lm)).to(device)
                norm_in = torch.FloatTensor(prepare_graph_stream(norm)).to(device)
                diff_in = torch.FloatTensor(prepare_graph_stream(diff_)).to(device)
                feat_in = torch.FloatTensor(feat).to(device)
                with torch.no_grad():
                    outputs, _ = model(landmark_in, norm_in, diff_in, feat_in)
                    tta_probs.append(F.softmax(outputs, dim=1))
            model_probs.append(torch.stack(tta_probs).mean(0))
        ensemble = torch.stack(model_probs).mean(0)
        out_probs.append(ensemble.cpu().numpy())
    return np.concatenate(out_probs, axis=0)


def generate_pseudo_labels(cfg, threshold=PSEUDO_THRESHOLD):
    """Charge l'ensemble round-1, predit le test, retourne (landmarks, labels) confiants."""
    print("\n--- Generation des pseudo-labels (round 2) ---")
    models, scalers = load_ensemble(cfg)
    if len(models) == 0:
        print("Aucun modele round-1 trouve -> pas de pseudo-labels.")
        return None, None

    test_landmarks = load_test_data()
    test_landmarks = scale_normalize(test_landmarks)
    probs = predict_test_probs(models, scalers, test_landmarks)

    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1) + 1                      # labels 1-indexes
    mask = conf > threshold
    n_conf = int(mask.sum())
    print(f"Pseudo-labels retenus: {n_conf}/{len(test_landmarks)} "
          f"(confiance > {threshold}) = {100.*n_conf/len(test_landmarks):.1f}% du test")
    if n_conf > 0:
        uniq, cnt = np.unique(pred[mask], return_counts=True)
        print(f"Repartition des pseudo-labels: {dict(zip(uniq.tolist(), cnt.tolist()))}")

    pl_landmarks = test_landmarks[mask].astype('float32')
    pl_labels = pred[mask].astype(str)                   # meme dtype (str) que train_labels

    for m in models:
        del m
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return (pl_landmarks, pl_labels) if n_conf > 0 else (None, None)


def train_single_config(run_id, cfg, train_landmarks_raw, train_labels_raw,
                        val_landmarks_raw, val_labels,
                        pseudo_landmarks=None, pseudo_labels=None):
    print(f"\n========== RUN {run_id}: {cfg['name']} ==========")
    print(cfg)

    torch.manual_seed(cfg.get('seed', 42) + run_id)
    np.random.seed(cfg.get('seed', 42) + run_id)

    target_count = cfg.get('target_count', 450)

    # Pseudo-labels : ajoutes UNIQUEMENT au train (jamais a la val)
    if pseudo_landmarks is not None and len(pseudo_landmarks) > 0:
        print(f"Ajout de {len(pseudo_landmarks)} pseudo-labels au train du run {run_id}")
        train_landmarks_raw = np.concatenate([train_landmarks_raw, pseudo_landmarks], axis=0)
        train_labels_raw = np.concatenate([train_labels_raw, pseudo_labels])

    # Scaler fit sur les features du train (rotation-invariantes en majorite)
    _, _, _, feat_for_scaler = create_features(train_landmarks_raw, with_normalized=True)
    scaler = StandardScaler().fit(feat_for_scaler.reshape(-1, 31))
    with open(f'scaler_run{run_id}.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    # Datasets : train = augmentation EN LIGNE + equilibrage target_count ; val = brut
    train_dataset = HandGestureOnlineDataset(
        train_landmarks_raw, train_labels_raw, scaler,
        target_count=target_count, augment=True
    )
    val_dataset = HandGestureOnlineDataset(
        val_landmarks_raw, val_labels, scaler, target_count=None, augment=False
    )
    train_loader = make_loader(train_dataset, cfg['batch_size'], shuffle=True)
    val_loader = make_loader(val_dataset, cfg['batch_size'], shuffle=False)

    alpha = compute_focal_alpha(train_dataset.balanced_labels, NUM_CLASSES, device)

    model = build_model(cfg)
    ema_model = build_model(cfg)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Run {run_id} total parameters: {total_params:,}")
    print(f"Run {run_id} train samples (equilibres): {len(train_dataset)} | val: {len(val_dataset)}")

    criterion = FocalLoss(alpha=alpha, gamma=cfg['focal_gamma'], label_smoothing=cfg['label_smoothing'])
    optimizer = optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=cfg['T_0'], T_mult=1, eta_min=1e-6)
    ema = EMA(model, decay=cfg['ema_decay'])

    history = {"train_loss": [], "train_aux_loss": [], "train_acc": [],
               "val_loss": [], "val_aux_loss": [], "val_acc": []}

    best_val_acc, best_train_acc, best_epoch, patience_counter = 0.0, 0.0, 0, 0
    save_path = f'best_model_run{run_id}.pth'

    for epoch in range(cfg['epochs']):
        model.train()
        correct, total, running_loss, running_aux = 0, 0, 0.0, 0.0

        for inputs, targets in train_loader:
            landmark_in, norm_in, diff_in, feat_in = [inp.to(device) for inp in inputs]
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs, moe_info = model(landmark_in, norm_in, diff_in, feat_in)
            main_loss = criterion(outputs, targets)
            total_loss = main_loss + cfg['aux_loss_weight'] * moe_info['aux_loss']
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            ema.update(model)

            running_loss += main_loss.item()
            running_aux += moe_info['aux_loss'].item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        scheduler.step()
        train_acc = 100. * correct / total
        train_loss = running_loss / len(train_loader)
        train_aux = running_aux / len(train_loader)

        ema_model.load_state_dict(ema.state_dict())
        val_acc, val_loss, val_aux, _, _ = evaluate(ema_model, val_loader, criterion, cfg['aux_loss_weight'])

        history["train_loss"].append(train_loss)
        history["train_aux_loss"].append(train_aux)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_aux_loss"].append(val_aux)
        history["val_acc"].append(val_acc)

        wandb.log({'run_id': run_id, 'epoch': epoch + 1,
                   'train_loss': train_loss, 'train_aux_loss': train_aux, 'train_accuracy': train_acc,
                   'val_loss': val_loss, 'val_aux_loss': val_aux, 'val_accuracy': val_acc})

        print(f'Run {run_id} Epoch {epoch+1}/{cfg["epochs"]}: '
              f'Train Loss {train_loss:.4f} Acc {train_acc:.2f}% | '
              f'EMA Val Loss {val_loss:.4f} Acc {val_acc:.2f}%')

        if val_acc > best_val_acc:
            best_val_acc, best_train_acc, best_epoch, patience_counter = val_acc, train_acc, epoch + 1, 0
            torch.save(ema_model.state_dict(), save_path)
            with open(f'scaler_best_run{run_id}.pkl', 'wb') as f:
                pickle.dump(scaler, f)
        else:
            patience_counter += 1
            if patience_counter >= cfg['patience']:
                print(f'Run {run_id} early stopping at epoch {epoch+1}')
                break

    plot_training_curves(history, run_id)

    if os.path.exists(save_path):
        ema_model.load_state_dict(torch.load(save_path, map_location=device))
        _, _, _, val_preds, val_targets = evaluate(ema_model, val_loader, criterion, cfg['aux_loss_weight'])
        plot_val_evaluation(val_preds, val_targets, run_id)

    del model, ema_model
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return {'run_id': run_id, 'name': cfg['name'], 'best_epoch': best_epoch,
            'best_train_acc': best_train_acc, 'best_val_acc': best_val_acc,
            'save_path': save_path, 'scaler_path': f'scaler_best_run{run_id}.pkl',
            'total_params': total_params, **cfg}


BASE = {
    'batch_size': 32,
    'epochs': 150,
    'patience': 15,
    'learning_rate': 7e-5,
    'weight_decay': 8e-4,
    'dropout': 0.35,
    'drop_path_rate': 0.2,
    'aux_loss_weight': 0.02,
    'num_blocks': [64, 64, 128, 128, 256, 384],
    'transformer_embed_dim': 80,
    'transformer_heads': 4,
    'transformer_layers': 3,
    'transformer_mlp_ratio': 4.0,
    'hidden_size': 384,
    'num_experts': 10,
    'k': 4,
    'num_moe_layers': 2,
    'focal_gamma': 2.2,
    'label_smoothing': 0.08,
    'ema_decay': 0.999,
    'T_0': 25,
    'target_count': 650,
    'seed': 42,
}

CONFIGS = [
    {**BASE, 'name': 'fixed_v2', 'learning_rate': 1e-4, 'target_count': 800},
]

if __name__ == "__main__":
    if not INFERENCE_ONLY:
        print("Loading training data...")
        train_landmarks, train_labels = load_train_data()
        print(f"train_landmarks shape: {train_landmarks.shape}")

        plot_data_exploration(train_landmarks, train_labels)
        plot_augmentation_effect(train_landmarks, train_labels, CONFIGS[0]['target_count'])
        plot_model_complexity(CONFIGS[0])

        train_landmarks = scale_normalize(train_landmarks)

        # --- Round 2 : generation des pseudo-labels a partir de l'ensemble round-1 ---
        pseudo_landmarks, pseudo_labels = None, None
        if USE_PSEUDO_LABELS:
            backup_round1_models()   # securise les modeles round-1 (suffixe _r1) avant ecrasement
            pseudo_landmarks, pseudo_labels = generate_pseudo_labels(CONFIGS[0], PSEUDO_THRESHOLD)

        if USE_KFOLD:
            print(f"\nUtilisation de la validation croisee {N_FOLDS}-fold")
            skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
            all_val_scores = []

            for fold, (train_idx, val_idx) in enumerate(skf.split(train_landmarks, train_labels.astype(int))):
                print(f"\n{'='*60}\nFOLD {fold+1}/{N_FOLDS}\n{'='*60}")
                cfg = CONFIGS[0].copy()
                cfg['name'] = f"{cfg['name']}_fold{fold+1}"
                res = train_single_config(
                    fold + 1, cfg,
                    train_landmarks[train_idx], train_labels[train_idx],
                    train_landmarks[val_idx], train_labels[val_idx],
                    pseudo_landmarks=pseudo_landmarks, pseudo_labels=pseudo_labels
                )
                all_val_scores.append(res['best_val_acc'])

            print(f"\n{'='*60}\nRESULTATS VALIDATION CROISEE {N_FOLDS}-FOLD\n{'='*60}")
            print(f"Scores par fold: {[f'{s:.2f}%' for s in all_val_scores]}")
            print(f"Moyenne: {np.mean(all_val_scores):.2f}% +/- {np.std(all_val_scores):.2f}%")

        else:
            indices = np.arange(len(train_landmarks))
            train_idx, val_idx = train_test_split(
                indices, test_size=0.2, random_state=42, stratify=train_labels.astype(int)
            )
            print(f"Train raw samples: {len(train_idx)} | Validation samples: {len(val_idx)}")
            results = []
            for i, cfg in enumerate(CONFIGS, start=1):
                res = train_single_config(
                    i, cfg,
                    train_landmarks[train_idx], train_labels[train_idx],
                    train_landmarks[val_idx], train_labels[val_idx],
                    pseudo_landmarks=pseudo_landmarks, pseudo_labels=pseudo_labels
                )
                results.append(res)
            df = pd.DataFrame(results).sort_values('best_val_acc', ascending=False).reset_index(drop=True)
            print("\n=== RESULTATS ===")
            print(df[['run_id', 'name', 'best_val_acc', 'best_train_acc', 'best_epoch', 'total_params']].to_string(index=False))
            best_row = df.iloc[0]
            print(f"\nMEILLEUR MODELE: {best_row['name']} (run {int(best_row['run_id'])})")
            print(f"Val acc: {float(best_row['best_val_acc']):.2f}%")

    else:
        print("\n" + "=" * 60)
        print("MODE INFERENCE AVEC ENSEMBLE + TTA (rotations seules)")
        print("=" * 60)

        cfg = CONFIGS[0].copy()
        models, scalers = load_ensemble(cfg)

        print("Loading test data...")
        test_landmarks = load_test_data()
        test_landmarks = scale_normalize(test_landmarks)

        probs = predict_test_probs(models, scalers, test_landmarks)
        predictions = probs.argmax(axis=1) + 1

        submission = pd.DataFrame({'Id': range(len(predictions)), 'Gesture': predictions})
        submission.to_csv('submission_ensemble_final.csv', index=False)
        print("Submission saved: submission_ensemble_final.csv")
        print(submission.head())
        print("\nDistribution des predictions:")
        print(pd.Series(predictions).value_counts().sort_index())

        if MAKE_PLOTS:
            pred_unique, pred_counts = np.unique(predictions, return_counts=True)
            plt.figure(figsize=(9, 4))
            plt.bar(pred_unique, pred_counts, color="#9467bd")
            plt.xlabel("Predicted gesture class"); plt.ylabel("Number of test sequences")
            plt.title("Predicted class distribution on the (unlabelled) test set")
            plt.xticks(pred_unique)
            plt.tight_layout()
            save_and_show("11_test_prediction_distribution")

    wandb.finish()
