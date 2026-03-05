import numpy as np
from pathlib import Path
import json
import cv2
from copy import deepcopy
import random
import spectral.io.envi as envi

def get_nonzero_polygon(cube):
    mask = cube.sum(axis=0) != 0
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    y1, y2 = ys.min(), ys.max() + 1
    x1, x2 = xs.min(), xs.max() + 1
    polygon = [[int(x1), int(y1)], [int(x2), int(y1)], [int(x2), int(y2)], [int(x1), int(y2)]]
    return polygon, (y1, y2, x1, x2)

def replace_extrude_with_augmented(plot_cube, extrude_polygon, available_extrudes_dict):
    y1, y2 = extrude_polygon[0][1], extrude_polygon[2][1]
    x1, x2 = extrude_polygon[0][0], extrude_polygon[2][0]
    H_ext = y2 - y1
    W_ext = x2 - x1
    B = plot_cube.shape[0]

    classes = list(available_extrudes_dict.keys())
    random.shuffle(classes)
    for cls in classes:
        random.shuffle(available_extrudes_dict[cls])
        for idx, aug_cube in enumerate(available_extrudes_dict[cls]):
            polygon_aug, (y1a, y2a, x1a, x2a) = get_nonzero_polygon(aug_cube)
            if polygon_aug is None:
                continue
            H_aug = y2a - y1a
            W_aug = x2a - x1a
            if H_aug >= H_ext and W_aug >= W_ext:
                patch = aug_cube[:, y1a:y1a+H_ext, x1a:x1a+W_ext]
                plot_cube[:, y1:y2, x1:x2] = patch
                # Supprimer l'extrude utilisé pour ne pas le réutiliser
                del available_extrudes_dict[cls][idx]
                if len(available_extrudes_dict[cls]) == 0:
                    del available_extrudes_dict[cls]
                return polygon_aug, cls
    return None, None

def process_plot_with_augmented(plot_cube, labelme_shapes, available_extrudes_dict):
    filled_cube = deepcopy(plot_cube)
    new_shapes = []

    for shape in labelme_shapes:
        polygon = shape["points"]
        polygon_aug, cls_used = replace_extrude_with_augmented(filled_cube, polygon, available_extrudes_dict)
        if polygon_aug is not None:
            new_shapes.append({
                "label": cls_used,
                "points": polygon_aug,
                "group_id": None,
                "shape_type": "polygon",
                "flags": {}
            })
        else:
            # Si aucun extrude compatible n'est trouvé, garder l'extrude original
            new_shapes.append(shape)
    return filled_cube, new_shapes

def save_labelme_json(shapes, cube_name, H, W, out_path):
    labelme_json = {
        "version": "5.0.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": f"{cube_name}.hdr",
        "imageData": None,
        "imageHeight": H,
        "imageWidth": W
    }
    with open(out_path, "w") as f:
        json.dump(labelme_json, f, indent=2)

def load_augmented_cubes(augmented_dir):
    augmented_dir = Path(augmented_dir)
    extrudes_dict = {}
    for class_dir in augmented_dir.iterdir():
        if not class_dir.is_dir():
            continue
        cubes_list = []
        for cube_path in class_dir.glob("*.hdr"):
            hdr_img = envi.open(str(cube_path))
            cube = np.array(hdr_img.load(), dtype=np.float32)
            cubes_list.append(cube)
        if cubes_list:
            extrudes_dict[class_dir.name] = cubes_list
    return extrudes_dict

def generate_all_plots_with_augmented(plot_cubes_dir, augmented_dir, out_dir):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    plot_paths = list(Path(plot_cubes_dir).glob("*.hdr"))
    if not plot_paths:
        print("Aucun plot trouvé")
        return

    available_extrudes = load_augmented_cubes(augmented_dir)
    cube_count = 0

    while available_extrudes:  # Tant qu'il reste des extrudes augmentés
        plot_path = random.choice(plot_paths)
        hdr_img = envi.open(str(plot_path))
        plot_cube = np.array(hdr_img.load(), dtype=np.float32)

        # Charger LabelMe JSON
        json_path = plot_path.with_suffix(".json")
        with open(json_path) as f:
            labelme_data = json.load(f)
        labelme_shapes = labelme_data["shapes"]

        # Remplacer les extrudes
        filled_cube, new_shapes = process_plot_with_augmented(plot_cube, labelme_shapes, available_extrudes)

        # Sauvegarde cube
        out_cube_path = Path(out_dir)/f"plot_aug_{cube_count}.hdr"
        envi.save_image(str(out_cube_path), filled_cube, dtype=np.float32, force=True, interleave="bil")

        # Sauvegarde LabelMe
        out_json_path = Path(out_dir)/f"plot_aug_{cube_count}.json"
        save_labelme_json(new_shapes, f"plot_aug_{cube_count}", filled_cube.shape[1], filled_cube.shape[2], out_json_path)

        cube_count += 1
        print(f"Cube généré : {out_cube_path.name}, extrudes restants : {sum(len(v) for v in available_extrudes.values())}")

        #######################################3333
        import numpy as np
from pathlib import Path
import json
import cv2
from copy import deepcopy
import random
import spectral.io.envi as envi

# --- Utilitaires ---
def get_nonzero_polygon(cube):
    mask = cube.sum(axis=0) != 0
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    y1, y2 = ys.min(), ys.max() + 1
    x1, x2 = xs.min(), xs.max() + 1
    polygon = [[int(x1), int(y1)], [int(x2), int(y1)], [int(x2), int(y2)], [int(x1), int(y2)]]
    return polygon, (y1, y2, x1, x2)

def replace_extrude(plot_cube, extrude_polygon, available_extrudes_dict):
    y1, y2 = extrude_polygon[0][1], extrude_polygon[2][1]
    x1, x2 = extrude_polygon[0][0], extrude_polygon[2][0]
    H_ext = y2 - y1
    W_ext = x2 - x1

    classes = list(available_extrudes_dict.keys())
    random.shuffle(classes)
    for cls in classes:
        random.shuffle(available_extrudes_dict[cls])
        for idx, aug_cube in enumerate(available_extrudes_dict[cls]):
            polygon_aug, (y1a, y2a, x1a, x2a) = get_nonzero_polygon(aug_cube)
            if polygon_aug is None:
                continue
            H_aug = y2a - y1a
            W_aug = x2a - x1a
            if H_aug >= H_ext and W_aug >= W_ext:
                patch = aug_cube[:, y1a:y1a+H_ext, x1a:x1a+W_ext]
                plot_cube[:, y1:y2, x1:x2] = patch
                # Supprimer l'extrude utilisé pour ne pas le réutiliser
                del available_extrudes_dict[cls][idx]
                if len(available_extrudes_dict[cls]) == 0:
                    del available_extrudes_dict[cls]
                return polygon_aug, cls
    return None, None

def process_plot(plot_cube, labelme_shapes, available_extrudes_dict):
    filled_cube = deepcopy(plot_cube)
    new_shapes = []
    for shape in labelme_shapes:
        polygon = shape["points"]
        polygon_aug, cls_used = replace_extrude(filled_cube, polygon, available_extrudes_dict)
        if polygon_aug is not None:
            new_shapes.append({
                "label": cls_used,
                "points": polygon_aug,
                "group_id": None,
                "shape_type": "polygon",
                "flags": {}
            })
        else:
            # Si aucun extrude compatible n'est trouvé, garder l'extrude original
            new_shapes.append(shape)
    return filled_cube, new_shapes

def save_labelme_json(shapes, cube_name, H, W, out_path):
    labelme_json = {
        "version": "5.0.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": f"{cube_name}.hdr",
        "imageData": None,
        "imageHeight": H,
        "imageWidth": W
    }
    with open(out_path, "w") as f:
        json.dump(labelme_json, f, indent=2)

# --- Chargement initial ---
def load_augmented_cubes(augmented_dir):
    augmented_dir = Path(augmented_dir)
    extrudes_dict = {}
    for class_dir in augmented_dir.iterdir():
        if not class_dir.is_dir():
            continue
        cubes_list = []
        for cube_path in class_dir.glob("*.hdr"):
            hdr_img = envi.open(str(cube_path))
            cube = np.array(hdr_img.load(), dtype=np.float32)
            cubes_list.append(cube)
        if cubes_list:
            extrudes_dict[class_dir.name] = cubes_list
    return extrudes_dict

def load_labelme_json(json_path):
    with open(json_path) as f:
        data = json.load(f)
    return data["shapes"]

# --- Pipeline principal ---
def generate_augmented_plots(plot_cubes_dir, augmented_dir, out_dir):
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Précharger plots de base
    plot_paths = list(Path(plot_cubes_dir).glob("*.hdr"))
    if not plot_paths:
        print("Aucun plot trouvé")
        return
    plots = []
    for plot_path in plot_paths:
        hdr_img = envi.open(str(plot_path))
        cube = np.array(hdr_img.load(), dtype=np.float32)
        json_path = plot_path.with_suffix(".json")
        shapes = load_labelme_json(json_path)
        plots.append((cube, shapes))

    # Précharger extrudes augmentés
    available_extrudes = load_augmented_cubes(augmented_dir)

    cube_count = 0
    while available_extrudes:  # Tant qu'il reste des extrudes augmentés
        # Sélection aléatoire d'un plot de base
        plot_cube, labelme_shapes = random.choice(plots)

        # Génération du cube augmenté
        filled_cube, new_shapes = process_plot(plot_cube, labelme_shapes, available_extrudes)

        # Sauvegarde cube
        out_cube_path = Path(out_dir)/f"plot_aug_{cube_count}.hdr"
        envi.save_image(str(out_cube_path), filled_cube, dtype=np.float32, force=True, interleave="bil")

        # Sauvegarde LabelMe
        out_json_path = Path(out_dir)/f"plot_aug_{cube_count}.json"
        save_labelme_json(new_shapes, f"plot_aug_{cube_count}", filled_cube.shape[1], filled_cube.shape[2], out_json_path)

        cube_count += 1
        total_remaining = sum(len(v) for v in available_extrudes.values())
        print(f"Cube généré : {out_cube_path.name}, extrudes restants : {total_remaining}")
        ######################################3


import numpy as np
from pathlib import Path
import json
from copy import deepcopy
import random
import spectral.io.envi as envi
import matplotlib.pyplot as plt

# --- Utilitaires ---
def get_nonzero_polygon(cube):
    """Calcule la bounding box 2D d'un cube en supprimant les pixels nuls sur les contours"""
    mask = cube.sum(axis=0) != 0
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    y1, y2 = ys.min(), ys.max() + 1
    x1, x2 = xs.min(), xs.max() + 1
    polygon = [[int(x1), int(y1)], [int(x2), int(y1)], [int(x2), int(y2)], [int(x1), int(y2)]]
    return polygon, (y1, y2, x1, x2)

def replace_extrude(plot_cube, extrude_polygon, available_extrudes_dict):
    """Remplace un extrude par un extrude augmenté compatible et le retire de la liste"""
    y1, y2 = extrude_polygon[0][1], extrude_polygon[2][1]
    x1, x2 = extrude_polygon[0][0], extrude_polygon[2][0]
    H_ext = y2 - y1
    W_ext = x2 - x1

    classes = list(available_extrudes_dict.keys())
    random.shuffle(classes)
    for cls in classes:
        random.shuffle(available_extrudes_dict[cls])
        for idx, aug_cube in enumerate(available_extrudes_dict[cls]):
            polygon_aug, (y1a, y2a, x1a, x2a) = get_nonzero_polygon(aug_cube)
            if polygon_aug is None:
                continue
            H_aug = y2a - y1a
            W_aug = x2a - x1a
            if H_aug >= H_ext and W_aug >= W_ext:
                patch = aug_cube[:, y1a:y1a+H_ext, x1a:x1a+W_ext]
                plot_cube[:, y1:y2, x1:x2] = patch
                # Supprimer l'extrude utilisé
                del available_extrudes_dict[cls][idx]
                if len(available_extrudes_dict[cls]) == 0:
                    del available_extrudes_dict[cls]
                return polygon_aug, cls
    return None, None

def process_plot(plot_cube, labelme_shapes, available_extrudes_dict):
    """Remplace tous les extrudes du plot par des extrudes augmentés"""
    filled_cube = deepcopy(plot_cube)
    new_shapes = []
    for shape in labelme_shapes:
        polygon = shape["points"]
        polygon_aug, cls_used = replace_extrude(filled_cube, polygon, available_extrudes_dict)
        if polygon_aug is not None:
            new_shapes.append({
                "label": cls_used,
                "points": polygon_aug,
                "group_id": None,
                "shape_type": "polygon",
                "flags": {}
            })
        else:
            new_shapes.append(shape)
    return filled_cube, new_shapes

def save_labelme_json(shapes, cube_name, H, W, out_path):
    labelme_json = {
        "version": "5.0.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": f"{cube_name}.hdr",
        "imageData": None,
        "imageHeight": H,
        "imageWidth": W
    }
    with open(out_path, "w") as f:
        json.dump(labelme_json, f, indent=2)

# --- Chargement ---
def load_augmented_cubes(augmented_dir):
    augmented_dir = Path(augmented_dir)
    extrudes_dict = {}
    for class_dir in augmented_dir.iterdir():
        if not class_dir.is_dir():
            continue
        cubes_list = []
        for cube_path in class_dir.glob("*.hdr"):
            hdr_img = envi.open(str(cube_path))
            cube = np.array(hdr_img.load(), dtype=np.float32)
            cubes_list.append(cube)
        if cubes_list:
            extrudes_dict[class_dir.name] = cubes_list
    return extrudes_dict

def load_labelme_json(json_path):
    with open(json_path) as f:
        data = json.load(f)
    return data["shapes"]

# --- Pipeline principal ---
def generate_augmented_plots(plot_cubes_dir, augmented_dir, out_dir):
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Précharger plots de base
    plot_paths = list(Path(plot_cubes_dir).glob("*.hdr"))
    if not plot_paths:
        print("Aucun plot trouvé")
        return
    plots = []
    for plot_path in plot_paths:
        hdr_img = envi.open(str(plot_path))
        cube = np.array(hdr_img.load(), dtype=np.float32)
        wavelengths = np.array(hdr_img.metadata["wavelength"])  # Utilisation du metadata
        json_path = plot_path.with_suffix(".json")
        shapes = load_labelme_json(json_path)
        plots.append((cube, shapes, wavelengths))

    # Précharger extrudes augmentés
    available_extrudes = load_augmented_cubes(augmented_dir)

    cube_count = 0
    while available_extrudes:
        plot_cube, labelme_shapes, wavelengths = random.choice(plots)

        # Génération cube augmenté
        filled_cube, new_shapes = process_plot(plot_cube, labelme_shapes, available_extrudes)

        # --- Cartographie ratio 996/1197 ---
        i1 = np.argmin(np.abs(wavelengths - 996))
        i2 = np.argmin(np.abs(wavelengths - 1197))
        ratio = filled_cube[i1] / (filled_cube[i2] + 1e-8)
        plt.figure(figsize=(5,5))
        plt.imshow(ratio, cmap="inferno")
        plt.axis("off")
        plt.title(f"Plot {cube_count} ratio 996/1197")
        plt.show()
        plt.savefig(Path(out_dir)/f"plot_aug_{cube_count}_ratio.png")
        plt.close()

        # Sauvegarde cube
        out_cube_path = Path(out_dir)/f"plot_aug_{cube_count}.hdr"
        envi.save_image(str(out_cube_path), filled_cube, dtype=np.float32, force=True, interleave="bil")

        # Sauvegarde LabelMe
        out_json_path = Path(out_dir)/f"plot_aug_{cube_count}.json"
        save_labelme_json(new_shapes, f"plot_aug_{cube_count}", filled_cube.shape[1], filled_cube.shape[2], out_json_path)

        cube_count += 1
        total_remaining = sum(len(v) for v in available_extrudes.values())
        print(f"Cube généré : {out_cube_path.name}, extrudes restants : {total_remaining}")