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

x0, y0, x1, y1 = polygon_bbox(poly)
real_H, real_W = y1 - y0, x1 - x0

candidates = [e for e in extrudes if e["cube"].shape[0] >= real_H and e["cube"].shape[2] >= real_W]
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



import os
import json
import random
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import specarray as sa
import spectral.io.envi as envi
from scipy.ndimage import binary_erosion
from skimage.draw import polygon as ski_polygon


# ─────────────────────────────────────────────
# UTILS
# ─────────────────────────────────────────────

def find_band_index(wavelengths, wavelength):
    waves = np.array(wavelengths, dtype=float)
    return np.argmin(np.abs(waves - wavelength))


def spectral_ratio_map(cube, wavelengths, save_path=None):
    i1 = find_band_index(wavelengths, 996)
    i2 = find_band_index(wavelengths, 1197)
    ratio = cube[:, i1, :] / (cube[:, i2, :] + 1e-6)
    plt.figure()
    plt.imshow(ratio, cmap="RdYlGn")
    plt.colorbar()
    plt.title("Ratio 996 / 1197 nm")
    if save_path:
        plt.savefig(save_path, dpi=100)
    plt.close()


def contour_pixels(mask):
    """Contour vectorisé via érosion morphologique."""
    eroded = binary_erosion(mask)
    contour_mask = mask & ~eroded
    ys, xs = np.where(contour_mask)
    return [[float(x), float(y)] for x, y in zip(xs, ys)]


def polygon_to_mask(points, H, W):
    """Convertit une liste de points polygon en masque binaire (H, W)."""
    xs = np.array([p[0] for p in points])
    ys = np.array([p[1] for p in points])
    rr, cc = ski_polygon(ys, xs, shape=(H, W))
    mask = np.zeros((H, W), dtype=bool)
    mask[rr, cc] = True
    return mask


def polygon_bbox(poly):
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))


def polygon_area(poly):
    """Aire via formule du lacet (Shoelace)."""
    n = len(poly)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += poly[i][0] * poly[j][1]
        area -= poly[j][0] * poly[i][1]
    return abs(area) / 2.0


def offset_polygon(poly, dx, dy):
    return [[p[0] + dx, p[1] + dy] for p in poly]


# ─────────────────────────────────────────────
# CHARGEMENT
# ─────────────────────────────────────────────

def load_cube_specarray(folder):
    """Charge un cube HSI via specarray, retourne (cube H×B×W, wavelengths)."""
    arr = sa.SpecArray.from_folder(folder)
    arr = arr.spectral_albedo()
    cube = arr.array  # (H, B, W)
    wavelengths = list(arr.wavelengths)
    return cube, wavelengths


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
        rects.append((min(x1,x2), min(y1,y2), max(x1,x2), max(y1,y2)))
    return rects


def load_labelme_polygons(json_path):
    """Retourne liste de (label, points)."""
    with open(json_path) as f:
        data = json.load(f)
    polygons = []
    for shape in data["shapes"]:
        if shape["shape_type"] != "polygon":
            continue
        polygons.append((shape.get("label", "unknown"), shape["points"]))
    return polygons


def crop_plots_from_cube(cube, rectangles):
    """Découpe les plots selon les rectangles annotation. cube: H×B×W"""
    plots = []
    for (x1, y1, x2, y2) in rectangles:
        plot_cube = cube[y1:y2, :, x1:x2]
        plots.append(plot_cube)
    return plots


# ─────────────────────────────────────────────
# CHARGEMENT DES EXTRUDES AUGMENTÉS
# ─────────────────────────────────────────────

def load_augmented_extrudes(aug_folder):
    """
    Charge les extrudes augmentés depuis aug_folder/classe/*.hdr
    Pour chaque extrude :
      - supprime les pixels nuls en bordure (contour_pixels)
      - calcule la bounding box réelle sans pixels nuls
      - calcule l'aire du polygone contour
    """
    extrudes = []
    for class_name in os.listdir(aug_folder):
        class_folder = os.path.join(aug_folder, class_name)
        if not os.path.isdir(class_folder):
            continue
        hdr_files = [f for f in os.listdir(class_folder) if f.endswith(".hdr")]
        print(f"  Classe '{class_name}' : {len(hdr_files)} extrudes trouvés")
        for file in hdr_files:
            hdr_path = os.path.join(class_folder, file)
            try:
                cube_envi = envi.open(hdr_path)
                cube = cube_envi.load().astype(np.float32)  # H×B×W
            except Exception as e:
                print(f"    ⚠ Erreur lecture {hdr_path}: {e}")
                continue

            H, B, W = cube.shape

            # Masque des pixels non nuls (au moins une bande non nulle)
            mask = np.any(cube != 0, axis=1)  # (H, W)

            if not np.any(mask):
                print(f"    ⚠ Extrude vide ignoré: {file}")
                continue

            # Bounding box réelle sans pixels nuls
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]

            # Crop du cube sur la bbox réelle
            cube_cropped = cube[rmin:rmax+1, :, cmin:cmax+1]
            mask_cropped = mask[rmin:rmax+1, cmin:cmax+1]

            # Contour polygone sur le masque croppé
            polygon = contour_pixels(mask_cropped)

            if len(polygon) < 3:
                print(f"    ⚠ Contour insuffisant pour {file}, ignoré")
                continue

            area = polygon_area(polygon)
            eff_H, _, eff_W = cube_cropped.shape

            extrudes.append({
                "cube": cube_cropped,       # H×B×W sans pixels nuls en bordure
                "mask": mask_cropped,       # masque (H, W)
                "polygon": polygon,         # contour local (coordonnées dans cube_cropped)
                "area": area,               # aire du contour
                "H": eff_H,
                "W": eff_W,
                "class": class_name,
                "file": file
            })

    print(f"\nTotal extrudes augmentés chargés : {len(extrudes)}")
    random.shuffle(extrudes)
    return extrudes


# ─────────────────────────────────────────────
# GÉNÉRATION DES PLOTS AUGMENTÉS
# ─────────────────────────────────────────────

def generate_augmented_plots(annotation_folder, aug_folder, output_folder):
    """
    Pour chaque cube dans ann/hsi_cube :
      1. Charge le cube via specarray
      2. Lit les rectangles → crop les plots
      3. Lit les polygones réels (extrudes réels dans le même json)
      4. Pour chaque polygone réel, cherche un extrude augmenté dont
         l'aire couvre celle du polygone réel
      5. Remplace le contenu du polygone réel par l'extrude augmenté
      6. Sauvegarde le plot augmenté + annotation mise à jour
    """
    os.makedirs(output_folder, exist_ok=True)

    # Charger tous les extrudes augmentés
    print("=== Chargement des extrudes augmentés ===")
    extrudes = load_augmented_extrudes(aug_folder)
    if not extrudes:
        print("❌ Aucun extrude augmenté trouvé. Vérifiez aug_folder.")
        return

    hsi_folder = os.path.join(annotation_folder, "hsi_cube")
    cube_folders = [
        d for d in os.listdir(hsi_folder)
        if os.path.isdir(os.path.join(hsi_folder, d))
    ]
    print(f"\n=== {len(cube_folders)} cubes trouvés dans {hsi_folder} ===\n")

    plot_id = 0

    for cube_name in cube_folders:
        cube_folder = os.path.join(hsi_folder, cube_name)
        json_path = os.path.join(annotation_folder, cube_name + ".json")

        if not os.path.exists(json_path):
            print(f"⚠ Pas d'annotation pour {cube_name}, ignoré.")
            continue

        print(f"--- Traitement cube : {cube_name} ---")

        # 1. Charger le cube
        cube, wavelengths = load_cube_specarray(cube_folder)
        H, B, W = cube.shape
        print(f"  Cube shape: {H}×{B}×{W}")

        # 2. Charger rectangles → crop plots
        rectangles = load_labelme_rectangles(json_path)
        print(f"  {len(rectangles)} plots (rectangles) trouvés")
        plots = crop_plots_from_cube(cube, rectangles)

        # 3. Charger polygones réels (extrudes réels à remplacer)
        real_polygons = load_labelme_polygons(json_path)
        print(f"  {len(real_polygons)} polygones réels trouvés")

        # 4. Pour chaque plot croppé, on va remplacer les extrudes réels
        for idx, (plot_cube, rect) in enumerate(zip(plots, rectangles)):
            x1, y1, x2, y2 = rect
            plot_H, plot_B, plot_W = plot_cube.shape
            print(f"\n  Plot {idx} | bbox=({x1},{y1})-({x2},{y2}) | shape={plot_H}×{plot_B}×{plot_W}")

            # Polygones réels qui tombent dans ce plot
            plot_polygons = []
            for label, poly in real_polygons:
                px0, py0, px1, py1 = polygon_bbox(poly)
                # Vérifie que le polygone est bien dans ce plot
                if px0 >= x1 and py0 >= y1 and px1 <= x2 and py1 <= y2:
                    # Recentre le polygone dans les coordonnées du plot
                    local_poly = [[p[0] - x1, p[1] - y1] for p in poly]
                    plot_polygons.append((label, local_poly))

            print(f"    {len(plot_polygons)} polygones dans ce plot")

            shapes_out = []
            plot_cube_aug = plot_cube.copy()

            for label, local_poly in plot_polygons:
                real_area = polygon_area(local_poly)
                px0, py0, px1, py1 = polygon_bbox(local_poly)
                real_H = py1 - py0
                real_W = px1 - px0

                print(f"    Polygone '{label}' | aire={real_area:.1f} | bbox={real_W}×{real_H}")

                # 5. Cherche un extrude augmenté dont l'aire >= aire réelle
                #    ET dont les dimensions couvrent la bbox
                candidates = [
                    e for e in extrudes
                    if e["area"] >= real_area
                    and e["H"] >= real_H
                    and e["W"] >= real_W
                ]

                if not candidates:
                    print(f"    ⚠ Aucun extrude compatible → polygone conservé original")
                    shapes_out.append({
                        "label": label,
                        "points": [[p[0] + x1, p[1] + y1] for p in local_poly],
                        "group_id": None,
                        "shape_type": "polygon",
                        "flags": {}
                    })
                    continue

                ext = random.choice(candidates)
                print(f"    ✔ Extrude sélectionné: classe={ext['class']} H={ext['H']} W={ext['W']} aire={ext['area']:.1f}")

                # Extraire un patch de la taille réelle depuis l'extrude
                ext_cube = ext["cube"]
                ext_H, ext_B, ext_W = ext_cube.shape

                # Position aléatoire dans l'extrude pour extraire le patch
                dy = random.randint(0, ext_H - real_H) if ext_H > real_H else 0
                dx = random.randint(0, ext_W - real_W) if ext_W > real_W else 0
                patch = ext_cube[dy:dy+real_H, :, dx:dx+real_W]

                if patch.shape[0] != real_H or patch.shape[2] != real_W:
                    print(f"    ⚠ Patch mal extrait, ignoré")
                    shapes_out.append({
                        "label": label,
                        "points": [[p[0] + x1, p[1] + y1] for p in local_poly],
                        "group_id": None,
                        "shape_type": "polygon",
                        "flags": {}
                    })
                    continue

                # Masque du patch extrait
                patch_mask = ext["mask"][dy:dy+real_H, dx:dx+real_W]

                # Masque du polygone réel dans le plot
                real_mask = polygon_to_mask(local_poly, plot_H, plot_W)

                # Remplacer uniquement les pixels du polygone réel par le patch
                for band in range(plot_B):
                    band_slice = plot_cube_aug[py0:py0+real_H, band, px0:px0+real_W]
                    patch_band = patch[:, band, :]
                    # On remplace là où le masque réel ET le patch sont valides
                    combined_mask = real_mask[py0:py0+real_H, px0:px0+real_W] & patch_mask
                    band_slice[combined_mask] = patch_band[combined_mask]
                    plot_cube_aug[py0:py0+real_H, band, px0:px0+real_W] = band_slice

                # Nouveau polygone = contour du patch_mask, offset dans le plot global
                new_local_poly = contour_pixels(patch_mask)
                new_global_poly = [[p[0] + px0 + x1, p[1] + py0 + y1] for p in new_local_poly]

                shapes_out.append({
                    "label": f"aug_{ext['class']}",
                    "points": new_global_poly,
                    "group_id": None,
                    "shape_type": "polygon",
                    "flags": {}
                })

            # 6. Sauvegarder le plot augmenté
            out_name = f"plot_{plot_id:04d}"
            out_cube_path = os.path.join(output_folder, out_name)
            out_json_path = os.path.join(output_folder, out_name + ".json")
            out_ratio_path = os.path.join(output_folder, out_name + "_ratio.png")

            save_envi_cube(out_cube_path, plot_cube_aug, wavelengths)
            save_labelme(out_json_path, shapes_out, plot_cube_aug.shape)
            spectral_ratio_map(plot_cube_aug, wavelengths, save_path=out_ratio_path)

            print(f"  ✅ Plot sauvegardé: {out_name}")
            plot_id += 1

    print(f"\n=== Terminé. {plot_id} plots générés dans '{output_folder}' ===")


# ─────────────────────────────────────────────
# SAVE UTILS
# ─────────────────────────────────────────────

def save_envi_cube(file_path, cube, wavelengths):
    envi.save_image(
        file_path + ".hdr",
        cube.astype(np.float32),
        dtype=np.float32,
        interleave="bil",
        metadata={
            "wavelength": wavelengths,
            "wavelength units": "nm"
        },
        force=True
    )


def save_labelme(file_path, shapes, cube_shape):
    H = cube_shape[0]
    W = cube_shape[2]
    data = {
        "version": "5.0.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": Path(file_path).stem + ".png",
        "imageData": None,
        "imageHeight": H,
        "imageWidth": W
    }
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    annotation_folder = "ann"
    aug_folder = "aug_data"
    output_folder = "generated_dataset"
    generate_augmented_plots(annotation_folder, aug_folder, output_folder)