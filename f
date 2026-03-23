import numpy as np
import random
from pathlib import Path
import spectral.io.envi as envi
import cv2
import json
from collections import defaultdict
import matplotlib.pyplot as plt

MAX_PLOTS = 500
SINGLE_CLASSES = []          # ex: ["mildiou", "rouille"]
MIN_PATCHES = 10
MAX_PATCHES = 14
WAVELENGTH_1 = 996
WAVELENGTH_2 = 1197

# ──────────────────────────────────────────────
# I/O
# ──────────────────────────────────────────────

def load_cube(hdr_path):
    img = envi.open(str(hdr_path))
    return np.array(img.load(), dtype=np.float32), img

def save_envi_cube(cube, path):
    envi.save_image(str(path) + ".hdr", cube, dtype=np.float32,
                    interleave="bsq", force=True)

def get_band_index(img, wavelength):
    wl = np.array(img.metadata["wavelength"], dtype=float)
    return np.argmin(np.abs(wl - wavelength))

def compute_ratio_map(cube, b1, b2):
    band1 = cube[b1]
    band2 = cube[b2]
    ratio = np.zeros_like(band1)
    valid = band2 > 1e-6
    ratio[valid] = band1[valid] / band2[valid]
    return ratio

def save_ratio_map(ratio, polygons, out_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(ratio, cmap="viridis")
    plt.colorbar(im, ax=ax, fraction=0.03)
    for poly in polygons:
        if poly is None or len(poly) < 3:
            continue
        p = np.array(poly)
        p = np.vstack([p, p[0]])
        ax.plot(p[:, 0], p[:, 1], color="red", linewidth=1.5)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()

# ──────────────────────────────────────────────
# Masques
# ──────────────────────────────────────────────

def get_valid_mask_fast(cube):
    return cube.var(axis=0) > 1e-6

def crop_to_valid(cube, mask):
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return cube, mask
    return (cube[:, rows[0]:rows[-1]+1, cols[0]:cols[-1]+1],
            mask[rows[0]:rows[-1]+1, cols[0]:cols[-1]+1])

def find_contour_polygons(mask):
    mask_u8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in contours:
        pts = c.reshape(-1, 2).tolist()
        if len(pts) >= 3:
            polys.append(pts)
    return polys

def polygon_to_mask(poly, H, W):
    m = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(m, [np.array(poly, dtype=np.int32)], 1)
    return m.astype(bool)

# ──────────────────────────────────────────────
# Positions valides (érosion, anchor coin TL)
# ──────────────────────────────────────────────

def compute_valid_positions(zone_mask, extrude_mask):
    kernel = extrude_mask.astype(np.uint8)
    valid = cv2.erode(
        zone_mask.astype(np.uint8) * 255,
        kernel,
        anchor=(0, 0),
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0
    ).astype(bool)
    return valid

# ──────────────────────────────────────────────
# Occupation
# ──────────────────────────────────────────────

def can_place(occ, mask, x, y):
    h, w = mask.shape
    H, W = occ.shape
    if y + h > H or x + w > W:
        return False
    return not (occ[y:y+h, x:x+w] & mask).any()

def update_occ(occ, mask, x, y):
    h, w = mask.shape
    occ[y:y+h, x:x+w] |= mask

# ──────────────────────────────────────────────
# Placement full
# ──────────────────────────────────────────────

def place_full(plot, occ, cube, extrude_mask, zone_poly):
    _, H, W = plot.shape
    zone_mask = polygon_to_mask(zone_poly, H, W)
    valid = compute_valid_positions(zone_mask, extrude_mask)

    ys, xs = np.where(valid)
    free = [i for i in range(len(xs))
            if can_place(occ, extrude_mask, xs[i], ys[i])]
    if not free:
        return False, None

    idx = random.choice(free)
    y, x = ys[idx], xs[idx]
    h, w = extrude_mask.shape

    for b in range(plot.shape[0]):
        patch = plot[b, y:y+h, x:x+w].copy()
        patch[extrude_mask] = cube[b][extrude_mask]
        plot[b, y:y+h, x:x+w] = patch

    update_occ(occ, extrude_mask, x, y)

    contours = find_contour_polygons(extrude_mask)
    if not contours:
        return False, None
    poly = np.array(contours[0])
    poly[:, 0] += x
    poly[:, 1] += y
    return True, poly.tolist()

# ──────────────────────────────────────────────
# Placement patch circulaire (single, strat 2)
# ──────────────────────────────────────────────

def place_circle_patch(plot, occ, cube, zone_poly):
    _, H, W = plot.shape
    zone_mask = polygon_to_mask(zone_poly, H, W)

    r = random.randint(5, 15)   # diamètre 10–30 px
    d = 2 * r + 1

    circle_mask = np.zeros((d, d), dtype=np.uint8)
    cv2.circle(circle_mask, (r, r), r, 1, -1)
    circle_mask = circle_mask.astype(bool)

    valid = compute_valid_positions(zone_mask, circle_mask)

    ys, xs = np.where(valid)
    free = [i for i in range(len(xs))
            if can_place(occ, circle_mask, xs[i], ys[i])]
    if not free:
        return False, None

    idx = random.choice(free)
    y_dst, x_dst = ys[idx], xs[idx]

    _, Hc, Wc = cube.shape
    src_y = random.randint(0, max(0, Hc - d))
    src_x = random.randint(0, max(0, Wc - d))

    for b in range(plot.shape[0]):
        dst_patch = plot[b, y_dst:y_dst+d, x_dst:x_dst+d].copy()
        src_patch = np.tile(
            cube[b, src_y:src_y+min(d, Hc), src_x:src_x+min(d, Wc)],
            (d // max(1, Hc - src_y) + 1, d // max(1, Wc - src_x) + 1)
        )[:d, :d]
        dst_patch[circle_mask] = src_patch[circle_mask]
        plot[b, y_dst:y_dst+d, x_dst:x_dst+d] = dst_patch

    update_occ(occ, circle_mask, x_dst, y_dst)

    contours = find_contour_polygons(circle_mask)
    if not contours:
        return False, None
    poly = np.array(contours[0])
    poly[:, 0] += x_dst
    poly[:, 1] += y_dst
    return True, poly.tolist()

# ──────────────────────────────────────────────
# Helpers : essaie toutes les zones
# ──────────────────────────────────────────────

def try_place_full(plot, occ, cube, mask, zones):
    for z in zones:
        ok, poly = place_full(plot, occ, cube, mask, z)
        if ok:
            return True, poly
    return False, None

def try_place_circle_patch(plot, occ, cube, zones):
    for z in zones:
        ok, poly = place_circle_patch(plot, occ, cube, z)
        if ok:
            return True, poly
    return False, None

# ──────────────────────────────────────────────
# Pool
# ──────────────────────────────────────────────

def get_class(name):
    return name.split("_")[0]

def load_pool(root):
    entries = [{"hdr": f, "class": get_class(f.stem)}
               for f in Path(root).rglob("*.hdr")]
    return entries

def load_plot(p):
    cube, img = load_cube(p)
    with open(p.with_suffix(".json")) as f:
        data = json.load(f)
    zones = [s["points"] for s in data["shapes"]
             if s["label"] == "zone_placement"]
    return cube, zones, img

# ──────────────────────────────────────────────
# Tirage équilibré
#
# Principe :
#   - On regroupe les extrudes par classe.
#   - On calcule combien de fois chaque classe doit être tirée pour
#     que toutes les classes soient tirées exactement le même nombre
#     de fois sur l'ensemble des MAX_PLOTS.
#   - On construit une "queue" globale équilibrée : chaque classe
#     contribue exactement `quota` tirages, répartis uniformément
#     dans les plots.
#   - La queue est mélangée aléatoirement puis consommée plot par plot.
#
# Détail du calcul du quota :
#   Chaque plot tire N patches (∈ [MIN_PATCHES, MAX_PATCHES]).
#   Sur MAX_PLOTS plots, le total de tirages est entre
#   MAX_PLOTS*MIN_PATCHES et MAX_PLOTS*MAX_PATCHES.
#   On veut que chaque classe soit tirée `quota` fois.
#   quota = floor(total_tirages / nb_classes), avec
#   total_tirages = MAX_PLOTS * avg_patches (avg = (MIN+MAX)//2).
# ──────────────────────────────────────────────

def build_balanced_queue(pool, max_plots, min_p, max_p):
    """
    Retourne une liste de listes :
      queue[i] = liste des extrudes à utiliser pour le plot i.
    Chaque classe apparaît exactement le même nombre de fois
    dans l'ensemble de tous les plots.
    """
    # Regrouper par classe
    by_class = defaultdict(list)
    for e in pool:
        by_class[e["class"]].append(e)

    nb_classes = len(by_class)
    if nb_classes == 0:
        return []

    # Nombre moyen de patches par plot
    avg_patches = (min_p + max_p) // 2
    total_slots = max_plots * avg_patches   # total de tirages sur tous les plots

    # Quota par classe (identique pour toutes)
    quota = total_slots // nb_classes

    # Pour chaque classe, on construit une liste de `quota` extrudes
    # en répétant / échantillonnant dans les extrudes disponibles
    class_queues = {}
    for cls, items in by_class.items():
        if len(items) >= quota:
            # On a assez → on tire sans remise (garder la diversité)
            sampled = random.sample(items, quota)
        else:
            # Pas assez → on répète les items jusqu'à atteindre le quota
            reps = (quota // len(items)) + 1
            extended = (items * reps)[:quota]
            random.shuffle(extended)
            sampled = extended
        class_queues[cls] = sampled

    # Fusionner toutes les listes en une queue globale plate et mélangée
    flat = []
    for cls_list in class_queues.values():
        flat.extend(cls_list)
    random.shuffle(flat)

    # Découper la queue globale en tranches pour chaque plot
    # Chaque plot reçoit un nombre aléatoire de patches ∈ [min_p, max_p]
    plots_batches = []
    idx = 0
    for _ in range(max_plots):
        n = random.randint(min_p, max_p)
        if idx + n > len(flat):
            break
        plots_batches.append(flat[idx:idx+n])
        idx += n

    return plots_batches

# ──────────────────────────────────────────────
# Génération
# ──────────────────────────────────────────────

def generate(plot_root, extrude_root, out_dir):
    plots = list(Path(plot_root).rglob("*.hdr"))
    raw_pool = load_pool(extrude_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    # ── Construire la queue équilibrée ────────────────────────────────
    plots_batches = build_balanced_queue(raw_pool, MAX_PLOTS, MIN_PATCHES, MAX_PATCHES)
    print(f"[INFO] {len(plots_batches)} plots planifiés "
          f"({MIN_PATCHES}–{MAX_PATCHES} patches/plot)")

    # Vérification de l'équilibre au démarrage
    all_entries = [e for batch in plots_batches for e in batch]
    class_counts = defaultdict(int)
    for e in all_entries:
        class_counts[e["class"]] += 1
    print("[INFO] Tirages par classe (équilibre) :")
    for cls, cnt in sorted(class_counts.items()):
        print(f"       {cls}: {cnt}")

    counter = defaultdict(int)   # compteur de placements réels

    for i, batch in enumerate(plots_batches):
        selected = batch           # N extrudes pour ce plot (10–14)
        singles = [e for e in selected if e["class"] in SINGLE_CLASSES]

        plot_hdr = random.choice(plots)
        plot, zones, img = load_plot(plot_hdr)
        _, H, W = plot.shape
        occ = np.zeros((H, W), bool)
        shapes = []

        # ── CAS : au moins une classe single ──────────────────────────
        if singles:
            e = random.choice(singles)
            cube_e, _ = load_cube(e["hdr"])
            mask_e = get_valid_mask_fast(cube_e)
            cube_e, mask_e = crop_to_valid(cube_e, mask_e)

            strat = random.choice([1, 2])

            if strat == 1:
                # Stratégie 1 : full unique, rien d'autre
                ok, poly = try_place_full(plot, occ, cube_e, mask_e, zones)
                if ok:
                    shapes = [{"label": e["class"], "points": poly,
                                "shape_type": "polygon"}]
                    counter[e["class"]] += 1

            else:
                # Stratégie 2 : patch circulaire single + autres en full
                ok, poly = try_place_circle_patch(plot, occ, cube_e, zones)
                if ok:
                    shapes.append({"label": e["class"], "points": poly,
                                   "shape_type": "polygon"})
                    counter[e["class"]] += 1

                others = [x for x in selected if x != e]
                for e2 in others:
                    cube2, _ = load_cube(e2["hdr"])
                    mask2 = get_valid_mask_fast(cube2)
                    cube2, mask2 = crop_to_valid(cube2, mask2)
                    ok2, poly2 = try_place_full(plot, occ, cube2, mask2, zones)
                    if ok2:
                        shapes.append({"label": e2["class"], "points": poly2,
                                       "shape_type": "polygon"})
                        counter[e2["class"]] += 1

        # ── CAS : aucune classe single → tous en full ─────────────────
        else:
            for e in selected:
                cube, _ = load_cube(e["hdr"])
                mask = get_valid_mask_fast(cube)
                cube, mask = crop_to_valid(cube, mask)
                ok, poly = try_place_full(plot, occ, cube, mask, zones)
                if ok:
                    shapes.append({"label": e["class"], "points": poly,
                                   "shape_type": "polygon"})
                    counter[e["class"]] += 1

        if not shapes:
            continue

        name = f"aug_{i}"
        save_envi_cube(plot, out_dir / name)

        with open(out_dir / f"{name}.json", "w") as f:
            json.dump({
                "version": "5.0.1",
                "shapes": shapes,
                "imagePath": f"{name}.png",
                "imageHeight": H,
                "imageWidth": W
            }, f, indent=2)

        b1 = get_band_index(img, WAVELENGTH_1)
        b2 = get_band_index(img, WAVELENGTH_2)
        ratio = compute_ratio_map(plot, b1, b2)
        save_ratio_map(ratio, [s["points"] for s in shapes],
                       out_dir / f"{name}_ratio.png")

        print(f"Plot {i:>4}  patches placés: {len(shapes):>2}  "
              f"compteur global: {dict(counter)}")

    # ── Résumé final ──────────────────────────────────────────────────
    print("\n[RÉSUMÉ] Placements réels par classe :")
    for cls, cnt in sorted(counter.items()):
        print(f"  {cls}: {cnt}")


if __name__ == "__main__":
    generate("plots_zone_to_add", "aug_after_rot_flip", "output")