import numpy as np
import random
from pathlib import Path
import spectral.io.envi as envi
import cv2
import json
from collections import defaultdict
import matplotlib.pyplot as plt

MAX_PLOTS   = 500
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
# Positions valides (érosion anchor coin TL)
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
# Placement patch circulaire (single strat 2)
# ──────────────────────────────────────────────

def place_circle_patch(plot, occ, cube, zone_poly):
    _, H, W = plot.shape
    zone_mask = polygon_to_mask(zone_poly, H, W)

    r = random.randint(5, 15)
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
# Helpers zones
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
    return [{"hdr": f, "class": get_class(f.stem)}
            for f in Path(root).rglob("*.hdr")]

def load_plot(p):
    cube, img = load_cube(p)
    with open(p.with_suffix(".json")) as f:
        data = json.load(f)
    zones = [s["points"] for s in data["shapes"]
             if s["label"] == "zone_placement"]
    return cube, zones, img

# ──────────────────────────────────────────────
# Queue PARFAITEMENT équilibrée
#
# Principe :
#   On génère d'abord les tailles de batches (10–14) pour tous les plots.
#   On connaît donc le nombre exact de slots par plot.
#   On répartit les slots entre les classes de façon EXACTEMENT égale,
#   en distribuant les éventuels restes un par un.
#   Résultat : toutes les classes ont exactement le même compte,
#   à ±1 près (inévitable si total_slots n'est pas divisible par nb_classes).
#
#   Pour chaque classe, on remplit sa liste jusqu'au quota exact
#   (répétition si nécessaire, shuffle, sans remise si assez d'exemples).
#
#   Enfin, on construit chaque batch en piochant round-robin dans
#   les queues de classes → équilibre GARANTI plot par plot et global.
# ──────────────────────────────────────────────

def make_class_queue(items, quota):
    """
    Retourne une liste de `quota` entrées issues de `items`.
    Sans répétition si len(items) >= quota, avec répétition sinon.
    """
    if len(items) >= quota:
        q = random.sample(items, quota)
    else:
        reps = (quota // len(items)) + 1
        extended = (items * reps)[:quota]
        random.shuffle(extended)
        q = extended
    return q


def build_balanced_batches(pool, max_plots, min_p, max_p):
    """
    Retourne une liste de listes :
      batches[i] = liste des extrudes pour le plot i.

    Garantie : chaque classe apparaît EXACTEMENT le même nombre de fois
    dans l'ensemble de tous les batches (à ±1 près si le total n'est pas
    divisible par le nombre de classes — différence max = 1 tirage).
    """
    # ── 1. Regrouper par classe ────────────────────────────────────
    by_class = defaultdict(list)
    for e in pool:
        by_class[e["class"]].append(e)
    classes = sorted(by_class.keys())
    nb_classes = len(classes)
    if nb_classes == 0:
        return []

    # ── 2. Générer les tailles de batches ─────────────────────────
    batch_sizes = [random.randint(min_p, max_p) for _ in range(max_plots)]
    total_slots = sum(batch_sizes)

    # ── 3. Quota EXACT par classe ──────────────────────────────────
    # base : chaque classe reçoit floor(total / nb_classes)
    # reste : les `r` premières classes reçoivent 1 slot de plus
    base_quota = total_slots // nb_classes
    remainder  = total_slots % nb_classes   # 0 ≤ remainder < nb_classes

    quotas = {}
    for k, cls in enumerate(classes):
        quotas[cls] = base_quota + (1 if k < remainder else 0)

    # Vérification : somme des quotas == total_slots
    assert sum(quotas.values()) == total_slots, "Erreur quota"

    # ── 4. Construire une queue par classe ─────────────────────────
    class_queues = {cls: make_class_queue(by_class[cls], quotas[cls])
                    for cls in classes}

    # ── 5. Construire les batches en distribuant round-robin ───────
    # On distribue les slots de chaque batch entre les classes
    # de façon cyclique → équilibre local dans chaque batch aussi.
    #
    # Pour chaque batch de taille N :
    #   - base_per_class = N // nb_classes  (chaque classe reçoit ça)
    #   - les (N % nb_classes) premières classes (dans un ordre
    #     aléatoire pour ce batch) reçoivent 1 slot de plus.
    #
    # On pioche dans class_queues dans l'ordre.

    # Pointeurs courants dans chaque queue
    pointers = {cls: 0 for cls in classes}

    batches = []
    for size in batch_sizes:
        base   = size // nb_classes
        extra  = size % nb_classes
        # ordre des classes aléatoire pour ce batch (pour répartir le reste équitablement)
        cls_order = classes[:]
        random.shuffle(cls_order)

        batch = []
        for k, cls in enumerate(cls_order):
            n = base + (1 if k < extra else 0)
            p = pointers[cls]
            q = class_queues[cls]
            # sécurité : ne pas dépasser la queue
            n = min(n, len(q) - p)
            batch.extend(q[p:p+n])
            pointers[cls] += n

        random.shuffle(batch)   # mélanger l'ordre à l'intérieur du batch
        batches.append(batch)

    return batches, quotas


# ──────────────────────────────────────────────
# Génération
# ──────────────────────────────────────────────

def generate(plot_root, extrude_root, out_dir):
    plots    = list(Path(plot_root).rglob("*.hdr"))
    raw_pool = load_pool(extrude_root)
    out_dir  = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    # ── Queue parfaitement équilibrée ─────────────────────────────
    batches, quotas = build_balanced_batches(
        raw_pool, MAX_PLOTS, MIN_PATCHES, MAX_PATCHES)

    print(f"[INFO] {len(batches)} plots planifiés "
          f"({MIN_PATCHES}–{MAX_PATCHES} patches/plot)")
    print("[INFO] Quota prévu par classe (parfaitement équilibré) :")
    for cls, q in sorted(quotas.items()):
        print(f"       {cls}: {q}")

    counter = defaultdict(int)

    for i, batch in enumerate(batches):
        selected = batch
        singles  = [e for e in selected if e["class"] in SINGLE_CLASSES]

        plot_hdr = random.choice(plots)
        plot, zones, img = load_plot(plot_hdr)
        _, H, W = plot.shape
        occ    = np.zeros((H, W), bool)
        shapes = []

        # ── CAS : au moins une classe single ──────────────────────
        if singles:
            e = random.choice(singles)
            cube_e, _ = load_cube(e["hdr"])
            mask_e    = get_valid_mask_fast(cube_e)
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
                # Stratégie 2 : patch circulaire + autres en full
                ok, poly = try_place_circle_patch(plot, occ, cube_e, zones)
                if ok:
                    shapes.append({"label": e["class"], "points": poly,
                                   "shape_type": "polygon"})
                    counter[e["class"]] += 1

                for e2 in [x for x in selected if x != e]:
                    cube2, _ = load_cube(e2["hdr"])
                    mask2    = get_valid_mask_fast(cube2)
                    cube2, mask2 = crop_to_valid(cube2, mask2)
                    ok2, poly2 = try_place_full(plot, occ, cube2, mask2, zones)
                    if ok2:
                        shapes.append({"label": e2["class"], "points": poly2,
                                       "shape_type": "polygon"})
                        counter[e2["class"]] += 1

        # ── CAS : aucune classe single → tous en full ─────────────
        else:
            for e in selected:
                cube, _ = load_cube(e["hdr"])
                mask    = get_valid_mask_fast(cube)
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

        b1    = get_band_index(img, WAVELENGTH_1)
        b2    = get_band_index(img, WAVELENGTH_2)
        ratio = compute_ratio_map(plot, b1, b2)
        save_ratio_map(ratio, [s["points"] for s in shapes],
                       out_dir / f"{name}_ratio.png")

        print(f"Plot {i:>4}  patches placés: {len(shapes):>2}  "
              f"compteur global: {dict(counter)}")

    # ── Résumé final ──────────────────────────────────────────────
    print("\n[RÉSUMÉ] Placements réels par classe :")
    for cls, cnt in sorted(counter.items()):
        print(f"  {cls}: {cnt}")

    counts = list(counter.values())
    if counts:
        print(f"\n  min={min(counts)}  max={max(counts)}  "
              f"écart={max(counts)-min(counts)}")


if __name__ == "__main__":
    generate("plots_zone_to_add", "aug_after_rot_flip", "output")