import numpy as np
import random
from pathlib import Path
import spectral.io.envi as envi
import cv2
import json
from collections import defaultdict
import matplotlib.pyplot as plt

N_EXTRUDES_PER_PLOT = 12
MAX_PLOTS = 500
SINGLE_CLASSES = []
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
# Masques utilitaires
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
# Positions valides STRICTEMENT dans la zone
# anchor=(0,0) → coin TL du noyau = ancre de l'érosion
# garantit que chaque pixel actif de l'extrude reste dans zone_mask
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
# Placement – full extrude (strictement dans zone)
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
# Placement – patch circulaire aléatoire (stratégie 2 single)
#
# On génère un disque de rayon r ∈ [5, 15] px (diamètre 10–30 px)
# centré en un point aléatoire DANS la zone du plot.
# On copie les spectres du cube source depuis la zone homologue
# (même bbox, mêmes coordonnées relatives).
# ──────────────────────────────────────────────

def place_circle_patch(plot, occ, cube, zone_poly):
    """
    Génère un polygone circulaire de diamètre aléatoire (10–30 px),
    le place dans zone_poly, puis remplit les pixels du cercle avec
    les spectres d'une zone de même taille extraite du cube source.
    Retourne (ok, poly_in_plot).
    """
    _, H, W = plot.shape
    zone_mask = polygon_to_mask(zone_poly, H, W)

    # Rayon aléatoire → diamètre entre 10 et 30 px
    r = random.randint(5, 15)
    d = 2 * r + 1  # taille de la boîte englobante

    # Construire le masque circulaire local (d × d)
    circle_mask = np.zeros((d, d), dtype=np.uint8)
    cv2.circle(circle_mask, (r, r), r, 1, -1)
    circle_mask = circle_mask.astype(bool)

    # Trouver les positions valides dans la zone (érosion)
    valid = compute_valid_positions(zone_mask, circle_mask)

    ys, xs = np.where(valid)
    free = [i for i in range(len(xs))
            if can_place(occ, circle_mask, xs[i], ys[i])]
    if not free:
        return False, None

    # Position de dépôt dans le plot
    idx = random.choice(free)
    y_dst, x_dst = ys[idx], xs[idx]

    # Zone source dans le cube extrude : on prend une zone de taille d×d
    # à une position aléatoire dans le cube (en restant dans les limites)
    _, Hc, Wc = cube.shape
    if Hc < d or Wc < d:
        # Cube trop petit pour fournir une zone d×d → on tile
        src_y, src_x = 0, 0
    else:
        src_y = random.randint(0, Hc - d)
        src_x = random.randint(0, Wc - d)

    # Copie spectrale pixel par pixel dans le cercle
    for b in range(plot.shape[0]):
        dst_patch = plot[b, y_dst:y_dst+d, x_dst:x_dst+d].copy()
        # Source : on tile si nécessaire
        src_patch = np.tile(
            cube[b, src_y:src_y+min(d, Hc), src_x:src_x+min(d, Wc)],
            (d // max(1, Hc - src_y) + 1, d // max(1, Wc - src_x) + 1)
        )[:d, :d]
        dst_patch[circle_mask] = src_patch[circle_mask]
        plot[b, y_dst:y_dst+d, x_dst:x_dst+d] = dst_patch

    update_occ(occ, circle_mask, x_dst, y_dst)

    # Polygone du cercle dans le repère du plot
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
# Pool / plot
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
# Génération
# ──────────────────────────────────────────────

def generate(plot_root, extrude_root, out_dir):
    plots = list(Path(plot_root).rglob("*.hdr"))
    pool = load_pool(extrude_root)
    counter = defaultdict(int)
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    for i in range(MAX_PLOTS):
        if len(pool) < N_EXTRUDES_PER_PLOT:
            break

        # ── Tirage de 12 extrudes ─────────────────────────────────────
        selected = random.sample(pool, N_EXTRUDES_PER_PLOT)
        singles = [e for e in selected if e["class"] in SINGLE_CLASSES]

        # ── Chargement du plot ────────────────────────────────────────
        plot_hdr = random.choice(plots)
        plot, zones, img = load_plot(plot_hdr)
        _, H, W = plot.shape
        occ = np.zeros((H, W), bool)
        shapes = []

        # ── CAS : au moins une classe single ─────────────────────────
        if singles:
            e = random.choice(singles)
            cube_e, _ = load_cube(e["hdr"])
            mask_e = get_valid_mask_fast(cube_e)
            cube_e, mask_e = crop_to_valid(cube_e, mask_e)

            strat = random.choice([1, 2])

            if strat == 1:
                # ── Stratégie 1 : full extrude unique ─────────────────
                # On place l'extrude en full ; si d'autres avaient été
                # placés avant (il n'y en a pas ici car on commence),
                # on repart d'un plot + occ vierge → garanti propre.
                ok, poly = try_place_full(plot, occ, cube_e, mask_e, zones)
                if ok:
                    shapes = [{"label": e["class"], "points": poly,
                                "shape_type": "polygon"}]
                    counter[e["class"]] += 1
                    pool.remove(e)
                # Stratégie 1 : on s'arrête là, rien d'autre placé.

            else:
                # ── Stratégie 2 : patch circulaire single + 11 full ───
                # 1) Patch circulaire du single (diamètre 10–30 px)
                ok, poly = try_place_circle_patch(plot, occ, cube_e, zones)
                if ok:
                    shapes.append({"label": e["class"], "points": poly,
                                   "shape_type": "polygon"})
                    counter[e["class"]] += 1
                    pool.remove(e)

                # 2) Les 11 autres en full, dans la zone
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
                        pool.remove(e2)

        # ── CAS : aucune classe single → 12 full dans la zone ─────────
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
                    pool.remove(e)

        # ── Pas de sauvegarde si rien n'a été placé ───────────────────
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

        print(f"Plot {i} -> {dict(counter)}")


if __name__ == "__main__":
    generate("plots_zone_to_add", "aug_after_rot_flip", "output")