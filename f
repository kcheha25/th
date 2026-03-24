import numpy as np
import random
from pathlib import Path
import spectral.io.envi as envi
import cv2
import json
from collections import defaultdict
import matplotlib.pyplot as plt

MAX_PLOTS      = 500
SINGLE_CLASSES = []          # ex: ["mildiou", "rouille"]
MIN_PATCHES    = 10
MAX_PATCHES    = 14
WAVELENGTH_1   = 996
WAVELENGTH_2   = 1197

# ═══════════════════════════════════════════════════════════════
# I/O  –  cubes ENVI
# ═══════════════════════════════════════════════════════════════

def load_cube(hdr_path):
    img = envi.open(str(hdr_path))
    return np.array(img.load(), dtype=np.float32), img

def save_envi_cube(cube, path):
    envi.save_image(str(path) + ".hdr", cube, dtype=np.float32,
                    interleave="bsq", force=True)

def get_band_index(img, wavelength):
    wl = np.array(img.metadata["wavelength"], dtype=float)
    return np.argmin(np.abs(wl - wavelength))

# ═══════════════════════════════════════════════════════════════
# Carte ratio
# ═══════════════════════════════════════════════════════════════

def compute_ratio_map(cube, b1, b2):
    band1 = cube[b1]
    band2 = cube[b2]
    ratio = np.zeros_like(band1)
    valid = band2 > 1e-6
    ratio[valid] = band1[valid] / band2[valid]
    return ratio

def _ratio_to_bgr(ratio):
    """Normalise ratio → uint8 puis applique colormap viridis (BGR)."""
    r = ratio.copy()
    rmin, rmax = r.min(), r.max()
    if rmax - rmin > 1e-8:
        r = (r - rmin) / (rmax - rmin)
    else:
        r = np.zeros_like(r)
    u8 = (r * 255).astype(np.uint8)
    return cv2.applyColorMap(u8, cv2.COLORMAP_VIRIDIS)   # shape H×W×3 BGR

def save_ratio_plain(ratio, out_path):
    """
    PNG ratio SANS contours.
    Même nom que le .json → c'est l'image référencée par LabelMe
    (imagePath = "<name>.png").
    """
    bgr = _ratio_to_bgr(ratio)
    cv2.imwrite(str(out_path), bgr)

def save_ratio_with_contours(ratio, polygons, out_path):
    """
    PNG ratio AVEC contours rouges des polygones annotés.
    Nom : "<name>_ratio.png"  (fichier de visualisation).
    """
    bgr = _ratio_to_bgr(ratio).copy()
    for poly in polygons:
        if poly is None or len(poly) < 3:
            continue
        pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(bgr, [pts], isClosed=True,
                      color=(0, 0, 255), thickness=2)   # rouge en BGR
    cv2.imwrite(str(out_path), bgr)

# ═══════════════════════════════════════════════════════════════
# Masques utilitaires
# ═══════════════════════════════════════════════════════════════

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

# ═══════════════════════════════════════════════════════════════
# Positions valides (érosion, anchor coin TL)
# Garantit que chaque pixel actif de l'extrude reste dans zone_mask
# ═══════════════════════════════════════════════════════════════

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

# ═══════════════════════════════════════════════════════════════
# Occupation
# ═══════════════════════════════════════════════════════════════

def can_place(occ, mask, x, y):
    h, w = mask.shape
    H, W = occ.shape
    if y + h > H or x + w > W:
        return False
    return not (occ[y:y+h, x:x+w] & mask).any()

def update_occ(occ, mask, x, y):
    h, w = mask.shape
    occ[y:y+h, x:x+w] |= mask

# ═══════════════════════════════════════════════════════════════
# Placement – full extrude (dans zone)
# ═══════════════════════════════════════════════════════════════

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

# ═══════════════════════════════════════════════════════════════
# Placement – patch circulaire (single, stratégie 2)
# Diamètre aléatoire 10–30 px, spectres copiés depuis une zone
# de même taille dans le cube source
# ═══════════════════════════════════════════════════════════════

def place_circle_patch(plot, occ, cube, zone_poly):
    _, H, W = plot.shape
    zone_mask = polygon_to_mask(zone_poly, H, W)

    r = random.randint(5, 15)   # rayon → diamètre 10–30 px
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

# ═══════════════════════════════════════════════════════════════
# Helpers : essaie toutes les zones disponibles
# ═══════════════════════════════════════════════════════════════

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

# ═══════════════════════════════════════════════════════════════
# Pool
# ═══════════════════════════════════════════════════════════════

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

# ═══════════════════════════════════════════════════════════════
# Queue PARFAITEMENT équilibrée
#
# Principe :
#   1. On connaît à l'avance le nombre exact de slots (sum batch_sizes).
#   2. Quota identique pour toutes les classes (±1 si non divisible).
#   3. Distribution round-robin dans chaque batch → équilibre local
#      ET global garanti.
# ═══════════════════════════════════════════════════════════════

def make_class_queue(items, quota):
    """quota entrees issues de items (sans repetition si possible)."""
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
    Garantie ecart = 0 entre toutes les classes, sur tous les tirages.

    Principe en 5 etapes :
      1. Tirer les tailles de batches aleatoirement (min_p..max_p).
      2. Ajuster le total pour qu'il soit exactement divisible par
         nb_classes en ajoutant au plus (nb_classes-1) slots sur les
         derniers batches -> ecart = 0 mathematiquement garanti.
      3. Q = total / nb_classes : quota identique pour toutes les classes.
      4. Construire la queue globale en mode interleaved :
         pour chaque round r, on prend l'element r de chaque classe
         dans un ordre aleatoire. Resultat : chaque classe apparait
         exactement Q fois, dans un ordre varie.
      5. Decouper la queue globale en batches selon les tailles calculees.
    """

    # 1. Regrouper par classe
    by_class = defaultdict(list)
    for e in pool:
        by_class[e["class"]].append(e)
    classes    = sorted(by_class.keys())
    nb_classes = len(classes)
    if nb_classes == 0:
        return [], {}

    # 2. Tirer les tailles de batches
    batch_sizes = [random.randint(min_p, max_p) for _ in range(max_plots)]
    raw_total   = sum(batch_sizes)

    # 3. Ajuster pour que raw_total % nb_classes == 0
    #    On ajoute au plus (nb_classes - 1) slots sur les derniers batches
    remainder = raw_total % nb_classes
    if remainder != 0:
        extra_needed = nb_classes - remainder
        for k in range(extra_needed):
            idx = len(batch_sizes) - 1 - k
            batch_sizes[idx] += 1

    total_slots = sum(batch_sizes)
    assert total_slots % nb_classes == 0
    Q = total_slots // nb_classes   # quota identique pour toutes les classes

    quotas = {cls: Q for cls in classes}   # ecart = 0 garanti

    # 4. Queue de Q elements par classe
    class_queues = {cls: make_class_queue(by_class[cls], Q)
                    for cls in classes}

    # 5. Queue globale interleaved (ABCABC... avec ordre aleatoire par round)
    global_queue = []
    for r in range(Q):
        cls_order = classes[:]
        random.shuffle(cls_order)
        for cls in cls_order:
            global_queue.append(class_queues[cls][r])
    assert len(global_queue) == total_slots

    # 6. Decouper exactement selon batch_sizes
    batches = []
    pos = 0
    for size in batch_sizes:
        batch = global_queue[pos:pos + size]
        random.shuffle(batch)   # ordre interne aleatoire
        batches.append(batch)
        pos += size

    return batches, quotas

# ═══════════════════════════════════════════════════════════════
# Sauvegarde JSON compatible LabelMe
# ═══════════════════════════════════════════════════════════════

def save_labelme_json(path, shapes, image_name, H, W):
    """
    Écrit un JSON lisible par LabelMe.
    - imageData: null  → LabelMe charge l'image depuis le disque
    - imagePath pointe vers "<name>.png" (image ratio sans contours)
    - Tous les champs obligatoires de chaque shape sont présents
    """
    labelme_shapes = [
        {
            "label":       s["label"],
            "points":      s["points"],
            "group_id":    None,
            "description": "",
            "shape_type":  s["shape_type"],
            "flags":       {},
            "mask":        None,
        }
        for s in shapes
    ]
    with open(path, "w") as f:
        json.dump({
            "version":     "5.4.1",
            "flags":       {},
            "shapes":      labelme_shapes,
            "imagePath":   image_name,   # "<name>.png" dans le même dossier
            "imageData":   None,         # null explicite = LabelMe lit depuis disque
            "imageHeight": H,
            "imageWidth":  W,
        }, f, indent=2)

# ═══════════════════════════════════════════════════════════════
# Génération principale
# ═══════════════════════════════════════════════════════════════

def generate(plot_root, extrude_root, out_dir):
    plots    = list(Path(plot_root).rglob("*.hdr"))
    raw_pool = load_pool(extrude_root)
    out_dir  = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    # ── Queue équilibrée ──────────────────────────────────────────
    batches, quotas = build_balanced_batches(
        raw_pool, MAX_PLOTS, MIN_PATCHES, MAX_PATCHES)

    print(f"[INFO] {len(batches)} plots planifiés "
          f"({MIN_PATCHES}–{MAX_PATCHES} patches/plot)")
    print("[INFO] Quota prévu par classe :")
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
                # Stratégie 2 : patch circulaire single + autres en full
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

        # ── 1. Sauvegarde du cube hyperspectral ───────────────────
        save_envi_cube(plot, out_dir / name)

        # ── 2. Carte ratio ────────────────────────────────────────
        b1    = get_band_index(img, WAVELENGTH_1)
        b2    = get_band_index(img, WAVELENGTH_2)
        ratio = compute_ratio_map(plot, b1, b2)

        # ── 3. PNG SANS contours → même nom que le JSON (pour LabelMe)
        save_ratio_plain(ratio, out_dir / f"{name}.png")

        # ── 4. PNG AVEC contours rouges → fichier de visualisation
        save_ratio_with_contours(
            ratio,
            [s["points"] for s in shapes],
            out_dir / f"{name}_ratio.png"
        )

        # ── 5. JSON compatible LabelMe ────────────────────────────
        save_labelme_json(
            path        = out_dir / f"{name}.json",
            shapes      = shapes,
            image_name  = f"{name}.png",   # pointe vers le PNG sans contours
            H           = H,
            W           = W,
        )

        print(f"Plot {i:>4}  patches: {len(shapes):>2}  {dict(counter)}")

    # ── Résumé final ──────────────────────────────────────────────
    print("\n[RÉSUMÉ] Placements réels par classe :")
    for cls, cnt in sorted(counter.items()):
        print(f"  {cls}: {cnt}")
    counts = list(counter.values())
    if counts:
        print(f"  min={min(counts)}  max={max(counts)}  "
              f"écart={max(counts) - min(counts)}")


if __name__ == "__main__":
    generate("plots_zone_to_add", "aug_after_rot_flip", "output")




import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection

wl = np.array(wavelengths)
idx_996  = np.argmin(np.abs(wl - 996))
idx_1197 = np.argmin(np.abs(wl - 1197))

band_996  = cube_BHW[idx_996]
band_1197 = cube_BHW[idx_1197]

ratio = np.where(band_1197 > 1e-6, band_996 / band_1197, np.nan)

fig, ax = plt.subplots(figsize=(14, 10))
im = ax.imshow(ratio, cmap="RdYlGn", interpolation="nearest")
plt.colorbar(im, ax=ax, label="Ratio 996/1197")
ax.set_title("Ratio bandes 996nm / 1197nm avec contours annotations")

colors = {"plot": "blue", "trou": "red"}
default_color = "yellow"

for shape in original_json_data["shapes"]:
    label = shape["label"]
    pts   = np.array(shape["points"])

    if label == "plot":
        x0, y0 = pts[:,0].min(), pts[:,1].min()
        x1, y1 = pts[:,0].max(), pts[:,1].max()
        rect = mpatches.Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            linewidth=2, edgecolor="blue", facecolor="none", label="plot"
        )
        ax.add_patch(rect)
        ax.text(x0 + 2, y0 + 12, "plot", color="blue", fontsize=7)

    elif label == "trou":
        poly = MplPolygon(pts, closed=True, linewidth=1.5,
                          edgecolor="red", facecolor="none")
        ax.add_patch(poly)

    else:
        color = default_color
        poly = MplPolygon(pts, closed=True, linewidth=1.5,
                          edgecolor=color, facecolor="none")
        ax.add_patch(poly)
        cx, cy = pts[:,0].mean(), pts[:,1].mean()
        ax.text(cx, cy, label, color=color, fontsize=6, ha="center")

legend_elements = [
    mpatches.Patch(edgecolor="blue",   facecolor="none", label="plot"),
    mpatches.Patch(edgecolor="red",    facecolor="none", label="trou"),
    mpatches.Patch(edgecolor="yellow", facecolor="none", label="extrude"),
]
ax.legend(handles=legend_elements, loc="upper right", fontsize=8)

plt.tight_layout()
plt.show()