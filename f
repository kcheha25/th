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




import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon as MplPolygon
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from pathlib import Path
from spectral import envi


def load_cube(cube_folder):
    spec        = SpecArray.from_folder(str(cube_folder))
    cube_BHW    = np.array(spec.spectral_albedo).transpose(1, 0, 2)
    wavelengths = np.array(spec.wavelengths)
    return cube_BHW, wavelengths


def parse_labelme(json_path):
    with open(json_path) as f:
        data = json.load(f)

    plots, extrudes, trous = [], [], []

    for shape in data["shapes"]:
        label = shape["label"]
        pts   = shape["points"]

        if label == "plot":
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            plots.append({
                "label" : label,
                "rect"  : (min(xs), min(ys), max(xs), max(ys)),
                "shape" : shape,
            })
        elif label == "trou":
            trous.append(Polygon(pts))
        else:
            extrudes.append({
                "label"   : label,
                "polygon" : Polygon(pts),
                "shape"   : shape,
            })

    return plots, extrudes, trous, data


def assign_shapes_to_plots(plots, extrudes, trous):
    for plot in plots:
        x0, y0, x1, y1 = plot["rect"]
        plot_box = Polygon([(x0,y0),(x1,y0),(x1,y1),(x0,y1)])

        plot["extrudes"] = [
            e for e in extrudes
            if plot_box.contains(e["polygon"].centroid)
        ]
        plot["trous"] = [
            t for t in trous
            if plot_box.contains(t.centroid)
        ]

    return plots


def get_classes_with_overlapping_trou(plot):
    trou_union = unary_union(plot["trous"]) if plot["trous"] else None
    if trou_union is None:
        return set()

    classes = set()
    for ext in plot["extrudes"]:
        if ext["polygon"].intersects(trou_union):
            classes.add(ext["label"])
    return classes


def polygon_to_relative(polygon, x0, y0):
    return [(x - x0, y - y0) for x, y in polygon.exterior.coords[:-1]]


def build_output_json(plot, original_json_data):
    x0, y0, x1, y1 = plot["rect"]

    trou_union     = unary_union(plot["trous"]) if plot["trous"] else None
    target_classes = get_classes_with_overlapping_trou(plot)

    shapes_out = []

    for ext in plot["extrudes"]:
        if ext["label"] not in target_classes:
            continue

        geom = ext["polygon"]

        if trou_union is not None:
            geom = geom.difference(trou_union)

        if geom.is_empty:
            continue

        parts = geom.geoms if isinstance(geom, MultiPolygon) else [geom]

        for part in parts:
            if part.is_empty or part.area < 1:
                continue
            rel_pts = polygon_to_relative(part, x0, y0)
            shapes_out.append({
                "label"     : ext["label"],
                "points"    : [[round(x, 2), round(y, 2)] for x, y in rel_pts],
                "shape_type": "polygon",
                "flags"     : {},
                "group_id"  : None,
            })

    return {
        "version"    : original_json_data.get("version", "5.0.1"),
        "flags"      : {},
        "shapes"     : shapes_out,
        "imagePath"  : f"plot_{int(x0)}_{int(y0)}.png",
        "imageData"  : None,
        "imageHeight": int(y1 - y0),
        "imageWidth" : int(x1 - x0),
    }


def visualize_ratio(cube_BHW, wavelengths, original_json_data):
    wl       = np.array(wavelengths)
    idx_996  = np.argmin(np.abs(wl - 996))
    idx_1197 = np.argmin(np.abs(wl - 1197))

    band_996  = cube_BHW[idx_996]
    band_1197 = cube_BHW[idx_1197]
    ratio     = np.where(band_1197 > 1e-6, band_996 / band_1197, np.nan)

    fig, ax = plt.subplots(figsize=(14, 10))
    im = ax.imshow(ratio, cmap="RdYlGn", interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Ratio 996/1197")
    ax.set_title("Ratio bandes 996nm / 1197nm avec contours annotations")

    for shape in original_json_data["shapes"]:
        label = shape["label"]
        pts   = np.array(shape["points"])

        if label == "plot":
            x0, y0 = pts[:,0].min(), pts[:,1].min()
            x1, y1 = pts[:,0].max(), pts[:,1].max()
            rect = mpatches.Rectangle(
                (x0, y0), x1 - x0, y1 - y0,
                linewidth=2, edgecolor="blue", facecolor="none"
            )
            ax.add_patch(rect)
            ax.text(x0 + 2, y0 + 12, "plot", color="blue", fontsize=7)

        elif label == "trou":
            poly = MplPolygon(pts, closed=True, linewidth=1.5,
                              edgecolor="red", facecolor="none")
            ax.add_patch(poly)

        else:
            poly = MplPolygon(pts, closed=True, linewidth=1.5,
                              edgecolor="yellow", facecolor="none")
            ax.add_patch(poly)
            cx, cy = pts[:,0].mean(), pts[:,1].mean()
            ax.text(cx, cy, label, color="yellow", fontsize=6, ha="center")

    legend_elements = [
        mpatches.Patch(edgecolor="blue",   facecolor="none", label="plot"),
        mpatches.Patch(edgecolor="red",    facecolor="none", label="trou"),
        mpatches.Patch(edgecolor="yellow", facecolor="none", label="extrude"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.show()


def process_cubes(cube_folders, json_paths, output_dir, visualize=True):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for cube_folder, json_path in zip(cube_folders, json_paths):
        cube_folder = Path(cube_folder)
        json_path   = Path(json_path)
        cube_name   = cube_folder.stem

        print(f"\n── Cube: {cube_name}")

        cube_BHW, wavelengths = load_cube(cube_folder)

        plots, extrudes, trous, original_json_data = parse_labelme(json_path)
        plots = assign_shapes_to_plots(plots, extrudes, trous)

        print(f"   {len(plots)} plots | {len(extrudes)} extrudes | {len(trous)} trous")

        if visualize:
            visualize_ratio(cube_BHW, wavelengths, original_json_data)

        cube_out_dir = output_dir / cube_name
        cube_out_dir.mkdir(exist_ok=True)

        for i, plot in enumerate(plots):
            x0, y0, x1, y1 = plot["rect"]
            x0i, y0i = int(x0), int(y0)
            x1i, y1i = int(x1), int(y1)

            target_classes = get_classes_with_overlapping_trou(plot)
            if not target_classes:
                print(f"   Plot {i}: aucun trou superposé → ignoré")
                continue

            crop      = cube_BHW[:, y0i:y1i, x0i:x1i]
            json_out  = build_output_json(plot, original_json_data)
            plot_id   = f"plot_{i:03d}_{x0i}_{y0i}"
            json_file = cube_out_dir / f"{plot_id}.json"

            with open(json_file, "w") as f:
                json.dump(json_out, f, indent=2)

            print(f"   Plot {i} ({x0i},{y0i})→({x1i},{y1i}) | classes: {target_classes} | {len(json_out['shapes'])} polygones → {json_file.name}")


if __name__ == "__main__":
    process_cubes(
        cube_folders = ["data/cube1", "data/cube2"],
        json_paths   = ["data/cube1.json", "data/cube2.json"],
        output_dir   = "output/crops",
        visualize    = True,
    )


import json
import numpy as np
import cv2
from pathlib import Path
from specarray import SpecArray
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union

DATA_ROOT   = Path("data_cubes")
LABELME_DIR = Path("labelme")
OUTPUT_DIR  = Path("extrudes_eroded")

CLASSES_A_IGNORER = ["trou", "plots"]
json_files        = list(LABELME_DIR.glob("*.json"))
kernel            = np.ones((3, 3), np.uint8)

for json_file in json_files:
    with open(json_file) as f:
        data = json.load(f)

    base     = json_file.stem
    cube_dir = DATA_ROOT / base
    if not cube_dir.exists():
        continue

    spec = SpecArray.from_folder(cube_dir)
    cube = np.array(spec.spectral_albedo)
    H, B, W = cube.shape

    trous = [
        Polygon(s["points"])
        for s in data["shapes"]
        if s["label"] == "trou" and len(s["points"]) >= 4
    ]
    trou_union = unary_union(trous) if trous else None

    class_counter = {}  # reset par cube

    for shape in data["shapes"]:
        label = shape["label"]
        if label in CLASSES_A_IGNORER:
            continue
        if len(shape["points"]) < 4:
            continue

        pts         = np.array(shape["points"], dtype=np.int32)
        ext_polygon = Polygon(shape["points"])

        mask        = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 1)
        mask_eroded = cv2.erode(mask, kernel, iterations=1)
        if mask_eroded.sum() == 0:
            continue

        ys, xs       = np.where(mask_eroded > 0)
        y_min, y_max = ys.min(), ys.max()
        x_min, x_max = xs.min(), xs.max()

        if label not in class_counter:
            class_counter[label] = 0
        idx = class_counter[label]
        class_counter[label] += 1

        if trou_union is None or not ext_polygon.intersects(trou_union):
            continue

        shapes_out = []
        for trou in trous:
            if not ext_polygon.intersects(trou):
                continue
            geom = trou.intersection(ext_polygon)
            if geom.is_empty:
                continue
            parts = geom.geoms if isinstance(geom, MultiPolygon) else [geom]
            for part in parts:
                if part.is_empty or part.area < 1:
                    continue
                coords = list(part.exterior.coords[:-1])
                if len(coords) < 3:
                    continue
                rel_pts = [
                    [round(x - x_min, 2), round(y - y_min, 2)]
                    for x, y in coords
                ]
                shapes_out.append({
                    "label"     : "trou",
                    "points"    : rel_pts,
                    "shape_type": "polygon",
                    "flags"     : {},
                    "group_id"  : None,
                })

        if not shapes_out:
            continue

    json_out = {
        "version"    : data.get("version", "5.0.1"),
        "flags"      : {},
        "shapes"     : shapes_out,
        "imagePath"  : f"{label}_{idx}.png",
        "imageData"  : None,
        "imageHeight": int(y_max - y_min + 1),
        "imageWidth" : int(x_max - x_min + 1),
        "extrude_info": {
            "x_min"  : int(x_min),
            "y_min"  : int(y_min),
            "x_max"  : int(x_max),
            "y_max"  : int(y_max),
            "height" : int(y_max - y_min + 1),
            "width"  : int(x_max - x_min + 1),
        }
    }

        out_dir       = OUTPUT_DIR / f"{label}_{idx}"
        json_out_path = out_dir / f"{label}_{idx}.json"
        with open(json_out_path, "w") as f:
            json.dump(json_out, f, indent=2)
        print(f"   → {len(shapes_out)} trou(s) → {json_out_path.name}")

print("JSON trous sauvegardés.")



import numpy as np
import json
from pathlib import Path
from spectral import envi
import matplotlib.pyplot as plt
import random
from scipy.ndimage import rotate


def transform_cube_spatial(cube, min_angle=0, max_angle=360):
    B, H, W = cube.shape
    angle   = random.uniform(min_angle, max_angle)
    operation = f"rot_{angle:.1f}°"

    rotated_bands = []
    for b in range(B):
        rotated_band = rotate(cube[b], angle, reshape=True, order=0, mode='constant', cval=0)
        rotated_bands.append(rotated_band)

    max_h = max(band.shape[0] for band in rotated_bands)
    max_w = max(band.shape[1] for band in rotated_bands)

    transformed_cube = np.zeros((B, max_h, max_w), dtype=cube.dtype)
    for b, band in enumerate(rotated_bands):
        h, w = band.shape
        transformed_cube[b, :h, :w] = band

    H_new, W_new = max_h, max_w
    return transformed_cube, operation, H_new, W_new, angle


def transform_points_rotation(points, angle, H, W, H_new, W_new):
    rad    = np.deg2rad(angle)
    cx, cy = W / 2, H / 2
    transformed = []
    for x, y in points:
        x_new = cx + (x - cx) * np.cos(rad) - (y - cy) * np.sin(rad)
        y_new = cy + (x - cx) * np.sin(rad) + (y - cy) * np.cos(rad)
        transformed.append([round(x_new, 2), round(y_new, 2)])
    return transformed


def find_matching_json(label, H, W, json_dir):
    for jpath in Path(json_dir).rglob(f"{label}_*.json"):
        with open(jpath) as f:
            data = json.load(f)
        if data.get("imageHeight") == H and data.get("imageWidth") == W:
            return jpath, data
    return None, None


def augment_existing_cubes(cube_paths, json_dir, output_dir, visualize=True, min_angle=0, max_angle=360):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for cube_path in cube_paths:
        cube_path = Path(cube_path)
        hdr_img   = envi.open(str(cube_path))
        cube      = np.array(hdr_img.load(), dtype=np.float32)
        H, B, W   = cube.shape
        cube_BHW  = cube.transpose(1, 0, 2)

        transformed_cube, operation, H_new, W_new, angle = transform_cube_spatial(cube_BHW, min_angle, max_angle)

        if visualize:
            plt.figure(figsize=(8, 4))
            plt.subplot(1, 2, 1)
            plt.imshow(np.mean(cube_BHW, axis=0), cmap="gray")
            plt.title("Avant")
            plt.subplot(1, 2, 2)
            plt.imshow(np.mean(transformed_cube, axis=0), cmap="gray")
            plt.title(f"Après {operation}")
            plt.show()

        meta            = hdr_img.metadata.copy()
        meta['lines']   = int(H_new)
        meta['samples'] = int(W_new)
        meta['bands']   = int(transformed_cube.shape[0])

        safe_op   = operation.replace('.', '_').replace('°', '')
        new_name  = f"{cube_path.stem}_{safe_op}{cube_path.suffix}"
        save_path = output_dir / new_name

        envi.save_image(
            str(save_path),
            transformed_cube.astype(np.float32),
            force=True, interleave="bil", metadata=meta
        )
        print(f"✔ {cube_path.name} transformé ({operation}) → {save_path.name}")

        label                = cube_path.stem.split("_")[0]
        json_path, json_data = find_matching_json(label, H, W, json_dir)

        if json_data is None:
            print(f"   ↳ aucun JSON trou trouvé pour {label} ({H}x{W})")
            continue

        shapes_out = []
        for shape in json_data["shapes"]:
            new_pts = transform_points_rotation(shape["points"], angle, H, W, H_new, W_new)
            shapes_out.append({
                "label"     : "trou",
                "points"    : new_pts,
                "shape_type": "polygon",
                "flags"     : {},
                "group_id"  : None,
            })

        json_out = {
            "version"    : json_data.get("version", "5.0.1"),
            "flags"      : {},
            "shapes"     : shapes_out,
            "imagePath"  : f"{save_path.stem}.png",
            "imageData"  : None,
            "imageHeight": int(H_new),
            "imageWidth" : int(W_new),
            "extrude_info": {
                "x_min" : json_data.get("extrude_info", {}).get("x_min"),
                "y_min" : json_data.get("extrude_info", {}).get("y_min"),
                "x_max" : json_data.get("extrude_info", {}).get("x_max"),
                "y_max" : json_data.get("extrude_info", {}).get("y_max"),
                "height": int(H_new),
                "width" : int(W_new),
            }
        }

        json_out_path = output_dir / f"{save_path.stem}.json"
        with open(json_out_path, "w") as f:
            json.dump(json_out, f, indent=2)
        print(f"   ↳ JSON trous → {json_out_path.name} ({len(shapes_out)} trou(s))")


if __name__ == "__main__":
    cube_paths = list(Path("extrudes_eroded").rglob("*.hdr"))
    augment_existing_cubes(
        cube_paths = cube_paths,
        json_dir   = "extrudes_eroded",
        output_dir = "aug_data",
        visualize  = False,
    )