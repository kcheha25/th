import numpy as np
from pathlib import Path
import json
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon as MplPolygon
import spectral.io.envi as envi


PLOTS_DIR      = "output_annotated"
EXTRUDE_ROOT   = "aug_data"
OUT_DIR        = "output_placed"
N_PER_PLOT     = 5
TARGET_TOTAL   = 200
THRESHOLD      = 1e-6
SINGLE_CLASSES = ["A1", "B2"]

COLORS = ["#f38ba8", "#a6e3a1", "#fab387", "#89b4fa",
          "#f9e2af", "#cba6f7", "#94e2d5", "#eba0ac"]


def load_hdr(hdr_path):
    img  = envi.open(str(hdr_path))
    cube = np.array(img.load(), dtype=np.float32)
    meta = img.metadata.copy()
    return cube, meta


def get_valid_mask(cube, threshold=1e-6):
    return cube.var(axis=0) > threshold


def crop_to_valid(cube, valid_mask):
    rows = np.where(valid_mask.any(axis=1))[0]
    cols = np.where(valid_mask.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return cube, valid_mask, 0, 0
    r0, r1 = rows[0], rows[-1]+1
    c0, c1 = cols[0], cols[-1]+1
    return cube[:, r0:r1, c0:c1], valid_mask[r0:r1, c0:c1], r0, c0


def find_contour_polygons(valid_mask):
    """
    Détecte les contours exacts des pixels valides.
    Retourne une liste de polygons [[x,y],...] en coordonnées image.
    """
    mask_u8     = valid_mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for cnt in contours:
        pts = cnt.reshape(-1, 2).tolist()
        if len(pts) >= 3:
            polygons.append(pts)
    return polygons


def polygon_mask_2d(H, W, pts):
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(pts, dtype=np.int32)], 1)
    return mask.astype(bool)


def update_occupied_mask(occupied_mask, new_pts):
    pts_arr = np.array(new_pts, dtype=np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(occupied_mask, [pts_arr], 1)
    return occupied_mask


def clip_pts_to_zone(pts, zone_mask):
    zone_rows, zone_cols = np.where(zone_mask)
    if len(zone_rows) == 0:
        return pts
    zone_r0 = int(zone_rows.min()); zone_r1 = int(zone_rows.max())
    zone_c0 = int(zone_cols.min()); zone_c1 = int(zone_cols.max())
    return [
        [max(zone_c0, min(zone_c1, int(x))),
         max(zone_r0, min(zone_r1, int(y)))]
        for x, y in pts
    ]


def translate_polygon(poly, offset_x, offset_y):
    """Translate un polygon de offset_x, offset_y."""
    return [[int(x + offset_x), int(y + offset_y)] for x, y in poly]


def insert_spectra(plot_cube, ext_cube, ext_valid_mask, new_pts, poly_extrude, H, W):
    """
    Copie les spectres valides de ext_cube dans plot_cube
    aux pixels du polygon new_pts (translaté dans le plot).
    - ext_valid_mask : masque des pixels valides dans l extrude croppé
    - poly_extrude   : polygon de contour de l extrude (coords dans l extrude croppé)
    - new_pts        : polygon translaté dans le plot
    """
    result   = plot_cube.copy()
    Be, He, We = ext_cube.shape

    # masque du polygon placé dans le plot (H x W)
    placed_mask = polygon_mask_2d(H, W, [(int(y), int(x)) for x, y in new_pts])

    # masque du polygon de contour dans l extrude croppé (He x We)
    ext_poly_mask = polygon_mask_2d(He, We, [(int(y), int(x)) for x, y in poly_extrude])

    # pixels valides dans l extrude = dans le polygon ET non nuls
    ext_valid = ext_poly_mask & ext_valid_mask

    # offset de translation : coin haut-gauche du polygon placé
    pts_arr   = np.array(new_pts)
    placed_c0 = int(pts_arr[:, 0].min())
    placed_r0 = int(pts_arr[:, 1].min())

    # coin haut-gauche du polygon dans l extrude
    ext_pts = np.array(poly_extrude)
    ext_c0  = int(ext_pts[:, 0].min())
    ext_r0  = int(ext_pts[:, 1].min())
    ext_h   = max(1, int(ext_pts[:, 1].max()) - ext_r0)
    ext_w   = max(1, int(ext_pts[:, 0].max()) - ext_c0)

    rows_placed, cols_placed = np.where(placed_mask)
    for r, c in zip(rows_placed, cols_placed):
        # coordonnée relative dans le polygon placé
        dr = r - placed_r0
        dc = c - placed_c0

        # coordonnée dans l extrude avec wrapping
        er = ext_r0 + (dr % ext_h)
        ec = ext_c0 + (dc % ext_w)

        if er >= He or ec >= We:
            continue
        # ne copier que les pixels valides
        if not ext_valid[er, ec]:
            continue

        result[:, r, c] = ext_cube[:, er, ec]

    return result


def find_valid_position(zone_mask, poly_pts, H, W, rng, occupied_mask):
    """
    Cherche une position dans zone_mask où le bounding box du polygon rentre.
    Retourne (offset_x, offset_y) ou None.
    """
    poly_arr   = np.array(poly_pts)
    poly_min_x = int(poly_arr[:, 0].min())
    poly_min_y = int(poly_arr[:, 1].min())
    pw         = int(poly_arr[:, 0].max() - poly_min_x)
    ph         = int(poly_arr[:, 1].max() - poly_min_y)

    zone_rows, zone_cols = np.where(zone_mask)
    valid_positions      = []
    for r, c in zip(zone_rows, zone_cols):
        if r + ph >= H or c + pw >= W:
            continue
        if not zone_mask[r:r+ph, c:c+pw].all():
            continue
        if occupied_mask[r:r+ph, c:c+pw].any():
            continue
        valid_positions.append((r, c))

    if not valid_positions:
        return None

    idx      = rng.integers(len(valid_positions))
    dr, dc   = valid_positions[idx]
    offset_x = int(dc) - poly_min_x
    offset_y = int(dr) - poly_min_y
    return offset_x, offset_y


def strategy_normal(ext, zone_mask, H, W, rng, occupied_mask):
    """
    Place l extrude entier dans la zone.
    Retourne (new_pts, poly_used) ou (None, None).
    """
    poly_indices = list(range(len(ext["polygons"])))
    rng.shuffle(poly_indices)

    for j in poly_indices:
        poly   = ext["polygons"][j]
        result = find_valid_position(zone_mask, poly, H, W, rng, occupied_mask)
        if result is None:
            continue
        offset_x, offset_y = result
        new_pts = translate_polygon(poly, offset_x, offset_y)
        new_pts = clip_pts_to_zone(new_pts, zone_mask)
        return new_pts, poly

    return None, None


def strategy1_single(ext_list, zone_mask, H, W, rng, occupied_mask):
    """
    Stratégie 1 : extrude single entier ou croppé aux bords de la zone.
    Retourne (new_pts, ext_cube, poly_used) ou (None, None, None).
    """
    indices = list(range(len(ext_list)))
    rng.shuffle(indices)

    # essai 1 : trouver un extrude qui rentre entièrement
    for i in indices:
        ext          = ext_list[i]
        poly_indices = list(range(len(ext["polygons"])))
        rng.shuffle(poly_indices)

        for j in poly_indices:
            poly   = ext["polygons"][j]
            result = find_valid_position(zone_mask, poly, H, W, rng, occupied_mask)
            if result is None:
                continue
            offset_x, offset_y = result
            new_pts = translate_polygon(poly, offset_x, offset_y)
            new_pts = clip_pts_to_zone(new_pts, zone_mask)
            print(f"    S1 : extrude entier placé")
            return new_pts, ext["cube"], poly

    # fallback : crop aux bords de la zone
    print(f"    S1 : aucun ne rentre → crop")
    zone_rows, zone_cols = np.where(zone_mask)
    if len(zone_rows) == 0:
        return None, None, None

    zone_r0 = int(zone_rows.min()); zone_r1 = int(zone_rows.max())
    zone_c0 = int(zone_cols.min()); zone_c1 = int(zone_cols.max())

    i   = rng.integers(len(ext_list))
    ext = ext_list[i]
    j   = rng.integers(len(ext["polygons"]))
    poly     = ext["polygons"][j]
    poly_arr = np.array(poly)

    offset_x = zone_c0 - int(poly_arr[:, 0].min())
    offset_y = zone_r0 - int(poly_arr[:, 1].min())

    new_pts = [
        [max(zone_c0, min(zone_c1, int(x + offset_x))),
         max(zone_r0, min(zone_r1, int(y + offset_y)))]
        for x, y in poly
    ]

    if occupied_mask[zone_r0:zone_r1, zone_c0:zone_c1].any():
        return None, None, None

    return clip_pts_to_zone(new_pts, zone_mask), ext["cube"], poly


def strategy2_single(ext, zone_mask, H, W, rng, occupied_mask):
    """
    Stratégie 2 : place le vrai polygon de l extrude dans la zone.
    Les extrudes normaux déjà placés restent.
    Retourne (new_pts, poly_used) ou (None, None).
    """
    poly_indices = list(range(len(ext["polygons"])))
    rng.shuffle(poly_indices)

    for j in poly_indices:
        poly   = ext["polygons"][j]
        result = find_valid_position(zone_mask, poly, H, W, rng, occupied_mask)
        if result is None:
            continue
        offset_x, offset_y = result
        new_pts = translate_polygon(poly, offset_x, offset_y)
        new_pts = clip_pts_to_zone(new_pts, zone_mask)
        print(f"    S2 : polygon placé")
        return new_pts, poly

    # fallback crop
    print(f"    S2 : aucun ne rentre → crop")
    zone_rows, zone_cols = np.where(zone_mask)
    if len(zone_rows) == 0:
        return None, None

    zone_r0 = int(zone_rows.min()); zone_r1 = int(zone_rows.max())
    zone_c0 = int(zone_cols.min()); zone_c1 = int(zone_cols.max())

    j        = rng.integers(len(ext["polygons"]))
    poly     = ext["polygons"][j]
    poly_arr = np.array(poly)

    offset_x = zone_c0 - int(poly_arr[:, 0].min())
    offset_y = zone_r0 - int(poly_arr[:, 1].min())

    new_pts = [
        [max(zone_c0, min(zone_c1, int(x + offset_x))),
         max(zone_r0, min(zone_r1, int(y + offset_y)))]
        for x, y in poly
    ]

    if occupied_mask[zone_r0:zone_r1, zone_c0:zone_c1].any():
        return None, None

    return clip_pts_to_zone(new_pts, zone_mask), poly


def make_labelme_json(image_name, H, W, annotations):
    shapes = []
    for ann in annotations:
        shapes.append({
            "label"     : ann["label"],
            "points"    : ann["points"],
            "group_id"  : None,
            "shape_type": "polygon",
            "flags"     : {}
        })
    return {
        "version"    : "5.0.1",
        "flags"      : {},
        "shapes"     : shapes,
        "imagePath"  : image_name,
        "imageData"  : None,
        "imageHeight": H,
        "imageWidth" : W
    }


def load_plot_polygons(json_path):
    with open(json_path) as f:
        data = json.load(f)
    polygons = []
    for shape in data["shapes"]:
        if shape["shape_type"] == "polygon":
            polygons.append(shape["points"])
    return polygons


def save_cartography(mean_map, annotations, H, W, out_path):
    fig, ax = plt.subplots(1, 1, figsize=(W / 100, H / 100), dpi=100)
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_facecolor("#181825")
    ax.imshow(mean_map, cmap="inferno", aspect="auto", extent=[0, W, H, 0])
    ax.set_xlim(0, W); ax.set_ylim(H, 0)

    legend_classes = {}
    color_idx      = 0

    for shape in annotations:
        label = shape["label"]
        pts   = np.array(shape["points"])
        if label == "zone_placement":
            ax.add_patch(MplPolygon(pts, closed=True, linewidth=1.5,
                                    edgecolor="#89b4fa", facecolor="none",
                                    linestyle="--"))
            continue
        if label not in legend_classes:
            legend_classes[label] = COLORS[color_idx % len(COLORS)]
            color_idx += 1
        color = legend_classes[label]
        ax.add_patch(MplPolygon(pts, closed=True, linewidth=1.5,
                                edgecolor=color, facecolor=color, alpha=0.35))
        ax.text(pts[:, 0].mean(), pts[:, 1].mean(), label,
                color=color, fontsize=6, ha="center", va="center",
                bbox=dict(fc="#1e1e2e", alpha=0.7, pad=1))

    if legend_classes:
        legend_patches = [
            mpatches.Patch(color=c, label=cls, alpha=0.7)
            for cls, c in legend_classes.items()
        ]
        ax.legend(handles=legend_patches, loc="upper right", fontsize=7,
                  framealpha=0.7, facecolor="#1e1e2e", labelcolor="#cdd6f4")

    ax.axis("off")
    plt.savefig(out_path, dpi=100, facecolor="#181825",
                bbox_inches="tight", pad_inches=0)
    plt.show()
    plt.close()


def main():
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng     = np.random.default_rng()

    # ── charger les plots ─────────────────────────────────────────────────
    plots = []
    for hdr in sorted(Path(PLOTS_DIR).rglob("*.hdr")):
        json_path = hdr.parent / f"{hdr.stem}.json"
        if not json_path.exists():
            print(f"  JSON manquant pour {hdr.stem} → ignoré")
            continue
        try:
            cube, meta = load_hdr(hdr)
            polygons   = load_plot_polygons(json_path)
            if not polygons:
                continue
            plots.append({
                "name"    : hdr.stem,
                "cube"    : cube,
                "meta"    : meta,
                "path"    : str(hdr),
                "polygons": polygons
            })
        except Exception as e:
            print(f"  Erreur {hdr.name}: {e}")

    print(f"{len(plots)} plot(s) chargé(s)")
    if not plots:
        return

    # ── charger les extrudes avec contours exacts ─────────────────────────
    all_extrudes = {}
    for hdr in sorted(Path(EXTRUDE_ROOT).rglob("*.hdr")):
        try:
            class_name       = hdr.stem.split("_")[0]
            cube, _          = load_hdr(hdr)
            valid_mask       = get_valid_mask(cube, THRESHOLD)
            cube_c, mask_c, _, _ = crop_to_valid(cube, valid_mask)
            polygons         = find_contour_polygons(mask_c)
            if not polygons:
                continue
            if class_name not in all_extrudes:
                all_extrudes[class_name] = []
            all_extrudes[class_name].append({
                "name"       : hdr.stem,
                "cube"       : cube_c,          # cube croppé
                "valid_mask" : mask_c,           # masque valide croppé
                "polygons"   : polygons          # contours exacts
            })
        except Exception as e:
            print(f"  Erreur extrude {hdr.name}: {e}")

    print(f"Extrudes chargés : { {k: len(v) for k,v in all_extrudes.items()} }")
    if not all_extrudes:
        return

    class_counts = {cls: 0 for cls in all_extrudes}
    all_classes  = sorted(class_counts.keys())
    print(f"Toutes les classes : {all_classes}")
    print(f"Classes single     : {SINGLE_CLASSES}")

    generated = 0

    while generated < TARGET_TOTAL:
        plot      = plots[rng.integers(len(plots))]
        B, H, W   = plot["cube"].shape

        polygon_roi = plot["polygons"][rng.integers(len(plot["polygons"]))]
        zone_pts    = [(int(round(x)), int(round(y))) for x, y in polygon_roi]
        zone_mask   = polygon_mask_2d(H, W, [(y, x) for x, y in zone_pts])

        out_name      = f"{plot['name']}_aug_{generated:04d}"
        annotations   = [{"label": "zone_placement", "points": polygon_roi}]
        # placed : liste de (cls, new_pts, ext_cube, ext_valid_mask, poly_extrude)
        placed        = []
        occupied_mask = np.zeros((H, W), dtype=np.uint8)

        print(f"\n{out_name}")

        for k in range(N_PER_PLOT):
            min_count  = min(class_counts[cls] for cls in all_classes)
            least_used = [cls for cls in all_classes
                          if class_counts[cls] == min_count]
            cls        = least_used[rng.integers(len(least_used))]
            is_single  = cls in SINGLE_CLASSES

            if is_single:
                ext      = all_extrudes[cls][rng.integers(len(all_extrudes[cls]))]
                poly_arr = np.array(ext["polygons"][0])
                pw       = int(poly_arr[:, 0].max() - poly_arr[:, 0].min())
                ph       = int(poly_arr[:, 1].max() - poly_arr[:, 1].min())

                if pw < 20 and ph < 20:
                    print(f"  [{k+1}] Single {cls} petit ({pw}x{ph}px) → normal")
                    new_pts, poly = strategy_normal(ext, zone_mask, H, W, rng, occupied_mask)
                    if new_pts is not None:
                        occupied_mask = update_occupied_mask(occupied_mask, new_pts)
                        annotations.append({"label": cls, "points": new_pts})
                        placed.append((cls, new_pts, ext["cube"], ext["valid_mask"], poly))
                        class_counts[cls] += 1
                else:
                    strategy = int(rng.integers(1, 3))
                    print(f"  [{k+1}] Single {cls} → Stratégie {strategy}")

                    if strategy == 1:
                        # annuler les placements normaux déjà faits
                        for _, _, _, _, _ in placed:
                            pass
                        for cls_p, _, _, _, _ in placed:
                            class_counts[cls_p] -= 1
                        placed        = []
                        annotations   = [{"label": "zone_placement", "points": polygon_roi}]
                        occupied_mask = np.zeros((H, W), dtype=np.uint8)

                        new_pts, ext_cube, poly = strategy1_single(
                            all_extrudes[cls], zone_mask, H, W, rng, occupied_mask
                        )
                        if new_pts is not None:
                            # récupérer valid_mask de l extrude choisi
                            ext_vm = next(
                                e["valid_mask"] for e in all_extrudes[cls]
                                if e["cube"] is ext_cube
                            )
                            occupied_mask = update_occupied_mask(occupied_mask, new_pts)
                            annotations.append({"label": cls, "points": new_pts})
                            placed.append((cls, new_pts, ext_cube, ext_vm, poly))
                            class_counts[cls] += 1
                        break

                    else:
                        new_pts, poly = strategy2_single(
                            ext, zone_mask, H, W, rng, occupied_mask
                        )
                        if new_pts is not None:
                            occupied_mask = update_occupied_mask(occupied_mask, new_pts)
                            annotations.append({"label": cls, "points": new_pts})
                            placed.append((cls, new_pts, ext["cube"], ext["valid_mask"], poly))
                            class_counts[cls] += 1

            else:
                print(f"  [{k+1}] Normale {cls}")
                ext     = all_extrudes[cls][rng.integers(len(all_extrudes[cls]))]
                new_pts, poly = strategy_normal(ext, zone_mask, H, W, rng, occupied_mask)
                if new_pts is None:
                    print(f"    Pas de place pour {cls}")
                    continue
                occupied_mask = update_occupied_mask(occupied_mask, new_pts)
                annotations.append({"label": cls, "points": new_pts})
                placed.append((cls, new_pts, ext["cube"], ext["valid_mask"], poly))
                class_counts[cls] += 1

        if not placed:
            print(f"  Aucun placement réussi pour {out_name}, on réessaie")
            continue

        # ── insertion des spectres valides dans le cube du plot ───────────
        result_cube = plot["cube"].copy()
        for cls_p, pts_p, ext_cube, ext_vm, poly_ext in placed:
            result_cube = insert_spectra(
                result_cube, ext_cube, ext_vm, pts_p, poly_ext, H, W
            )

        # ── sauvegarde JSON labelme ───────────────────────────────────────
        json_data = make_labelme_json(f"{out_name}.png", H, W, annotations)
        json_out  = out_dir / f"{out_name}.json"
        with open(json_out, "w") as f:
            json.dump(json_data, f, indent=2)

        # ── sauvegarde HDR avec spectres insérés ──────────────────────────
        m = plot["meta"].copy()
        m["lines"]   = H
        m["samples"] = W
        m["bands"]   = B
        envi.save_image(str(out_dir / f"{out_name}.hdr"),
                        result_cube, dtype=np.float32,
                        interleave="bil", force=True, metadata=m)

        # ── cartographie depuis le cube augmenté ──────────────────────────
        mean_map_result = result_cube.mean(axis=0)
        png_out         = out_dir / f"{out_name}.png"
        save_cartography(mean_map_result, annotations, H, W, png_out)

        generated += 1
        print(f"  [{generated}/{TARGET_TOTAL}] {out_name} — {len(placed)} extrude(s)")
        print(f"  Compteurs : { {k:v for k,v in sorted(class_counts.items())} }")

    print(f"\n✓ {generated} plots générés dans {out_dir}")
    print(f"Bilan final : { {k:v for k,v in sorted(class_counts.items())} }")


if __name__ == "__main__":
    main()