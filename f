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

def load_cube(hdr_path):
    img = envi.open(str(hdr_path))
    return np.array(img.load(), dtype=np.float32), img

def save_envi_cube(cube, path):
    envi.save_image(str(path)+".hdr", cube, dtype=np.float32, interleave="bsq", force=True)

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
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(ratio, cmap="viridis")
    plt.colorbar(im, ax=ax, fraction=0.03)
    for poly in polygons:
        if poly is None or len(poly)<3:
            continue
        p = np.array(poly)
        p = np.vstack([p, p[0]])
        ax.plot(p[:,0], p[:,1], color="red", linewidth=1.5)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()

def get_valid_mask_fast(cube):
    return cube.var(axis=0) > 1e-6

def crop_to_valid(cube, mask):
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if len(rows)==0 or len(cols)==0:
        return cube, mask
    return cube[:, rows[0]:rows[-1]+1, cols[0]:cols[-1]+1], mask[rows[0]:rows[-1]+1, cols[0]:cols[-1]+1]

def find_contour_polygons(mask):
    mask_u8 = mask.astype(np.uint8)*255
    contours,_ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys=[]
    for c in contours:
        pts=c.reshape(-1,2).tolist()
        if len(pts)>=3:
            polys.append(pts)
    return polys

def polygon_to_mask(poly,H,W):
    m=np.zeros((H,W),dtype=np.uint8)
    cv2.fillPoly(m,[np.array(poly,dtype=np.int32)],1)
    return m.astype(bool)

def compute_valid_positions(zone_mask, extrude_mask):
    return cv2.erode(zone_mask.astype(np.uint8)*255, extrude_mask.astype(np.uint8)).astype(bool)

def can_place(mask, occ, x, y):
    h,w=mask.shape
    return not (occ[y:y+h,x:x+w] & mask).any()

def update_occ(occ, mask, x, y):
    h,w=mask.shape
    occ[y:y+h,x:x+w] |= mask

def place_full(plot, occ, cube, mask, zone_poly):
    _,H,W=plot.shape
    zone_mask=polygon_to_mask(zone_poly,H,W)
    valid=compute_valid_positions(zone_mask,mask)
    ys,xs=np.where(valid)
    if len(xs)==0:
        return False,None
    idx=random.randint(0,len(xs)-1)
    y,x=ys[idx],xs[idx]
    h,w=mask.shape
    for b in range(plot.shape[0]):
        patch=plot[b,y:y+h,x:x+w]
        patch[mask]=cube[b][mask]
        plot[b,y:y+h,x:x+w]=patch
    update_occ(occ,mask,x,y)
    poly=find_contour_polygons(mask)[0]
    poly=np.array(poly); poly[:,0]+=x; poly[:,1]+=y
    return True,poly.tolist()

def random_patch(mask):
    h,w=mask.shape
    m=np.zeros((h,w),dtype=np.uint8)
    cx=random.randint(0,w-1); cy=random.randint(0,h-1)
    r=random.randint(10,40)
    pts=[]
    for _ in range(random.randint(5,10)):
        a=random.uniform(0,2*np.pi)
        rr=random.uniform(r*0.5,r)
        x=int(cx+rr*np.cos(a)); y=int(cy+rr*np.sin(a))
        x=np.clip(x,0,w-1); y=np.clip(y,0,h-1)
        pts.append([x,y])
    cv2.fillPoly(m,[np.array(pts,dtype=np.int32)],1)
    return m.astype(bool)

def place_patch(plot, occ, cube, mask, zone_poly):
    _,H,W=plot.shape
    zone_mask=polygon_to_mask(zone_poly,H,W)
    rp=random_patch(mask)
    combined=rp & mask
    valid=compute_valid_positions(zone_mask,combined)
    ys,xs=np.where(valid)
    if len(xs)==0:
        return False,None
    idx=random.randint(0,len(xs)-1)
    y,x=ys[idx],xs[idx]
    h,w=combined.shape
    for b in range(plot.shape[0]):
        patch=plot[b,y:y+h,x:x+w]
        patch[combined]=cube[b][combined]
        plot[b,y:y+h,x:x+w]=patch
    update_occ(occ,combined,x,y)
    poly=find_contour_polygons(combined)[0]
    poly=np.array(poly); poly[:,0]+=x; poly[:,1]+=y
    return True,poly.tolist()

def get_class(name):
    return name.split("_")[0]

def load_pool(root):
    return [{"hdr":f,"class":get_class(f.stem)} for f in Path(root).rglob("*.hdr")]

def load_plot(p):
    cube,img=load_cube(p)
    with open(p.with_suffix(".json")) as f:
        data=json.load(f)
    zones=[s["points"] for s in data["shapes"] if s["label"]=="zone_placement"]
    return cube,zones,img

def generate(plot_root, extrude_root, out_dir):
    plots=list(Path(plot_root).rglob("*.hdr"))
    pool=load_pool(extrude_root)
    counter=defaultdict(int)
    out_dir=Path(out_dir); out_dir.mkdir(exist_ok=True)

    for i in range(MAX_PLOTS):
        if len(pool)<12:
            break

        selected=random.sample(pool,12)
        singles=[e for e in selected if e["class"] in SINGLE_CLASSES]

        plot_hdr=random.choice(plots)
        plot,zones,img=load_plot(plot_hdr)
        _,H,W=plot.shape
        occ=np.zeros((H,W),bool)
        shapes=[]

        if singles:
            e=random.choice(singles)
            cube,_=load_cube(e["hdr"])
            mask=get_valid_mask_fast(cube)
            cube,mask=crop_to_valid(cube,mask)

            strat=random.choice([1,2])

            if strat==1:
                for z in zones:
                    ok,poly=place_full(plot,occ,cube,mask,z)
                    if ok:
                        shapes=[{"label":e["class"],"points":poly,"shape_type":"polygon"}]
                        counter[e["class"]]+=1
                        pool.remove(e)
                        break

            else:
                for z in zones:
                    ok,poly=place_patch(plot,occ,cube,mask,z)
                    if ok:
                        shapes.append({"label":e["class"],"points":poly,"shape_type":"polygon"})
                        counter[e["class"]]+=1
                        pool.remove(e)
                        break

                others=[x for x in selected if x!=e]
                for e2 in others:
                    cube,_=load_cube(e2["hdr"])
                    mask=get_valid_mask_fast(cube)
                    cube,mask=crop_to_valid(cube,mask)
                    for z in zones:
                        ok,poly=place_full(plot,occ,cube,mask,z)
                        if ok:
                            shapes.append({"label":e2["class"],"points":poly,"shape_type":"polygon"})
                            counter[e2["class"]]+=1
                            pool.remove(e2)
                            break

        else:
            for e in selected:
                cube,_=load_cube(e["hdr"])
                mask=get_valid_mask_fast(cube)
                cube,mask=crop_to_valid(cube,mask)
                for z in zones:
                    ok,poly=place_full(plot,occ,cube,mask,z)
                    if ok:
                        shapes.append({"label":e["class"],"points":poly,"shape_type":"polygon"})
                        counter[e["class"]]+=1
                        pool.remove(e)
                        break

        if not shapes:
            continue

        name=f"aug_{i}"
        save_envi_cube(plot,out_dir/name)

        with open(out_dir/f"{name}.json","w") as f:
            json.dump({
                "version":"5.0.1",
                "shapes":shapes,
                "imagePath":f"{name}.png",
                "imageHeight":H,
                "imageWidth":W
            },f,indent=2)

        b1=get_band_index(img,WAVELENGTH_1)
        b2=get_band_index(img,WAVELENGTH_2)
        ratio=compute_ratio_map(plot,b1,b2)
        save_ratio_map(ratio,[s["points"] for s in shapes],out_dir/f"{name}_ratio.png")

        print("Plot",i,"->",dict(counter))

if __name__=="__main__":
    generate("plots_zone_to_add","aug_after_rot_flip","output")