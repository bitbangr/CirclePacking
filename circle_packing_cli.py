#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, uuid, math
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import cv2
from sklearn.cluster import KMeans
from itertools import permutations
import yaml  # <-- YAML support

# Balance speed/quality
reasoning_effort = "medium"

# ---------- helpers ----------
def announce(step: str, inputs: Dict[str, Any]):
    print(f"[STEP] {step} | inputs: " + ", ".join(f"{k}={v}" for k, v in inputs.items()))

def ensure_bool(cond: bool, msg: str):
    if not cond:
        raise RuntimeError(msg)

def within_bounds(pt: Tuple[int, int], w: int, h: int, r: int=0) -> bool:
    x, y = pt
    return (r <= x < (w - r)) and (r <= y < (h - r))

def circle_fits(mask: np.ndarray, center: Tuple[int,int], radius: int) -> bool:
    x0, y0 = center
    h, w = mask.shape
    if not within_bounds((x0, y0), w, h, r=radius):
        return False
    angles = np.linspace(0, 2*np.pi, num=36, endpoint=False)
    rings = [radius, int(radius*0.7), int(radius*0.4)]
    for rr in rings:
        if rr <= 0:
            continue
        xs = (x0 + (rr*np.cos(angles))).astype(int)
        ys = (y0 + (rr*np.sin(angles))).astype(int)
        if np.any(mask[ys, xs] == 0):
            return False
    return True

def no_overlap(center: Tuple[int,int], radius: int, placed: List[Tuple[Tuple[int,int], int]]) -> bool:
    cx, cy = center
    for (px, py), pr in placed:
        if (cx - px)**2 + (cy - py)**2 <= (radius + pr)**2:
            return False
    return True

def map_clusters_to_user_colors(cluster_centers_bgr: np.ndarray, user_colors_bgr: List[Tuple[int,int,int]]) -> List[int]:
    D = np.zeros((6, 6), dtype=float)
    for i, uc in enumerate(user_colors_bgr):
        uc_arr = np.array(uc, dtype=float)
        for j, cc in enumerate(cluster_centers_bgr):
            D[i, j] = np.linalg.norm(uc_arr - cc.astype(float))
    from itertools import permutations
    best_perm, best_cost = None, float("inf")
    for perm in permutations(range(6)):
        cost = sum(D[i, perm[i]] for i in range(6))
        if cost < best_cost:
            best_cost, best_perm = cost, perm
    return list(best_perm)

def candidate_points_for_region(mask: np.ndarray, edge_map: np.ndarray, max_samples: int = 6000) -> np.ndarray:
    h, w = mask.shape
    kernel = np.ones((3,3), np.uint8)
    edge_dil = cv2.dilate(edge_map, kernel, iterations=1)
    edge_zone = (edge_dil > 0) & (mask > 0)
    edge_pts = np.column_stack(np.where(edge_zone))[:, [1, 0]]  # (x,y)

    keep_edge = int(0.6 * max_samples)
    if edge_pts.shape[0] > keep_edge > 0:
        sel = np.random.choice(edge_pts.shape[0], size=keep_edge, replace=False)
        edge_pts = edge_pts[sel]

    stride = max(4, int(round(math.sqrt((w*h)/max(1, int(0.4*max_samples))))))  # coarse grid
    grid_y, grid_x = np.mgrid[0:h:stride, 0:w:stride]
    grid = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    grid = grid[(mask[grid[:,1], grid[:,0]] > 0)]
    keep_grid = int(0.5 * max_samples)
    if grid.shape[0] > keep_grid > 0:
        sel = np.random.choice(grid.shape[0], size=keep_grid, replace=False)
        grid = grid[sel]

    ys, xs = np.where(mask > 0)
    centroid = np.array([[int(xs.mean()), int(ys.mean())]]) if xs.size else np.empty((0,2), dtype=int)

    pts = np.vstack([centroid, edge_pts, grid]) if edge_pts.size else np.vstack([centroid, grid])
    if pts.shape[0] > 1:
        idx = np.arange(pts.shape[0]); np.random.shuffle(idx); pts = pts[idx]
    return pts

def pack_region_with_circles(mask: np.ndarray, edge_map: np.ndarray, diameters: List[int]) -> List[Dict[str, Any]]:
    circles, placed = [], []
    candidates = candidate_points_for_region(mask, edge_map, max_samples=6000)
    for d in sorted(set(diameters), reverse=True):
        r = max(1, int(round(d/2)))
        tried = 0
        for (x, y) in candidates:
            tried += 1
            if circle_fits(mask, (x, y), r) and no_overlap((x, y), r, placed):
                placed.append(((x, y), r))
                circles.append({'center': (int(x), int(y)), 'radius': int(r)})
        print(f"[PACK] diameter={d} tried={tried} placed={sum(1 for c in circles if c['radius']==r)}")
    return circles

def pack_circles_from_image(
    img_path: str,
    user_colors: List[Tuple[int,int,int]],
    circle_sizes: Optional[List[int]] = None,
    output_size: Tuple[int,int] = (1000, 1000),
    visualization_outdir: str = "./circle_packing_outputs"
) -> Dict[str, Any]:
    try:
        announce("LOAD_IMAGE", {"img_path": img_path, "output_size": output_size})
        ensure_bool(isinstance(user_colors, list) and len(user_colors) == 6, "Exactly six user RGB colors are required.")
        for tup in user_colors:
            ensure_bool(isinstance(tup, (list, tuple)) and len(tup) == 3, "Each user color must be a 3-tuple (R,G,B).")
        if circle_sizes is None:
            circle_sizes = [10, 30, 20, 50, 100, 150]
        ensure_bool(isinstance(circle_sizes, list) and len(circle_sizes) == 6, "circle_sizes must be a list of six diameters.")
        ensure_bool(isinstance(output_size, (list, tuple)) and len(output_size) == 2, "output_size must be (width, height).")

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        ensure_bool(img is not None, f"Image loading failed for path: {img_path}")
        width, height = int(output_size[0]), int(output_size[1])
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]
        print("[OK] Image loaded & resized.")

        announce("KMEANS_CLUSTERING", {"k": 6, "pixels": h*w})
        pixels = img.reshape(-1, 3).astype(np.float32)
        try:
            kmeans = KMeans(n_clusters=6, n_init=10, random_state=42)
            labels = kmeans.fit_predict(pixels)
            centers = kmeans.cluster_centers_.astype(np.uint8)
        except Exception as e:
            return {"error": f"K-means failed: {str(e)}"}
        ensure_bool(len(np.unique(labels)) == 6, "K-means did not produce 6 distinct clusters.")
        label_img = labels.reshape(h, w)
        print("[OK] K-means produced 6 clusters.")

        announce("MAP_CLUSTERS_TO_USER_COLORS", {"method": "minimize total Euclidean distance"})
        user_colors_bgr = [(c[2], c[1], c[0]) for c in user_colors]
        mapping = map_clusters_to_user_colors(centers, user_colors_bgr)
        ensure_bool(len(mapping) == 6 and len(set(mapping)) == 6, "Failed to map clusters to user colors uniquely.")
        print("[OK] Clusters mapped to user colors.")

        region_masks: List[np.ndarray] = []
        for user_idx in range(6):
            cluster_idx = mapping[user_idx]
            mask = (label_img == cluster_idx).astype(np.uint8) * 255
            region_masks.append(mask)

        combined = np.clip(sum((m > 0).astype(np.uint8) for m in region_masks), 0, 1)
        coverage = combined.mean()
        ensure_bool(coverage > 0.95, "Region masks do not sufficiently cover the image.")
        print("[OK] Region masks built with adequate coverage.")

        announce("EDGE_DETECTION", {"method": "Canny", "note": "prioritize placements"})
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        v = np.median(gray)
        lower = int(max(0, 0.66 * v))
        upper = int(min(255, 1.33 * v))
        edges = cv2.Canny(gray, lower, upper)
        print("[OK] Edge map computed.")

        diameters = circle_sizes[:]
        all_regions_output: List[Dict[str, Any]] = []
        packed_visual = img.copy()

        for idx, (mask, rgb_color) in enumerate(zip(region_masks, user_colors)):
            announce("PACK_REGION", {"region_index": idx, "allowed_diameters": diameters})
            region_edges = cv2.bitwise_and(edges, edges, mask=mask)
            region_circles = pack_region_with_circles(mask, region_edges, diameters)

            valid = []
            for c in region_circles:
                (x, y), r = c['center'], c['radius']
                if within_bounds((x, y), w, h, r=r) and circle_fits(mask, (x, y), r):
                    valid.append(c)
            if len(valid) != len(region_circles):
                print(f"[WARN] Removed {len(region_circles) - len(valid)} invalid circles after validation.")
            region_circles = valid

            # FILLED circles + subtle outline for visibility
            bgr = (int(rgb_color[2]), int(rgb_color[1]), int(rgb_color[0]))
            outline = tuple(int(v * 0.7) for v in bgr)
            for c in region_circles:
                cv2.circle(packed_visual, (c['center'][0], c['center'][1]), c['radius'], bgr, thickness=-1, lineType=cv2.LINE_AA)
                cv2.circle(packed_visual, (c['center'][0], c['center'][1]), c['radius'], outline, thickness=2, lineType=cv2.LINE_AA)

            counts = {}
            for c in region_circles:
                counts[c['radius']] = counts.get(c['radius'], 0) + 1
            circle_size_counts = sorted([(int(r), int(n)) for r, n in counts.items()], key=lambda t: -t[0])

            all_regions_output.append({
                "color": (int(rgb_color[0]), int(rgb_color[1]), int(rgb_color[2])),
                "circles": [
                    {"center": (int(c['center'][0]), int(c['center'][1])),
                     "radius": int(c['radius']),
                     "color": (int(rgb_color[0]), int(rgb_color[1]), int(rgb_color[2]))}
                    for c in region_circles
                ],
                "circle_size_counts": circle_size_counts
            })
            print(f"[OK] Region {idx} packed: {len(region_circles)} circles.")

        print("[NOTE] Packing is NP-hard; alternate diameters or local optimization can yield denser fills.")

        os.makedirs(visualization_outdir, exist_ok=True)
        vis_path = os.path.join(visualization_outdir, f"packing_{uuid.uuid4().hex}.png")
        announce("SAVE_VISUALIZATION", {"path": vis_path, "size": (w, h)})
        ok = cv2.imwrite(vis_path, packed_visual)
        ensure_bool(ok, "Failed to save visualization image.")
        print("[OK] Visualization saved.")

        announce("VALIDATE_OUTPUT_SCHEMA", {"regions": 6})
        ensure_bool(isinstance(all_regions_output, list) and len(all_regions_output) == 6, "regions must be a list of six dicts.")
        for reg in all_regions_output:
            ensure_bool(set(reg.keys()) == {"color", "circles", "circle_size_counts"}, "Region dict keys mismatch.")
            ensure_bool(isinstance(reg["color"], tuple) and len(reg["color"]) == 3, "Region color must be an (R,G,B) tuple.")
            for c in reg["circles"]:
                ensure_bool(set(c.keys()) == {"center", "radius", "color"}, "Circle dict keys mismatch.")
                ensure_bool(isinstance(c["center"], tuple) and len(c["center"]) == 2, "Circle center must be (x,y).")
                ensure_bool(isinstance(c["radius"], int) and c["radius"] > 0, "Circle radius must be positive int.")
                ensure_bool(isinstance(c["color"], tuple) and len(c["color"]) == 3, "Circle color must be an (R,G,B) tuple.")
            for rc in reg["circle_size_counts"]:
                ensure_bool(isinstance(rc, tuple) and len(rc) == 2, "circle_size_counts entries must be (radius, count).")
        print("[OK] Output schema validated.")

        return {
            "regions": all_regions_output,
            "visualization": vis_path,
            "visualization_type": "file",
            "image_size": (int(w), int(h))
        }

    except Exception as e:
        return {"error": str(e)}

# ---------- CLI ----------
def main():
    p = argparse.ArgumentParser(description="RGB-guided segmentation + discrete circle packing (YAML config).")
    p.add_argument("--config", required=True, help="Path to YAML config file (e.g., config.yaml).")
    p.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    args = p.parse_args()

    # Read YAML config
    try:
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
    except Exception as e:
        print(json.dumps({"error": f"Failed to read config: {e}"}))
        return

    img_path = cfg.get("img_path")
    user_colors = cfg.get("user_colors")
    circle_sizes = cfg.get("circle_sizes", None)
    output_size = tuple(cfg.get("output_size", [1000, 1000]))
    visualization_outdir = cfg.get("visualization_outdir", "./circle_packing_outputs")

    result = pack_circles_from_image(
        img_path=img_path,
        user_colors=[tuple(map(int, c)) for c in user_colors] if user_colors else None,
        circle_sizes=[int(x) for x in circle_sizes] if circle_sizes is not None else None,
        output_size=(int(output_size[0]), int(output_size[1])),
        visualization_outdir=visualization_outdir
    )

    # Convert tuples to lists for JSON printing
    def tuplify(o):
        if isinstance(o, tuple):
            return list(o)
        if isinstance(o, list):
            return [tuplify(v) for v in o]
        if isinstance(o, dict):
            return {k: tuplify(v) for k, v in o.items()}
        return o

    out = tuplify(result)
    print(json.dumps(out, indent=2 if args.pretty else None))

if __name__ == "__main__":
    main()
