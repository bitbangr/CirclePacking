#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, uuid, math
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import cv2
from sklearn.cluster import KMeans
from itertools import permutations
import yaml

# =========================
# Configurable constants
# =========================
# General behavior
REASONING_EFFORT = "medium"

# Defaults for optional inputs
DEFAULT_CIRCLE_SIZES = [10, 30, 20, 50, 100, 150]   # diameters (px)
DEFAULT_OUTPUT_SIZE = (1000, 1000)                  # (width, height)

# K-means parameters
KMEANS_N_INIT = 10
KMEANS_RANDOM_STATE = 42

# Edge detection (Canny) — thresholds = [LOWER_MULT * median, UPPER_MULT * median]
CANNY_LOWER_MULT = 0.66
CANNY_UPPER_MULT = 1.33

# Candidate generation
CANDIDATE_MAX_SAMPLES =20000  # was 6000
EDGE_SAMPLE_FRACTION = 0.4 # was 0.6    # portion of candidates prioritized near edges
GRID_SAMPLE_FRACTION = 0.5    # cap for grid-derived candidates
EDGE_DILATE_KERNEL_SIZE = 3   # kernel side (pixels) for dilating edges
GRID_STRIDE_COEFF = 0.7 # was 1.0       # tune the density of grid candidates (higher -> sparser)

# Circle containment/overlap checks
ANGLES_ON_RING = 36                   # samples per ring
RING_FRACTIONS = [1.0, 0.7, 0.4]      # rings to sample for containment
DISALLOW_TOUCHING = False # was True              # if True, require distance > r1 + r2

# Region coverage sanity-check
COVERAGE_THRESHOLD = 0.95

# Visualization
DRAW_FILLED_THICKNESS = -1
DRAW_OUTLINE_DARKEN = 0.7
DRAW_OUTLINE_THICKNESS = 1 # was 2
DRAW_LINE_TYPE = cv2.LINE_AA

# Assignment strategy
# For num_regions <= BRUTE_FORCE_LIMIT, use exact (permutation) assignment;
# otherwise use a greedy assignment (to avoid factorial blow-up).
BRUTE_FORCE_LIMIT = 8


# =========================
# Utility helpers
# =========================
def announce(step: str, inputs: Dict[str, Any]):
    """Per requirements: state purpose & minimal inputs before significant calls."""
    print(f"[STEP] {step} | inputs: " + ", ".join(f"{k}={v}" for k, v in inputs.items()))

def ensure_bool(cond: bool, msg: str):
    if not cond:
        raise RuntimeError(msg)

def within_bounds(pt: Tuple[int, int], w: int, h: int, r: int = 0) -> bool:
    x, y = pt
    return (r <= x < (w - r)) and (r <= y < (h - r))

def circle_fits(mask: np.ndarray, center: Tuple[int, int], radius: int) -> bool:
    """Check that a circle of radius lies entirely within a binary region mask."""
    x0, y0 = center
    h, w = mask.shape
    if not within_bounds((x0, y0), w, h, r=radius):
        return False
    # Sample points on multiple rings to check interior containment
    angles = np.linspace(0, 2 * np.pi, num=ANGLES_ON_RING, endpoint=False)
    for frac in RING_FRACTIONS:
        rr = int(round(radius * frac))
        if rr <= 0:
            continue
        xs = (x0 + (rr * np.cos(angles))).astype(int)
        ys = (y0 + (rr * np.sin(angles))).astype(int)
        if np.any(mask[ys, xs] == 0):
            return False
    return True

def no_overlap(center: Tuple[int, int], radius: int, placed: List[Tuple[Tuple[int, int], int]]) -> bool:
    """Return True if circle does not overlap previously placed circles."""
    cx, cy = center
    for (px, py), pr in placed:
        # distance^2 > (r1 + r2)^2; if touching is disallowed, strictly greater
        dist2 = (cx - px) ** 2 + (cy - py) ** 2
        min_d = radius + pr
        if DISALLOW_TOUCHING:
            if dist2 <= (min_d * min_d):
                return False
        else:
            if dist2 < (min_d * min_d):
                return False
    return True

import scipy.ndimage as ndi  # pip install scipy

def pack_region_with_circles_dt(mask: np.ndarray,
                                diameters: List[int]) -> List[Dict[str, Any]]:
    """
    High-density packing using distance transform (maximal discs).
    For each diameter (largest→smallest):
      - Compute distance to boundary of the *currently available* area.
      - Place a circle where distance is maximal and ≥ radius.
      - Carve the circle out of the available area and repeat.
    Returns [{'center': (x,y), 'radius': r}, ...]
    """
    # Working availability mask (1 inside region and not yet occupied)
    avail = (mask > 0).astype(np.uint8)

    circles: List[Dict[str, Any]] = []

    # Structuring element used to "carve out" placed circles
    def carve_circle(a: np.ndarray, cx: int, cy: int, r: int):
        cv2.circle(a, (cx, cy), r, 0, thickness=-1)

    for d in sorted(set(int(x) for x in diameters if x > 1), reverse=True):
        r = max(1, int(round(d / 2)))
        placed_this_d = 0

        # Loop until no space for this radius
        while True:
            if avail.max() == 0:
                break

            # Distance to the nearest zero (boundary); multiply by mask to zero-out outside
            # Use Euclidean distance (pixels).
            dt = cv2.distanceTransform(avail, distanceType=cv2.DIST_L2, maskSize=5)

            # Find the best location (global maximum)
            minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(dt)
            if maxVal < r:        # no place large enough for this radius
                break

            cx, cy = maxLoc       # OpenCV points are (x, y)
            # Safety check: ensure the circle lies fully in avail (rounding guards)
            if cy < 0 or cy >= dt.shape[0] or cx < 0 or cx >= dt.shape[1]:
                break

            # Record the circle
            circles.append({'center': (int(cx), int(cy)), 'radius': int(r)})
            placed_this_d += 1

            # Carve out the circle (and a tiny buffer to avoid numerical edge touching)
            carve_circle(avail, cx, cy, r)

        print(f"[DT] diameter={d} placed={placed_this_d}")

    return circles



# =========================
# Cluster-to-color assignment
# =========================
def _distance_matrix_bgr(cluster_centers_bgr: np.ndarray, user_colors_bgr: List[Tuple[int, int, int]]) -> np.ndarray:
    num_regions = len(user_colors_bgr)
    D = np.zeros((num_regions, num_regions), dtype=float)
    for i, uc in enumerate(user_colors_bgr):
        uc_arr = np.array(uc, dtype=float)
        for j, cc in enumerate(cluster_centers_bgr):
            D[i, j] = np.linalg.norm(uc_arr - cc.astype(float))
    return D

def _assign_bruteforce(D: np.ndarray) -> List[int]:
    """Exact assignment: returns perm mapping user_index -> cluster_index (min total cost)."""
    num_regions = D.shape[0]
    best_perm, best_cost = None, float("inf")
    for perm in permutations(range(num_regions)):
        cost = sum(D[i, perm[i]] for i in range(num_regions))
        if cost < best_cost:
            best_cost, best_perm = cost, perm
    return list(best_perm)

def _assign_greedy(D: np.ndarray) -> List[int]:
    """Greedy assignment for larger num_regions (avoids factorial blow-up)."""
    num_regions = D.shape[0]
    remaining_clusters = set(range(num_regions))
    mapping = [-1] * num_regions
    for i in range(num_regions):
        # pick nearest available cluster for user color i
        j = min(remaining_clusters, key=lambda c: D[i, c])
        mapping[i] = j
        remaining_clusters.remove(j)
    return mapping

def map_clusters_to_user_colors(cluster_centers_bgr: np.ndarray,
                                user_colors_bgr: List[Tuple[int, int, int]]) -> List[int]:
    """
    Returns a permutation mapping: user_index -> assigned_cluster_index,
    minimizing total Euclidean distance in BGR space.
    Uses brute-force for small num_regions, greedy otherwise.
    """
    num_regions = len(user_colors_bgr)
    D = _distance_matrix_bgr(cluster_centers_bgr, user_colors_bgr)
    if num_regions <= BRUTE_FORCE_LIMIT:
        return _assign_bruteforce(D)
    print(f"[INFO] Using greedy assignment (num_regions={num_regions} > {BRUTE_FORCE_LIMIT}).")
    return _assign_greedy(D)


# =========================
# Candidate generation (edge-aware)
# =========================
def candidate_points_for_region(mask: np.ndarray, edge_map: np.ndarray, max_samples: int = CANDIDATE_MAX_SAMPLES) -> np.ndarray:
    """
    Candidate (x,y) points prioritized near edges and spread across region.
    - EDGE_SAMPLE_FRACTION: points near edges (dilated Canny)
    - GRID_SAMPLE_FRACTION: cap for blue-noise-like grid over the region interior
    """
    h, w = mask.shape

    # Edge-prioritized candidates
    kernel = np.ones((EDGE_DILATE_KERNEL_SIZE, EDGE_DILATE_KERNEL_SIZE), np.uint8)
    edge_dil = cv2.dilate(edge_map, kernel, iterations=1)
    edge_zone = (edge_dil > 0) & (mask > 0)
    edge_pts = np.column_stack(np.where(edge_zone))  # (y, x)
    if edge_pts.size > 0:
        edge_pts = edge_pts[:, [1, 0]]  # -> (x, y)

    keep_edge = int(EDGE_SAMPLE_FRACTION * max_samples)
    if edge_pts.shape[0] > keep_edge > 0:
        sel = np.random.choice(edge_pts.shape[0], size=keep_edge, replace=False)
        edge_pts = edge_pts[sel]

    # Grid candidates (coarse grid approximating blue-noise)
    # Compute a stride that scales with image area and desired candidate count
    stride = max(
        4,
        int(round(
            math.sqrt((w * h) / max(1, int(GRID_SAMPLE_FRACTION * max_samples))) * GRID_STRIDE_COEFF
        ))
    )
    grid_y, grid_x = np.mgrid[0:h:stride, 0:w:stride]
    grid = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    grid = grid[(mask[grid[:, 1], grid[:, 0]] > 0)]
    keep_grid = int(GRID_SAMPLE_FRACTION * max_samples)
    if grid.shape[0] > keep_grid > 0:
        sel = np.random.choice(grid.shape[0], size=keep_grid, replace=False)
        grid = grid[sel]

    # Region centroid (acts as a high-priority seed)
    ys, xs = np.where(mask > 0)
    centroid = np.array([[int(xs.mean()), int(ys.mean())]]) if xs.size else np.empty((0, 2), dtype=int)

    # Combine & shuffle
    pts = np.vstack([centroid, edge_pts, grid]) if edge_pts.size else np.vstack([centroid, grid])
    if pts.shape[0] > 1:
        idx = np.arange(pts.shape[0]); np.random.shuffle(idx); pts = pts[idx]
    return pts


# =========================
# Packing heuristic
# =========================
def pack_region_with_circles(mask: np.ndarray, edge_map: np.ndarray, diameters: List[int]) -> List[Dict[str, Any]]:
    """
    Greedy multi-diameter packing:
      1) Generate candidates prioritized by edges & centroid.
      2) For each diameter (largest→smallest), place circles if:
         - Entirely inside mask (containment sampling)
         - No overlap with previously placed circles
    Returns list of circle dicts: {'center': (x,y), 'radius': r}
    """
    circles = []
    placed = []  # list of ((x,y), radius)
    candidates = candidate_points_for_region(mask, edge_map, max_samples=CANDIDATE_MAX_SAMPLES)

    for d in sorted(set(diameters), reverse=True):
        r = max(1, int(round(d / 2)))
        tried = 0
        for (x, y) in candidates:
            tried += 1
            if circle_fits(mask, (x, y), r) and no_overlap((x, y), r, placed):
                placed.append(((x, y), r))
                circles.append({'center': (int(x), int(y)), 'radius': int(r)})
        print(f"[PACK] diameter={d} tried={tried} placed={sum(1 for c in circles if c['radius'] == r)}")
    return circles


# =========================
# Main pipeline
# =========================
def pack_circles_from_image(
    img_path: str,
    user_colors: List[Tuple[int, int, int]],
    circle_sizes: Optional[List[int]] = None,
    output_size: Tuple[int, int] = DEFAULT_OUTPUT_SIZE,
    visualization_outdir: str = "./circle_packing_outputs"
) -> Dict[str, Any]:
    try:
        announce("LOAD_IMAGE", {"img_path": img_path, "output_size": output_size})

        # Validate inputs
        ensure_bool(isinstance(user_colors, list) and len(user_colors) >= 1,
                    "At least one user RGB color is required.")
        num_regions = len(user_colors)

        for tup in user_colors:
            ensure_bool(isinstance(tup, (list, tuple)) and len(tup) == 3,
                        "Each user color must be a 3-tuple/list (R,G,B).")

        if circle_sizes is None:
            circle_sizes = DEFAULT_CIRCLE_SIZES
        ensure_bool(isinstance(circle_sizes, list) and len(circle_sizes) >= 1,
                    "circle_sizes must be a non-empty list of diameters (ints).")

        ensure_bool(isinstance(output_size, (list, tuple)) and len(output_size) == 2,
                    "output_size must be (width, height).")

        # Load & resize image
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        ensure_bool(img is not None, f"Image loading failed for path: {img_path}")
        width, height = int(output_size[0]), int(output_size[1])
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]
        print("[OK] Image loaded & resized.")

        # K-means quantization
        announce("KMEANS_CLUSTERING", {"num_regions": num_regions, "pixels": h * w})
        pixels = img.reshape(-1, 3).astype(np.float32)
        try:
            kmeans = KMeans(n_clusters=num_regions, n_init=KMEANS_N_INIT, random_state=KMEANS_RANDOM_STATE)
            labels = kmeans.fit_predict(pixels)
            centers = kmeans.cluster_centers_.astype(np.uint8)  # BGR
        except Exception as e:
            return {"error": f"K-means failed: {str(e)}"}
        ensure_bool(len(np.unique(labels)) == num_regions,
                    f"K-means did not produce {num_regions} distinct clusters.")
        label_img = labels.reshape(h, w)
        print("[OK] K-means produced expected clusters.")

        # Map clusters to user colors (nearest in Euclidean BGR space)
        announce("MAP_CLUSTERS_TO_USER_COLORS", {"method": "minimize total Euclidean distance"})
        user_colors_bgr = [(c[2], c[1], c[0]) for c in user_colors]
        mapping = map_clusters_to_user_colors(centers, user_colors_bgr)
        ensure_bool(len(mapping) == num_regions and len(set(mapping)) == num_regions,
                    "Failed to map clusters to user colors uniquely.")
        print("[OK] Clusters mapped to user colors.")

        # Build region masks in the original input order
        region_masks: List[np.ndarray] = []
        for user_idx in range(num_regions):
            cluster_idx = mapping[user_idx]
            mask = (label_img == cluster_idx).astype(np.uint8) * 255
            region_masks.append(mask)

        # Coverage check
        combined = np.clip(sum((m > 0).astype(np.uint8) for m in region_masks), 0, 1)
        coverage = combined.mean()
        ensure_bool(coverage > COVERAGE_THRESHOLD,
                    f"Region masks do not sufficiently cover the image (coverage={coverage:.3f}).")
        print("[OK] Region masks built with adequate coverage.")

        # Edge / contour detection on grayscale
        announce("EDGE_DETECTION", {"method": "Canny", "lower_mult": CANNY_LOWER_MULT, "upper_mult": CANNY_UPPER_MULT})
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        v = np.median(gray)
        lower = int(max(0, CANNY_LOWER_MULT * v))
        upper = int(min(255, CANNY_UPPER_MULT * v))
        edges = cv2.Canny(gray, lower, upper)
        print("[OK] Edge map computed.")

        # Circle packing per region (multi-diameter set)
        diameters = circle_sizes[:]  # allowed set for all regions
        all_regions_output: List[Dict[str, Any]] = []
        packed_visual = img.copy()

        for idx, (mask, rgb_color) in enumerate(zip(region_masks, user_colors)):
            announce("PACK_REGION", {"region_index": idx, "allowed_diameters": diameters})
            region_edges = cv2.bitwise_and(edges, edges, mask=mask)
           #  original packing code
           #  region_circles = pack_region_with_circles(mask, region_edges, diameters)

            region_circles = pack_region_with_circles_dt(mask, diameters)


            # Validate placements: inside bounds and within mask
            valid = []
            for c in region_circles:
                (x, y), r = c['center'], c['radius']
                if within_bounds((x, y), w, h, r=r) and circle_fits(mask, (x, y), r):
                    valid.append(c)
            if len(valid) != len(region_circles):
                print(f"[WARN] Removed {len(region_circles) - len(valid)} invalid circles after validation.")
            region_circles = valid

            # Visualization: FILLED circles + subtle outline
            bgr = (int(rgb_color[2]), int(rgb_color[1]), int(rgb_color[0]))
            outline = tuple(int(v * DRAW_OUTLINE_DARKEN) for v in bgr)
            for c in region_circles:
                center_xy = (c['center'][0], c['center'][1])
                r = c['radius']
                cv2.circle(packed_visual, center_xy, r, bgr, thickness=DRAW_FILLED_THICKNESS, lineType=DRAW_LINE_TYPE)
                cv2.circle(packed_visual, center_xy, r, outline, thickness=DRAW_OUTLINE_THICKNESS, lineType=DRAW_LINE_TYPE)

            # Summaries
            counts: Dict[int, int] = {}
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

        print("[NOTE] Circle packing is NP-hard; alternative diameter sets or local refinements may yield denser packings.")

        # Save visualization
        os.makedirs(visualization_outdir, exist_ok=True)
        vis_path = os.path.join(visualization_outdir, f"packing_{uuid.uuid4().hex}.png")
        announce("SAVE_VISUALIZATION", {"path": vis_path, "size": (w, h)})
        ok = cv2.imwrite(vis_path, packed_visual)
        ensure_bool(ok, "Failed to save visualization image.")
        print("[OK] Visualization saved.")

        # Final structure validation
        announce("VALIDATE_OUTPUT_SCHEMA", {"regions": num_regions})
        ensure_bool(isinstance(all_regions_output, list) and len(all_regions_output) == num_regions,
                    f"regions must be a list of {num_regions} dicts.")
        for reg in all_regions_output:
            ensure_bool(set(reg.keys()) == {"color", "circles", "circle_size_counts"},
                        "Region dict keys mismatch.")
            ensure_bool(isinstance(reg["color"], tuple) and len(reg["color"]) == 3,
                        "Region color must be an (R,G,B) tuple.")
            for c in reg["circles"]:
                ensure_bool(set(c.keys()) == {"center", "radius", "color"}, "Circle dict keys mismatch.")
                ensure_bool(isinstance(c["center"], tuple) and len(c["center"]) == 2,
                            "Circle center must be (x,y).")
                ensure_bool(isinstance(c["radius"], int) and c["radius"] > 0,
                            "Circle radius must be positive int.")
                ensure_bool(isinstance(c["color"], tuple) and len(c["color"]) == 3,
                            "Circle color must be an (R,G,B) tuple.")
            for rc in reg["circle_size_counts"]:
                ensure_bool(isinstance(rc, tuple) and len(rc) == 2,
                            "circle_size_counts entries must be (radius, count).")
        print("[OK] Output schema validated.")

        return {
            "regions": all_regions_output,
            "visualization": vis_path,
            "visualization_type": "file",
            "image_size": (int(w), int(h))
        }

    except Exception as e:
        # Any caught error returns the mandated single-key structure
        return {"error": str(e)}


# =========================
# CLI
# =========================
def _normalize_rgb_list(seq):
    """Accept [r,g,b] or '(r, g, b)' and coerce to tuples of ints."""
    out = []
    for c in seq or []:
        if isinstance(c, str):
            s = c.strip().strip("()[]")
            parts = [p.strip() for p in s.split(",")]
            if len(parts) != 3:
                raise ValueError(f"Bad color: {c}")
            out.append(tuple(int(p) for p in parts))
        elif isinstance(c, (list, tuple)) and len(c) == 3:
            out.append(tuple(int(v) for v in c))
        else:
            raise ValueError(f"Bad color: {c}")
    return out

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
    user_colors = _normalize_rgb_list(cfg.get("user_colors"))
    circle_sizes = cfg.get("circle_sizes", None)
    output_size_raw = cfg.get("output_size", list(DEFAULT_OUTPUT_SIZE))
    output_size = (int(output_size_raw[0]), int(output_size_raw[1]))
    visualization_outdir = cfg.get("visualization_outdir", "./circle_packing_outputs")

    result = pack_circles_from_image(
        img_path=img_path,
        user_colors=user_colors,
        circle_sizes=[int(x) for x in circle_sizes] if circle_sizes is not None else None,
        output_size=output_size,
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
