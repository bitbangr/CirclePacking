"""
Circle Packing and Images — RGB-guided segmentation + discrete-diameter circle packing.

Usage (example):
result = pack_circles_from_image(
    img_path="path/to/photo.jpg",
    user_colors=[(200,20,20),(20,200,20),(20,20,200),(200,200,20),(200,20,200),(20,200,200)],
    circle_sizes=[10, 30, 20, 50, 100, 150],      # diameters (px); optional
    output_size=(1000, 1000)                      # (width, height); optional
)

The function returns a dictionary with the exact required schema, or {'error': '...'} on failure.

Mi-Tiente  Colours
user_colors = [
    # (89, 69, 57),     # Tobacco 501
    (178, 113, 14),   # Havana clear 502
    #(93, 64, 64),     # Winelees 503
    #(229, 14, 99),    # Bright red 505
    (237, 48, 35),    # Poppy red 506
    (160, 32, 140),   # Violet 507
    (253, 184, 19),   # Cadmium yellow deep 553
    (20, 144, 89),    # Viridian 575
    #(0, 105, 170),    # Ultramarine 590
    (7, 145, 176)     # Turquoise blue 595
]
"""
from __future__ import annotations
import cv2
import numpy as np
from sklearn.cluster import KMeans
from itertools import permutations
from typing import List, Tuple, Dict, Any, Optional
import math
import os
import uuid

# Global knob requested by user
reasoning_effort = "medium"

# -------------------------------
# Utility: minimal "announce" helper to state purpose + minimal inputs before significant calls (per instruction #3)
def announce(step: str, inputs: Dict[str, Any]):
    print(f"[STEP] {step} | inputs: " + ", ".join(f"{k}={v}" for k, v in inputs.items()))

# -------------------------------
# Validation helpers
def ensure_bool(cond: bool, msg: str):
    if not cond:
        raise RuntimeError(msg)

def within_bounds(pt: Tuple[int, int], w: int, h: int, r: int=0) -> bool:
    x, y = pt
    return (r <= x < (w - r)) and (r <= y < (h - r))

def circle_fits(mask: np.ndarray, center: Tuple[int,int], radius: int) -> bool:
    """Check that a circle of radius lies entirely within a binary region mask."""
    x0, y0 = center
    h, w = mask.shape
    if not within_bounds((x0, y0), w, h, r=radius):
        return False
    # Sample points on circle and a few interior rings to ensure containment
    angles = np.linspace(0, 2*np.pi, num=36, endpoint=False)
    rings = [radius, int(radius*0.7), int(radius*0.4)]
    for rr in rings:
        xs = (x0 + (rr*np.cos(angles))).astype(int)
        ys = (y0 + (rr*np.sin(angles))).astype(int)
        if np.any(mask[ys, xs] == 0):
            return False
    return True

def no_overlap(center: Tuple[int,int], radius: int, placed: List[Tuple[Tuple[int,int], int]]) -> bool:
    cx, cy = center
    for (px, py), pr in placed:
        # Strictly prohibit touching to avoid rendering artifacts: dist > r1 + r2
        if (cx - px)**2 + (cy - py)**2 <= (radius + pr)**2:
            return False
    return True

# -------------------------------
# Core: cluster mapping to user colors
def map_clusters_to_user_colors(cluster_centers_bgr: np.ndarray, user_colors_bgr: List[Tuple[int,int,int]]) -> List[int]:
    """
    Returns a permutation mapping: user_index -> assigned_cluster_index,
    minimizing total Euclidean distance in BGR space.
    """
    # Build distance matrix D[user_i, cluster_j]
    D = np.zeros((6, 6), dtype=float)
    for i, uc in enumerate(user_colors_bgr):
        uc_arr = np.array(uc, dtype=float)
        for j, cc in enumerate(cluster_centers_bgr):
            D[i, j] = np.linalg.norm(uc_arr - cc.astype(float))
    # Solve exact assignment by brute force (6! = 720)
    best_perm = None
    best_cost = float("inf")
    for perm in permutations(range(6)):
        cost = sum(D[i, perm[i]] for i in range(6))
        if cost < best_cost:
            best_cost = cost
            best_perm = perm
    return list(best_perm)

# -------------------------------
# Candidate point generation guided by edges and structure
def candidate_points_for_region(mask: np.ndarray, edge_map: np.ndarray, max_samples: int = 5000) -> np.ndarray:
    """
    Return candidate (x,y) points prioritized near edges and spread across region.
    - 60%: points near edges (dilated Canny)
    - 40%: blue-noise-like grid over the region interior
    """
    h, w = mask.shape

    # Edge-prioritized candidates
    kernel = np.ones((3,3), np.uint8)
    edge_dil = cv2.dilate(edge_map, kernel, iterations=1)
    edge_zone = (edge_dil > 0) & (mask > 0)
    edge_pts = np.column_stack(np.where(edge_zone))
    # Convert to (x,y)
    edge_pts = edge_pts[:, [1, 0]]
    if edge_pts.shape[0] > 0 and edge_pts.shape[0] > int(0.6 * max_samples):
        sel = np.random.choice(edge_pts.shape[0], size=int(0.6 * max_samples), replace=False)
        edge_pts = edge_pts[sel]
    # Grid candidates
    stride = max(4, int(round(math.sqrt((w*h)/max(1, int(0.4*max_samples))))))  # coarse grid
    grid_y, grid_x = np.mgrid[0:h:stride, 0:w:stride]
    grid = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    grid = grid[(mask[grid[:,1], grid[:,0]] > 0)]
    if grid.shape[0] > int(0.5 * max_samples):
        sel = np.random.choice(grid.shape[0], size=int(0.5 * max_samples), replace=False)
        grid = grid[sel]

    # Region centroid (acts as high-priority seed)
    ys, xs = np.where(mask > 0)
    centroid = np.array([[int(xs.mean()), int(ys.mean())]]) if xs.size else np.empty((0,2), dtype=int)

    pts = np.vstack([centroid, edge_pts, grid]) if edge_pts.size else np.vstack([centroid, grid])
    # Shuffle to mix sources a bit
    if pts.shape[0] > 1:
        idx = np.arange(pts.shape[0]); np.random.shuffle(idx); pts = pts[idx]
    return pts

# -------------------------------
# Packing heuristic (greedy, largest→smallest)
def pack_region_with_circles(mask: np.ndarray, edge_map: np.ndarray, diameters: List[int]) -> List[Dict[str, Any]]:
    """
    Greedy multi-diameter packing:
      1) Generate candidate points prioritized by edges & centroid.
      2) For each diameter (largest→smallest), attempt to place circles if:
         - Entirely inside mask
         - No overlap with previously placed circles
    Returns list of circle dicts: {'center': (x,y), 'radius': r}
    """
    circles = []
    placed = []  # list of ((x,y), radius)
    candidates = candidate_points_for_region(mask, edge_map, max_samples=6000)
    # Precompute integral image for fast area checks? (Not strictly needed with sampling in circle_fits)

    for d in sorted(set(diameters), reverse=True):
        r = max(1, int(round(d/2)))
        tried = 0
        for (x, y) in candidates:
            tried += 1
            if circle_fits(mask, (x, y), r) and no_overlap((x, y), r, placed):
                placed.append(((x, y), r))
                circles.append({'center': (int(x), int(y)), 'radius': int(r)})
        # Light validation summary per diameter tier (prints only)
        print(f"[PACK] diameter={d} tried={tried} placed={sum(1 for c in circles if c['radius']==r)}")
    return circles

# -------------------------------
# Main pipeline
def pack_circles_from_image(
    img_path: str,
    user_colors: List[Tuple[int,int,int]],
    circle_sizes: Optional[List[int]] = None,
    output_size: Tuple[int,int] = (1000, 1000),
    visualization_outdir: str = "./circle_packing_outputs"
) -> Dict[str, Any]:
    """
    - user_colors: list of six (R,G,B) tuples
    - circle_sizes: list of six diameters (px). If omitted, defaults are used.
      NOTE: The six sizes are treated as the discrete set of allowed diameters for *all* regions
            (multi-diameter packing as requested).
    - output_size: (width, height) for processing & visualization
    Returns: dictionary per required schema, or {'error': '...'} on failure.
    """
    try:
        announce("LOAD_IMAGE", {"img_path": img_path, "output_size": output_size})
        # Input validation
        ensure_bool(isinstance(user_colors, list) and len(user_colors) == 6, "Exactly six user RGB colors are required.")
        for tup in user_colors:
            ensure_bool(isinstance(tup, (list, tuple)) and len(tup) == 3, "Each user color must be a 3-tuple (R,G,B).")
        if circle_sizes is None:
            circle_sizes = [10, 30, 20, 50, 100, 150]
        ensure_bool(isinstance(circle_sizes, list) and len(circle_sizes) == 6, "circle_sizes must be a list of six diameters.")
        ensure_bool(isinstance(output_size, (list, tuple)) and len(output_size) == 2, "output_size must be (width, height).")

        # Load image
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        ensure_bool(img is not None, f"Image loading failed for path: {img_path}")
        # Resize (OpenCV uses (width,height) when given as tuple for resize? It's (width,height) indeed)
        width, height = int(output_size[0]), int(output_size[1])
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]
        print("[OK] Image loaded & resized.")

        # K-means quantization
        announce("KMEANS_CLUSTERING", {"k": 6, "pixels": h*w})
        pixels = img.reshape(-1, 3).astype(np.float32)
        try:
            kmeans = KMeans(n_clusters=6, n_init=10, random_state=42)
            labels = kmeans.fit_predict(pixels)
            centers = kmeans.cluster_centers_.astype(np.uint8)  # in BGR
        except Exception as e:
            return {"error": f"K-means failed: {str(e)}"}
        ensure_bool(len(np.unique(labels)) == 6, "K-means did not produce 6 distinct clusters.")
        label_img = labels.reshape(h, w)
        print("[OK] K-means produced 6 clusters.")

        # Map clusters to user colors (nearest in Euclidean RGB/BGR space)
        announce("MAP_CLUSTERS_TO_USER_COLORS", {"method": "minimize total Euclidean distance"})
        # Convert user colors from RGB to BGR for OpenCV consistency
        user_colors_bgr = [(c[2], c[1], c[0]) for c in user_colors]
        mapping = map_clusters_to_user_colors(centers, user_colors_bgr)
        ensure_bool(len(mapping) == 6 and len(set(mapping)) == 6, "Failed to map clusters to user colors uniquely.")
        print("[OK] Clusters mapped to user colors.")

        # Build region masks in the *original input order*
        region_masks: List[np.ndarray] = []
        for user_idx in range(6):
            cluster_idx = mapping[user_idx]
            mask = (label_img == cluster_idx).astype(np.uint8) * 255
            region_masks.append(mask)

        # Quick validation: ensure masks cover the frame (allow tiny holes)
        combined = np.clip(sum((m > 0).astype(np.uint8) for m in region_masks), 0, 1)
        coverage = combined.mean()
        ensure_bool(coverage > 0.95, "Region masks do not sufficiently cover the image.")
        print("[OK] Region masks built with adequate coverage.")

        # Edge / contour detection on grayscale
        announce("EDGE_DETECTION", {"method": "Canny", "note": "used to prioritize placements"})
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        v = np.median(gray)
        lower = int(max(0, 0.66 * v))
        upper = int(min(255, 1.33 * v))
        edges = cv2.Canny(gray, lower, upper)
        print("[OK] Edge map computed.")

        # Circle packing per region (multi-diameter: same allowed set for each region)
        diameters = circle_sizes[:]  # treat as allowed set for all regions
        all_regions_output: List[Dict[str, Any]] = []
        packed_visual = img.copy()

        for idx, (mask, rgb_color) in enumerate(zip(region_masks, user_colors)):
            announce("PACK_REGION", {"region_index": idx, "allowed_diameters": diameters})
            region_edges = cv2.bitwise_and(edges, edges, mask=mask)
            region_circles = pack_region_with_circles(mask, region_edges, diameters)

            # Validate placements: inside bounds and within mask
            valid = []
            for c in region_circles:
                (x, y), r = c['center'], c['radius']
                if within_bounds((x, y), w, h, r=r) and circle_fits(mask, (x, y), r):
                    valid.append(c)
            if len(valid) != len(region_circles):
                print(f"[WARN] Removed {len(region_circles) - len(valid)} invalid circles after validation.")
            region_circles = valid

            # Summarize counts by radius
            counts = {}
            for c in region_circles:
                counts[c['radius']] = counts.get(c['radius'], 0) + 1
                # Draw on visualization (BGR expected)
                bgr = (int(rgb_color[2]), int(rgb_color[1]), int(rgb_color[0]))
                #cv2.circle(packed_visual, (c['center'][0], c['center'][1]), c['radius'], bgr, thickness=2)
                # Draw a filled disc in the region color, then a thin darker outline for legibility
                cv2.circle(
                    packed_visual,
                    (c['center'][0], c['center'][1]),
                    c['radius'],
                    bgr,
                    thickness=-1,                 # <-- filled
                    lineType=cv2.LINE_AA
                )
                outline = tuple(int(v * 0.7) for v in bgr)  # slightly darker edge
                cv2.circle(
                    packed_visual,
                    (c['center'][0], c['center'][1]),
                    c['radius'],
                    outline,
                    thickness=2,
                    lineType=cv2.LINE_AA
                )


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

        # Inform about possible better packings (NP-hard note)
        print("[NOTE] Circle packing is NP-hard; alternative diameter sets or annealing/metaheuristics may yield denser packings.")

        # Save visualization
        os.makedirs(visualization_outdir, exist_ok=True)
        vis_path = os.path.join(visualization_outdir, f"packing_{uuid.uuid4().hex}.png")
        announce("SAVE_VISUALIZATION", {"path": vis_path, "size": (w, h)})
        ok = cv2.imwrite(vis_path, packed_visual)
        ensure_bool(ok, "Failed to save visualization image.")
        print("[OK] Visualization saved.")

        # Final structure validation
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

        result = {
            "regions": all_regions_output,
            "visualization": vis_path,
            "visualization_type": "file",
            "image_size": (int(w), int(h))
        }
        return result

    except Exception as e:
        # Any caught error returns the mandated single-key structure
        return {"error": str(e)}

# -------------------------------
# Efficiency notes (not relaxing constraints, just transparency):
# - Greedy largest→smallest placement with edge-prioritized candidates gives reasonable fill
#   while keeping runtime practical. For denser packing, consider:
#   * Adaptive Poisson-disk sampling per diameter tier
#   * Local relocation / hill-climbing of recently placed circles
#   * Simulated annealing over centers for the largest 10–50 circles per region
#   * Alternate diameter sets: try small geometric progressions (e.g., [128, 90, 64, 45, 32, 22])
#   These may increase packing density but at higher compute cost.

if __name__ == "__main__":
    result = pack_circles_from_image(
        img_path="./4x4_Kroma_16.png",  # 👈 Replace with your JPG path
        #user_colors=[(200,20,20),(20,200,20),(20,20,200),(200,200,20),(200,20,200),(20,200,200)]
        user_colors = [
    # (89, 69, 57),     # Tobacco 501
    (178, 113, 14),   # Havana clear 502
    #(93, 64, 64),     # Winelees 503
    #(229, 14, 99),    # Bright red 505
    (237, 48, 35),    # Poppy red 506
    (160, 32, 140),   # Violet 507
    (253, 184, 19),   # Cadmium yellow deep 553
    (20, 144, 89),    # Viridian 575
    #(0, 105, 170),    # Ultramarine 590
    (7, 145, 176)     # Turquoise blue 595
]
    )
    print(result)
