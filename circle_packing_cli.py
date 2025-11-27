#!/usr/bin/env python3

""" 
circle_packing_cli.py 

CLI tools for generating circle-packing layouts from an input image.

This module loads an image, computes a circle-packing solution within a
physical board size, exports diagnostic PNGs, writes a CSV layout, and
optionally creates an SVG accurate to millimeter scaling.

Typical usage:
    $ python3 circle_packing_cli.py input.jpg --config config.json

The public entry point is :func:`main`.

Generates a circle-packed representation and a build CSV from an input image,
respecting a physical board size (mm), circle size set (mm), and color palette.

Outputs per run:
  - packing_<ID>.png                (overlay: original/cropped image + circles)
  - packing_<ID>_circles_only.png   (circles drawn on black)
  - packing_<ID>_layout.csv         (grid_cell, position_in_mm, diameter_in_mm, color_rgb, color_name)
and prints a JSON summary to stdout (optionally pretty).
"""

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
# Preprocessing: mosaic labeling
# =========================
def mosaic_labels_by_user_colors(img_bgr: np.ndarray,
                                 user_colors_bgr: List[Tuple[int,int,int]],
                                 tile: int) -> np.ndarray:
    """
    Divide image into tile x tile cells.
    For each cell, compute mean BGR and assign the cell to the nearest user color.
    Return a label image (H x W) with values in [0, num_regions-1].
    """
    h, w = img_bgr.shape[:2]
    num_regions = len(user_colors_bgr)
    labels = np.empty((h, w), dtype=np.int32)

    # Precomputed array for fast distance calculation
    U = np.array(user_colors_bgr, dtype=np.float32)  # (num_regions, 3)

    for y0 in range(0, h, tile):
        y1 = min(y0 + tile, h)
        for x0 in range(0, w, tile):
            x1 = min(x0 + tile, w)
            cell = img_bgr[y0:y1, x0:x1]
            mean_bgr = cell.reshape(-1, 3).mean(axis=0).astype(np.float32)  # (3,)
            # nearest user color (Euclidean in BGR)
            diffs = U - mean_bgr[None, :]
            dists = np.sqrt((diffs ** 2).sum(axis=1))
            idx = int(np.argmin(dists))
            labels[y0:y1, x0:x1] = idx
    return labels


# =========================
# Utility helpers
# =========================
def announce(step: str, inputs: Dict[str, Any]):
    """
    Log a structured event to stdout for debugging and automation.
    Per requirements: state purpose & minimal inputs before significant calls.
    """
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
# Helpers for mm/grid/CSV
# =========================
DEFAULT_GRID_LABEL_BASE = "A"
CSV_FIELDS = ["grid_cell","position_in_mm","diameter_in_mm","color_rgb","color_name"]
ROUND_MODES = {"nearest","floor","ceil"}

def center_crop_to_aspect(img, target_w_mm: float, target_h_mm: float):
    h, w = img.shape[:2]
    tgt = float(target_w_mm) / float(target_h_mm)
    cur = float(w) / float(h)
    if abs(cur - tgt) < 1e-12:
        return img
    if cur > tgt:
        new_w = int(round(h * tgt))
        x0 = (w - new_w) // 2
        return img[:, x0:x0+new_w]
    new_h = int(round(w / tgt))
    y0 = (h - new_h) // 2
    return img[y0:y0+new_h, :]

def round_mm(value_mm: float, mode: str, step_mm: float) -> int:
    import math
    if step_mm <= 0: raise ValueError("mm_rounding.step_mm must be > 0")
    q = value_mm / step_mm
    if mode == "nearest": r = round(q)
    elif mode == "floor": r = math.floor(q)
    elif mode == "ceil": r = math.ceil(q)
    else: raise ValueError(f"Unsupported rounding mode: {mode}")
    return int(r * step_mm)

def grid_cell_for(x_px: int, y_px: int, img_w: int, img_h: int, grid_n: int, epsilon: float,
                  base_letter: str = DEFAULT_GRID_LABEL_BASE) -> str:
    import math
    cell_w = img_w / float(grid_n)
    cell_h = img_h / float(grid_n)
    col = int(math.floor((x_px - epsilon) / cell_w))
    row = int(math.floor((y_px - epsilon) / cell_h))
    col = max(0, min(grid_n - 1, col))
    row = max(0, min(grid_n - 1, row))
    return chr(ord(base_letter) + col) + str(row + 1)

def write_layout_csv(
    csv_path: str,
    regions: list[dict],
    img_w: int,
    img_h: int,
    user_colors: list[tuple[int,int,int]],
    color_names: list[str] | None,
    board_w_mm: float,
    board_h_mm: float,
    grid_n: int,
    boundary_epsilon: float,
    mm_round_mode: str,
    mm_round_step: float,
) -> None:
    """
    Write a CSV containing the physical circle layout in millimeters.

    The CSV contains one row per circle and includes the grid cell index,
    millimeter coordinates, diameter, RGB color, and optionally a color name.

    Args:
        csv_path: Output CSV file path.
        regions: Circle packing results in pixel units.
        img_w: Image width in pixels.
        img_h: Image height in pixels.
        user_colors: RGB tuples for all palette entries.
        color_names: Optional list of color names.
        board_w_mm: Board width in millimeters.
        board_h_mm: Board height in millimeters.
        grid_n: Number of grid divisions.
        boundary_epsilon: Margin used when interpreting border cells.
        mm_round_mode: Rounding mode for coordinates and diameters.
        mm_round_step: Step size for rounding in mm.

    Returns:
        None.
    """
    import csv, os
    if mm_round_mode not in ROUND_MODES:
        raise ValueError(f"mm_rounding.mode must be one of {sorted(ROUND_MODES)}")
    px_to_mm_x = board_w_mm / float(img_w)
    px_to_mm_y = board_h_mm / float(img_h)

    name_map = {}
    if color_names and len(color_names) >= len(user_colors):
        for rgb, nm in zip(user_colors, color_names):
            name_map[tuple(int(v) for v in rgb)] = nm

    rows = []
    for reg in regions:
        rgb = tuple(int(v) for v in reg["color"])
        nm = name_map.get(rgb, "")
        for c in reg["circles"]:
            cx, cy = c["center"]
            d_px = int(c["radius"]) * 2

            x_mm_f = cx * px_to_mm_x
            y_mm_f = cy * px_to_mm_y
            d_mm_f = d_px * px_to_mm_x  # square pixels → x-scale ok

            x_mm = round_mm(x_mm_f, mm_round_mode, mm_round_step)
            y_mm = round_mm(y_mm_f, mm_round_mode, mm_round_step)
            d_mm = round_mm(d_mm_f, mm_round_mode, mm_round_step)

            cell = grid_cell_for(cx, cy, img_w, img_h, grid_n, boundary_epsilon)
            rows.append({
                "grid_cell": cell,
                "position_in_mm": f"[{x_mm}, {y_mm}]",
                "diameter_in_mm": d_mm,
                "color_rgb": f"[{rgb[0]}, {rgb[1]}, {rgb[2]}]",
                "color_name": nm
            })

    def sort_key(r):
        col = ord(r["grid_cell"][0]) - ord(DEFAULT_GRID_LABEL_BASE)
        row = int(r["grid_cell"][1:]) - 1
        return (row, col, -int(r["diameter_in_mm"]))

    rows.sort(key=sort_key)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)

# =========================
# SVG Placement Guide Helpers 
# =========================
import xml.etree.ElementTree as ET

def _sanitize_id(text: str) -> str:
    s = "".join(ch if ch.isalnum() or ch in "-_." else "-" for ch in (text or ""))
    return s.strip("-") or "unnamed"

def _identifier(color_name: str, d_mm: float, seq: int) -> str:
    # two decimals so 19.05 is preserved; replace '.' with '_' in label if you prefer
    return f"{_sanitize_id(color_name)}_{d_mm:.2f}_{seq}"

# =========================
# Write Assembly Aid CSV
# =========================
def write_assembly_aid_csv(
    csv_path: str,
    regions: list[dict],
    img_w: int,
    img_h: int,
    color_rgb_values: list[tuple[int, int, int]],
    color_names: list[str] | None,
    board_w_mm: float,
    board_h_mm: float,
    mm_round_mode: str,
    mm_round_step: float,
) -> None:
    import csv
    px_to_mm_x = board_w_mm / float(img_w)
    px_to_mm_y = board_h_mm / float(img_h)

    # RGB -> color name map
    name_map = {}
    if color_names and len(color_names) >= len(color_rgb_values):
        for rgb, nm in zip(color_rgb_values, color_names):
            name_map[tuple(int(v) for v in rgb)] = nm

    counters: dict[tuple[str, float], int] = {}
    rows = []

    for reg in regions:
        rgb = tuple(int(v) for v in reg["color"])
        cname = name_map.get(rgb, f"rgb-{rgb[0]}-{rgb[1]}-{rgb[2]}")

        for c in reg["circles"]:
            cx_px, cy_px = c["center"]
            d_px = int(c["radius"]) * 2

            cx_mm = round_mm(cx_px * px_to_mm_x, mm_round_mode, mm_round_step)
            cy_mm = round_mm(cy_px * px_to_mm_y, mm_round_mode, mm_round_step)
            d_mm  = round_mm(d_px * px_to_mm_x,  mm_round_mode, mm_round_step)

            key = (cname, float(d_mm))
            counters[key] = counters.get(key, 0) + 1
            ident = _identifier(cname, float(d_mm), counters[key])

            rows.append({
                "Color": cname,
                "Diameter": float(d_mm),
                "CenterX": float(cx_mm),
                "CenterY": float(cy_mm),
                "Identifier": ident,
            })

    # stable order: by color, then diameter desc, then X,Y
    rows.sort(key=lambda r: (r["Color"], -r["Diameter"], r["CenterY"], r["CenterX"]))

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["Color", "Diameter", "CenterX", "CenterY", "Identifier"])
        w.writeheader()
        w.writerows(rows)

# =========================
# Write Assembly Aid SVG
# =========================
def write_assembly_aid_svg(
    svg_path: str,
    regions: list[dict],
    img_w: int,
    img_h: int,
    color_rgb_values: list[tuple[int,int,int]],
    color_names: list[str] | None,
    board_w_mm: float,
    board_h_mm: float,
    mm_round_mode: str,
    mm_round_step: float,
    arc_radius_mm: float = 2.0,
    cross_stroke_mm: float = 0.6,
    label_font_mm: float = 3.0,
    label_offset_mm: float = 2.0,
    make_black_background: bool = True,
) -> None:
    px_to_mm_x = board_w_mm / float(img_w)
    px_to_mm_y = board_h_mm / float(img_h)

    # RGB -> name
    name_map = {}
    if color_names and len(color_names) >= len(color_rgb_values):
        for rgb, nm in zip(color_rgb_values, color_names):
            name_map[tuple(int(v) for v in rgb)] = nm

    # counters for unique IDs
    counters: dict[tuple[str, float], int] = {}

    svg = ET.Element(
        "svg",
        xmlns="http://www.w3.org/2000/svg",
        version="1.1",
        width=f"{board_w_mm}mm",
        height=f"{board_h_mm}mm",
        viewBox=f"0 0 {board_w_mm} {board_h_mm}",
    )

    if make_black_background:
        ET.SubElement(svg, "rect", x="0", y="0",
                      width=str(board_w_mm), height=str(board_h_mm),
                      fill="black")

    def arc_path(cx, cy, dx, dy, r):
        # small arc centered at (cx+dx, cy+dy) oriented outward
        # draw as 180-degree small arc (symmetric end cap)
        x0 = cx + dx - r
        y0 = cy + dy
        x1 = cx + dx + r
        y1 = cy + dy
        # sweep-flag flips visually the same for semicircle; keep 0
        return f"M {x0:.3f} {y0:.3f} A {r:.3f} {r:.3f} 0 0 1 {x1:.3f} {y1:.3f}"

    # per-color layers
    for reg in regions:
        rgb = tuple(int(v) for v in reg["color"])
        cname = name_map.get(rgb, f"rgb-{rgb[0]}-{rgb[1]}-{rgb[2]}")
        color_str = f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"

        layer_cross = ET.SubElement(
            svg, "g",
            **{
                "id": f"layer-crosses-{_sanitize_id(cname)}",
                "inkscape:groupmode": "layer",
                "inkscape:label": f"Crosses – {cname}",
                "fill": "none",
                "stroke": color_str,
                "stroke-width": str(cross_stroke_mm),
                "stroke-linecap": "round"
            }
        )
        layer_labels = ET.SubElement(
            svg, "g",
            **{
                "id": f"layer-labels-{_sanitize_id(cname)}",
                "inkscape:groupmode": "layer",
                "inkscape:label": f"Labels – {cname}",
                "fill": color_str
            }
        )

        for c in reg["circles"]:
            cx_px, cy_px = c["center"]
            d_px = int(c["radius"]) * 2

            cx_mm = round_mm(cx_px * px_to_mm_x, mm_round_mode, mm_round_step)
            cy_mm = round_mm(cy_px * px_to_mm_y, mm_round_mode, mm_round_step)
            d_mm  = round_mm(d_px * px_to_mm_x,  mm_round_mode, mm_round_step)
            r_mm  = d_mm / 2.0

            # unique identifier per (color, diameter)
            key = (cname, float(d_mm))
            counters[key] = counters.get(key, 0) + 1
            ident = _identifier(cname, float(d_mm), counters[key])

            # cross lines (full diameter)
            # horizontal
            ET.SubElement(layer_cross, "line",
                          x1=str(cx_mm - r_mm), y1=str(cy_mm),
                          x2=str(cx_mm + r_mm), y2=str(cy_mm))
            # vertical
            ET.SubElement(layer_cross, "line",
                          x1=str(cx_mm), y1=str(cy_mm - r_mm),
                          x2=str(cx_mm), y2=str(cy_mm + r_mm))

            # # arc tips at each end (four semicircles pointing outward)
            # ar = min(arc_radius_mm, max(0.5, r_mm*0.08))  # cap at ~8% of radius, min 0.5mm
            # # right, left, top, bottom
            # ET.SubElement(layer_cross, "path", d=arc_path(cx_mm, cy_mm, +r_mm, 0.0, ar))
            # ET.SubElement(layer_cross, "path", d=arc_path(cx_mm, cy_mm, -r_mm, 0.0, ar))
            # ET.SubElement(layer_cross, "path", d=arc_path(cx_mm, cy_mm, 0.0, -r_mm, ar))
            # ET.SubElement(layer_cross, "path", d=arc_path(cx_mm, cy_mm, 0.0, +r_mm, ar))

            # --- Tick marks at cross-arm ends (short perpendicular lines) ---
            tick_len = min(arc_radius_mm, max(0.5, r_mm * 0.08))  # reuse same scale
            # right end
            ET.SubElement(layer_cross, "line",
                        x1=str(cx_mm + r_mm), y1=str(cy_mm - tick_len),
                        x2=str(cx_mm + r_mm), y2=str(cy_mm + tick_len))
            # left end
            ET.SubElement(layer_cross, "line",
                        x1=str(cx_mm - r_mm), y1=str(cy_mm - tick_len),
                        x2=str(cx_mm - r_mm), y2=str(cy_mm + tick_len))
            # top end
            ET.SubElement(layer_cross, "line",
                        x1=str(cx_mm - tick_len), y1=str(cy_mm - r_mm),
                        x2=str(cx_mm + tick_len), y2=str(cy_mm - r_mm))
            # bottom end
            ET.SubElement(layer_cross, "line",
                        x1=str(cx_mm - tick_len), y1=str(cy_mm + r_mm),
                        x2=str(cx_mm + tick_len), y2=str(cy_mm + r_mm))


            # label (own layer), centered then nudged down a bit
            ET.SubElement(layer_labels, "text",
                          x=str(cx_mm), y=str(cy_mm + label_offset_mm),
                          **{
                              "font-size": f"{label_font_mm}mm",
                              "text-anchor": "middle",
                              "dominant-baseline": "middle"
                          }).text = ident

    ET.ElementTree(svg).write(svg_path, encoding="utf-8", xml_declaration=True)



# =========================
# SVG Export (mm-accurate)
# =========================
import xml.etree.ElementTree as ET

def _sanitize_id(text: str) -> str:
    s = "".join(ch if ch.isalnum() or ch in "-_." else "-" for ch in (text or ""))
    return s.strip("-") or "unnamed"

def write_layout_svg(
    svg_path: str,
    regions: list[dict],
    img_w: int,
    img_h: int,
    color_rgb_values: list[tuple[int,int,int]],
    color_names: list[str] | None,
    board_w_mm: float,
    board_h_mm: float,
    grid_n: int,
    boundary_epsilon: float,   # unused in SVG; kept for parity with CSV
    mm_round_mode: str,
    mm_round_step: float,
    include_grid: bool = True,
    transparent_bg: bool = False,
    circle_stroke_mm: float = 0.2,
) -> None:
    """
    Write an SVG layout file representing the computed circles in millimeter units.

    Each region entry should contain pixel-space circle geometry generated by
    the packing algorithm. This function converts all pixel values into
    millimeters and produces a scale-accurate SVG layout.

    Args:
        svg_path: Output SVG file path.
        regions: Circle regions with pixel coordinates and diameters.
        img_w: Width of the original image in pixels.
        img_h: Height of the original image in pixels.
        color_rgb_values: RGB tuples for each palette entry.
        color_names: Optional human-readable color names aligned with the palette.
        board_w_mm: Physical board width in millimeters.
        board_h_mm: Physical board height in millimeters.
        grid_n: Number of grid subdivisions used during packing.
        boundary_epsilon: Margin applied at borders (kept for parity with CSV).
        mm_round_mode: Rounding mode used for exported millimeter values.
        mm_round_step: Increment used when rounding millimeter values.
        include_grid: Whether to draw grid lines in the output.
        transparent_bg: Whether to omit the background rectangle.
        circle_stroke_mm: Visual stroke width for circles.

    Returns:
        None. Writes an SVG file to ``svg_path``.
    """

    # px -> mm scale
    px_to_mm_x = board_w_mm / float(img_w)
    px_to_mm_y = board_h_mm / float(img_h)

    # Map exact RGB to name if available
    name_map = {}
    if color_names and len(color_names) >= len(color_rgb_values):
        for rgb, nm in zip(color_rgb_values, color_names):
            name_map[tuple(int(v) for v in rgb)] = nm

    # Root <svg> with physical size in mm + matching viewBox
    svg = ET.Element(
        "svg",
        xmlns="http://www.w3.org/2000/svg",
        version="1.1",
        width=f"{board_w_mm}mm",
        height=f"{board_h_mm}mm",
        viewBox=f"0 0 {board_w_mm} {board_h_mm}",
    )

    # Optional background (keep transparent by default)
    if not transparent_bg:
        ET.SubElement(
            svg, "rect",
            x="0", y="0",
            width=str(board_w_mm), height=str(board_h_mm),
            fill="black"
        )

    # Optional grid overlay (A–J, 1–10)
    if include_grid and grid_n > 0:
        grid = ET.SubElement(svg, "g", id="grid", **{"stroke":"#888","stroke-width":"0.2","fill":"none","opacity":"0.35"})
        # verticals
        step_x = board_w_mm / float(grid_n)
        for i in range(grid_n + 1):
            x = i * step_x
            ET.SubElement(grid, "line", x1=str(x), y1="0", x2=str(x), y2=str(board_h_mm))
        # horizontals
        step_y = board_h_mm / float(grid_n)
        for j in range(grid_n + 1):
            y = j * step_y
            ET.SubElement(grid, "line", x1="0", y1=str(y), x2=str(board_w_mm), y2=str(y))
        # labels (small)
        labels = ET.SubElement(svg, "g", id="grid-labels", fill="#666", **{"font-size":"3"})
        for i in range(grid_n):
            x = i * step_x + 1.5
            col = chr(ord('A') + i)
            ET.SubElement(labels, "text", x=str(x), y="3.5").text = col
        for j in range(grid_n):
            y = j * step_y + 4.5
            ET.SubElement(labels, "text", x="1.5", y=str(y)).text = str(j+1)

    # Circles grouped by color
    for reg in regions:
        rgb = tuple(int(v) for v in reg["color"])
        nm = name_map.get(rgb, f"rgb-{rgb[0]}-{rgb[1]}-{rgb[2]}")
        gid = f"color-{_sanitize_id(nm)}"
        grp = ET.SubElement(svg, "g", id=gid, fill=f"rgb({rgb[0]},{rgb[1]},{rgb[2]})", stroke="black", **{"stroke-width":str(circle_stroke_mm), "stroke-opacity":"0.35"})

        # largest-first is nice for overlap ordering (already sorted in your pipeline)
        for c in reg["circles"]:
            cx_px, cy_px = c["center"]
            r_px = int(c["radius"])
            d_mm = (2 * r_px) * px_to_mm_x  # px are square; x-scale is fine

            # Use your rounding policy to get integer/step mm in SVG too
            # (reusing round_mm from your helpers)
            from math import isfinite
            cx_mm = round_mm(cx_px * px_to_mm_x, mm_round_mode, mm_round_step)
            cy_mm = round_mm(cy_px * px_to_mm_y, mm_round_mode, mm_round_step)
            r_mm  = round_mm(d_mm,             mm_round_mode, mm_round_step) / 2.0

            el = ET.SubElement(grp, "circle",
                               cx=str(cx_mm), cy=str(cy_mm), r=str(r_mm))
            # embed a few data-* attrs to mirror CSV
            el.set("data-rgb", f"[{rgb[0]},{rgb[1]},{rgb[2]}]")
            el.set("data-name", nm)
            el.set("data-d_mm", str(int(round(r_mm*2))))

    # Write file
    ET.ElementTree(svg).write(svg_path, encoding="utf-8", xml_declaration=True)

    announce("WRITE_LAYOUT_SVG",{"svg_path":svg_path})


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
    visualization_outdir: str = "./circle_packing_outputs",
    preprocess_cfg: Optional[Dict[str, Any]] = None   # <-- add this line
) -> Dict[str, Any]:
    
    # UNPACK *immediately* — now these names exist in this scope
    out_w_mm, out_h_mm = preprocess_cfg["__output_size_mm__"]
    grid_divs        = int(preprocess_cfg["__grid_divisions__"])
    boundary_epsilon = float(preprocess_cfg["__boundary_epsilon__"])
    mm_round_mode    = str(preprocess_cfg["__mm_round_mode__"])
    mm_round_step    = float(preprocess_cfg["__mm_round_step__"])
    color_names      = preprocess_cfg.get("__color_names__", [])
    export_svg       = bool(preprocess_cfg.get("__export_svg__", True))
    svg_include_grid         = bool(preprocess_cfg.get("__svg_include_grid__", True))
    svg_transparent  = bool(preprocess_cfg.get("__svg_transparent_bg__", True))
    
    try:
        announce("LOAD_IMAGE", {"img_path": img_path, "output_size": output_size})

        # Validate inputs
        ensure_bool(isinstance(user_colors, list) and len(user_colors) >= 1,
                    "At least one user RGB color is required.")
        
        num_regions = len(user_colors)

        for tup in user_colors:
            ensure_bool(isinstance(tup, (list, tuple)) and len(tup) == 3,
                        "Each user color must be a 3-tuple/list (R,G,B).")

        ensure_bool(isinstance(output_size, (list, tuple)) and len(output_size) == 2,
                    "output_size must be (width, height).")

        # Load & resize image
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        ensure_bool(img is not None, f"Image loading failed for path: {img_path}")
        
        # *****************************
        # Respect final physical aspect ratio in mm, then resize to pixel canvas
        img = center_crop_to_aspect(img, out_w_mm, out_h_mm)
        # ******************************
        
        width, height = int(output_size[0]), int(output_size[1])
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]
        print("[OK] Image loaded & resized.")

        # --- Optional mosaic pre-quantization ---
        mode_cfg = preprocess_cfg or {}
        mode = mode_cfg.get("mode", "none")
        tile = int(mode_cfg.get("tile", 16))

        # mgj TODO check this logic.  mosaic was supposed to run AND THEN kmeans still done on this result
        if mode == "mosaic":
            announce("PREPROCESS", {"mode": "mosaic", "tile": tile})
            user_colors_bgr = [(c[2], c[1], c[0]) for c in user_colors]
            label_img = mosaic_labels_by_user_colors(img, user_colors_bgr, tile)
            num_regions = len(user_colors)  # already set earlier
            centers = np.array(user_colors_bgr, dtype=np.uint8)  # “cluster centers” = user colors
            print("[OK] Mosaic labels generated; skipping KMeans.")
        else:
            # (existing KMeans block stays as-is)
            announce("KMEANS_CLUSTERING", {"num_regions": num_regions, "pixels": h * w})
            pixels = img.reshape(-1, 3).astype(np.float32)
            try:
                kmeans = KMeans(n_clusters=num_regions, n_init=KMEANS_N_INIT, random_state=KMEANS_RANDOM_STATE)
                labels = kmeans.fit_predict(pixels)
                centers = kmeans.cluster_centers_.astype(np.uint8)  # BGR
            except Exception as e:
                return {"error": f"K-means failed: {str(e)}"}
            ensure_bool(len(np.unique(labels)) == num_regions, f"K-means did not produce {num_regions} distinct clusters.")
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
        packed_visual = img.copy()                     # existing (photo + circles)
        packed_circles_only = np.zeros((h, w, 4), dtype=np.uint8)  # NEW: BGRA, fully transparent

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
            # existing colors for BGR drawing
            bgr     = (int(rgb_color[2]), int(rgb_color[1]), int(rgb_color[0]))
            outline = tuple(int(v * DRAW_OUTLINE_DARKEN) for v in bgr)

            # NEW: colors with alpha for the circles-only canvas
            bgr_a     = (bgr[0],     bgr[1],     bgr[2],     255)
            outline_a = (outline[0], outline[1], outline[2], 255)

            for c in region_circles:
                center_xy = (c['center'][0], c['center'][1])
                r = c['radius']

                # 1) Draw on top of the original image (as you already do)
                cv2.circle(packed_visual, center_xy, r, bgr,     thickness=DRAW_FILLED_THICKNESS, lineType=DRAW_LINE_TYPE)
                cv2.circle(packed_visual, center_xy, r, outline, thickness=DRAW_OUTLINE_THICKNESS, lineType=DRAW_LINE_TYPE)

                # 2) Draw on the transparent canvas (no photo background)
                cv2.circle(packed_circles_only, center_xy, r, bgr_a,     thickness=DRAW_FILLED_THICKNESS, lineType=DRAW_LINE_TYPE)
                cv2.circle(packed_circles_only, center_xy, r, outline_a, thickness=DRAW_OUTLINE_THICKNESS, lineType=DRAW_LINE_TYPE)

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

        # One session ID for all outputs
        session_id = uuid.uuid4().hex

        # Save visualization
        os.makedirs(visualization_outdir, exist_ok=True)
        vis_path = os.path.join(visualization_outdir, f"packing_{session_id}.png")
        announce("SAVE_VISUALIZATION", {"path": vis_path, "size": (w, h)})
        ok = cv2.imwrite(vis_path, packed_visual)
        ensure_bool(ok, "Failed to save visualization image.")
        print("[OK] Visualization saved.")

        # NEW: circles-only (transparent PNG)
        circles_only_path = os.path.join(visualization_outdir, f"packing_{session_id}_circles_only.png")
        announce("SAVE_VISUALIZATION", {"path": circles_only_path, "type": "circles_only_bgra"})
        ok2 = cv2.imwrite(circles_only_path, packed_circles_only)
        ensure_bool(ok2, "Failed to save circles-only visualization.")
        print(f"[OK] Circles-only visualization saved: {circles_only_path}")

        # --- NEW: Save CSV using your computed circles (no detection) ---
        csv_path = os.path.join(visualization_outdir, f"packing_{session_id}_layout.csv")
        write_layout_csv(
            csv_path=csv_path,
            regions=all_regions_output,             # <-- use your computed circles
            img_w=w, img_h=h,
            user_colors=[(int(c[0]), int(c[1]), int(c[2])) for c in user_colors],
            color_names=color_names,
            board_w_mm=out_w_mm,
            board_h_mm=out_h_mm,
            grid_n=grid_divs,
            boundary_epsilon=boundary_epsilon,
            mm_round_mode=mm_round_mode,
            mm_round_step=mm_round_step,
        )
        print(f"[OK] CSV layout saved: {csv_path}")

        # --- SVG export (mm-accurate) ---
        if export_svg:
            svg_path = os.path.join(visualization_outdir, f"packing_{session_id}_layout.svg")
            write_layout_svg(
                svg_path=svg_path,
                regions=all_regions_output,            # your computed circles
                img_w=w, img_h=h,
                color_rgb_values=user_colors,     # (R,G,B) tuples
                color_names=color_names,
                board_w_mm=float(out_w_mm),
                board_h_mm=float(out_h_mm),
                grid_n=grid_divs,
                boundary_epsilon=boundary_epsilon,
                mm_round_mode=mm_round_mode,
                mm_round_step=mm_round_step,
                include_grid=svg_include_grid,
                transparent_bg=svg_transparent,
            )
            print(f"[OK] SVG layout saved: {svg_path}")


            # --- Assembly-aid CSV ---
            aid_csv_path = os.path.join(visualization_outdir, f"packing_{session_id}_assembly.csv")
            write_assembly_aid_csv(
                csv_path=aid_csv_path,
                regions=all_regions_output,
                img_w=w, img_h=h,
                color_rgb_values=user_colors,
                color_names=color_names,
                board_w_mm=float(out_w_mm), board_h_mm=float(out_h_mm),
                mm_round_mode=mm_round_mode, mm_round_step=mm_round_step,
            )
            print(f"[OK] Assembly CSV saved: {aid_csv_path}")

            # --- Assembly-aid SVG ---
            aid_svg_path = os.path.join(visualization_outdir, f"packing_{session_id}_assembly.svg")
            write_assembly_aid_svg(
                svg_path=aid_svg_path,
                regions=all_regions_output,
                img_w=w, img_h=h,
                color_rgb_values=user_colors,
                color_names=color_names,
                board_w_mm=float(out_w_mm), board_h_mm=float(out_h_mm),
                mm_round_mode=mm_round_mode, mm_round_step=mm_round_step,
                arc_radius_mm=2.0,           # tweak if you like
                cross_stroke_mm=0.6,
                label_font_mm=3.0,
                label_offset_mm=2.0,
                make_black_background=True,  # solid base for print
            )
            print(f"[OK] Assembly SVG saved: {aid_svg_path}")

            # include in result dict
            result_paths = {
                "assembly_csv": aid_csv_path,
                "assembly_svg": aid_svg_path,
            }




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
            "circles_only": circles_only_path,
            "csv_layout": csv_path,
            "image_size": (int(w), int(h)),
            "output_size_mm": preprocess_cfg.get("__output_size_mm__", None),
            "grid_divisions": int(preprocess_cfg.get("__grid_divisions__", 10)),
            "mm_rounding": {
                "mode": str(preprocess_cfg.get("__mm_round_mode__", "nearest")),
                "step_mm": float(preprocess_cfg.get("__mm_round_step__", 1)),
            },
            "session_id": session_id
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
    """
    Entry point for the circle-packing command-line interface.

    Parses command-line arguments, loads the image and configuration file,
    runs preprocessing and the circle-packing algorithm, generates visual
    outputs (PNGs), and writes layout files (CSV and SVG).

    This function prints a JSON summary to stdout that includes paths to
    generated files and metadata about the run.

    Returns:
        None.
    """
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

# ********************************
    # after cfg = yaml.safe_load(...)
    # --- 1. Physical settings (from YAML) ---
    phys = cfg["physical"]
    out_w_mm, out_h_mm = map(float, phys["output_size_mm"])
    min_d_mm = float(phys.get("min_circle_diameter_mm", 13))
    grid_divs = int(phys.get("grid_divisions", 10))
    boundary_epsilon = float(phys.get("boundary_epsilon", 1e-9))
    mmr = phys.get("mm_rounding", {})
    mm_round_mode = str(mmr.get("mode", "nearest")).lower()
    mm_round_step = float(mmr.get("step_mm", 1))

    # --- 2. Rendering settings ---
    # These are just your pixel canvas dimensions and output directory
    render = argparse.Namespace(
        output_w_px=int(cfg.get("output_size", [1000, 1000])[0]),
        output_h_px=int(cfg.get("output_size", [1000, 1000])[1]),
        outdir=str(cfg.get("visualization_outdir", "./circle_packing_outputs"))
    )

    # --- render (pixels) ---
    out_w_px, out_h_px = map(int, cfg.get("output_size", [1000, 1000]))

    # --- 3. Circle sizes conversion (mm → px) ---
    circle_sizes_mm = cfg.get("circle_sizes_mm", [])
    if not circle_sizes_mm:
        raise ValueError("Missing 'circle_sizes_mm' in config.yaml")

    px_per_mm_x = render.output_w_px / float(out_w_mm)
    circle_sizes_px_desc = sorted(
        {int(round(d_mm * px_per_mm_x)) for d_mm in circle_sizes_mm if d_mm >= min_d_mm},
        reverse=True
    )

    if not circle_sizes_px_desc:
        raise ValueError("No valid circle sizes (after filtering by min_d_mm).")

    img_path = cfg.get("img_path")
  
    # --- Extract user colors in {name, rgb} format ---
    # --- colors (strict {name,rgb}) ---
    color_cfg = cfg["user_colors"]
    user_colors  = [tuple(int(v) for v in e["rgb"]) for e in color_cfg]
    color_names  = [str(e["name"]) for e in color_cfg]

    circle_sizes = cfg.get("circle_sizes", None)
    output_size_raw = cfg.get("output_size", list(DEFAULT_OUTPUT_SIZE))
    output_size = (int(output_size_raw[0]), int(output_size_raw[1]))
    visualization_outdir = cfg.get("visualization_outdir", "./circle_packing_outputs")


    # --- 4. Now build preprocess_cfg (this is what pack_circles_from_image uses) ---
    preprocess_cfg = {
            "__color_names__": color_names,
            "__output_size_mm__": (out_w_mm, out_h_mm),   # <-- this is what your crop block uses
            "__grid_divisions__": grid_divs,
            "__boundary_epsilon__": boundary_epsilon,
            "__mm_round_mode__": mm_round_mode,
            "__mm_round_step__": mm_round_step,
            "__min_d_mm__": float(phys.get("min_circle_diameter_mm", 13)),
            "__export_svg__": bool(cfg.get("export_svg", True)),
            "__svg_include_grid__": bool(cfg.get("svg_include_grid", True)),
            "__svg_transparent_bg__": bool(cfg.get("svg_transparent_bg", True)),
        }

    result = pack_circles_from_image(
            img_path=img_path,
            user_colors=user_colors,
            circle_sizes=circle_sizes_px_desc,
            output_size=(render.output_w_px, render.output_h_px),
            visualization_outdir=render.outdir,
            preprocess_cfg=preprocess_cfg      # <-- must be passed in
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
