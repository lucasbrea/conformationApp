## Use the env_tr_june2025 conda env

import pandas as pd
import csv
import polars as pl
import numpy as np
from scipy.spatial import ConvexHull
from pathlib import Path
from itertools import chain
import os, shutil
import json

def parse_dlc_csv_one(fp_csv: Path, scorer: str | None = None) -> pd.DataFrame:
    """
    Parse ONE DLC CSV (3 header rows) into a tidy pandas df:
    columns = [id, <bodypart_x>, <bodypart_y>, <bodypart_l> ...]
    """

    fp_csv = Path(fp_csv)

    if not fp_csv.exists():
        raise FileNotFoundError(fp_csv)

    # quick empty-file guard
    if fp_csv.stat().st_size < 16:
        raise ValueError(f"CSV too small/empty: {fp_csv}")

    # read 3 header rows
    with open(fp_csv, newline="") as f:
        r = csv.reader(f)
        h1, h2, h3 = next(r), next(r), next(r)

    # sanity check header widths
    if not (len(h1) == len(h2) == len(h3) and len(h1) > 1):
        raise ValueError(f"Bad header widths: {len(h1)},{len(h2)},{len(h3)} in {fp_csv}")

    triples = list(zip(h1, h2, h3))[1:]  # drop index col
    flat = []
    for _, bodypart, coord in triples:
        col = f"{bodypart}_{coord}"
        if col.endswith("_likelihood"):
            col = col[:-len("_likelihood")] + "_l"
        flat.append(col)

    # load data (rows after 3 header rows)
    raw = pl.read_csv(fp_csv, has_header=False, skip_rows=3)

    if len(raw.columns) < 1:
        raise ValueError(f"No columns parsed from {fp_csv}")

    df = raw.drop(raw.columns[0])  # drop index col
    if len(df.columns) != len(flat):
        raise ValueError(f"Cols mismatch {len(df.columns)} != {len(flat)} for {fp_csv}")

    df.columns = flat

    # create deterministic id (strip scorer suffix if provided)
    if scorer:
        stem = fp_csv.stem
        # robust removal even if scorer is not at end
        new_name = stem.replace(scorer, "").rstrip("_-")
        id_val = str(fp_csv.with_name(new_name))
    else:
        id_val = str(fp_csv)

    df = df.with_columns(pl.lit(id_val).alias("id")).select(["id"] + [c for c in df.columns if c != "id"])

    return df.to_pandas()

def calculate_features(df):
    # Auxiliary functions
    def add_synthetic_point(df, name, func, *args):
        x_s, y_s = func(*args)
        df[f"{name}_x"] = x_s
        df[f"{name}_y"] = y_s

    def get_xy(part):
        return df[f"{part}_x"], df[f"{part}_y"]

    def dist(p1, p2):
        # euclidean distance beween two points
        x1, y1 = get_xy(p1)
        x2, y2 = get_xy(p2)
        return np.hypot(x2 - x1, y2 - y1)

    def dist_ratio(p1, p2, p3, p4):
        # ratio of distances p1-p2 and p3-p4
        return dist(p1, p2) / dist(p3, p4)
    
    def dist_x(p1, p2):
        # horizontal distance (x component only)
        x1, _ = get_xy(p1)
        x2, _ = get_xy(p2)
        return abs(x2 - x1)

    def dist_y(p1, p2):
        # vertical distance (y component only)
        _, y1 = get_xy(p1)
        _, y2 = get_xy(p2)
        return abs(y2 - y1)

    # def angle(A, B, C):
    #     #angle A-B-C in degrees
    #     xA, yA = get_xy(A)
    #     xB, yB = get_xy(B)
    #     xC, yC = get_xy(C)
    #     BA = np.vstack([xA - xB, yA - yB]).T
    #     BC = np.vstack([xC - xB, yC - yB]).T
    #     dot = np.einsum("ij,ij->i", BA, BC)
    #     norm = np.linalg.norm(BA, axis=1) * np.linalg.norm(BC, axis=1)
    #     cosθ = np.clip(dot / norm, -1, 1)
    #     return np.degrees(np.arccos(cosθ))

    # def segment_orientation(p1, p2):
    #     # orientation of line p1-p2 with respect to the horizontal axis
    #     x1, y1 = get_xy(p1)
    #     x2, y2 = get_xy(p2)
    #     θ = np.degrees(np.arctan2(y2 - y1, x2 - x1))
    #     return np.mod(θ + 360, 360)

    # def segments_orientation_diff(pA, pB, pC, pD):
    #     # how different the orientation of AB is versus CD
    #     o1 = segment_orientation(pA, pB)
    #     o2 = segment_orientation(pC, pD)
    #     Δ = o1 - o2
    #     return ((Δ + 180) % 360) - 180
    def angle(A, B, C, baseline=0.0, direction="counterclockwise"):
        """
        ∠ABC in degrees, measured from BA to BC.
        By default measured counterclockwise; set direction="clockwise" for CW.
        Returns signed deviation around `baseline` in (-180, 180].
        """
        xA, yA = get_xy(A)
        xB, yB = get_xy(B)
        xC, yC = get_xy(C)
        a1 = np.degrees(np.arctan2(yA - yB, xA - xB))
        a2 = np.degrees(np.arctan2(yC - yB, xC - xB))
        diff = ((a2 - a1 + 180.0) % 360.0) - 180.0
        if direction == "clockwise":
            diff = -diff
        return ((diff - baseline + 180.0) % 360.0) - 180.0

    def segment_orientation(p1, p2, baseline=0.0, direction="counterclockwise"):
        """
        Orientation of segment p1→p2 (degrees).
        By default measured counterclockwise from x-axis; set direction="clockwise" for CW.
        Returns signed deviation around `baseline` in (-180, 180].
        """
        x1, y1 = get_xy(p1)
        x2, y2 = get_xy(p2)
        theta = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if direction == "clockwise":
            theta = -theta
        return ((theta - baseline + 180.0) % 360.0) - 180.0

    def segments_orientation_diff(pA, pB, pC, pD, baseline=0.0, direction="counterclockwise"):
        """
        Signed angle from segment AB to CD (degrees).
        By default measured counterclockwise; set direction="clockwise" for CW.
        Returned as deviation around `baseline` in (-180, 180].
        """
        xA, yA = get_xy(pA)
        xB, yB = get_xy(pB)
        xC, yC = get_xy(pC)
        xD, yD = get_xy(pD)
        o1 = np.degrees(np.arctan2(yB - yA, xB - xA))
        o2 = np.degrees(np.arctan2(yD - yC, xD - xC))
        diff = ((o2 - o1 + 180.0) % 360.0) - 180.0
        if direction == "clockwise":
            diff = -diff
        return ((diff - baseline + 180.0) % 360.0) - 180.0

    def segment_midpoint(p1, p2):
        # point halfway between p1 and p2
        x1, y1 = get_xy(p1)
        x2, y2 = get_xy(p2)
        return (x1 + x2) / 2, (y1 + y2) / 2

    def polygon_centroid(*parts):
        # centroid (arithmetic mean) of many parts in a polygon
        if len(parts) == 1 and hasattr(parts[0], "__iter__"):
            parts = parts[0]
        xs = [get_xy(p)[0] for p in parts]
        ys = [get_xy(p)[1] for p in parts]
        return sum(xs) / len(xs), sum(ys) / len(ys)

    def triangle_height_width_ratio(A, B, C): 
        # measures how much C (top) protrudes off the segment AB (base), normalized by AB
        AC = dist(A, C)
        angle_deg = angle(B, A, C)
        angle_rad = np.deg2rad(angle_deg)
        h = AC * np.sin(angle_rad)
        AB = dist(A, B)
        return h / AB

    def polygon_area(*parts):
        # area of a polygon with the shoelace formula (https://en.wikipedia.org/wiki/Shoelace_formula)
        if len(parts) == 1 and hasattr(parts[0], "__iter__"):
            parts = parts[0]
        xs = np.vstack([get_xy(p)[0] for p in parts])
        ys = np.vstack([get_xy(p)[1] for p in parts])
        xs_next = np.roll(xs, -1, axis=0)
        ys_next = np.roll(ys, -1, axis=0)
        cross = xs * ys_next - ys * xs_next
        return 0.5 * np.abs(cross.sum(axis=0))

    def polygon_bounding_box_aspect_ratio(*parts):
        # aspect ratio of the bounding box of a polygon
        if len(parts)==1 and hasattr(parts[0],"__iter__"):
            parts = parts[0]
        xs = np.vstack([get_xy(p)[0] for p in parts])
        ys = np.vstack([get_xy(p)[1] for p in parts])
        width_box = xs.max(axis=0) - xs.min(axis=0)
        height_box = ys.max(axis=0) - ys.min(axis=0)
        return width_box / height_box

    def polygon_rectangularity(*parts):
        # measures how "rectangular" a polygon is (compared tot it's bounding box)
        if len(parts) == 1 and hasattr(parts[0], "__iter__"):
            parts = parts[0]
        xs = np.vstack([get_xy(p)[0] for p in parts])
        ys = np.vstack([get_xy(p)[1] for p in parts])
        width_box = xs.max(axis=0) - xs.min(axis=0)
        height_box = ys.max(axis=0) - ys.min(axis=0)
        bbox_area = width_box * height_box
        return polygon_area(*parts) / bbox_area

    def polygon_edge_length_variation(*parts):
        # cefficient of variation (std/mean) of all edge lengths in the polygon, higher means more irregular
        if len(parts) == 1 and hasattr(parts[0], "__iter__"):
            parts = parts[0]
        edges = [dist(parts[i], parts[(i+1)%len(parts)]) 
                for i in range(len(parts))]
        E = np.vstack(edges)
        return E.std(axis=0) / E.mean(axis=0)

    def polygon_perimeter(*parts):
        # perimeter of a polygon
        if len(parts)==1 and hasattr(parts[0],"__iter__"):
            parts = parts[0]
        perim = None
        for i in range(len(parts)):
            p1 = parts[i]
            p2 = parts[(i+1) % len(parts)]
            edge = dist(p1, p2)
            perim = edge if perim is None else perim + edge
        return perim

    def polygon_circularity(*parts):
        # how circular a polygon is (1 means it's a perfect circles)
        if len(parts)==1 and hasattr(parts[0],"__iter__"):
            parts = parts[0]
        area = polygon_area(*parts)
        perimeter = polygon_perimeter(*parts)
        return 4 * np.pi * area / (perimeter**2)

    def polygon_convexity(*parts):
        # measures how convex a polygon is (1 maeans it's convex)
        if len(parts) == 1 and hasattr(parts[0], "__iter__"):
            parts = parts[0]
        pts = np.array([
        (get_xy(p)[0].iloc[0], get_xy(p)[1].iloc[0])
        for p in parts
        ])
        hull_area = ConvexHull(pts).volume
        area = polygon_area(*parts)
        return area/hull_area
    

    # Adding synthetic points to the df, we can call them in the feats later
    # add_synthetic_point(df, "Mid_Chest_Withers", segment_midpoint, "Withers", "Chest_top")
    # add_synthetic_point(df, "Back_Centroid", polygon_centroid,  "Dock", "Croup_top", "Withers", "Chest_top")
    add_synthetic_point(df, "Upper_neck_midpoint", segment_midpoint, "Poll", "Throat_latch_top")
    add_synthetic_point(df, "Lower_neck_midpoint", segment_midpoint, "Neck_base_bottom", "Neck_base_top")
    add_synthetic_point(df, "Knee_mid", segment_midpoint, "Knee_front", "Knee_back")

    add_synthetic_point(df, "Topline_mid",  segment_midpoint, "Withers", "Croup_top")
    add_synthetic_point(df, "Shoulder_mid", segment_midpoint, "Withers", "Chest_top")
    add_synthetic_point(df, "Hip_mid",      segment_midpoint, "Croup_top", "Buttock")
    add_synthetic_point(df, "Underline_mid",segment_midpoint, "Barrel_below_withers", "Stiffle_front")

    feats = pd.DataFrame(
        {
            # =========================
            # REGION: HEAD
            # =========================
            # --- Orientation diffs ---
            "head_axis_topline":             segments_orientation_diff("Poll","Nose","Withers","Croup_top"),
            "head_axis_shldr_diff":          segments_orientation_diff("Poll","Nose","Withers","Chest_top"),
            "head_ang":                      segments_orientation_diff("Forehead","Nose","Throat_latch_bottom","Chin"),
            "poll_ang":                      angle("Nose", "Poll", "Crest_above_chest"),  
            "throat_poll_nose_ang":          angle("Throat_latch_bottom", "Poll", "Nose"), 
            # --- Dist ratios (head/face) ---
            "head_neck_len_ratio":           dist_ratio("Poll","Nose","Poll","Withers"),
            "eye_nose_head_ratio":           dist_ratio("Eye","Nose","Poll","Nose"),
            "ear_head_ratio":                dist_ratio("Ear_tip", "Poll", "Nose", "Poll"),

            # =========================
            # REGION: NECK & CHEST / SHOULDER
            # =========================
            # --- Segment orientations / diffs ---
            "downhill_angle":                segment_orientation("Withers","Croup_top"), # different effect for yearling vs HORA... (well developed hind is good for 1yos, even if it's causing a tetmporary downhill ang)
            "downhill_diff_vs_elbow":        segments_orientation_diff("Withers","Croup_top", "Elbow_front", "Elbow_back"),
            "shoulder_slope_ang":            segments_orientation_diff("Withers","Chest_top","Neck_base_top","Knee_mid"),
            "neck_axis_topline":             angle("Poll","Withers","Croup_top"),
            "throatlatch_topline":           segments_orientation_diff("Throat_latch_top","Neck_base_top","Withers","Croup_top"),
            "neck_axis_shldr_diff":          angle("Poll","Withers","Chest_top"),
            # --- Dist ratios (neck & chest proportions) ---
            "neck_len_base_ratio":           dist_ratio("Upper_neck_midpoint","Lower_neck_midpoint","Neck_base_bottom","Neck_base_top"),
            "top_neck_ratio":                dist_ratio("Withers","Croup_top","Neck_base_bottom","Neck_base_top"),
            "neck_top_bottom_ratio":         dist_ratio("Poll","Neck_base_top","Throat_latch_top","Neck_base_bottom"),
            "throatlatch_head_ratio":        dist_ratio("Poll","Throat_latch_top","Poll","Nose"),
            "throatlatch_head_ratio2":       dist_ratio("Poll","Throat_latch_bottom","Poll","Nose"),
            "head_neckbase_ratio":           dist_ratio("Poll","Nose","Neck_base_bottom","Neck_base_top"),
            "neck_top_und_ratio":            dist_ratio("Poll","Withers","Throat_latch_top","Chest_top"),
            "neck_tie_in_height":            dist_ratio("Neck_base_top","Chest_top","Withers","Croup_top"),
            "neck_body_ratio":               dist_ratio("Poll","Withers","Chest_top","Buttock"),
            "chest_depth_norm_len":          dist_ratio("Chest_top","Chest_botom","Chest_top","Buttock"),
            "chest_depth_norm_elbow":        dist_ratio("Chest_top","Elbow_front","Chest_top","Buttock"),
            # --- Thickness ratios (neck/throat base) ---
            "neckbase_thick_top":            dist_ratio("Neck_base_bottom","Neck_base_top","Withers","Croup_top"),
            "throat_vs_neckbase":            dist_ratio("Throat_latch_bottom","Throat_latch_top","Neck_base_bottom","Neck_base_top"),
            # --- Local curvature / bulge ---
            "crest_bulge_ratio":             triangle_height_width_ratio("Poll","Neck_base_top","Crest_above_chest"),
            "shoulder_triangle_proj":        triangle_height_width_ratio("Withers","Barrel_below_withers","Neck_base_bottom"),
            # --- Chest & heartgirth placements ---
            "heart_floor_ratio":             dist_ratio("Withers","Barrel_below_withers","Elbow_back","Hoof_lower_back"),
            "heart_flank_ratio":             dist_ratio("Withers","Barrel_below_withers","Croup_top","Stiffle_front"),
            "heart_length_ratio":            dist_ratio("Withers","Barrel_below_withers","Chest_top","Buttock"),
            "heartgirth_ground_ratio":       dist_ratio("Withers","Barrel_below_withers","Barrel_below_withers","Hoof_lower_front"),
            "rib_spring_ratio":              dist_ratio("Barrel_below_withers","Mid_belly","Withers","Croup_top"),
            "chest_proj_ratio":              triangle_height_width_ratio("Chest_top","Elbow_front","Chest_botom"),
            "ribcage_body_ratio":          dist_y("Withers","Barrel_below_withers") / dist_x("Chest_top","Buttock"),

            # --- Shoulder component (research) ---
            "humerus_scapula_ratio":         dist_ratio("Withers","Elbow_back","Withers","Chest_top"),
            # Neck rectangularity
            "neck_rectangularity":          polygon_rectangularity("Neck_base_top", "Neck_base_bottom", "Throat_latch_top", "Poll", "Crest_above_chest"),
            

            # =========================
            # REGION: FORELIMBS
            # =========================
            # --- Thickness (top or bodylength -normalized) ---
            "pect_depth_top_norm":           dist_ratio("Chest_top","Chest_botom","Withers","Croup_top"),
            "forearm_thick_top_norm":        dist_ratio("Mid_forearm_front","Mid_forearm_back","Withers","Croup_top"),
            "knee_w_top_norm":               dist_ratio("Knee_front","Knee_back","Withers","Croup_top"),
            "cannon_w_top_norm":             dist_ratio("Cannon_front","Cannon_back","Withers","Croup_top"),
            "fetlock_w_top_norm":            dist_ratio("Fetlock_front","Fetlock_back","Withers","Croup_top"),
            "pect_depth_bl_norm":            dist_ratio("Chest_top","Chest_botom","Chest_top","Buttock"),
            "forearm_thick_bl_norm":         dist_ratio("Mid_forearm_front","Mid_forearm_back","Chest_top","Buttock"),
            "knee_w_bl_norm":                dist_ratio("Knee_front","Knee_back","Chest_top","Buttock"),
            "cannon_w_bl_norm":              dist_ratio("Cannon_front","Cannon_back","Chest_top","Buttock"),
            "fetlock_w_bl_norm":             dist_ratio("Fetlock_front","Fetlock_back","Chest_top","Buttock"),
            "elbow_barrel_angle":            segment_orientation("Elbow_back","Barrel_below_withers"),
            # --- Angles & orientation diffs ---
            "cannon_ang_front":              segment_orientation("Knee_front","Fetlock_front"),
            "pastern_ang_front":             segment_orientation("Fetlock_front","Hoof_upper_front"),
            "hoof_wall_ang_front":           segment_orientation("Hoof_upper_front","Hoof_lower_front"),
            "hoof_wall_ang_back":            segment_orientation("Hoof_upper_back","Hoof_lower_back"),
            "pastern_hoof_diff":             segments_orientation_diff("Fetlock_front","Hoof_upper_front","Hoof_upper_front","Hoof_lower_front", baseline=180),
            "cannon_pastern_diff":           segments_orientation_diff("Knee_front","Fetlock_front","Fetlock_front","Hoof_upper_front", baseline=180),
            "knee_back_ang":                 angle("Mid_forearm_back","Knee_back","Cannon_back", baseline=180),
            "shoulder_pastern_diff":         segments_orientation_diff("Withers","Chest_top","Fetlock_front","Hoof_upper_front"),
            "shoulder_pastern_back_diff":    segments_orientation_diff("Withers","Chest_top","Fetlock_back","Hoof_upper_back"),
            "forearm_barrel_ang":            angle("Mid_forearm_back", "Elbow_back","Barrel_below_withers"),
            "hoof_mid_pastern_ang":              angle("Pastern_front","Hoof_upper_front", "Hoof_lower_front"),
            "hoof_fetlock_ang":              angle("Fetlock_front","Hoof_upper_front", "Hoof_lower_front"),
            # --- Lengths (top or bodylength -normalized) ---
            "forearm_len_top_norm":          dist_ratio("Elbow_back","Knee_back","Withers","Croup_top"),
            "fore_cannon_len_top_norm":      dist_ratio("Knee_back","Fetlock_back","Withers","Croup_top"),
            "fore_pastern_len_top_norm":     dist_ratio("Fetlock_front","Hoof_upper_front","Withers","Croup_top"),
            "fore_hoof_h_top_norm":          dist_ratio("Hoof_upper_front","Hoof_lower_front","Withers","Croup_top"),
            "forearm_len_bl_norm":           dist_ratio("Elbow_back","Knee_back","Chest_top","Buttock"),
            "fore_cannon_len_bl_norm":       dist_ratio("Knee_back","Fetlock_back","Chest_top","Buttock"),
            "fore_pastern_len_bl_norm":      dist_ratio("Fetlock_front","Hoof_upper_front","Chest_top","Buttock"),
            "fore_hoof_h_bl_norm":           dist_ratio("Hoof_upper_front","Hoof_lower_front","Chest_top","Buttock"),
            # --- Slenderness / within-limb proportions ---
            "cannon_ratio":                  dist_ratio("Knee_front","Fetlock_front","Cannon_front","Cannon_back"),
            "pastern_slen_front":            dist_ratio("Fetlock_front","Hoof_upper_front","Pastern_front","Pastern_back"),
            "fore_pas_can_ratio":            dist_ratio("Fetlock_front","Hoof_upper_front","Knee_back","Fetlock_back"),
            "fore_arm_pas_ratio":            dist_ratio("Elbow_back","Knee_back","Fetlock_front","Hoof_upper_front"),
            # --- Width proportions within limb ---
            "knee_w_can_w_ratio":            dist_ratio("Knee_front","Knee_back","Cannon_front","Cannon_back"),
            "fetlock_w_can_w_ratio":         dist_ratio("Fetlock_front","Fetlock_back","Cannon_front","Cannon_back"),
            # --- Hoof front/back height balance (fore only) ---
            "hoof_fb_h_ratio":               dist_ratio("Hoof_upper_front","Hoof_lower_front","Hoof_upper_back","Hoof_lower_back"),
            "hoof_ang_diff_horiz":           segments_orientation_diff("Hoof_upper_front","Hoof_upper_back","Hoof_lower_front","Hoof_lower_back"),
            "hoof_ang_diff_vert":            segments_orientation_diff("Hoof_upper_front","Hoof_lower_front","Hoof_upper_back","Hoof_lower_back"),
            # --- Forelimb vs body (research) ---
            "forearm_cannon_ratio":          dist_ratio("Elbow_back","Knee_back","Knee_back","Fetlock_back"),
            "forearm_len_body_norm":         dist_ratio("Elbow_back","Knee_back","Chest_top","Buttock"),
            "cannon_len_body_norm":          dist_ratio("Knee_back","Fetlock_back","Chest_top","Buttock"),
            "pastern_len_body_norm":         dist_ratio("Fetlock_front","Hoof_upper_front","Chest_top","Buttock"),
            "hoof_h_body_norm":              dist_ratio("Hoof_upper_front","Hoof_lower_front","Chest_top","Buttock"),
            # --- Joint angle (fore) ---
            "shoulder_joint_ang":            angle("Withers","Elbow_back","Knee_back", baseline=180),

            # =========================
            # REGION: MID-BODY / BARREL
            # =========================
            # --- Body polygon shape (global) ---
            "top_rect":                      polygon_rectangularity("Withers","Back_bottom","Croup_top"),
            "body_circ":                     polygon_circularity("Chest_top","Barrel_below_withers","Mid_belly","Stiffle_front","Croup_top","Buttock"),
            "body_aspect":                   polygon_bounding_box_aspect_ratio("Chest_top","Barrel_below_withers","Mid_belly","Stiffle_front","Croup_top","Buttock"),
            "body_edge_cv":                  polygon_edge_length_variation("Chest_top","Barrel_below_withers","Mid_belly","Stiffle_front","Croup_top","Buttock"),
            # --- Barrel-specific shape indices ---
            "barrel_rect":                   polygon_rectangularity("Chest_top","Barrel_below_withers","Mid_belly","Stiffle_front"),
            "barrel_aspect2":                polygon_bounding_box_aspect_ratio("Chest_top","Barrel_below_withers","Mid_belly","Stiffle_front"),
            "barrel_edge_cv2":               polygon_edge_length_variation("Chest_top","Barrel_below_withers","Mid_belly","Stiffle_front"),
            # --- Curvature / protrusion (midline) ---
            "belly_sag_ratio":               triangle_height_width_ratio("Barrel_below_withers","Stiffle_front","Mid_belly"),
            # --- Alternative normalizations (robustness) ---
            "underline_top_ratio":           dist_ratio("Barrel_below_withers","Stiffle_front","Withers","Croup_top"),

            # =========================
            # REGION: HINDQUARTERS / HINDLIMBS
            # =========================
            # --- Thickness (top-normalized) ---
            "gaskin_w_top_norm":             dist_ratio("Gaskin_front","Gaskin_back","Withers","Croup_top"),
            "hock_w_top_norm":               dist_ratio("Hock_front","Hock_back","Withers","Croup_top"),
            "hind_cannon_w_top_norm":        dist_ratio("Hind_cannon_front","Hind_cannon_back","Withers","Croup_top"),
            "hind_fetlock_w_top_norm":       dist_ratio("Hind_fetlock_front","Hind_fetlock_back","Withers","Croup_top"),
            # --- Angles & orientation diffs ---
            "hind_cannon_ang":               segment_orientation("Hock_front","Hind_fetlock_front"),
            "hind_pastern_ang":              segment_orientation("Hind_fetlock_front","Hind_hoof_upper_front"),
            "hind_hoof_wall_ang":            segment_orientation("Hind_hoof_upper_front","Hind_hoof_lower_front"),
            "hind_pastern_hoof_diff":        angle("Hind_fetlock_front","Hind_hoof_upper_front","Hind_hoof_lower_front", baseline=180),
            "hock_back_ang":                 angle("Gaskin_back","Hock_back","Hind_fetlock_back"),
            "hock_back_buttock_ang":         angle("Buttock","Hock_back","Hind_fetlock_back", baseline=180),
            "croup_slope":                   segment_orientation("Croup_top","Buttock"),
            "buttock_hock_slope":            segment_orientation("Hock_back", "Buttock"),
            # --- Lengths (top-normalized) ---
            "gaskin_len_top_norm":           dist_ratio("Gaskin_back","Hock_back","Withers","Croup_top"),
            "hind_cannon_len_top_norm":      dist_ratio("Hock_back","Hind_fetlock_back","Withers","Croup_top"),
            "hind_pastern_len_top_norm":     dist_ratio("Hind_fetlock_front","Hind_hoof_upper_front","Withers","Croup_top"),
            "hind_hoof_h_top_norm":          dist_ratio("Hind_hoof_upper_front","Hind_hoof_lower_front","Withers","Croup_top"),
            # --- Slenderness / within-limb proportions ---
            "pastern_slen_hind":             dist_ratio("Hind_fetlock_front","Hind_hoof_upper_front","Hind_pastern_front","Hind_pastern_back"),
            "hind_cannon_slen":              dist_ratio("Hock_back","Hind_fetlock_back","Hind_cannon_front","Hind_cannon_back"),
            "hind_pas_can_ratio":            dist_ratio("Hind_fetlock_front","Hind_hoof_upper_front","Hock_back","Hind_fetlock_back"),
            "hind_gask_pas_ratio":           dist_ratio("Gaskin_back","Hock_back","Hind_fetlock_front","Hind_hoof_upper_front"),
            # --- Width proportions within limb ---
            "hind_hock_w_can_w_ratio":       dist_ratio("Hock_front","Hock_back","Hind_cannon_front","Hind_cannon_back"),
            "hind_fet_w_can_w_ratio":        dist_ratio("Hind_fetlock_front","Hind_fetlock_back","Hind_cannon_front","Hind_cannon_back"),
            # --- Hoof front/back height balance (hind only) ---
            "hind_hoof_fb_h_ratio":          dist_ratio("Hind_hoof_upper_front","Hind_hoof_lower_front","Hind_hoof_upper_back","Hind_hoof_lower_back"),
            # --- Hind vs body (research) ---
            "hip_len_body_ratio":            dist_ratio("Croup_top","Buttock","Chest_top","Buttock"),
            "stifle_pos_body_ratio":         dist_ratio("Stiffle_front","Buttock","Chest_top","Buttock"),
            "gaskin_len_body_norm":          dist_ratio("Gaskin_back","Hock_back","Chest_top","Buttock"),
            "hind_cannon_len_norm":          dist_ratio("Hock_back","Hind_fetlock_back","Chest_top","Buttock"),
            "hind_cannon_short_ratio":       dist_ratio("Hock_back","Hind_fetlock_back","Gaskin_back","Hock_back"),
            # --- Joint angle (hind) ---
            "stifle_joint_ang":              angle("Croup_top","Stiffle_front","Hock_front"),
            "hind_leg_ang":                  angle("Croup_top","Stiffle_front","Hock_back"),
            # --- Hind thickness normalized by body length (research) ---
            "gaskin_thick_len_norm":         dist_ratio("Gaskin_front","Gaskin_back","Chest_top","Buttock"),

            # =========================
            # REGION: TOP LINE & BALANCE
            # =========================
            # --- Orientation vs horizontal / global balance ---
            "shldr_hip_slope_diff":          segments_orientation_diff("Withers","Chest_top","Croup_top","Buttock"),
            "shldr_vs_horiz":                angle("Chest_top","Withers","Croup_top"),
            "croup_vs_horiz":                angle("Withers","Croup_top","Buttock"),
            "back_vs_horiz":                 angle("Back_bottom","Withers","Croup_top"),
            "withers_ang":                   angle("Neck_base_top", "Withers", "Croup_top"),
            # --- Proportionate lengths across top/underline ---
            "top_und_ratio":                 dist_ratio("Withers","Croup_top","Elbow_back","Gaskin_front"),
            "top_croup_ratio":               dist_ratio("Withers","Croup_top","Croup_top","Buttock"),
            "shldr_hip_len_ratio":           dist_ratio("Withers","Chest_top","Croup_top","Buttock"),
            "back_hip_len_ratio":            dist_ratio("Withers","Back_bottom","Back_bottom","Buttock"),
            "top_und_len_ratio":             dist_ratio("Withers","Back_bottom","Elbow_back","Stiffle_front"),
            "shoulder_back_hip_eq":          dist_ratio("Withers","Chest_top","Chest_top","Back_bottom"),
            "neck_withers_back_ratio":       dist_ratio("Neck_base_top", "Withers", "Withers", "Back_bottom"),
            "lumbar_body_ratio":             dist_ratio("Back_bottom", "Croup_top","Chest_top","Buttock"),
            "lumbar_body_ratio_h":           dist("Back_bottom", "Croup_top")/dist_x("Chest_top","Buttock"),
            # --- Vertical/topline curvature & vertical metrics ---
            "back_sag_ratio":                triangle_height_width_ratio("Withers","Croup_top","Back_bottom"),
            "withers_hip_vert_ratio":        dist_y("Withers","Croup_top") / dist("Withers","Croup_top"),

            # =========================
            # REGION: CROSS-REGION COMPARISONS
            # =========================
            # --- Fore vs hind orientation/length comparisons ---
            "fore_hind_cannon_diff":         segments_orientation_diff("Knee_front","Fetlock_front","Hock_front","Hind_fetlock_front"),
            "hoof_wall_pair_diff":           segments_orientation_diff("Hoof_upper_front","Hoof_lower_front","Hind_hoof_upper_front","Hind_hoof_lower_front"),
            "pastern_pair_diff":             segments_orientation_diff("Fetlock_front","Hoof_upper_front","Hind_fetlock_front","Hind_hoof_upper_front"),
            "croup_pastern_diff":            segments_orientation_diff("Croup_top","Buttock","Hind_fetlock_front","Hind_hoof_upper_front"),
            "cannon_shldr_diff":             segments_orientation_diff("Knee_front","Fetlock_front","Withers","Chest_top"),
            "hind_cannon_croup_diff":        segments_orientation_diff("Hock_front","Hind_fetlock_front","Croup_top","Buttock"),
            # --- Simple cross-region ratios ---
            "fore_hind_pas_len_ratio":       dist_ratio("Fetlock_front","Hoof_upper_front","Hind_fetlock_front","Hind_hoof_upper_front"),
            "fore_hind_hoof_h_ratio":        dist_ratio("Hoof_upper_front","Hoof_lower_front","Hind_hoof_upper_front","Hind_hoof_lower_front"),
            # --- Multi-region joint/triangle angles (research) ---
            "wither_stifle_elbow_ang":       angle("Withers","Stiffle_front","Elbow_back"),
            "head_shldr_elbow_ang":          angle("Poll","Withers","Elbow_back"),
            "stifle_hip_buttock_ang":        angle("Stiffle_front","Croup_top","Buttock"),
            "outer_hip_ang":                 angle("Croup_top","Buttock","Dock"),
            "dock_angle":                    angle("Croup_top","Dock","Buttock"),

            # =========================
            # REGION: REGIONAL AREAS & SHAPE INDICES
            # =========================
            # --- Area ratios (regions / body polygon) ---
            "barrel_area_ratio":             (
                                                polygon_area("Chest_top","Barrel_below_withers","Mid_belly","Stiffle_front") /
                                                polygon_area("Chest_top","Barrel_below_withers","Mid_belly","Stiffle_front","Croup_top","Buttock")
                                            ),
            "hiptri_area_ratio":             (
                                                polygon_area("Croup_top","Buttock","Dock") /
                                                polygon_area("Chest_top","Barrel_below_withers","Mid_belly","Stiffle_front","Croup_top","Buttock")
                                            ),
            "lowerneck_area_ratio":          (
                                                polygon_area("Withers","Neck_base_bottom","Neck_base_top","Throat_latch_top") /
                                                polygon_area("Chest_top","Barrel_below_withers","Mid_belly","Stiffle_front","Croup_top","Buttock")
                                            ),
            "loin_area_ratio":               (
                                                polygon_area("Back_bottom","Croup_top","Stiffle_front") /
                                                polygon_area("Chest_top","Barrel_below_withers","Mid_belly","Stiffle_front","Croup_top","Buttock")
                                            ),
            "foretri_area_ratio":            (
                                                polygon_area("Withers","Chest_top","Elbow_back") /
                                                polygon_area("Chest_top","Barrel_below_withers","Mid_belly","Stiffle_front","Croup_top","Buttock")
                                            ),
            "hindtri_area_ratio":            (
                                                polygon_area("Croup_top","Buttock","Stiffle_front") /
                                                polygon_area("Chest_top","Barrel_below_withers","Mid_belly","Stiffle_front","Croup_top","Buttock")
                                            ),
            "neckpatch_area_ratio":          (
                                                polygon_area("Withers","Neck_base_top","Throat_latch_top","Crest_above_chest") /
                                                polygon_area("Chest_top","Barrel_below_withers","Mid_belly","Stiffle_front","Croup_top","Buttock")
                                            ),
            # --- Area ratios (tiny regional patches) ---
            "pastern_tri_area_ratio":        (
                                                polygon_area("Fetlock_front","Hoof_upper_front","Hoof_lower_front") /
                                                polygon_area("Chest_top","Barrel_below_withers","Mid_belly","Stiffle_front","Croup_top","Buttock")
                                            ),
            "hind_pas_tri_area_ratio":       (
                                                polygon_area("Hind_fetlock_front","Hind_hoof_upper_front","Hind_hoof_lower_front") /
                                                polygon_area("Chest_top","Barrel_below_withers","Mid_belly","Stiffle_front","Croup_top","Buttock")
                                            ),
                                            # --- Area ratios (regions / body polygon) ---
            "head_area_ratio":             (
                                                polygon_area("Nose","Chin", "Throat_latch_bottom","Throat_latch_top","Poll", "Forehead", "Forehead_eye") /
                                                polygon_area("Neck_base_bottom", "Chest_top","Chest_botom", "Barrel_below_withers","Mid_belly","Stiffle_front","Buttock", "Dock", "Croup_top", "Back_bottom", "Withers", "Neck_base_top")
                                            ),
            # --- Regional shape indices / aspects ---
            "fore_rect":                     polygon_rectangularity("Withers","Chest_top","Elbow_back"),
            "hind_rect":                     polygon_rectangularity("Stiffle_front","Croup_top","Buttock"),
            "top_aspect":                    polygon_bounding_box_aspect_ratio("Withers","Back_bottom","Croup_top"),
            "hip_aspect":                    polygon_bounding_box_aspect_ratio("Croup_top","Buttock","Dock"),
            "fore_aspect":                   polygon_bounding_box_aspect_ratio("Withers","Chest_top","Elbow_back"),
        },
        index=df.index
    )

    feats.insert(loc=0, column='id', value=df['id'])

    return feats

# --- extracting CSV's ---
scorer = "DLC_Resnet50_pose_analysisJun27shuffle1_snapshot_680"



def export_features_json(df, out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Accept df as pandas DataFrame OR dict
    if hasattr(df, "iloc"):  # pandas DataFrame
        if len(df) != 1:
            raise ValueError(f"Expected exactly one row, got {len(df)}")
        record = df.iloc[0].to_dict()
    elif isinstance(df, dict):
        record = df
    else:
        raise TypeError("export_features_json expects a pandas DataFrame (1 row) or dict")

    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)

    return out_path


def gen_features(csv_path, scorer, out_path: Path) -> Path:
    df_keypoints = parse_dlc_csv_one(csv_path, scorer)     # pandas df
    df_features  = calculate_features(df_keypoints)        # ideally 1-row df or dict
    return export_features_json(df_features, out_path)

