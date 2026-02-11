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

with open("/models/feature_bounds.json", "r") as f:
    bounds_list = json.load(f)


bounds = (
    pd.DataFrame(bounds_list)
      .set_index("feature")[["p01", "p99"]]
)


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
            #Minimal Model Feats
            "croup_vs_horiz":                angle("Withers","Croup_top","Buttock"),
            "fore_pastern_len_top_norm":     dist_ratio("Fetlock_front","Hoof_upper_front","Withers","Croup_top"),
            #"chest_depth_norm_elbow":        dist_ratio("Chest_top","Elbow_front","Chest_top","Buttock"),
            "forearm_len_top_norm":          dist_ratio("Elbow_back","Knee_back","Withers","Croup_top"),
            "head_area_ratio":             (
                                                polygon_area("Nose","Chin", "Throat_latch_bottom","Throat_latch_top","Poll", "Forehead", "Forehead_eye") /
                                                polygon_area("Neck_base_bottom", "Chest_top","Chest_botom", "Barrel_below_withers","Mid_belly","Stiffle_front","Buttock", "Dock", "Croup_top", "Back_bottom", "Withers", "Neck_base_top")
                                            ),
            "hind_pastern_ang":              segment_orientation("Hind_fetlock_front","Hind_hoof_upper_front"),
            "stifle_hip_buttock_ang":        angle("Stiffle_front","Croup_top","Buttock"),
        },
        index=df.index
    )

    ##Clip outlier features(P01,P99 for now)
    feat_cols = [c for c in feats.columns if c != "id"]

    for c in feat_cols:
        if c in bounds.index:
            lo = float(bounds.loc[c, "p01"])
            hi = float(bounds.loc[c, "p99"])
            feats[c] = feats[c].clip(lower=lo, upper=hi)

    feats["fore_pastern_len_top_norm_sq"] = feats["fore_pastern_len_top_norm"] ** 2
    #feats["chest_depth_norm_elbow_sq"]    = feats["chest_depth_norm_elbow"] ** 2
    feats["forearm_len_top_norm_sq"]      = feats["forearm_len_top_norm"] ** 2
    feats["head_area_ratio_sq"]           = feats["head_area_ratio"] ** 2
    feats["hind_pastern_ang_sq"]          = feats["hind_pastern_ang"] ** 2

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

USED_PARTS = sorted({
    "Withers","Croup_top","Buttock",
    "Fetlock_front","Hoof_upper_front",
    "Chest_top","Elbow_front",
    "Elbow_back","Knee_back",
    "Nose","Chin","Throat_latch_bottom","Throat_latch_top","Poll","Forehead","Forehead_eye",
    "Neck_base_bottom","Chest_botom","Barrel_below_withers","Mid_belly","Stiffle_front",
    "Dock","Back_bottom","Neck_base_top",
    "Hind_fetlock_front","Hind_hoof_upper_front",
})

def likelihood_stats(df_keypoints):
    lk_cols = [f"{p}_l" for p in USED_PARTS if f"{p}_l" in df_keypoints.columns]
    if not lk_cols:
        raise ValueError("No likelihood columns found for USED_PARTS")

    # df_keypoints. Flatten likelihoods across used points.
    lk = df_keypoints[lk_cols].to_numpy(dtype=float).ravel()
    lk = lk[~np.isnan(lk)]

    return {
        "lk_mean": float(lk.mean()),
        "lk_median": float(np.median(lk)),
        "lk_min": float(lk.min()),
        "lk_max": float(lk.max()),
        "lk_n": int(lk.size),
    }

def gen_features(csv_path, scorer, out_path: Path):
    df_keypoints = parse_dlc_csv_one(csv_path, scorer)

    feats_df = calculate_features(df_keypoints)
    feats_path = export_features_json(feats_df, out_path)

    lk_stats = likelihood_stats(df_keypoints)

    return feats_path, lk_stats
