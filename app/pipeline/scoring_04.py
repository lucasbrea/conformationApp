from __future__ import annotations
from pathlib import Path
import json
import math
import numpy as np

def _load_json(path: Path) -> dict:
    path = Path(path)
    with open(path, "r") as f:
        return json.load(f)

def _sigmoid(z: float) -> float:
    # stable enough for your range; keep simple
    return 1.0 / (1.0 + math.exp(-z))

def score_from_features(
    features_json: Path,
    coeffs_json: Path,
    xb_range_json: Path,
) -> dict:
    """
    Reads:
      - features_json: output from your pipeline (single horse)
      - coeffs_json: intercept + coeffs (your Stata model export)
      - xb_range_json: xb_lo/xb_hi used for min-max scaling to 0-100
    Returns dict with xb, probit_prob, score_0_100.
    """

    feats = _load_json(Path(features_json))
    model = _load_json(Path(coeffs_json))
    rng = _load_json(Path(xb_range_json))

    intercept = float(model["intercept"])
    coeffs = model["coeffs"]

    # compute xb = intercept + sum(beta_j * x_j)
    xb = intercept
    missing = []
    for k, beta in coeffs.items():
        if k not in feats:
            missing.append(k)
            continue
        xb += float(beta) * float(feats[k])

    if missing:
        raise KeyError(f"Missing features in features.json: {missing}")

    xb_lo = float(rng["xb_lo"])
    xb_hi = float(rng["xb_hi"])
    xb_mu = float(rng["xb_mu"])
    xb_sd = float(rng["xb_sd"])
    if xb_hi <= xb_lo:
        raise ValueError(f"Invalid xb range: xb_hi ({xb_hi}) <= xb_lo ({xb_lo})")

    # linear index 0-100 (clipped)
    # score = (xb - xb_lo) / (xb_hi - xb_lo) * 100.0
    # score = max(0.0, min(100.0, score))

    #Z-score
    # z=(xb-xb_mu)/xb_sd
    # score=50 + (10*z)
    # score = max(0,min(100,score))
    #Percentiles
    xb_ref = np.load("/models/xb_ref_v1.npy")

    pct = np.searchsorted(xb_ref, xb, side="right") / len(xb_ref)
    score = pct * 100
    score = min(99.9, max(0.1, score))



    return {
        "xb": xb,
        "score_0_100": score,
        "prob_proxy_logit": -1 ,
        "xb_range": {"xb_lo": xb_lo, "xb_hi": xb_hi, "method": rng.get("method")},
        "model": model.get("name"),
    }