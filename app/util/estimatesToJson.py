import csv
import json
import re
from pathlib import Path

STAR_RE = re.compile(r"\*+$")

def clean_cell(x: str) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    # remove leading Excel formula prefix like ="" or ="_cons"
    if s.startswith("="):
        s = s[1:].strip()
    # strip surrounding quotes
    if len(s) >= 2 and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
        s = s[1:-1].strip()
    return s

def parse_coef(s: str) -> float:
    s = clean_cell(s)
    if not s:
        raise ValueError("empty coef string")
    s = STAR_RE.sub("", s)       # remove trailing stars
    s = s.replace(",", "")       # remove thousands separators
    return float(s)

def first_nonempty(row):
    for c in row:
        v = clean_cell(c)
        if v != "":
            return v
    return ""

def second_nonempty(row):
    seen = 0
    for c in row:
        v = clean_cell(c)
        if v != "":
            seen += 1
            if seen == 2:
                return v
    return ""

def esttab_csv_to_json(csv_path: Path, out_json: Path, model_name="model", link="probit"):
    coeffs = {}
    intercept = None
    n = None

    skip_vars = {
        "(1)", "WonGroup",
        "t statistics in parentheses",
        "* p<0.05, ** p<0.01, *** p<0.001",
    }

    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        r = csv.reader(f)
        for row in r:
            if not row:
                continue

            var = first_nonempty(row)
            val = second_nonempty(row)

            if not var:
                continue

            if var in skip_vars:
                continue

            if var == "N":
                try:
                    n = int(val.replace(",", ""))
                except Exception:
                    pass
                continue

            # t-stat rows often look like "(-2.81)" in val; we ignore them automatically
            if val.startswith("(") and val.endswith(")"):
                continue

            if var == "_cons":
                intercept = parse_coef(val)
                continue

            # try parse coefficient; if fails, ignore row
            try:
                coeffs[var] = parse_coef(val)
            except Exception:
                continue

    if intercept is None:
        # Helpful debug: show what rows *look* like
        raise RuntimeError(
            "Could not find _cons in the file. "
            "Likely the export has extra columns or formula quoting. "
            "Open the CSV and search for _cons; confirm it exists."
        )

    payload = {
        "name": model_name,
        "link": link,
        "intercept": intercept,
        "coeffs": coeffs,
        "n": n,
    }

    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return out_json
# Example usage
esttab_csv_to_json(
    csv_path=Path(r"W:\ComputerVision\Web App\Estimates\estimates_noInter.csv"),
    out_json=Path(r"W:\ComputerVision\Web App\models\fundamental_cr_coeffs.json"),
    model_name="WonGroup_probit_noFE",
    link="probit",
)