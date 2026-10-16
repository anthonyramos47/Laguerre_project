"""
Normalize B-spline surfaces stored as JSON so they fit in the box [-half, half]^3.

The control points are translated and scaled with hanan's normalize_vertices
(longest bounding-box side -> 2*half, centred at the origin). Because the surface
lies in the convex hull of its control points and the map is affine, the surface
ends up inside the same box. Weights and knots are left unchanged.

Usage:
    python scripts/normalize_bsplines.py                    # all of data/bsplines/*.json, box [-1, 1]
    python scripts/normalize_bsplines.py --half 1 a.json    # selected files
"""

import argparse
import json
from pathlib import Path

import numpy as np
from hanan.geometry.construction import normalize_vertices

DEFAULT_DIR = Path(__file__).resolve().parent.parent / "data" / "bsplines"


def normalize_bspline_json(path, half=1.0):
    data = json.loads(Path(path).read_text())
    cp = np.asarray(data["controlPoints"], dtype=float)
    cp[:, :3] = normalize_vertices(cp[:, :3], factor=2 * half)
    data["controlPoints"] = cp.tolist()
    Path(path).write_text(json.dumps(data, indent=4))
    return cp[:, :3].min(axis=0), cp[:, :3].max(axis=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", type=Path, help="JSON files (default: data/bsplines/*.json)")
    ap.add_argument("--half", type=float, default=1.0, help="half-size of the target box (default 1)")
    args = ap.parse_args()

    for f in args.files or sorted(DEFAULT_DIR.glob("*.json")):
        lo, hi = normalize_bspline_json(f, args.half)
        print(f"{f.name}: control points in [{lo.round(3)}, {hi.round(3)}]")


if __name__ == "__main__":
    main()
