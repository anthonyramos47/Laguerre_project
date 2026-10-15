#!/usr/bin/env python
"""Build docs/bspline_examples.html: the catalogue of Lconjugacy.surface_examples.

Every registered example plus the random lofts loft_00000 ... loft_<n-1> is fitted as in
the notebook (ex.fit(name)), sampled, and written into the page with its curvature and
parameters. Rerun after adding or changing examples:

    python scripts/build_example_catalog.py            # 24 random lofts
    python scripts/build_example_catalog.py --lofts 48
"""
from __future__ import annotations

import argparse
import inspect
import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import Lconjugacy.surface_examples as ex  # noqa: E402

TEMPLATE = Path(__file__).with_name("example_catalog_template.html")
OUT = ROOT / "docs" / "bspline_examples.html"
N_SAMPLES = 33                     # viewer grid per surface (n x n vertices)

RANGE_DOC = {
    "u_range": "range of the function's u, mapped to the B-spline's u in [0, 1]",
    "v_range": "range of the function's v, mapped to the B-spline's v in [0, 1]",
}


def param_docs(func) -> dict[str, str]:
    """'name : text' lines of a docstring (continuation lines appended)."""
    docs, last = {}, None
    names = set(inspect.signature(func).parameters)
    for line in (inspect.getdoc(func) or "").splitlines():
        m = re.match(r"^\s*([A-Za-z_]\w*)(?:\s*,\s*[A-Za-z_]\w*)*\s*:\s*(.+)$", line)
        if m and m.group(1) in names:
            keys = [k.strip() for k in line.split(":", 1)[0].split(",")]
            for k in keys:
                docs[k] = m.group(2).strip()
            last = keys
        elif last and line.startswith("  ") and line.strip() and ":" not in line[:20]:
            for k in last:
                docs[k] += " " + line.strip()
        else:
            last = None
    return docs


def fmt_value(v) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    if isinstance(v, (tuple, list)):
        return "(" + ", ".join(fmt_value(x) for x in v) + ")"
    if isinstance(v, str):
        return f'"{v}"'
    return repr(v)


def entry(name: str, group: str) -> dict:
    example, params = ex._resolve(name)
    func = example.func
    bsp = ex.fit(name)
    s = np.linspace(0, 1, N_SAMPLES)
    P = bsp(s, s).reshape(-1, 3)
    c = 0.5 * (P.min(0) + P.max(0))
    P = (P - c) * (2.0 / np.ptp(P, axis=0).max())
    _, _, K, _ = ex.curvatures(bsp, N_SAMPLES, N_SAMPLES)
    K = K.ravel()
    rep = ex.curvature_report(bsp)
    docs = {**RANGE_DOC, **param_docs(func)}
    if group.startswith("random"):
        kind = ex._random_params(name)[0]
        kind_label, note = f"random {kind}", f"random {name.split('_')[0]}, kind {kind}"
    else:
        kind_label = func.__name__ if func.__name__ != name else (
            "architectural" if example.architectural else "parametric")
        note = example.note
    return dict(
        name=name, group=group, func=func.__name__, kind_label=kind_label, note=note,
        arch=bool(example.architectural), k_positive=bool(example.k_positive),
        fit=ex.fit_settings(name),
        params=[[k, fmt_value(v), docs.get(k, "")] for k, v in params.items()],
        doc=inspect.getdoc(func) or "",
        n=N_SAMPLES, m=N_SAMPLES,
        P=np.round(P, 4).ravel().tolist(),
        K=[float(f"{k:.4g}") if np.isfinite(k) else 0.0 for k in K],
        Kmin=rep["K_min"], Kmax=rep["K_max"],
        Kratio=rep["K_ratio"] if np.isfinite(rep["K_ratio"]) else None,
        Kneg=rep["frac_K_nonpositive"],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lofts", type=int, default=24, help="random lofts loft_00000.. to include")
    ap.add_argument("--shells", type=int, default=16, help="random shells shell_00000..")
    ap.add_argument("--roofs", type=int, default=16, help="random roofs / twisted lofts roof_00000..")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    arch = ex.names(architectural=True)
    rest = [n for n in ex.names() if n not in arch]
    data = [entry(n, "registry") for n in arch]
    data += [entry(f"shell_{i:05d}", "random_shell") for i in range(args.shells)]
    data += [entry(f"roof_{i:05d}", "random_roof") for i in range(args.roofs)]
    data += [entry(f"loft_{i:05d}", "random_loft") for i in range(args.lofts)]
    data += [entry(n, "registry") for n in rest]

    html = TEMPLATE.read_text().replace("/*__DATA__*/", json.dumps(data, separators=(",", ":"))
                                        .replace("NaN", "null").replace("Infinity", "null"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html)
    print(f"{len(data)} examples -> {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
