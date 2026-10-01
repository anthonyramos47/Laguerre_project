# Original L-conjugacy (Polyscope) — reference snapshot

Read-only copy of `Code/LconjugacyGUI/` at commit **`9624f23`**, the last commit before the
web port (`7d7dc8f`, 2026-05-12, "Refactor LconjugacyGUI and PolyhedralMeshGUI to new webstack
front-end"). Kept to verify the web version (`../web_app.py`, `../computations.py`,
`../state.py`) against the original and fix the differences.

These files are **not importable from here** (they import `LconjugacyGUI.*` and `hananLab`
paths of that time). To *run* the original, check out the whole repository at that commit:

```bash
git worktree add ../lconj_original 9624f23
cd ../lconj_original/Code
PYTHONPATH="$(pwd):$(pwd)/../hananLab" python -m LconjugacyGUI.main <surface_name>
```

`_hanan_Optimizer_at_9624f23.py` is hanan's optimizer at the same commit. The web-port commit
changed it only by adding the thread pool (terms evaluated in parallel); the maths is the same.

## What the web port changed (found so far — to review together)

| | Original (`lconjugacy_panel.py`) | Web (`web_app.py::_setup_df_opt`) |
|---|---|---|
| `Lconjugacy` term arguments | `(k1, k2, state.radiusFunction, u_pts, v_pts)`, `k1, k2` from `principal_frame` | `(state.bspline, u_pts, v_pts)` → **`TypeError`, the setup cannot run** |
| Solver mode | `initialize_optimizer(verbose=True, adaptive_mu=False)` — fixed damping | `initialize_optimizer(verbose=True)` — **adaptive** LM (default) |

The same commit rewrote `computations.py` (515 lines changed) and `state.py` (127), and
replaced the Polyscope panels (`app.py`, `*_panel.py`, `visualizer.py`, …) with
`web_app.py` + `lconjugacy_panel.js`; those have not been compared yet.

Since then (2026-09): the `Code/surfaces/` data moved to `Code/data/bsplines/`, Polyscope
calls were removed from the shared `utils/`, and `utils/LConjugacy/bsplineOptimization.py`
was deleted (unused).
