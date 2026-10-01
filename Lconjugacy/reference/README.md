# Reference code (not maintained, not imported)

Kept for reading and for checking the notebook pipeline against. Nothing here is imported
by `Lconjugacy/pipeline.py` or the notebooks, and import paths inside may be stale
(they still refer to the old `LconjugacyGUI` package and `Code/` layout).

| Folder | What |
|---|---|
| `legacy_polyscope/` | the original L-conjugacy app (Polyscope desktop GUI, commit `9624f23`). `pipeline.py` is a port of it and was checked against it; its own `README.md` lists the known differences. |
| `web_gui/` | the later browser GUI (`web_app.py`, `computations.py`, `state.py`, `lconjugacy_panel.js`). Known broken: its direction-field optimizer passes the wrong arguments to the `Lconjugacy` term and switched to adaptive damping. Do not fix without reviewing it against `legacy_polyscope/`. |
| `exampleGenerator/` | functions that generated the example B-spline surfaces. |

The L-conjugacy workflow now runs in `notebooks/lconjugacy_lab.ipynb` on `Lconjugacy/pipeline.py`.
