# Laguerre project

This repository contains the code used for the computation of **sphere meshes in Laguerre
geometry** (Laguerre meshes, L-meshes), together with research code on L-conjugate nets and
Dupin cyclide patches. `Lconjugacy/` contains the code used for the paper
[*Computer Aided Geometric Design* (2026)](https://www.sciencedirect.com/science/article/abs/pii/S0167839626000981).

The geometry and optimization library (**hanan**) and the browser viewer (**kayviz**) come
from the public [Chakana_Geo](https://github.com/anthonyramos47/Chakana_Geo) repository,
included here as a git submodule.

> **Under construction.** This is research code. It is being cleaned up and documented to
> make it easier to use; expect rough edges and changes.

## Layout

| Path | Contents |
|---|---|
| `Chakana_Geo/` | submodule: hanan + kayviz (public, GPL-3.0) |
| `LMesh/` | `Lmesh` (spheres on `top.vertices`, oriented planes on `top.faces`), its objective terms (`Optimization/`), research geometry (`utils/laguerre.py`, `utils/dupin.py`), and display (`lmesh_serializers.py`) |
| `Lconjugacy/` | L-conjugacy pipeline (`pipeline.py`: B-spline, sphere fit, L-conjugate directions, remeshing, L-mesh) and its utilities (`utils/`) |
| `Lconjugacy/reference/` | old L-conjugacy GUIs kept for reference only (original Polyscope app, later web GUI); not imported, see its README |
| `notebooks/` | the main entry points (below) |
| `notebooks/out/` | results, one folder per project (`lmesh/`, `lconjugacy/`, `dupin/`); **git-ignored**, kept locally |
| `data/` | example input meshes and B-spline surfaces used by the notebooks and tests (see `data/README.md`) |
| `bin/` | `quadRemesher`, the quad remesher of the L-conjugacy stage 3 (Linux x86-64 build) |
| `docs/` | notes on the Dupin cyclide propagation |
| `tests/` | tests of the L-mesh code |

## Setup

The code runs in a conda environment (`environment.yml`, Python 3.11):

```bash
git clone --recurse-submodules git@github.com:anthonyramos47/Laguerre_project.git
cd Laguerre_project

conda env create -f environment.yml       # creates the "Lproj" environment
conda activate Lproj
pip install -e Chakana_Geo/hanan -e Chakana_Geo/kayviz
python -m ipykernel install --user --name lproj --display-name "Python (Lproj)"
```

Then open the notebooks (`jupyter lab`, or VS Code) with the **Python (Lproj)** kernel.
In an existing clone without the libraries: `git submodule update --init`.

To update the environment after a change to `environment.yml`:
`conda env update -f environment.yml --prune`.

## Notebooks

| Notebook | What it does |
|---|---|
| `tl_mesh_lab`, `hl_mesh_lab`, `ql_mesh_lab` | triangular / hexagonal / quad L-meshes from input meshes: initialization, optimization, export, images and tables |
| `lmesh_support_strcutures` | HL-meshes whose spheres lie in a linear sphere complex (support structures), sphere sampling, planarity of the sphere centres |
| `lconjugacy_lab` | L-conjugacy pipeline, stages 0–4 (B-spline range, sphere fit, L-conjugate directions, external quad remesher, L-mesh on the remeshed quads). Ready-made examples: `nautilus`, `seashell`, `testSimpleOptimized` |
| `dupin_quad_test`, `dupin_initialization` | Dupin cyclide patches from a quad mesh (rectangle propagation) |
| `subdivision_lab` | subdivided L-meshes |
| `visualizer_ph` | viewer for the `ph_results` meshes |
| `tl_mesh_support_hyperplane` | old (pre-swap) notebook kept for reference; does not run as is |

Each notebook starts with a short description of what it does and how to run it. Only a few
example meshes are included in `data/` for now; the mesh lists in the notebooks name them, and
the other names are commented out.

Each notebook starts its own kayviz viewer (`init()` prints the URL). To share one viewer
between notebooks, run `python -m kayviz` first.

### L-conjugacy examples

In `lconjugacy_lab`, set `SURFACE` in the first cell to `"nautilus"`, `"seashell"` (examples of
the paper) or `"testSimpleOptimized"`; the (u, v) range, sampling, sphere radius and angle are
then set to the values of the original runs. The B-splines are in `data/bsplines/`; the other files
there (K > 0 shells, lofts, tubes, ...) come from the catalogue in `../Bspline_lab`, and
`bsplines_log.jsonl` records the example and parameters each one was made from. Stages 0–2
run on any machine. Stage 3 runs the quad remesher shipped in `bin/quadRemesher` (a Linux x86-64
build; it needs `libblas.so.3`, e.g. `sudo apt install libblas3`). To use another build, do one of:

- set `st.path_remesher = "/path/to/quadRemesher"` in the stage 3 cell of the notebook,
- `export QUAD_REMESHER=/path/to/quadRemesher` before starting Jupyter,
- edit `REMESHER_CANDIDATES` at the top of `Lconjugacy/pipeline.py`.

`pl.check_remesher()` prints which binary is used.

## Tests

```bash
python -m pytest tests/                          # L-mesh
(cd Chakana_Geo/hanan && python -m pytest)       # libraries
(cd Chakana_Geo/kayviz && python -m pytest)
```

## Working with the submodule

`Chakana_Geo/` is its own repository, checked out at the commit this project uses. To change
the libraries: `cd Chakana_Geo && git checkout main`, edit, commit and push there, then commit
in this repository to record the new library version (`git add Chakana_Geo`).

## License

The code in this repository is released under the [MIT License](LICENSE).

- The libraries it uses, hanan and kayviz ([Chakana_Geo](https://github.com/anthonyramos47/Chakana_Geo)),
  are licensed separately under GPL-3.0-or-later.
- The planar hexagonal meshes in `data/ph_results/` (`*.out*`) were computed with
  [PH-CPF](https://github.com/michaled/PH-CPF) (K. Pluta, M. Edelstein, A. Vaxman, M. Ben-Chen,
  "PH-CPF: Planar Hexagonal Meshing using Coordinate Power Fields", ACM TOG 40(4), 2021).
