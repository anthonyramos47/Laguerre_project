# Data

Input data for the notebooks and tests. Results are written to `notebooks/out/`, not here.
Only a few example meshes are published for now (the list is in `.gitignore`).

| Path | Contents | Used by |
|---|---|---|
| `bsplines/nautilus.json`, `seashell.json`, `testSimpleOptimized.json` | B-spline surfaces | `lconjugacy_lab` |
| `bsplines/*.json` (the others), `bsplines/bsplines_log.jsonl` | K > 0 B-spline surfaces from `../Bspline_lab`, and the example + parameters of each | `lconjugacy_lab` |
| `ph_results/aspiralBdef*`, `ph_results/ph_input_nautilus_ex2*` | triangle mesh (`.obj`), PH-CPF result (`.out`) and its planar hexagonal dual (`.out_d_p`) | `tl_mesh_lab`, `hl_mesh_lab`, `lmesh_support_strcutures`, `visualizer_ph` |
| `ph_results/triMeshTest.obj`, `ph_results/hexMeshTest.obj` | small triangle / hexagonal test meshes | `tl_mesh_lab`, `hl_mesh_lab` |
| `cone_mesh_0.obj`, `cone_mesh_h.obj`, `nautilus_lconj.obj` | conical quad meshes | `ql_mesh_lab`, `dupin_quad_test`, `dupin_initialization` |
| `cone_mesh_small.obj`, `triMeshTest.obj`, `triMeshTest_small.obj` | small test meshes | `tests/`, `subdivision_lab` |

Meshes are read with `hanan.geometry.utils.read_obj` and usually scaled with
`normalize_vertices`.
