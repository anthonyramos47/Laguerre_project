# Rectangle propagation, optimization, and cyclidic patch generation

> **Code location (2026-09):** the propagation, optimization and cyclide-patch functions
> described here live in `LMesh/utils/dupin.py` (moved out of `hanan`'s old
> `geo_processing.py`); the pipeline runs in `notebooks/dupin_quad_test.ipynb`. The
> `Code/Dupin_Propagation` app and its legacy Polyscope files were removed, and
> `recenter_rectangle_in_face` / `create_rectangle_from_frame` were dropped as unused in the
> geometry split. Mentions of removed files below are kept as history.

Status: **propagation implemented 2026-08-12; optimization and patch
generation added 2026-08-14.** All three test meshes pass every propagation
acceptance check (0 corner-key mismatches, 0 non-rectangular, 0 non-planar,
0 bowties, 0 outside-face, full grid coverage) — see "Result".

This document is the design reference for the pipeline implemented in
`LMesh/utils/dupin.py` and run from `notebooks/dupin_quad_test.ipynb`.

## Pipeline overview

Three stages, each a section below:

| stage | input | output | section |
|---|---|---|---|
| **1. Propagation** | conical quad mesh | one rectangle per face, edge-compatible across shared edges | Steps 1–3 |
| **2. Optimization** | propagated rectangle corners | the same configuration with accumulated propagation error tightened | [Optimization](#optimization) |
| **3. Patch generation** | optimized corners + frames | one Dupin cyclide patch per interior vertex | [Cyclidic patch generation](#cyclidic-patch-generation) |

Propagation is *sequential and local*: it satisfies the Dupin condition by
construction, but each reflection step carries a small numerical error which
accumulates along the sweep. Optimization redistributes that error globally
rather than letting it compound away from the seed. Patch generation then
fills each interior vertex's four-corner cell with a smooth patch.

## Why

The current propagation moves each rectangle to its neighbour by *rigid
reflection* (`reflect_rectangle_rigid`). Reflection is a rigid motion, so every
rectangle stays congruent to the seed — and therefore cannot adapt to faces of
different size. Measured after propagation (rectangle corners outside their own
face, 2026-08-12):

| mesh | faces | outside their face | corner-key mismatches |
|---|---|---|---|
| `cone_mesh_small.obj` | 72 | 0 (0%) | 0 |
| `cone_mesh_0.obj` | 484 | 98 (20%) | 0 |
| `cone_mesh_small2.obj` | 300 | 66 (22%) | 0 |

(Produced by the audit harness in "How to verify" below, so the new
implementation's numbers are directly comparable. Non-rectangular, non-planar
and bowtie counts are 0 everywhere today — those must stay at 0.)

**Test meshes: `cone_mesh_small.obj`, `cone_mesh_0.obj`, `cone_mesh_small2.obj`
only.** `cone_mesh_h.obj` is deliberately excluded for now — do not use it for
these examples or acceptance runs.

The earlier `grid_W`/`grid_H` + `set_rectangle_edge` approach avoided the
overflow but abandoned congruence and drifted badly (interior rectangles rotated
up to 42° off their faces). `recenter_rectangle_in_face` only *translates*, so it
cannot rescue a rectangle that is genuinely larger than its face.

**The insight this design rests on:** reflecting across the bisecting plane
preserves only the *shared edge*. The perpendicular dimension is a real degree of
freedom at every step. Sizing it per-strip — from the faces the strip actually
crosses — is what keeps rectangles inside their faces while keeping neighbouring
rectangles edge-compatible.

## Notation

Conical quad mesh as an `m x n` grid. `F_0` is a corner face with vertices
`v_0, v_1, v_m, v_{m+1}`. `Q_i` is the rectangle on face `F_i`. Its corners are
`q(f_id, v_id)` — indexed by **face id and the face's own vertex id**, so each
corner is the "interior version" of that vertex.

In code the dict key order stays `frames[(face_id, vertex_id)]`.

## Step 1 — Seed rectangle `Q_0`

At the corner face `F_0`:

1. `b_F0` = barycentre of `F_0`.
2. Build orthogonal in-plane `u`, `v` almost aligned with the face edges, with
   **`u` pointing toward `F_1`** and **`v` pointing toward `F_m`**. (Take the
   edge direction, project into the face plane, orthogonalise; then
   `v = n x u`, flipping if it points away from `F_m`.)
3. Choose `h_u`, `h_v`; place corners symmetrically about `b_F0`:

```
q(0, v_0)     = b_F0 - (h_u/2) u - (h_v/2) v
q(0, v_1)     = b_F0 + (h_u/2) u - (h_v/2) v
q(0, v_{m+1}) = b_F0 + (h_u/2) u + (h_v/2) v
q(0, v_m)     = b_F0 - (h_u/2) u + (h_v/2) v
```

Signs must be assigned so each corner lands nearest its own `v_id` — do not
assume the rectangle winds the same way as the face loop. Reuse
`_match_corners_to_vertices`; a previous bug came from exactly this assumption.

### Bounding `h_u`, `h_v` at the seed

To keep the rectangle from outgrowing faces it will later be reflected into:

- `h_u` max ← min edge length, over faces of the **U-arm** (`F_1 … F_{m-1}`),
  of the edges aligned with direction `v`.
- `h_v` max ← min edge length, over faces of the **V-arm**
  (`F_m … F_{m(n-1)}`), of the edges aligned with direction `u`.

Multiply by `factor ∈ (0,1)`, default **0.5**. Expose `factor` as a slider.

## Step 2 — Arms (U-arm and V-arm)

Walking `F_0 → F_1 → F_2 → …` along the U-arm:

1. Reflect **only the shared edge's two corners** across the bisecting plane of
   the two faces (`reflect_edge_frames`). That edge's length is preserved —
   it carries `h_v` into `F_1` unchanged.
2. The perpendicular dimension `h_u(1)` is **free**. Choose it as
   `factor * min(edge lengths aligned with u over the V-strip starting at F_1)`
   — i.e. faces `F_{m+1}, F_{2m+1}, …`.
3. Repeat for every face along the arm.

Symmetrically for the V-arm: `h_u` is carried by the reflected edge and `h_v(m)`
is the free dimension, sized from the transverse U-strip.

**So: for a U-arm face the transverse strip is its V-strip, and vice versa.**

Corners must again be re-keyed to the vertex each actually lands nearest —
reflection carries a corner to the *far* side of the shared edge, so it does not
stay near the vertex it came from.

## Step 3 — Interior faces

For `F_{m+1}`, which shares an edge with `F_m` and with `F_1`:

1. Reflect `Q_m`'s edge `q(m, v_{m+1}) – q(m, v_{2m+1})` onto `F_{m+1}`.
2. Reflect `Q_1`'s edge `q(1, v_{m+1}) – q(1, v_{m+2})` onto `F_{m+1}`.
3. Both produce a position for the shared corner `q(m+1, v_{m+1})`. They will
   not coincide exactly — **average the two positions and project onto
   `F_{m+1}`'s plane.**
4. Those two edges define the rectangle. Square it up (enforce 90°) and project
   into the face plane.

Sweep the interior so both predecessors are already placed before a cell is
built (row-major from the seed corner works).

## Implementation notes

**`Mesh.quad_face_strips(f)` is now correct — use it** for the per-strip sizing
in Steps 1 and 2. It returns `(u_faces, v_faces)`, the full grid row and column
through face `f`, each including `f` itself.

It was rewritten on 2026-08-12 on top of `grid_face_matrix` (cached). The old
half-edge walk was wrong on *every* face of every mesh tested — on
`cone_mesh_small.obj` it gave empty lists for 26 of 72 faces, duplicated entries
for 42, and matched the true strip for none. Anything written against its old
behaviour should be re-checked; there are callers in
`Code/LconjugacyGUI/local_analysis_panel.py`,
`Code/LconjugacyGUI/lmeshManipulation.py` and
`Code/Dupin_Propagation/Dupin_Quad_Prop.py` that were silently consuming garbage.

`grid_coordinates(mesh.faces, seed_face=…)` remains the right tool when the grid
must be **oriented with the seed at `[0, 0]`** (as the arms require) — it and
`grid_face_matrix` agree on coverage for every test mesh, differing only by
a transpose. Skip `-1` entries (holes) from either.

**Reuse rather than rewrite:**
- `bisecting_plane`, `reflect_frame_across_plane` — the reflection math is
  correct; leave it alone.
- `_match_corners_to_vertices` — corner→vertex assignment.
- `_inward_shift` / `recenter_rectangle_in_face` — still useful as a final
  safety net, but sizing should make it rarely necessary.
- `project_rectangle_to_face` — flattens onto the face plane and re-squares;
  this is what Step 3.4 needs.

**Functions that become unused** once this lands: `reflect_rectangle_rigid`,
`create_rectangle_from_frame`, `gram_schmidt_average`, `project_rectangle_to_face`
(unless reused for Step 3.4). `set_rectangle_edge` stays — `reflect_rectangle`
still calls it.

The legacy `Code/Dupin_Propagation/Dupin_Quad_Prop.py` and its
`legacy_polyscope/` copy define their own *methods* of the same names on the GUI
class; they do not import the library functions, so removing the library
versions will not break them. `create_rectangle_inside_face` is referenced only
by those legacy methods and by docstrings.

**Dead GUI controls to remove** from `dupin_quad_callback.py`: `u_dist_factor`,
`v_dist_factor`, `u_first_step_dist`, `v_first_step_dist`. Replace with the
single `factor` slider (default 0.5).

## Helpers to implement

The geometry helpers live in `LMesh/utils/dupin.py`
(`from LMesh.utils.dupin import ...`).

Existing signatures you will call, verified 2026-08-12:

```python
# LMesh/utils/dupin.py
bisecting_plane(mesh, f_normals, face_i, face_j)     -> (normal, offset, midpoint)
reflect_edge_frames(mesh, f_normals, frames, face_i, face_j)   # mutates frames
_match_corners_to_vertices(corner_pts, face_verts)   -> list[int]  (order[k] -> vertex slot)
_inward_shift(face_verts, normal, pts, max_passes=4, eps=1e-12)  -> (3,) shift
recenter_rectangle_in_face(mesh, frames, face_id, f_normals=None, ...)
project_rectangle_to_face(mesh, frames, face_id, f_normals=None)
get_rectangle(mesh, frames, face_id)                 -> (pts (4,3), w, h)
grid_coordinates(F, seed_face=0)                     -> (R, C) int array, -1 = hole
find_corner_faces(mesh, min_boundary_edges=2)
# algebraic.py
reflect_frame_across_plane(frame, plane_normal, plane_distance, face=None)
unit(v);  orth_proj(v, u);  rotation_matrix(axis, angle)
# Mesh methods
mesh.quad_face_strips(f_idx)      -> (u_faces, v_faces)   # row & column through f
mesh.shared_edge_vertices(fi, fj) -> (v_a, v_b)
mesh.max_min_edge(f)              -> (min_edge, max_edge)
mesh.face_normals                 # property
```

A *frame* is `{"pt", "u", "v", "n", "f"}`; `frames` is
`{(face_id, vertex_id): frame}`.

### H1 — `strip_min_edge_length(mesh, faces, direction, f_normals=None)`

Minimum length, over `faces`, of the face edges aligned with `direction`.
Backs the sizing rule in Steps 1 and 2.

- For each face, take its 4 edges; classify each as u-like or v-like by
  `abs(unit(edge) @ unit(direction))` — the two with the larger absolute dot
  are the aligned pair.
- Return the min length over all aligned edges of all faces.
- `direction` is a world vector; project it into each face's plane
  (`orth_proj(direction, n)`) before comparing, since faces are not coplanar.
- Return `inf` for an empty `faces` list so callers can `min()` safely.

### H2 — `seed_rectangle_at_barycentre(mesh, f_normals, face_id, u_dir, h_u, h_v)`

Step 1's construction, split out so it is testable without the GUI.

- Corners at `b_F ± (h_u/2) u ± (h_v/2) v`, with `v = cross(n, u)`.
- Assign corners to vertex ids via `_match_corners_to_vertices` — **do not**
  zip with the face loop.
- Set each corner's `u`/`v` to its own outward signs (`su * u`, `sv * v`), as
  `build_edge_aligned_rectangle` already does.
- Returns a fresh `frames` dict (4 entries).

Note `build_edge_aligned_rectangle` already does nearly this, but sizes from
*face edge fractions* and clamps the offset. Either generalise it with an
absolute-size mode or write H2 alongside it; do not silently change its
existing behaviour, the notebook's cell 2 sliders depend on it.

### H3 — `seed_direction_towards(mesh, f_normals, face_id, target_face)`

Returns the in-plane unit vector at `face_id` pointing toward `target_face`
(their shared-edge midpoint minus the barycentre, projected into the face
plane, normalised). Gives Step 1 its "`u` toward `F_1`, `v` toward `F_m`"
orientation without hand-picking edge indices.

### H4 — `place_rectangle_from_edge(mesh, f_normals, frames, face_id, edge_vertex_ids, depth)`

The arm step (Step 2), replacing `set_rectangle_edge`.

Given a face already carrying the two reflected corners at
`edge_vertex_ids`, complete the rectangle by stepping `depth` perpendicular to
that edge, *into* `face_id`.

- Step direction: in-plane normal to the edge; pick the sign whose dot with
  `(face_barycentre − edge_midpoint)` is **positive** so it goes inward. Do not
  infer direction from the frame's stored `u`/`v` signs — that was the source of
  an earlier transposition bug.
- Assign the two new corners with `_match_corners_to_vertices` over all four.
- Result must be an exact rectangle in the face plane.

### H5 — `rectangle_from_two_edges(mesh, f_normals, frames, face_id, edge_a, edge_b)`

The interior step (Step 3). `edge_a`, `edge_b` are the two reflected edges as
`(vertex_id, point)` pairs, sharing one vertex id.

- Average the two candidate positions for the shared corner; project onto the
  face plane.
- Build the rectangle those two edges span, then square it: keep the longer
  edge's direction as `u`, set `v = cross(n, u)`, and take extents from the two
  edge lengths.
- Finish with `project_rectangle_to_face` so it is exactly planar and
  right-angled.
- Return the max gap between the two candidate shared-corner positions — a
  useful drift diagnostic to print per sweep.

### H6 — `propagate_rectangles(mesh, f_normals, grid_faces, seed_frames, factor=0.5)`

Top-level driver: arms then interior, returning
`(frames, rectangles_placed, stats)`. Keeping this a pure function (no `ws.*`,
no globals) is what lets the verification script below run headlessly.

`stats` should carry per-sweep counts and the H5 drift values.

## How to verify

### 1. Unit-level, before wiring the GUI

```python
# H1: on a mesh with known geometry, the aligned-edge minimum must equal a
#     hand-computed value; empty face list -> inf.
# H2: 4 corners returned; all angles 90 deg; each corner nearest its own vertex.
# H3: dot(result, unit(shared_edge_midpoint - barycentre)) > 0.99.
# H4: shared edge unchanged; new edge length == depth; rectangle exact.
# H5: shared corner equals the mean of the two candidates (post-projection);
#     rectangle exact; returned drift >= 0.
```

### 2. Whole-mesh invariants

Run this against the three test meshes. It is the same harness used to produce the
baseline table at the top, so results are directly comparable:

```python
import numpy as np
from hanan.geometry.mesh import Mesh
from hanan.geometry.utils import (read_obj, normalize_vertices, unit,
                                  find_corner_faces, grid_coordinates, get_rectangle)
from hanan.geometry.algebraic import DupinPropagationError

def worst_outside(pts, face_verts, n):
    """>0 means some corner lies outside the face."""
    c, m, worst = face_verts.mean(axis=0), len(face_verts), -1e9
    for i in range(m):
        a, b = face_verts[i], face_verts[(i + 1) % m]
        ne = np.cross(n, b - a)
        if np.dot(ne, c - a) > 0:
            ne = -ne
        worst = max(worst, max(np.dot(p - a, unit(ne)) for p in pts))
    return worst

def audit(mesh, f_normals, frames, faces_placed):
    keymis = nonrect = nonplanar = outside = bowtie = 0
    for f in sorted(faces_placed):
        pts, w, h = get_rectangle(mesh, frames, f)
        fv, n = mesh.vertices[mesh.faces[f]], unit(f_normals[f])

        # (1) corner-key identity: q(f,v) must be nearest vertex v
        for k in [k for k in frames if k[0] == f]:
            d = {int(v): np.linalg.norm(frames[k]["pt"] - mesh.vertices[int(v)])
                 for v in mesh.faces[f]}
            if min(d, key=d.get) != int(k[1]):
                keymis += 1

        # (2) exact rectangle
        e = [pts[(i + 1) % 4] - pts[i] for i in range(4)]
        L = [np.linalg.norm(x) for x in e]
        ang = [np.degrees(np.arccos(np.clip(
            np.dot(e[i] / L[i], -e[(i - 1) % 4] / L[(i - 1) % 4]), -1, 1)))
            for i in range(4)]
        if not np.allclose(ang, 90, atol=1e-6):
            nonrect += 1

        # (3) planarity
        nr = np.cross(pts[1] - pts[0], pts[2] - pts[0])
        ln = np.linalg.norm(nr)
        if ln > 1e-14 and abs((pts[3] - pts[0]) @ (nr / ln)) > 1e-9:
            nonplanar += 1

        # (4) containment
        if worst_outside(pts, fv, n) > 1e-9:
            outside += 1

        # (5) bowtie: in a planar quad, the diagonals must cross
        o, e1 = pts[0], unit(pts[1] - pts[0])
        e2 = np.cross(n, e1)
        P = np.array([[(p - o) @ e1, (p - o) @ e2] for p in pts])
        def cr(a, b, c):
            return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
        def crosses(a, b, c, d):
            return ((cr(c,d,a) > 0) != (cr(c,d,b) > 0)) and \
                   ((cr(a,b,c) > 0) != (cr(a,b,d) > 0))
        if not crosses(P[0], P[2], P[1], P[3]):
            bowtie += 1

    return dict(rects=len(faces_placed), keymis=keymis, nonrect=nonrect,
                nonplanar=nonplanar, outside=outside, bowtie=bowtie)
```

Note check (5): for a *correctly ordered* convex quad the **diagonals** cross
and the sides do not. Testing that sides cross would flag every good rectangle.

### 3. Pass criteria

| check | required |
|---|---|
| corner-key mismatches | **0** on all three test meshes |
| non-rectangular | **0** (angles 90° to ~1e-12) |
| non-planar | **0** (to ~1e-16) |
| bowties | **0** |
| rectangles outside their face | **well below** the baseline 20–70%; 0 is the goal |
| faces placed | equal to the grid's covered-face count |

Also confirm shared edges still agree between neighbouring rectangles, and that
`h_u`/`h_v` shrink rather than grow as the sweep moves away from the seed (the
sizing rule is a `min`, so a growing dimension means a bug).

### 4. End to end

`notebooks/dupin_quad_test.ipynb` runs the whole pipeline in five cells:

| cell | contents |
|---|---|
| **1** | initialization — paths, viewer, imports, palette, `OUT_DIR` |
| **2** | optimization energy terms (`Parallelogram`, `StraightAngle`) |
| **3** | mesh + seed rectangle, with the five sliders |
| **4** | full pipeline: propagate → patches (both methods) → cones → circumcircles → optimize (60 steps) → rebuild everything from the optimized corners |
| **5** | export everything to `OUT_DIR` |

Cell 4 stores each stage's merged geometry in a `data` dict under
`init_*` (propagated) and `opt_*` (optimized) keys, so cell 5 exports without
recomputing:

```
data["init_rectangles"]        data["opt_rectangles"]
data["init_cyclides"]          data["opt_cyclides"]           # Bobenko
data["init_cyclides_bezier"]   data["opt_cyclides_bezier"]    # Bézier
data["init_cones"]             data["opt_cones"]
data["init_circles"]           data["opt_circles"]            # circumcircles
```

Execute it headlessly with:

```bash
cd notebooks && python3 -c "
import json; nb=json.load(open('dupin_quad_test.ipynb')); ns={'__name__':'__main__'}
[exec(compile(''.join(c['source']), f'<cell{i}>', 'exec'), ns)
 for i, c in enumerate(nb['cells'], 1)]"
```

Expected on `cone_mesh_0.obj` (484 faces, 441 interior vertices): 484
rectangles, 441 cyclides by each method, 840 cone strips, 441 circumcircles —
for both the `init` and `opt` passes.

**Remember:** after editing `LMesh/utils/dupin.py`, restart the notebook kernel
(or use `%autoreload 2`), otherwise the old functions stay loaded.

## Optimization

Propagation is a *sequential* construction: each rectangle is placed by
reflecting from an already-placed neighbour. It satisfies the Dupin condition
by construction, but every step carries a small numerical error, and those
errors accumulate along the sweep — the further a face is from the seed, the
more drift it inherits. The optimization's role is to **tighten** that
accumulated error, redistributing it globally instead of letting it compound
in one direction.

Implemented as a Levenberg–Marquardt solve (`hanan.optimization.Optimizer`)
over the rectangle corners.

### Variables

Writing `nF` for the face count and `nVi` for the interior-vertex count:

| name | shape | meaning |
|---|---|---|
| `p` | `(4·nF, 3)` | rectangle corner positions — the primary unknowns |
| `r_n` | `(nF, 3)` | per-rectangle normals |
| `aux_n` | `(nVi, 3)` | auxiliary cone axis, one per interior vertex |
| `cy_c` | `(nVi, 3)` | auxiliary circumcentre, one per interior vertex |
| `cy_n` | `(nVi, 3)` | auxiliary circle normal, one per interior vertex |

`cy_n` and `aux_n` are kept unit-length by `opt.unitize_variable(..., 3, 10)`.

### Index sets

Built once from the propagated frames:

```python
dict_indices[(f_id, v_id)] -> global corner index   # bijection
rectangle_faces  (nF, 4)    # the 4 corner indices of each rectangle
circles_faces    (nVi, 4)   # the 4 corners meeting at each interior vertex,
                            # in ring order (from vertex_face_adjacency_list)
```

`circles_faces` ring order matters: patch generation consumes it directly as
`F0, F1, F12, F2`, where `F1`/`F2` are `F0`'s ring neighbours and `F12` is
diagonal.

### Energy terms

| term | class | residual | tightens |
|---|---|---|---|
| `cyclicity` | `Cyclicity` | `(v_{i+1} − v_i)·(m_i − c_f)` and `(m_i − c_f)·n_f` | concircularity of the four corners at a vertex |
| `parallelogram` | `Parallelogram` | `v_2 + v_0 − v_3 − v_1` | each rectangle stays a parallelogram |
| `straight_angle` | `StraightAngle` | `(v_1−v_0)·(v_2−v_0) / (‖v_1−v_0‖‖v_2−v_0‖)` | right angle at each rectangle corner |
| `conicality` | `Planarity` (reused) | `(n_i − n_{i+1})·aux_n` | consecutive face normals around a vertex lie on a cone about `aux_n` |

All weights default to 1.0; the unitizing terms use weight 10.

Note `conicality` reuses `Planarity` with the *face normals* as its "vertices"
and `aux_n` as its "face normal". The residual `(n_i − n_{i+1})·aux_n` says
consecutive normals differ only orthogonally to `aux_n` — i.e. they lie on a
cone about that axis. This is the classical conical-mesh condition, not a
planarity constraint on points.

### Convergence

On `cone_mesh_0.obj` (484 faces, 441 interior vertices), 60 LM iterations:

```
cyclicity        6.07e-09  ->  2.32e-09     0.9% of final energy
parallelogram    2.55e-12  ->  9.48e-13     0.0%
straight_angle   1.45e-13  ->  1.57e-14     0.0%
conicality       8.61e-06  ->  2.62e-07    99.1%
                 ------------------------
total            8.62e-06  ->  2.65e-07
```

`conicality` dominates the residual throughout — it is the slowest term to
converge and the one that limits how tight the final configuration gets.

### Residual error after optimization

Measured on the optimized corners of `cone_mesh_0.obj`, against Property (d)
of Pottmann et al., *Circular Arc Structures* (Bo, Pottmann, Kilian, Wang,
Wallner, SIGGRAPH 2011):

> **(d)** A quadrilateral with circular edges is a principal quadrilateral of
> a cyclide **if and only if edges intersect orthogonally and each edge is
> symmetric** in the sense of Definition 5.
>
> **Definition 5.** An edge `vw` is symmetric iff reflection in the bisector
> plane of `v, w` maps the normals onto each other:
> `ρ_vw(n_v) = n_w`, `ρ_vw(n_w) = n_v`. The paper's energy is
> `f_symm = Σ ‖ρ_vw(n_v) − n_w‖²`.

| condition | mean | median | max |
|---|---|---|---|
| symmetry defect `‖ρ_vw(n_v) − n_w‖` | 0.0031 | 0.0025 | 0.0209 |
| orthogonality (deg off 90°) | 2.15° | 2.01° | 7.73° |

These are the residual propagation errors the current energy set does not
fully absorb. Neither condition has a *direct* term: `straight_angle`
enforces 90° *within* each rectangle, not between cell edges across
rectangles, and no term targets the normal-reflection symmetry. Adding
`f_symm` and an inter-rectangle orthogonality term would tighten these further.

## Cyclidic patch generation

Each interior vertex of valence 4 carries a cell of four concircular corners —
one from each incident rectangle — which is filled with a Dupin cyclide patch.
Two constructions are implemented.

### A. Lie-sphere construction (`Dupin_patch`)

Follows Bobenko & Huhnen-Venedey (arXiv:1101.5955), Eqs. 2.10–2.15. Anchors at
`x0`: takes `n0` as given, derives `n1`, `n2` by reflecting `n0` across the
perpendicular bisectors of `(x0,x1)` and `(x0,x2)` (Eq. 2.11), builds two
cyclidic families of spheres, and samples their envelope.

**Corner reproduction is asymmetric.** `x0`, `x1`, `x2` are exact by
construction. `x12` is *recovered* — it is where the two families must meet at
parameter `(1,1)` — so it absorbs the accumulated error as a visible gap.
Measured on `cone_mesh_0.obj`, before and after the optimization:

| pass | mean `x12` gap | worst |
|---|---|---|
| `init` (propagated corners) | 4.87e-04 | 1.92e-02 |
| `opt` (optimized corners) | 1.01e-03 | **2.60e-01** |

**The gap grows through optimization**, even though every energy term falls by
an order of magnitude. This is not a contradiction: the optimizer tightens
concircularity, parallelogram, right-angle and conicality residuals, none of
which is the `x12` reproduction error. Moving corners to satisfy those can push
a cell closer to the Lie-sphere construction's numerical failure mode (see
below). It is a reason to prefer the Bézier construction downstream, not a
sign the optimization is misbehaving.

The worst case (vertex 113) is 587% of the patch diagonal. That outlier is a
*numerical* degeneracy rather than accumulated drift:
`boundary_lie_sphere_cyclide` computes `value_r = ‖Δx‖² / (2 Δx·n)`, which
blows up as `Δx·n → 0`. Evidence:

- Re-anchoring the same cell at a different corner changes the gap from 0.259
  to 0.0045 — a 50× improvement from a pure relabelling.
- The gap correlates with **neither** Property (d) residual
  (Pearson −0.03 symmetry, −0.07 orthogonality), so it is not a restatement of
  the drift measured above.

### B. Rational biquadratic Bézier construction (`dupin_patch_bezier`)

Follows Pottmann et al., proof of Proposition 8:

> "Cyclidic patches have degree (2,2), with control points according to
> Figure 20. The control mesh with vertices `b_ij` has planar faces, with
> `b11` as the intersection point of tangent planes in the boundary vertices
> `b00, b20, b22, b02`."

#### The surface

With Bernstein polynomials `B_i^k(t) = C(k,i) (1−t)^(k−i) t^i`, a rational
degree-(2,2) Bézier surface is

```
            Σ_{i,j}  w_ij · b_ij · B_i²(s) · B_j²(t)
f(s,t)  =  ───────────────────────────────────────────
              Σ_{i,j}  w_ij · B_i²(s) · B_j²(t)
```

with `i, j ∈ {0,1,2}` and `s, t ∈ [0,1]`. The nine control points `b_ij` and
weights `w_ij` are laid out as

```
        j=0        j=1        j=2
 i=0   b00        b01        b02        <- corner, edge, corner
 i=1   b10        b11        b12        <- edge,  interior, edge
 i=2   b20        b21        b22        <- corner, edge, corner
```

#### Corners — `b00, b20, b22, b02`, weight 1

The four cell corners `x0, x1, x12, x2` are used *directly* as the four corner
control points, each with `w = 1`. A degree-2 Bézier interpolates its corner
control points exactly, so

```
f(0,0) = b00 ,   f(1,0) = b20 ,   f(1,1) = b22 ,   f(0,1) = b02
```

This is the structural difference from the Lie-sphere construction: no corner
is recovered, so none absorbs the accumulated propagation error.

#### Edge control points — `b10, b01, b21, b12`

Each patch boundary is a **circular arc**, which has an exact rational
quadratic Bézier form: control points `(p0, b1, p2)` with weights
`(1, cos(θ/2), 1)`, where `θ` is the arc's central angle.

`b1` is constructed from the arc's **own circle**. For an arc from `p0` to
`p2` with unit corner normal `n0` at the start:

```
chord = p2 − p0 ,   L = ‖chord‖ ,   d = chord / L

t0    = normalize( n0 × (d × n0) )        tangent at p0 (component of d
                                          orthogonal to n0; flip so t0·d > 0)

axis  = normalize( t0 × chord )           normal of the arc's plane
rdir  = normalize( axis × t0 )            in-plane, perpendicular to t0

ρ     = L² / (2 · chord·rdir)             signed distance to the centre
c     = p0 + ρ · rdir ,   R = |ρ|         circle centre and radius

sin(θ/2) = L / (2R)  ⇒  w1 = cos(θ/2)
b1    = c + (R / w1) · normalize( midpoint(p0,p2) − c )
```

A genuinely straight edge — `t0` parallel to the chord, or `R → ∞` — falls
back to the chord midpoint with `w1 = 1`, which is exactly the degree-2 form
of a line segment.

**Why not intersect the two endpoint tangents?** That is the textbook
construction (`b1` = intersection of the tangent lines at `p0` and `p2`) and
was the first implementation here, but it is **ill-conditioned on a conical
mesh** and produced visibly broken patches. Adjacent corner normals on such a
mesh are nearly equal, so the two tangents are nearly parallel and the
intersection is undefined; a least-squares solve still returns a point, but a
meaningless one. Measured on `cone_mesh_0.obj` (441 interior vertices, all of
them):

| | tangent intersection | circle-based |
|---|---|---|
| `‖b1 − midpoint‖ / chord`, mean | 3.21 | **0.037** |
| same, max | **882.2** (vertex 36) | **0.059** |
| vertices with `\|t0·t2\| > 0.99` | 441 / 441 | — |
| endpoint reproduction | — | 0.000e+00 |

At vertex 202 the `x1→x12` edge had `n1·n2 = 1.000000` exactly (smallest
singular value 8e-5), putting `b1` **0.0605 away from the chord midpoint on a
chord of length 0.0199** — the far-off point visible in the rendered patch.
The circle construction is well defined at every one of those vertices, and
correctly treats the near-straight edges as straight (at vertex 202 that edge
has `R = 55`, half-angle 0.01°) instead of blowing up.

#### Interior control point — `b11`

From Pottmann et al.: `b11` is the intersection point of the tangent planes at
the four corners. Each corner contributes a plane `nᵢ · x = nᵢ · pᵢ`, giving
four equations in three unknowns:

```
⎡ n00 ⎤        ⎡ n00 · p00 ⎤
⎢ n20 ⎥ · b11 = ⎢ n20 · p20 ⎥            solved in least squares
⎢ n22 ⎥        ⎢ n22 · p22 ⎥
⎣ n02 ⎦        ⎣ n02 · p02 ⎦
```

The system is over-determined; it has an exact solution precisely when the
cell is truly cyclidic, and the least-squares solution degrades gracefully
otherwise.

Its weight is taken as the product of the two adjacent edge weights,
`w11 = w10 · w01`, which is the standard tensor-product form for a patch built
from two families of conics.

#### Full weight matrix

```
        ⎡  1     w01    1   ⎤
W  =    ⎢ w10  w10·w01  w12 ⎥
        ⎣  1     w21    1   ⎦
```

Implemented as `dupin_patch_bezier` in the notebook (Cell 4), with
`_arc_bezier` for the per-edge conic.

Because the corners *are* control points, a degree-2 patch interpolates them
exactly — no corner absorbs the accumulated error, so the gap disappears
structurally rather than being reduced:

| | Lie-sphere | Bézier |
|---|---|---|
| corner reproduction | mean 1.0e-03, worst 2.6e-01 | **0.000e+00, all 441 vertices** |
| vertex 113 | 0.259 (degenerate) | exact |
| failure modes | division by `Δx·n → 0` | none (polynomial evaluation) |
| differentiability | hard (nested Lie algebra) | rational polynomial |

The two agree where the Lie-sphere version is healthy: at vertex 211 (smallest
Lie gap, 1.8e-04) the surfaces differ by 0.89% of the patch diagonal — so the
Bézier form reproduces the same geometry, not a different surface. The
remaining Property (d) residual shows up as a small deviation of the patch
from an exact cyclide rather than as a corner gap.

## Result (2026-08-12)

Final numbers via the audit harness above, run through the actual
`dupin_quad_callback` module (not a standalone reimplementation):

| mesh | rects | keymis | nonrect | nonplanar | outside | bowtie |
|---|---|---|---|---|---|---|
| `cone_mesh_small.obj` | 72/72 | 0 | 0 | 0 | **0** | 0 |
| `cone_mesh_0.obj` | 484/484 | 0 | 0 | 0 | **0** | 0 |
| `cone_mesh_small2.obj` | 300/300 | 0 | 0 | 0 | **0** | 0 |

Full grid coverage and 0 outside-face on every test mesh, against the pre-fix
baseline of 98/484 and 66/300.

### Deviations from the plan text above, both from mid-implementation
### corrections by the user — see [[project_dupin_propagation_redesign]]

**Transverse-strip identification is not row/column-based.** The plan's
worked description used `grid_faces[i, :]` / `grid_faces[:, j]` directly. That
depends on `grid_coordinates` happening to put the U-arm at row 0 — true by
construction for the *seed's own* strips, but `Mesh.quad_face_strips`'
`(u_faces, v_faces)` naming turned out to be the reverse of that convention
(verified on `cone_mesh_small.obj`: its first list is the grid column, not the
row). Per instruction, the algorithm instead identifies each arm step's
transverse strip **by membership** — at the newly-placed face, whichever of
`quad_face_strips`' two lists contains the already-placed predecessor is the
along-arm strip, so the other is transverse — with no dependency on row/column
position or a fixed U/V label. See `propagate_rectangles`'
`transverse_strip_at` closure in `dupin.py`.

**`bisecting_plane`'s normal formula changed**, not just the propagation
driver. The original `unit(n_i - n_j)` degenerates as two faces approach
coplanarity (n_i - n_j -> 0, direction dominated by float noise) — this
happens for real on `cone_mesh_0.obj` (two faces at n_i . n_j = 0.99999774)
and blocked 246 interior cells before the fix. Per instruction, the normal is
now `cross(unit(n_i + n_j), edge_direction)`: `n_i + n_j` stays well-
conditioned all the way to `n_i == n_j`, and crossing with the edge direction
keeps the plane through the shared edge by construction. Verified equivalent
to the old formula (anti-parallel normals, same plane) everywhere it isn't
degenerate. The exact-coplanar guard in `reflect_edge_frames` was also
loosened to only fire on `n_i . n_j == -1` (faces folded flat onto each
other, where `n_i + n_j` genuinely vanishes) rather than `== +1`, which is now
the well-handled case.

The plan named `reflect_rectangle_rigid` as unused once this landed; it was kept
and is still in `dupin.py`, as are `gram_schmidt_average`, `set_rectangle_edge`,
`create_rectangle_inside_face` and `build_edge_aligned_rectangle`, which the
propagation path no longer calls. `create_rectangle_from_frame` was removed in the
2026-09 geometry split.
