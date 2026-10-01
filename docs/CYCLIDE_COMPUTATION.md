# Cyclide computation

> **Code location (2026-09):** `Dupin_patch` and the rectangle propagation live in
> `LMesh/utils/dupin.py`; `dupin_patch_bezier` is defined in
> `notebooks/dupin_quad_test.ipynb`, which is also the entry point (the
> `Code/Dupin_Propagation` app was removed).

Two constructions of a cyclidic patch are implemented, from the same input
data — four (nearly) concircular corner points and a normal at each:

| | `Dupin_patch` | `dupin_patch_bezier` |
|---|---|---|
| basis | Lie-sphere envelope (Bobenko & Huhnen-Venedey) | rational biquadratic Bézier (Pottmann et al.) |
| corner reproduction | `x0, x1, x2` exact; **`x12` recovered** | **all four exact** |
| failure mode | division by `Δx·n → 0` | none (polynomial evaluation) |

The full pipeline that feeds them — propagation, optimization, and the patch
generation itself — is described in
[PROPAGATION_REDESIGN.md](PROPAGATION_REDESIGN.md). This document covers the
patch mathematics.

## A. Lie-sphere construction (`Dupin_patch`)

A cyclidic patch is determined by four concircular points
$(x_0, x_1, x_{12}, x_2)$ and one orthonormal frame $(t^{(1)}, t^{(2)}, n_0)$ at
the anchor $x_0$ (Bobenko & Huhnen-Venedey, arXiv:1101.5955, Thm. 2.14). The
frames at the other three corners are not independent data — they are obtained
from $n_0$ by reflecting across the plane perpendicularly bisecting the segment
between $x_0$ and the target corner (Eq. 2.11):

$$
n_i \;=\; n_0 - 2\,\frac{\langle x_i - x_0,\, n_0\rangle}{\langle x_i - x_0,\, x_i - x_0\rangle}\,(x_i - x_0), \qquad i \in \{1, 2\}.
$$

Each $n_i$ defines a *boundary sphere* $s^{(i)}$ tangent to the surface at
$x_0$ and passing through $x_i$ (Eq. 2.10), and a *shifted* boundary sphere at
the far corner using the shifted normal. Two *cyclidic families* of spheres
are built from these boundary spheres plus a third, derived sphere $\sigma^{(i)}$
(Eq. 2.15); evaluating each family and intersecting them (`envelope_point`)
samples the patch.

**Corner reproduction is asymmetric.** $x_0$, $x_1$, $x_2$ are exact by
construction, but $x_{12}$ is *recovered* — it is where the two cyclidic
families must meet at parameter $(1,1)$ — so it absorbs whatever error the
input data carries. On `cone_mesh_0.obj` the gap is mean 5.9e-04 / worst
8.3e-02 after optimization. The worst cases are a numerical degeneracy:
`boundary_lie_sphere_cyclide` computes $\|\Delta x\|^2 / (2\,\Delta x \cdot n)$,
which blows up as $\Delta x \cdot n \to 0$.

## B. Rational biquadratic Bézier construction (`dupin_patch_bezier`)

Pottmann et al., *Circular Arc Structures* (Bo, Pottmann, Kilian, Wang,
Wallner, SIGGRAPH 2011), proof of Prop. 8: cyclidic patches have degree
$(2,2)$, with $b_{11}$ at the intersection of the tangent planes at the four
boundary vertices.

$$
f(s,t) \;=\; \frac{\sum_{i,j} w_{ij}\, b_{ij}\, B_i^2(s)\, B_j^2(t)}
                  {\sum_{i,j} w_{ij}\, B_i^2(s)\, B_j^2(t)},
\qquad B_i^k(t) = \binom{k}{i}(1-t)^{k-i} t^i .
$$

- **Corners** $b_{00}, b_{20}, b_{22}, b_{02}$ are the four cell corners
  directly, with weight 1 — so a degree-2 patch interpolates them *exactly*.
  No corner is recovered, hence no $x_{12}$ gap.
- **Edge control points** come from each boundary being a circular arc, built
  from the arc's own circle rather than by intersecting endpoint tangents (see
  below).
- **Interior control point** $b_{11}$ solves $n_i \cdot x = n_i \cdot p_i$ for
  the four corners in least squares — exact when the cell is truly cyclidic.

Full formulas, including the weight matrix, are in
[PROPAGATION_REDESIGN.md](PROPAGATION_REDESIGN.md#b-rational-biquadratic-bézier-construction-dupin_patch_bezier).

### Why the arc control point is not a tangent intersection

The textbook construction puts $b_1$ at the intersection of the tangent lines
at the two arc endpoints. That is **ill-conditioned on a conical mesh**:
adjacent corner normals are nearly equal, so the tangents are nearly parallel
and the intersection is undefined. Measured on `cone_mesh_0.obj`, *every* one
of the 441 interior vertices had $|t_0 \cdot t_2| > 0.99$; the least-squares
intersection put $b_1$ up to **882× the chord length** away from the arc
(vertex 36), and ~1.5–1.9× at vertices 202/203 — the visible far-off points in
those patches.

Building the arc's circle instead is well conditioned everywhere: the centre
lies along the in-plane direction perpendicular to the tangent at $p_0$, at
the distance that makes it equidistant from both endpoints. That reduced the
worst offset to 0.059 and the mean to 0.037, with exact endpoint
reproduction, and correctly treats near-straight edges as straight instead of
diverging.

## Implementation

`notebooks/dupin_quad_test.ipynb` builds the four corner frames per grid vertex by
propagating rectangles across the mesh (`propagate_rectangles` in
`LMesh/utils/dupin.py`), then passes them to `Dupin_patch` or
`dupin_patch_bezier` (defined in the notebook). The notebook has five cells:
initialization / energy terms / seed + sliders / full pipeline / export.

### Current propagation (as of 2026-08-14)

The seed rectangle is centred on the corner face's **barycentre**
(`seed_rectangle_at_barycentre`), with `u` pointing toward the U-arm's first
neighbour and `v` toward the V-arm's (`seed_direction_towards`). Its sides
`h_u`, `h_v` are each bounded by the tightest edge on the strip they will
later be reflected across (`strip_min_edge_length`), then scaled by their own
slider; `theta` rotates the frame and `offset_u`/`offset_v` move the centre
within the face, clamped so it stays inside.

Propagation (`propagate_rectangles`) then reflects the **shared edge only**
across the bisecting plane, and re-sizes the free perpendicular dimension per
face from the transverse strip it crosses. The two boundary arms are walked
first, then the interior, where each cell reflects the two edges leaving its
already-placed predecessors and averages their shared corner.

This replaced an earlier *rigid* reflection (`reflect_rectangle_rigid`), which
kept every rectangle congruent to the seed and therefore could not adapt to
faces of differing size — it left 20–22% of rectangles outside their own face
on the larger meshes. See
[PROPAGATION_REDESIGN.md](PROPAGATION_REDESIGN.md) for the full algorithm,
the acceptance checks, and the optimization that tightens the accumulated
propagation error.

### Corner indexing, and the invariant that guards it

Rectangle corners are stored as `frames[(face_id, vertex_id)]`, so each corner is
the "interior version" of one of its face's vertices. Two silent bugs came from
getting this assignment wrong, and both were invisible in the render until
rectangles began straddling mesh edges:

- `reflect_edge_frames` keyed each reflected corner under the *source* vertex id.
  Reflection across the bisector carries a corner to the **far** side of the
  shared edge, so the corner keyed `v_11` physically landed near `v_21`.
- `build_edge_aligned_rectangle` assigned seed corners by walking the face loop
  in step with its sign pattern, assuming the rectangle winds the same way as
  the face. `v = n x u` need not follow the loop's orientation.

Both are fixed by assigning each corner to the vertex it actually lands nearest
(`_match_corners_to_vertices`, which picks the best rotation/winding and so
guarantees four *distinct* vertices — per-corner nearest-vertex can
double-assign on a small or skewed rectangle).

**Invariant to re-check after any change to propagation:** the corner stored at
`(f, v)` must be the one nearest vertex `v`. The `audit()` harness in
[PROPAGATION_REDESIGN.md](PROPAGATION_REDESIGN.md#2-whole-mesh-invariants)
checks it (`keymis` count) alongside rectangularity, planarity, containment
and bowties.

Note that the reflection math itself was never at fault.
`reflect_frame_across_plane` mirrors directions about the displacement
`d = p' - p` rather than about the bisector normal; all four corners' `d` come
out parallel, so it is a single rigid motion.

### Resolved: congruence versus containment

Two earlier approaches each failed one half of this trade-off:

- **Rigid reflection** (`reflect_rectangle_rigid`) kept every rectangle
  congruent to the seed, so it could not adapt to faces of differing size —
  0/72 rectangles outside their face on `cone_mesh_small.obj`, but 98/484 on
  `cone_mesh_0.obj` and 66/300 on `cone_mesh_small2.obj`.
- **Per-cell `grid_W`/`grid_H`** (via `set_rectangle_edge` /
  `create_rectangle_from_frame`) bought containment but gave up congruence, and
  its corner-anchored Gram-Schmidt construction for interior cells drifted
  badly — interior rectangles rotated up to 42° off their faces, against ~2.5°
  for the reflected arms. `recenter_rectangle_in_face` only translates, so it
  cannot help where a rectangle is genuinely larger than its face.

The current method resolves both: reflect the shared edge only (preserving
edge-compatibility with the neighbour) and re-size the free perpendicular
dimension per face from the strip it crosses. All three test meshes now pass
with **0 rectangles outside their face** and full grid coverage — see the
Result table in [PROPAGATION_REDESIGN.md](PROPAGATION_REDESIGN.md).

## Bug: wrong normal source

`Dupin_patch` read $n_1, n_2$ directly from the propagated frames (`F1["n"]`,
`F2["n"]`) instead of computing them via Eq. 2.11. The propagated normals are
reflections across the **mesh-face bisecting plane** (normal of face $i$ minus
normal of face $j$, `bisecting_plane` in `LMesh/utils/dupin.py`) — a plane
determined by the two adjacent mesh faces, not by the two points $x_0, x_i$.

These two planes coincide only on an exactly *conical* mesh, where each
vertex's four adjacent face normals meet the classical angle-sum condition. On
a mesh that only approximates this (`compute_conical` measured
$0.6°$–$1.6°$ error per vertex on `cone_mesh_small.obj`), the two normals
diverge by a comparable angle. `Dupin_patch`'s construction is sensitive to
this: the anchor corner $x_0$ always reproduced exactly, but $x_1, x_2, x_{12}$
were off by up to $\sim$15% of the patch size — the visible corner/circle
mismatch.

## Fix

`Dupin_patch` now derives $n_1, n_2$ internally via Eq. 2.11 from $n_0$ and the
four point positions, rather than trusting the stored frame normals. Verified
on `cone_mesh_small.obj` (56 interior vertices): corner-reproduction error
dropped from mean 0.03–0.07 (worst case 0.12–0.15, comparable to the patch
size) to mean 0.003, median 0.0024, max 0.015 — now tracking the mesh's
residual non-concircularity rather than a construction error.

## Why a local propagation fix does not work

A natural next step is correcting the stored frame normal at propagation time
(so `Dupin_patch` could trust it directly), using the same Eq. 2.11 reflection
from an already-placed diagonal neighbor. This was attempted and reverted:

- The four corner points around a vertex are supplied by the mesh/rectangle
  propagation, not derived from each other — the paper's construction assumes
  they are *already* concircular and only reflects frames between them. There
  is no valid formula for inferring a fourth position from three others
  (an initial attempt assumed the diagonal pair were diametrically opposite on
  the fitted circle; this is not implied by concircularity and produced wild
  results on near-collinear triples).
- Correcting only the *normal* at one corner still perturbs that corner's
  $u, v$ tangent basis, which is used by `create_rectangle_from_frame` to
  place the *other* corners of the same rectangle. Those corners feed later
  sweep steps, so a locally-correct fix propagates inconsistency sideways
  through the grid rather than removing it — mean/max corner error increased
  ~20$\times$ in testing rather than improving.

The propagation is not locally decomposable: each frame correction has
non-local effects on the rest of the grid. Tightening it requires solving for
all frames jointly (an optimization enforcing per-vertex concircularity and/or
conicality across the whole grid at once), not a sequential per-vertex patch.
