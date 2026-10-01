"""
Dupin cyclide patches and rectangle propagation over quad meshes.

Dupin_patch builds a cyclide patch from four Lie spheres; the rectangle
functions seed a rectangle in one face and propagate congruent rectangles
across the mesh by reflection (see docs/PROPAGATION_REDESIGN.md).

Moved out of hanan.geometry.{geo_processing, algebraic} — research code built on hanan.
"""

import numpy as np
from collections import defaultdict, deque

from hanan.geometry.lie import (
    boundary_lie_sphere_cyclide,
    cyclidic_family_lie,
    envelope_point,
    get_tangent_vector,
    lie_inner_prod,
    lie_midpoint,
    map_lie_s_to_sphere,
    normalize_lie_sphere,
)
from hanan.geometry.primitives import (
    clip_line_to_convex_face,
    plane_normal,
    plane_plane_intersection,
    reflect_point_plane,
)
from hanan.geometry.algebraic import (
    make_param_grid_vec,
    orth_proj,
    rotation_matrix,
    unit,
)


class DupinPropagationError(RuntimeError):
    """Raised when a frame cannot be reflected and propagation must stop."""


def reflect_frame_across_plane(frame, plane_normal, plane_distance, face=None):
    """
    Reflect a whole frame across the plane n · x + h = 0.

    The point is reflected across the plane; the directions (u, v, n) are then
    reflected in the plane orthogonal to the displacement d = pt_reflected - pt.
    Reflecting the directions about d rather than about the plane normal keeps
    the frame consistent with the reflected point when the two faces meet at an
    angle (this is the construction used for conical meshes).

    Args:
        frame: Frame dict {"pt", "u", "v", "n"}.
        plane_normal: Unit normal of the reflection plane (3,).
        plane_distance: Plane offset h in n · x + h = 0.
        face: Optional face index to tag the result with.

    Returns:
        Reflected frame dict {"pt", "u", "v", "n"} (plus "f" if face is given).

    Raises:
        DupinPropagationError: If the point lies on the plane, leaving the
            displacement zero so the directions cannot be reflected.
    """
    pt = np.asarray(frame["pt"], dtype=float)
    ref_pt = reflect_point_plane(pt, plane_normal, plane_distance)

    d = ref_pt - pt
    dd = d @ d
    if dd < 1e-20:
        raise DupinPropagationError(
            "Cannot reflect frame: the point lies on the reflection plane, so "
            "the displacement is zero and the directions are undetermined."
        )

    def _reflect(vec):
        vec = np.asarray(vec, dtype=float)
        return vec - 2.0 * (d @ vec) / dd * d

    out = {
        "pt": ref_pt,
        "u": _reflect(frame["u"]),
        "v": _reflect(frame["v"]),
        "n": _reflect(frame["n"]),
    }
    if face is not None:
        out["f"] = int(face)
    return out


def gram_schmidt_average(frame_u, frame_v):
    """
    Average two frames, re-orthonormalizing the result with Gram-Schmidt.

    Points and normals are averaged directly; u is averaged then projected off
    the averaged normal, and v is recovered as n x u with its orientation fixed
    to match the averaged v.

    Args:
        frame_u: First frame dict {"pt", "u", "v", "n"}.
        frame_v: Second frame dict {"pt", "u", "v", "n"}.

    Returns:
        Averaged orthonormal frame dict {"pt", "u", "v", "n"}.

    Raises:
        DupinPropagationError: If the averaged normal or u vector degenerates,
            which happens when the two frames are opposed.
    """
    pt_avg = (np.asarray(frame_u["pt"], float) + np.asarray(frame_v["pt"], float)) / 2.0

    n_avg = (np.asarray(frame_u["n"], float) + np.asarray(frame_v["n"], float)) / 2.0
    n_norm = np.linalg.norm(n_avg)
    if n_norm < 1e-12:
        raise DupinPropagationError(
            "Cannot average frames: the two normals are opposed, so the mean "
            "normal vanishes."
        )
    n_avg = n_avg / n_norm

    u_rough = (np.asarray(frame_u["u"], float) + np.asarray(frame_v["u"], float)) / 2.0
    v_rough = (np.asarray(frame_u["v"], float) + np.asarray(frame_v["v"], float)) / 2.0

    u_projected = u_rough - np.dot(u_rough, n_avg) * n_avg
    u_norm = np.linalg.norm(u_projected)
    if u_norm < 1e-12:
        raise DupinPropagationError(
            "Cannot average frames: the mean u direction is parallel to the "
            "mean normal, so the tangent frame is undetermined."
        )
    u_new = u_projected / u_norm

    v_new = np.cross(n_avg, u_new)
    if np.dot(v_new, v_rough) < 0:
        v_new = -v_new

    return {"pt": pt_avg, "u": u_new, "v": v_new, "n": n_avg}


def Dupin_patch(F0, F1, F12, F2, N=10, M=10):
    """
    Sample an N × M grid on the Dupin cyclide patch bounded by four surface frames.

    Args:
        F0: Frame dict {"pt", "u", "v", "n"} at corner 0.
        F1: Frame dict at corner 1.
        F12: Frame dict at corner 1-2 (diagonal).
        F2: Frame dict at corner 2.
        N: Grid resolution in the first parameter direction (default: 10).
        M: Grid resolution in the second parameter direction (default: 10).

    Returns:
        envelope_points: Sampled surface points (N*M, 3).
        face_grid: Quad face connectivity (N*M, 4).
    """
    from hanan.geometry.io import mathematica_v  # import locally to avoid circular dependencies

    x0,  u0,   v0,  n0  = F0["pt"],  F0["u"],  F0["v"],  F0["n"]
    x1,  u1,   v1,  n1  = F1["pt"],  F1["u"],  F1["v"],  F1["n"]
    x12, u12, v12, n12 = F12["pt"], F12["u"], F12["v"], F12["n"]
    x2,  u2,   v2,  n2  = F2["pt"],  F2["u"],  F2["v"],  F2["n"]

    t01 = get_tangent_vector(x0, x1, u0, v0)
    t02 = get_tangent_vector(x0, x2, u0, v0)

    # Shifted normals n1, n2 for the shifted boundary spheres s2_1, s1_2 (Eq. 2.11,
    # Bobenko & Huhnen-Venedey arXiv:1101.5955): reflect n0 — the frame normal at the
    # anchor corner x0 — across the bisecting plane of (x0, x1) resp. (x0, x2). This is
    # NOT the same as the frame normals n1, n2 stored on F1, F2: those come from whatever
    # independent construction placed those frames (e.g. mesh-driven propagation), and
    # using them in place of the Eq. 2.11 reflection breaks the patch's interpolation
    # property — it then only reproduces the anchor corner x0, not x1/x2/x12.
    def _reflect_normal(n, x, xi):
        d = xi - x
        return unit(n - 2 * (d @ n) / (d @ d) * d)

    n1_shift = _reflect_normal(n0, x0, x1)
    n2_shift = _reflect_normal(n0, x0, x2)

    # # Compute center of the cyclide
    # c, r, n = circle_3pts(x0, x1, x2)

    # # Fix orientation if needed 
    # barycenter = c[0]

    # # Check that u, v are pointin thowards the barycenter
    # if np.dot(u0, barycenter - x0) < 0:
    #     u0 = -u0
    # if np.dot(u1, barycenter - x1) < 0:
    #     u1 = -u1
    # if np.dot(u12, barycenter - x12) < 0:
    #     u12 = -u12
    # if np.dot(u2, barycenter - x2) < 0:
    #     u2 = -u2
    # if np.dot(v0, barycenter - x0) < 0:
    #     v0 = -v0
    # if np.dot(v1, barycenter - x1) < 0:
    #     v1 = -v1
    # if np.dot(v12, barycenter - x12) < 0:
    #     v12 = -v12
    # if np.dot(v2, barycenter - x2) < 0:
    #     v2 = -v2    

    # Define boundary spheres in Lie representation
    s1   = boundary_lie_sphere_cyclide(x0, x1,  n0)
    s2   = boundary_lie_sphere_cyclide(x0, x2,  n0)
    s1_2 = boundary_lie_sphere_cyclide(x2, x12, n2_shift)
    s2_1 = boundary_lie_sphere_cyclide(x1, x12, n1_shift)
    
    t1_12 = get_tangent_vector(x1, x12, u1, v1)
    t2_12 = get_tangent_vector(x2, x12, u2, v2)
    
    y1 = lie_midpoint(x0, x1, t01)
    y2 = lie_midpoint(x0, x2, t02)

    y1_12 = lie_midpoint(x1, x12, t1_12)
    y2_12 = lie_midpoint(x2, x12, t2_12)

    sigma2 = normalize_lie_sphere(lie_inner_prod(y1, s1_2)*s1 - lie_inner_prod(s1, s1_2)*y1)
    sigma1 = normalize_lie_sphere(lie_inner_prod(y2, s2_1)*s2 - lie_inner_prod(s2, s2_1)*y2)

    cyclidic_f1 = cyclidic_family_lie(s1, sigma1, s1_2)
    cyclidic_f2 = cyclidic_family_lie(s2, sigma2, s2_1)

    # Normalize sigma
    T, F, _, _ = make_param_grid_vec(I1=(0, 1), I2=(0, 1), n=N, m=M)

    # p0  = map_lie_s_to_sphere(envelope_point(cyclidic_f1, cyclidic_f2, 0, 0))[0]  # Test envelope point
    # p1  = map_lie_s_to_sphere(envelope_point(cyclidic_f1, cyclidic_f2, 1, 0))[0]  # Test envelope point
    # p2  = map_lie_s_to_sphere(envelope_point(cyclidic_f1, cyclidic_f2, 0, 1))[0]  # Test envelope point
    # p12 = map_lie_s_to_sphere(envelope_point(cyclidic_f1, cyclidic_f2, 1, 1))[0]  # Test envelope point

    # # Map back y1, y2
    # py1 = map_lie_s_to_sphere(y1)[0]  # Test envelope point
    # py2 = map_lie_s_to_sphere(y2)[0]  # Test envelope point

    # m1_12 = map_lie_s_to_sphere(y1_12)[0]  # Test envelope point
    # m2_12 = map_lie_s_to_sphere(y2_12)[0]  # Test envelope point

    # # Draw circles p0, y1 ,p1 
    # c1, r1, n1 = circle_3pts(x0, py1, x1)
    # c2, r2, n2 = circle_3pts(x0, py2, x2)

    # c1_12, r1_12, n1_12 = circle_3pts(x1, m1_12, x12)
    # c2_12, r2_12, n2_12 = circle_3pts(x2, m2_12, x12)

    # print(f"{(x0 - c1)[0]} n {n1[0]}")
    # # Tangents at x0
    # print(f"x0 t1.t2 : {unit(np.cross((x0 - c1)[0], n1[0]))@ unit(np.cross((x0 - c2)[0],n2[0]))}")
    # # Tangents at x1
    # print(f"x1 t1.t2 : {unit(np.cross((x1 - c1)[0],n1[0]))@ unit(np.cross((x1 - c1_12)[0],n1_12[0]))}")
    # # Tangents at x2
    # print(f"x2 t1.t2 : {unit(np.cross((x2 - c2)[0],n2[0]))@ unit(np.cross((x2 - c2_12)[0],n2_12[0]))}")
    # # Tangents at x12
    # print(f"x12 t1.t2 : {unit(np.cross((x12 - c1_12)[0],n1_12[0]))@ unit(np.cross((x12 - c2_12)[0],n2_12[0]))}")

    
    # draw_circle(c1, n1, r1, id="Circle 01", color=(0.3, 0, 0.3), radius=0.001)
    # draw_circle(c2, n2, r2, id="Circle 02", color=(  1, 1, 1), radius=0.001)
    # draw_circle(c1_12, n1_12, r1_12, id="Circle 12", color=(0.3, 0.3, 0), radius=0.001)
    # draw_circle(c2_12, n2_12, r2_12, id="Circle 21", color=(0.0, 0.8, 0.4), radius=0.001)

    #draw_polysphere(py1, 0.001, name="y1", color=[0.5, 0.5, 0.0], opacity=0.6)
    # Draw s1,s2,s12,s21
    # draw_polysphere(map_lie_s_to_sphere(s1)[0], map_lie_s_to_sphere(s1)[1], name="s1", color=[0.0, 0.0, 1.0], opacity=0.6)
    # draw_polysphere(map_lie_s_to_sphere(s2)[0], map_lie_s_to_sphere(s2)[1], name="s2", color=[1.0, 0.0, 0.0], opacity=0.6)
    #draw_polysphere(map_lie_s_to_sphere(s1_2)[0], map_lie_s_to_sphere(s1_2)[1], name="s1_2", color=[1.0, 1.0, 0.0], opacity=0.6)
    #draw_polysphere(map_lie_s_to_sphere(s2_1)[0], map_lie_s_to_sphere(s2_1)[1], name="s2_1", color=[0.0, 1.0, 0.0], opacity=0.6)

    # Evaluate the cyclidic families at the grid points
    envelope_points = np.array([map_lie_s_to_sphere(envelope_point(cyclidic_f1, cyclidic_f2, t1, t2))[0] for t1, t2 in T])

    #print(f"Envelope points shape: {envelope_points}")

    return envelope_points, F


def dupin_patch_boundary_curves(envelope_points, N, M):
    """
    The four boundary curves of a Dupin_patch grid, keyed by which pair of
    ring positions (0=x0, 1=x1, 2=x12, 3=x2 — the order Dupin_patch's inputs
    F0, F1, F12, F2 are given in) they run between.

    Dupin_patch samples row-major with t1 fast-varying: point k has
    (t1, t2) = (k % N, k // N). Corner (0,0)=x0, (N-1,0)=x2, (0,M-1)=x1,
    (N-1,M-1)=x12, so:
      - (0,1) [x0->x1]:  t1=0 column
      - (0,3) [x0->x2]:  t2=0 row
      - (1,2) [x1->x12]: t2=M-1 row
      - (2,3) [x12->x2]: t1=N-1 column

    Reading a boundary curve straight from the sampled grid is exact by
    construction (it IS the surface); reconstructing it from a normal at
    each endpoint (circular_arc_from_points_normals) is only exact for the
    two corners touching x0 (n0 is trusted, n1/n2 are exact reflections of
    it) — for the far edge x1->x12 or x2->x12 there is no exact normal at
    x12, only an averaged approximation, and the reconstructed arc measurably
    bulges off the true surface (~0.005-0.01 on cone_mesh_small.obj). Use
    this function instead wherever the real patch boundary is needed.

    Args:
        envelope_points: (N*M, 3) array returned by Dupin_patch.
        N: Grid resolution passed to Dupin_patch (t1 direction).
        M: Grid resolution passed to Dupin_patch (t2 direction).

    Returns:
        dict mapping frozenset({a, b}) for each adjacent ring-position pair
        (a, b) in {(0,1), (0,3), (1,2), (2,3)} to that boundary's points,
        ordered from the lower-indexed corner to the higher one.
    """
    envelope_points = np.asarray(envelope_points)
    return {
        frozenset((0, 1)): envelope_points[0::N],
        frozenset((0, 3)): envelope_points[0:N],
        frozenset((1, 2)): envelope_points[(M - 1) * N: M * N],
        frozenset((2, 3)): envelope_points[N - 1::N],
    }


def circular_arc_from_points_normals(p1, n1, p2, n2, n_samples=20):
    """
    Circular arc through p1, p2 whose center is the intersection of the two
    inward-normal rays p1 + t*n1 and p2 + s*n2.

    This is the boundary-circle construction of Def. 2.9 / Eq. 2.10
    (Bobenko & Huhnen-Venedey arXiv:1101.5955): p1, p2 are two points of a
    Dupin cyclide's boundary curve and n1, n2 the surface normals there. On
    the exact construction the two rays meet exactly (n2 is n1 reflected
    across the perpendicular bisecting plane of [p1, p2] — see Dupin_patch's
    n1_shift/n2_shift); this function also tolerates a near-miss (e.g. two
    independently-propagated frames) by using the closest-point intersection.

    Args:
        p1: First point (3,).
        n1: Inward normal at p1 (3,).
        p2: Second point (3,).
        n2: Inward normal at p2 (3,).
        n_samples: Number of samples along the arc (default: 20).

    Returns:
        Arc points (n_samples, 3), ordered from p1 to p2.

    Raises:
        DupinPropagationError: If the two normals are (nearly) parallel, so
            their rays do not determine a circle center.
    """
    from hanan.geometry.primitives import circle_3d

    p1 = np.asarray(p1, float)
    p2 = np.asarray(p2, float)
    n1 = np.asarray(n1, float)
    n2 = np.asarray(n2, float)

    # Closest-point intersection of the two rays p1 + t*n1 and p2 + s*n2.
    A = np.stack([n1, -n2], axis=1)
    b = p2 - p1
    sol, *_, sv = np.linalg.lstsq(A, b, rcond=None)
    if sv[-1] < 1e-9:
        raise DupinPropagationError(
            "Cannot build boundary arc: the two normals are (nearly) "
            "parallel, so their rays do not determine a circle center."
        )
    t, s = sol
    center = ((p1 + t * n1) + (p2 + s * n2)) / 2.0

    radius = (np.linalg.norm(p1 - center) + np.linalg.norm(p2 - center)) / 2.0
    plane_normal = unit(np.cross(p1 - center, p2 - center))

    full_circle = circle_3d(center, plane_normal, radius, pts=max(4 * n_samples, 200))
    i1 = int(np.argmin(np.linalg.norm(full_circle - p1, axis=1)))
    i2 = int(np.argmin(np.linalg.norm(full_circle - p2, axis=1)))

    m = len(full_circle)
    fwd = (i2 - i1) % m
    bwd = (i1 - i2) % m
    if fwd <= bwd:
        idx = [(i1 + k) % m for k in range(fwd + 1)]
    else:
        idx = [(i1 - k) % m for k in range(bwd + 1)]

    arc = full_circle[idx]
    sample_idx = np.linspace(0, len(arc) - 1, n_samples).round().astype(int)
    return arc[sample_idx]


def ruled_strip(arc_i, arc_j):
    """
    Quad mesh of the ruled strip connecting two point-sampled curves with
    straight lines, point-for-point (arc_i[k] to arc_j[k]).

    Mirrors cone_strip's structure but for two arbitrary polylines of equal
    length rather than two circles — used to join the matching boundary arcs
    of two Dupin cyclide patches along a shared mesh edge.

    Args:
        arc_i: First curve, ordered (K, 3).
        arc_j: Second curve, ordered (K, 3), same length and correspondence
            as arc_i (arc_j[k] is connected to arc_i[k]).

    Returns:
        vertices: Strip vertex positions (2*K, 3).
        faces: Quad face index lists.
    """
    arc_i = np.asarray(arc_i)
    arc_j = np.asarray(arc_j)
    if len(arc_i) != len(arc_j):
        raise ValueError(
            f"ruled_strip: arcs must have the same length, got {len(arc_i)} "
            f"and {len(arc_j)}."
        )

    K = len(arc_i)
    vertices = np.concatenate((arc_i, arc_j), axis=0)
    faces = [[k, k + 1, K + k + 1, K + k] for k in range(K - 1)]

    return vertices, faces


def build_edge_table(F):
    """
    Map every undirected edge of a quad mesh to the faces that use it.

    Args:
        F: Quad face array (nF, 4) of vertex indices in CCW order.

    Returns:
        dict: key (min(v, w), max(v, w)) -> list of (face_id, local_edge_idx).
    """
    edge_table = defaultdict(list)
    for f_id, face in enumerate(F):
        vs = list(face) + [face[0]]          # close the loop
        for e_idx in range(len(face)):
            a, b = vs[e_idx], vs[e_idx + 1]
            edge_table[tuple(sorted((a, b)))].append((f_id, e_idx))
    return edge_table


def find_corner_faces(mesh, min_boundary_edges=2):
    """
    Faces sitting at a corner of the mesh boundary.

    A corner face is one owning at least *min_boundary_edges* edges that belong
    to no other face. The test is purely combinatorial, so it works for any
    polygon degree — but note that the quad-grid propagation built on top of it
    (grid_coordinates, propagate_*) is quad-only.

    On a clean rectangular quad grid this returns exactly four faces. On a mesh
    with an irregular boundary it may return more; the caller chooses.

    Args:
        mesh: Mesh object with .faces.
        min_boundary_edges: Boundary edges a face needs to count (default: 2).

    Returns:
        np.ndarray: Sorted face indices, possibly empty.
    """
    edge_faces = defaultdict(list)
    for f_id, face in enumerate(mesh.faces):
        face = list(face)
        n = len(face)
        for i in range(n):
            edge_faces[tuple(sorted((face[i], face[(i + 1) % n])))].append(f_id)

    boundary_count = defaultdict(int)
    for faces_on_edge in edge_faces.values():
        if len(faces_on_edge) == 1:
            boundary_count[faces_on_edge[0]] += 1

    return np.array(sorted(f for f, c in boundary_count.items()
                           if c >= min_boundary_edges), dtype=int)


def grid_coordinates(F, seed_face=0):
    """
    Lay a quad mesh out on an integer (row, col) grid by BFS across shared edges.

    Each face carries its own rotation relative to the seed, so the walk does not
    assume that local edge index 0 points the same way on every face — on a real
    .obj it does not, since vertex order within a face is arbitrary.

    Args:
        F: Quad face array (nF, 4) of vertex indices.
        seed_face: Face placed at grid origin (default: 0). Pass a corner face
            (see find_corner_faces) so the origin sits at a grid corner.

    Returns:
        np.ndarray: Grid of face indices, -1 where no face maps to a cell.

    Raises:
        ValueError: If the faces are not all quads.
    """
    F = np.asarray(F)
    if F.ndim != 2 or F.shape[1] != 4:
        raise ValueError(
            f"grid_coordinates requires a quad mesh; got faces of shape {F.shape}."
        )

    edge_tab = build_edge_table(F)
    nF = len(F)

    coords = np.full((nF, 2), np.nan)
    # rot[f] = how many quarter-turns face f is rotated relative to the seed.
    rot = np.full(nF, -1, dtype=int)
    delta = np.array([[1, 0],    # local edge 0
                      [0, 1],    # local edge 1
                      [-1, 0],   # local edge 2
                      [0, -1]])  # local edge 3

    seed_face = int(seed_face)
    q = deque([seed_face])
    coords[seed_face] = (0, 0)
    rot[seed_face] = 0

    while q:
        f = q.popleft()
        for e_idx in range(4):
            vA, vB = F[f, e_idx], F[f, (e_idx + 1) % 4]
            faces_sharing = edge_tab[tuple(sorted((vA, vB)))]
            if len(faces_sharing) != 2:
                continue        # boundary edge

            neigh, e_neigh = (faces_sharing[0] if faces_sharing[1][0] == f
                              else faces_sharing[1])
            if not np.isnan(coords[neigh, 0]):
                continue

            # Direction of travel in seed frame, then the neighbour's own
            # rotation: crossing an edge flips the traversal sense, so the
            # neighbour's local edge e_neigh points back the way we came.
            step_dir = (e_idx + rot[f]) % 4
            coords[neigh] = coords[f] + delta[step_dir]
            rot[neigh] = (step_dir + 2 - e_neigh) % 4
            q.append(neigh)

    coords = coords.astype(int)
    min_row, min_col = coords.min(0)
    max_row, max_col = coords.max(0)

    grid = np.full((max_row - min_row + 1, max_col - min_col + 1), -1, int)
    for f, (r, c) in enumerate(coords):
        grid[r - min_row, c - min_col] = f

    # Orient the grid so the seed sits at [0, 0]. The BFS walks whichever way
    # the seed's winding points, so it can land at any of the four corners.
    if grid[-1, 0] == seed_face or grid[-1, -1] == seed_face:
        grid = grid[::-1, :]
    if grid[0, -1] == seed_face:
        grid = grid[:, ::-1]
    return grid


def closest_vertex_on_face(mesh, point, face_id):
    """
    Vertex of a face nearest to a point.

    Args:
        mesh: Mesh object with .vertices and .faces.
        point: Query point (3,).
        face_id: Face index.

    Returns:
        tuple: (vertex position (3,), global vertex index).
    """
    face_vertices = mesh.vertices[mesh.faces[face_id]]
    k = int(np.argmin(np.linalg.norm(face_vertices - point, axis=1)))
    return face_vertices[k], int(mesh.faces[face_id][k])


def bisecting_plane(mesh, f_normals, face_i, face_j):
    """
    Bisecting plane between two faces sharing an edge.

    On a conical mesh, reflecting across this plane maps one face's tangent
    frame onto its neighbour's.

    The normal is built as cross(unit(n_i + n_j), edge_direction) rather than
    the more direct unit(n_i - n_j). Both give the same plane (the two normals
    come out anti-parallel) whenever face_i and face_j are not nearly
    coplanar, but n_i - n_j is the difference of two nearly-identical unit
    vectors as the faces approach coplanarity, so its direction is dominated
    by floating-point noise exactly where correctness matters most — this
    happened in practice on cone_mesh_0.obj (n_i . n_j = 0.99999774), where it
    silently produced a near-arbitrary bisecting plane instead of raising the
    coplanar guard downstream in reflect_edge_frames. n_i + n_j stays
    well-conditioned (norm -> 2, not 0) all the way to exact coplanarity, and
    crossing it with the edge direction keeps the plane passing exactly
    through the shared edge by construction.

    Args:
        mesh: Mesh object with .vertices.
        f_normals: Per-face normals (nF, 3).
        face_i: First face index.
        face_j: Second face index, sharing an edge with face_i.

    Returns:
        tuple: (unit normal (3,), offset h in n · x + h = 0, edge midpoint (3,)).
    """
    id_vk, id_vl = mesh.shared_edge_vertices(face_i, face_j)
    p_k, p_l = mesh.vertices[id_vk], mesh.vertices[id_vl]
    mid_point = (p_k + p_l) / 2.0

    n_i = np.asarray(f_normals[face_i], float)
    n_j = np.asarray(f_normals[face_j], float)
    edge_dir = unit(p_l - p_k)
    avg_n = unit(n_i + n_j)
    normal_bisec = unit(np.cross(avg_n, edge_dir))

    return normal_bisec, -float(np.dot(mid_point, normal_bisec)), mid_point


def reflect_edge_frames(mesh, f_normals, frames, face_i, face_j):
    """
    Reflect both frames on a shared edge from one face onto its neighbour.

    Writes the two reflected frames into *frames* under (face_j, vertex_id).

    Propagation stops here rather than continuing with an ill-defined frame: if
    the two faces are coplanar their bisecting plane degenerates, so there is no
    reflection that carries the frame across the edge.

    Args:
        mesh: Mesh object.
        f_normals: Per-face normals (nF, 3).
        frames: Frame dict keyed by (face_id, vertex_id); mutated in place.
        face_i: Source face index.
        face_j: Target face index, sharing an edge with face_i.

    Returns:
        The frames dict, mutated in place.

    Raises:
        DupinPropagationError: If the two faces are coplanar, or if a required
            source frame is missing.
    """
    id_vk, id_vl = mesh.shared_edge_vertices(face_i, face_j)

    for v_id in (id_vk, id_vl):
        if (face_i, v_id) not in frames:
            raise DupinPropagationError(
                f"Cannot reflect edge frames from face {face_i} to {face_j}: "
                f"no frame stored at vertex {v_id} of face {face_i}."
            )

    normal_i = unit(np.asarray(f_normals[face_i], float))
    normal_j = unit(np.asarray(f_normals[face_j], float))

    # n_i . n_j == 1 (faces facing the same way, i.e. "nearly coplanar" in the
    # everyday sense) is exactly the case bisecting_plane's cross(n_i+n_j, e)
    # construction handles robustly — n_i+n_j stays well-conditioned all the
    # way to n_i == n_j. What that construction genuinely cannot recover from
    # is n_i . n_j == -1 (n_i + n_j == 0, e.g. two faces folded flat back onto
    # each other along the shared edge): the averaged normal vanishes and
    # there is no well-defined bisecting plane.
    if np.isclose(normal_i @ normal_j, -1.0):
        raise DupinPropagationError(
            f"Cannot reflect frames from face {face_i} to face {face_j}: the "
            f"faces are folded flat against each other (n_i · n_j = "
            f"{normal_i @ normal_j:.6f}), so n_i + n_j vanishes and their "
            f"bisecting plane is undefined."
        )

    normal_bisec, offset_bisec, _ = bisecting_plane(
        mesh, f_normals, face_i, face_j)

    # The shared-edge corners keep their own vertex ids. This is forced by
    # combinatorics, not measured: the bisecting plane contains the shared
    # edge, so reflection across it FIXES that edge pointwise. Both faces meet
    # at v_id, v_id lies on the mirror, and the corner keyed there on face_i
    # therefore stays associated with v_id on face_j.
    #
    # An earlier version assigned each reflected corner to the face_j vertex
    # it landed *nearest*. That agrees with the combinatorial answer wherever
    # it works (verified: 3460 reflections across cone_mesh_small, _0,
    # _small2 and _h — zero disagreements), but it is a heuristic and it
    # fails where the rectangle has drifted off-centre in its face: on
    # cone_mesh_h at face 300 -> 330 both reflected corners came out nearest
    # the same vertex 342 (0.0284 vs 0.0292 for the correct one, a 3% margin)
    # and propagation aborted with 330 of 900 faces placed. Reading the
    # assignment off the topology has no such failure mode.
    for v_id in (id_vk, id_vl):
        frames[(face_j, int(v_id))] = reflect_frame_across_plane(
            frames[(face_i, v_id)], normal_bisec, offset_bisec, face=face_j)

    return frames


def _inward_shift(face_verts, normal, pts, max_passes=4, eps=1e-12):
    """
    Translation that pushes a point set back inside a convex face.

    For each face edge, find the worst (most outside) point and push the whole
    set inward by that amount, repeating a few passes since fixing one edge can
    reveal a violation at another. Size and orientation are untouched — only
    the common translation is returned.
    """
    face_verts = np.asarray(face_verts, dtype=float)
    pts = np.asarray(pts, dtype=float)
    m = len(face_verts)
    inside_ref = face_verts.mean(axis=0)
    n = unit(np.asarray(normal, dtype=float))

    total_shift = np.zeros(3)
    for _ in range(max_passes):
        worst_violation = 0.0
        push_dir = None
        for i in range(m):
            a, b = face_verts[i], face_verts[(i + 1) % m]
            n_edge = np.cross(n, b - a)
            if np.dot(n_edge, inside_ref - a) > 0:
                n_edge = -n_edge  # n_edge now points outward
            n_edge = unit(n_edge)

            d = (pts + total_shift - a) @ n_edge  # positive = outside this edge
            worst = d.max()
            if worst > worst_violation:
                worst_violation = worst
                push_dir = n_edge

        if worst_violation <= eps:
            break
        total_shift -= worst_violation * push_dir

    return total_shift


def _match_corners_to_vertices(corner_pts, face_verts):
    """
    Pair each rectangle corner with a distinct face vertex.

    A rectangle inside a quad face keeps the face's cyclic order, so the
    assignment is one of the four rotations of the loop (both windings are
    covered because a rotation of the reversed loop is tried too). Picking the
    rotation with the smallest total distance keeps the four corners distinct,
    which per-corner nearest-vertex does not guarantee: on a small or skewed
    rectangle two corners can share a nearest vertex, leaving another vertex
    with no corner at all.

    Args:
        corner_pts: The four corner points, in rectangle order.
        face_verts: The four face vertex positions, in face-loop order.

    Returns:
        list[int]: order[k] is the index into face_verts for corner k.
    """
    corner_pts = np.asarray(corner_pts, dtype=float)
    face_verts = np.asarray(face_verts, dtype=float)
    n = len(corner_pts)

    best_order, best_cost = None, np.inf
    for winding in (1, -1):
        base = list(range(n))[::winding]
        for shift in range(n):
            order = [base[(i + shift) % n] for i in range(n)]
            cost = sum(np.linalg.norm(corner_pts[k] - face_verts[order[k]])
                       for k in range(n))
            if cost < best_cost:
                best_cost, best_order = cost, order
    return best_order


def strip_min_edge_length(mesh, faces, direction, f_normals=None):
    """
    Minimum length, over *faces*, of the face edges aligned with *direction*.

    For each face, classifies its 4 edges as aligned or transverse by
    |unit(edge) . unit(direction projected into the face plane)| — the two
    edges with the larger absolute dot are the aligned pair. Returns the
    smallest length among all aligned edges of all faces, so a rectangle sized
    from this value cannot outgrow the tightest face on the strip.

    Args:
        mesh: Mesh object with .vertices and .faces.
        faces: Iterable of face indices (e.g. one arm of quad_face_strips).
        direction: World-space direction (3,) to align against.
        f_normals: Per-face normals (nF, 3); defaults to mesh.face_normals.

    Returns:
        float: Minimum aligned edge length, or inf if *faces* is empty.
    """
    if f_normals is None:
        f_normals = mesh.face_normals
    direction = np.asarray(direction, dtype=float)

    best = np.inf
    for f in faces:
        f = int(f)
        fv = mesh.vertices[mesh.faces[f]]
        n = unit(np.asarray(f_normals[f], dtype=float))
        d = unit(orth_proj(direction, n))

        m = len(fv)
        edges = [fv[(i + 1) % m] - fv[i] for i in range(m)]
        lens = [np.linalg.norm(e) for e in edges]
        scores = [abs(unit(e) @ d) for e in edges]

        # The two edges with the larger |dot| are the aligned pair.
        order = sorted(range(m), key=lambda i: scores[i], reverse=True)
        for i in order[:2]:
            best = min(best, lens[i])

    return best


def seed_direction_towards(mesh, f_normals, face_id, target_face):
    """
    In-plane unit vector at *face_id* pointing toward *target_face*.

    Direction from the face barycentre to the shared edge's midpoint,
    projected into the face plane. Gives the seed rectangle its orientation
    ("u toward F_1", "v toward F_m") without hand-picking a local edge index.

    Args:
        mesh: Mesh object with .vertices and .faces.
        f_normals: Per-face normals (nF, 3).
        face_id: Face to compute the direction at.
        target_face: Edge-adjacent face to point toward.

    Returns:
        np.ndarray: Unit vector (3,) in the plane of face_id.
    """
    n = unit(np.asarray(f_normals[face_id], dtype=float))
    barycentre = mesh.vertices[mesh.faces[face_id]].mean(axis=0)
    id_a, id_b = mesh.shared_edge_vertices(face_id, target_face)
    mid = (mesh.vertices[id_a] + mesh.vertices[id_b]) / 2.0
    return unit(orth_proj(mid - barycentre, n))


def seed_rectangle_at_barycentre(mesh, f_normals, face_id, u_dir, h_u, h_v,
                                 theta=0.0, offset_u=0.0, offset_v=0.0):
    """
    Build a rectangle centred on a face's barycentre with absolute side lengths.

    Corners are placed at centre +/- (h_u/2) u +/- (h_v/2) v, with
    v = cross(n, u_dir projected into the face plane). This is Step 1 of
    docs/PROPAGATION_REDESIGN.md, split out from build_edge_aligned_rectangle
    (which sizes from face-edge fractions instead of absolute lengths) so a
    caller can size the seed from strip_min_edge_length in world units.

    theta rotates u (and therefore v) about the normal before placing corners
    — a free extra degree of freedom, independent of h_u/h_v. offset_u/
    offset_v shift the centre away from the barycentre along u/v, each as a
    fraction of the corresponding side length (so +-0.5 reaches a full half
    side); the shift is then clamped so the rectangle stays inside the face,
    same as build_edge_aligned_rectangle's offset handling.

    Args:
        mesh: Mesh object with .vertices and .faces.
        f_normals: Per-face normals (nF, 3).
        face_id: Face to seed in.
        u_dir: Approximate u direction (3,); projected into the face plane and
            normalised. Use seed_direction_towards to build one that points at
            a specific neighbour.
        h_u: Full side length along u.
        h_v: Full side length along v.
        theta: Rotation of (u, v) about the normal, in radians (default: 0).
        offset_u: Centre shift along u, as a fraction of h_u (default: 0).
        offset_v: Centre shift along v, as a fraction of h_v (default: 0).

    Returns:
        dict: Fresh {(face_id, vertex_id): frame}, one entry per corner.
    """
    face_loop = [int(w) for w in mesh.faces[face_id]]
    fv = mesh.vertices[face_loop]
    n = unit(np.asarray(f_normals[face_id], dtype=float))

    u_base = unit(orth_proj(np.asarray(u_dir, dtype=float), n))
    u = rotation_matrix(n, theta) @ u_base
    v = np.cross(n, u)

    centre = fv.mean(axis=0) + offset_u * h_u * u + offset_v * h_v * v
    half_w, half_h = h_u / 2.0, h_v / 2.0

    signs = ((-1, -1), (1, -1), (1, 1), (-1, 1))
    offsets = [su * half_w * u + sv * half_h * v for su, sv in signs]
    centre = centre + _inward_shift(fv, n, [centre + o for o in offsets])
    corner_pts = [centre + o for o in offsets]

    order = _match_corners_to_vertices(corner_pts, fv)

    frames = {}
    for k, (su, sv) in enumerate(signs):
        v_id = face_loop[order[k]]
        frames[(face_id, v_id)] = {"pt": corner_pts[k], "u": su * u, "v": sv * v,
                                   "n": n, "f": int(face_id)}
    return frames


def place_rectangle_from_edge(mesh, f_normals, frames, face_id,
                              edge_vertex_ids, depth):
    """
    Complete a rectangle on *face_id* from its two shared-edge corners.

    After reflect_edge_frames carries the shared edge onto face_id, this steps
    the remaining pair *depth* perpendicular to that edge, into the face, and
    assigns them to the vertices they land nearest. This is Step 2 of
    docs/PROPAGATION_REDESIGN.md: the shared edge's length is whatever
    reflection gave it; depth is the free dimension, chosen by the caller
    (typically from strip_min_edge_length over the transverse strip).

    Args:
        mesh: Mesh object.
        f_normals: Per-face normals (nF, 3).
        frames: Frame dict keyed by (face_id, vertex_id); mutated in place.
        face_id: Face index carrying exactly two frames, at edge_vertex_ids.
        edge_vertex_ids: The two vertex ids of the shared edge already placed
            on face_id.
        depth: Distance to extend perpendicular to the shared edge, in world
            units.

    Returns:
        The frames dict, mutated in place.

    Raises:
        DupinPropagationError: If face_id does not carry exactly the two given
            frames, or the two new corners land on the same vertex.
    """
    v_a, v_b = (int(v) for v in edge_vertex_ids)
    fid_a, fid_b = (face_id, v_a), (face_id, v_b)
    if fid_a not in frames or fid_b not in frames:
        raise DupinPropagationError(
            f"Cannot place rectangle on face {face_id}: expected frames at "
            f"{fid_a} and {fid_b}."
        )

    frame_a, frame_b = frames[fid_a], frames[fid_b]
    point_a, point_b = np.asarray(frame_a["pt"], float), np.asarray(frame_b["pt"], float)
    n = unit(np.asarray(f_normals[face_id], dtype=float))

    edge_mid = (point_a + point_b) / 2.0
    edge_dir = unit(point_b - point_a)

    # In-plane normal to the edge; pick the sign pointing into the face rather
    # than trusting the frames' stored u/v signs, which is what transposed
    # corners in an earlier version of this pipeline.
    step_dir = unit(np.cross(n, edge_dir))
    barycentre = mesh.vertices[mesh.faces[face_id]].mean(axis=0)
    if np.dot(step_dir, barycentre - edge_mid) < 0:
        step_dir = -step_dir

    point_a_new = point_a + depth * step_dir
    point_b_new = point_b + depth * step_dir

    face_verts = [int(v) for v in mesh.faces[face_id]]
    remaining = [v for v in face_verts if v not in (v_a, v_b)]
    if len(remaining) != 2:
        raise DupinPropagationError(
            f"Cannot place rectangle on face {face_id}: expected 2 remaining "
            f"vertices after excluding the shared edge, found {len(remaining)}."
        )

    order = _match_corners_to_vertices(
        [point_a_new, point_b_new], mesh.vertices[remaining])
    if order[0] == order[1]:
        raise DupinPropagationError(
            f"Cannot place rectangle on face {face_id}: both new corners land "
            f"nearest the same vertex {remaining[order[0]]}."
        )

    # Every corner's tangent vectors point AWAY from the rectangle's interior
    # along their own axis (verified against seed_rectangle_at_barycentre: at
    # every corner, both u and v point toward neither adjacent corner, i.e.
    # outward). The axis-of-travel component of src's frame — the one aligned
    # with step_dir, not with the shared edge — was pointing from the shared
    # edge INTO the face, i.e. toward where the new corner is being placed.
    # Copying it unflipped leaves the new corner's tangent pointing further
    # away from the rectangle (toward v_id's own far neighbour) instead of
    # back toward the shared edge it came from, breaking that convention.
    for pt, src, k in ((point_a_new, frame_a, order[0]),
                       (point_b_new, frame_b, order[1])):
        v_id = remaining[k]
        u_src, v_src = np.asarray(src["u"], float), np.asarray(src["v"], float)
        if abs(unit(u_src) @ step_dir) > abs(unit(v_src) @ step_dir):
            new_u, new_v = -u_src, v_src
        else:
            new_u, new_v = u_src, -v_src
        frames[(face_id, v_id)] = {
            "pt": pt, "u": new_u, "v": new_v, "n": n, "f": int(face_id)}

    return frames


def rectangle_from_two_edges(mesh, f_normals, frames, face_id, edge_a, edge_b):
    """
    Build an interior face's rectangle from two reflected edges.

    *edge_a* and *edge_b* are the images, under reflection from two different
    edge-adjacent predecessors, of edges that share one vertex. Their shared
    corner will not coincide exactly between the two reflections, so this
    averages the two candidate positions, projects onto face_id's plane, then
    builds the rectangle those two (now exact) edges span and squares it up
    via project_rectangle_to_face. This is Step 3 of
    docs/PROPAGATION_REDESIGN.md.

    Args:
        mesh: Mesh object.
        f_normals: Per-face normals (nF, 3).
        frames: Frame dict keyed by (face_id, vertex_id); mutated in place.
        face_id: Interior face index.
        edge_a: ((v_shared, pt_shared_a), (v_far_a, pt_far_a)) — an edge as two
            (vertex_id, point) pairs, sharing vertex v_shared with edge_b.
        edge_b: Same shape as edge_a, sharing v_shared with edge_a.

    Returns:
        float: Max gap between the two candidate positions for the shared
        corner (pre-averaging) — a drift diagnostic; 0 for an exact match.

    Raises:
        DupinPropagationError: If the two edges do not share a vertex, or the
            face does not end up with exactly 4 corners.
    """
    (v_shared_a, pt_shared_a), (v_far_a, pt_far_a) = edge_a
    (v_shared_b, pt_shared_b), (v_far_b, pt_far_b) = edge_b
    v_shared_a, v_far_a = int(v_shared_a), int(v_far_a)
    v_shared_b, v_far_b = int(v_shared_b), int(v_far_b)

    if v_shared_a != v_shared_b:
        raise DupinPropagationError(
            f"Cannot build face {face_id}: edges do not share a vertex "
            f"({v_shared_a} != {v_shared_b})."
        )
    if v_far_a == v_far_b:
        raise DupinPropagationError(
            f"Cannot build face {face_id}: both edges' far vertices are the "
            f"same ({v_far_a})."
        )

    n = unit(np.asarray(f_normals[face_id], dtype=float))
    pt_shared_a = np.asarray(pt_shared_a, dtype=float)
    pt_shared_b = np.asarray(pt_shared_b, dtype=float)
    drift = float(np.linalg.norm(pt_shared_a - pt_shared_b))

    shared_avg = (pt_shared_a + pt_shared_b) / 2.0
    pt_far_a = np.asarray(pt_far_a, dtype=float)
    pt_far_b = np.asarray(pt_far_b, dtype=float)

    face_verts = [int(v) for v in mesh.faces[face_id]]
    opp = [v for v in face_verts if v not in (v_shared_a, v_far_a, v_far_b)]
    if len(opp) != 1:
        raise DupinPropagationError(
            f"Cannot build face {face_id}: expected exactly 1 remaining "
            f"vertex opposite the shared corner, found {len(opp)}."
        )
    v_opp = opp[0]

    pt_opp_est = pt_far_a + pt_far_b - shared_avg   # parallelogram closure

    frames[(face_id, v_shared_a)] = {"pt": shared_avg, "u": pt_far_a - shared_avg,
                                     "v": pt_far_b - shared_avg, "n": n,
                                     "f": int(face_id)}
    frames[(face_id, v_far_a)] = {"pt": pt_far_a, "u": pt_far_a - shared_avg,
                                  "v": pt_opp_est - pt_far_a, "n": n,
                                  "f": int(face_id)}
    frames[(face_id, v_far_b)] = {"pt": pt_far_b, "u": pt_opp_est - pt_far_b,
                                  "v": pt_far_b - shared_avg, "n": n,
                                  "f": int(face_id)}
    frames[(face_id, v_opp)] = {"pt": pt_opp_est, "u": pt_opp_est - pt_far_b,
                                "v": pt_opp_est - pt_far_a, "n": n,
                                "f": int(face_id)}

    project_rectangle_to_face(mesh, frames, face_id, f_normals)

    return drift


def build_rectangle_frames(rectangle_corners, rectangle_faces, f_normals,
                           mesh_faces=None):
    """
    Build one orthogonal frame per rectangle corner, from post-optimization
    rectangle geometry rather than mesh-propagated frames.

    Used after an optimization pass has moved the rectangle corners
    (rectangle_corners) away from what propagate_rectangles produced: the
    (face_id, vertex_id)-keyed frames dict built during propagation is now
    stale, so this rebuilds it from the optimized corner positions and the
    fixed rectangle_faces connectivity.

    For rectangle i with corners rectangle_faces[i] = [c0, c1, c2, c3] (loop
    order), the normal is the cross of the two diagonals (plane_normal on 4
    points — robust to the rectangle no longer being exactly planar after
    optimization), sign-matched to f_normals[i] so it agrees with the
    original mesh's orientation. At each corner, the two in-plane tangent
    vectors run along that corner's two edges, pointing OUTWARD — away from
    the corner at the far end of that edge — matching the convention used
    everywhere else in this module (seed_rectangle_at_barycentre,
    place_rectangle_from_edge): e.g. at c0, the edge to c1 gives a tangent
    pointing from c1 through c0 and beyond, not from c0 toward c1.

    Args:
        rectangle_corners: (P, 3) array of corner positions.
        rectangle_faces: (R, 4) int array; rectangle_faces[i] are the 4
            indices into rectangle_corners for rectangle i, in loop order.
        f_normals: (R, 3) reference normals, one per rectangle (typically the
            original mesh's face_normals, same indexing as rectangle_faces).
        mesh_faces: (R, 4) int array of the ORIGINAL mesh face loops, so the
            result can be keyed by real vertex id — mesh_faces[i][k] is the
            mesh vertex that rectangle_faces[i][k] is the interior version of.
            If omitted, the key falls back to the corner slot (0-3).

    Returns:
        dict: {(face_id, vertex_id): {"pt","u","v","n","f"}} — the same keying
        propagate_rectangles produces, so downstream code (Dupin_patch lookups
        via vertex_face_adjacency_list, get_rectangle, ...) works unchanged.
        With mesh_faces omitted, vertex_id is the corner slot instead.
    """
    rectangle_corners = np.asarray(rectangle_corners, dtype=float)
    rectangle_faces = np.asarray(rectangle_faces, dtype=int)
    f_normals = np.asarray(f_normals, dtype=float)
    if mesh_faces is not None:
        mesh_faces = np.asarray(mesh_faces, dtype=int)

    frames = {}
    for rect_id, loop in enumerate(rectangle_faces):
        pts = rectangle_corners[loop]

        n = unit(plane_normal(pts))
        if n @ unit(f_normals[rect_id]) < 0:
            n = -n

        for k in range(4):
            pt = pts[k]
            nxt = pts[(k + 1) % 4]
            prv = pts[(k - 1) % 4]
            # Outward along each edge: from the neighbour, through this
            # corner, and beyond — not from this corner toward the neighbour.
            to_nxt = unit(pt - nxt)
            to_prv = unit(pt - prv)
            key_v = int(mesh_faces[rect_id][k]) if mesh_faces is not None else k
            frames[(rect_id, key_v)] = {
                "pt": pt, "u": to_prv, "v": to_nxt, "n": n, "f": int(rect_id)}

    return frames


def propagate_rectangles(mesh, f_normals, grid_faces, seed_frames,
                         u_factor=0.5, v_factor=0.5):
    """
    Propagate a seed rectangle across a quad grid, sizing each face from the
    strip it crosses rather than by rigid reflection.

    Implements docs/PROPAGATION_REDESIGN.md Steps 2-3 in full: the two
    boundary arms (row 0 and column 0 of grid_faces) are walked first, each
    step reflecting the shared edge (reflect_edge_frames) and then completing
    the rectangle with place_rectangle_from_edge, sized by
    factor * strip_min_edge_length over the transverse strip. The transverse
    strip at each arm step is identified locally (which of
    Mesh.quad_face_strips' two lists at the new face contains the
    already-placed predecessor — that one runs along the arm, so the other is
    transverse) rather than by trusting a fixed row/column position in
    grid_faces or quad_face_strips' own return order, both of which are
    accidents of how grid_coordinates happened to orient this particular
    mesh's grid (verified on cone_mesh_small.obj: quad_face_strips' first
    list is the grid column, not the row, the opposite of what its u_faces/
    v_faces naming suggests). The interior is then swept row-major; each cell
    reflects the two edges leaving its already-placed predecessors
    (u-predecessor and v-predecessor) and rebuilds via
    rectangle_from_two_edges.

    A pure function: no I/O, no GUI state, so it can run headlessly for
    testing (see docs/PROPAGATION_REDESIGN.md "How to verify").

    Args:
        mesh: Mesh object.
        f_normals: Per-face normals (nF, 3).
        grid_faces: (R, C) int array from grid_coordinates, seed at [0, 0].
            -1 marks a hole.
        seed_frames: frames dict for the seed face (grid_faces[0, 0]), e.g.
            from seed_rectangle_at_barycentre. Not mutated; copied internally.
        u_factor: Fraction in (0, 1] of the transverse strip's minimum aligned
            edge length used for each U-arm face's free dimension (default:
            0.5). Only the seed and the two arms have a free dimension to
            choose — the interior has none, it is fully determined by
            reflecting the two edges leaving its already-placed predecessors.
        v_factor: Same as u_factor, for the V-arm.

    Returns:
        tuple:
            frames (dict): (face_id, vertex_id) -> frame, for every placed face.
            placed (set[int]): Face indices successfully placed.
            stats (dict): {"arm_stopped": [...], "interior_skipped": [...],
                "interior_drift": [...]} — diagnostics, not required for
                correctness.
    """
    R, C = grid_faces.shape
    frames = {k: dict(v) for k, v in seed_frames.items()}
    seed_face = int(grid_faces[0, 0])
    placed = {seed_face}

    stats = {"arm_stopped": [], "interior_skipped": [], "interior_drift": []}

    def edge_step_direction(face_id, v_a, v_b):
        """In-plane unit vector at face_id perpendicular to edge (v_a, v_b)."""
        n = unit(np.asarray(f_normals[face_id], dtype=float))
        edge_dir = unit(mesh.vertices[int(v_b)] - mesh.vertices[int(v_a)])
        return unit(np.cross(n, edge_dir))

    def transverse_strip_at(fb, fa):
        """
        The strip through fb perpendicular to the arm fa->fb.

        Identified by membership, not by trusting quad_face_strips' u/v
        labels or a fixed row/column position in grid_faces: whichever of
        fb's two strips contains the already-placed predecessor fa is the
        along-arm strip, so the OTHER one is transverse. This makes each arm
        step self-contained — it only looks at the two faces it already has
        in hand — rather than depending on how grid_coordinates happened to
        orient this mesh's grid.
        """
        strip_a, strip_b = mesh.quad_face_strips(fb)
        if fa in strip_a and fa not in strip_b:
            return strip_b
        if fa in strip_b and fa not in strip_a:
            return strip_a
        raise DupinPropagationError(
            f"Cannot identify the transverse strip at face {fb}: predecessor "
            f"{fa} does not fall unambiguously on one of quad_face_strips' "
            f"two lists."
        )

    def walk_arm(steps, arm_name, arm_factor):
        for fa, fb in steps:
            if fa < 0 or fb < 0:
                break
            try:
                reflect_edge_frames(mesh, f_normals, frames, fa, fb)
                v_a, v_b = mesh.shared_edge_vertices(fa, fb)
                transverse = transverse_strip_at(fb, fa)
                direction = edge_step_direction(fb, v_a, v_b)
                depth = arm_factor * strip_min_edge_length(
                    mesh, transverse, direction, f_normals)
                place_rectangle_from_edge(mesh, f_normals, frames, fb,
                                          (v_a, v_b), depth)
            except DupinPropagationError as e:
                stats["arm_stopped"].append((arm_name, fa, fb, str(e)))
                break
            placed.add(fb)

    # -- U-arm: row 0 --
    walk_arm(((int(grid_faces[i, 0]), int(grid_faces[i + 1, 0]))
             for i in range(R - 1)), "u", u_factor)

    # -- V-arm: column 0 --
    walk_arm(((int(grid_faces[0, j]), int(grid_faces[0, j + 1]))
             for j in range(C - 1)), "v", v_factor)

    # -- interior, row-major so both predecessors are already placed --
    for i in range(1, R):
        for j in range(1, C):
            face_id = int(grid_faces[i, j])
            if face_id < 0 or face_id in placed:
                continue
            prev_u = int(grid_faces[i - 1, j])
            prev_v = int(grid_faces[i, j - 1])
            if prev_u not in placed or prev_v not in placed:
                stats["interior_skipped"].append((face_id, prev_u, prev_v))
                continue

            common_u = set(mesh.faces[prev_u]) & set(mesh.faces[face_id])
            common_v = set(mesh.faces[prev_v]) & set(mesh.faces[face_id])
            if len(common_u) != 2 or len(common_v) != 2:
                stats["interior_skipped"].append((face_id, prev_u, prev_v))
                continue

            try:
                shared = list((set(mesh.faces[prev_u]) & set(mesh.faces[prev_v]))
                             & set(mesh.faces[face_id]))
                if len(shared) != 1:
                    raise DupinPropagationError(
                        f"Cannot build face {face_id}: expected 1 vertex shared "
                        f"by both predecessors and this face, found "
                        f"{len(shared)}."
                    )
                v_shared = shared[0]
                v_far_u = next(iter(common_u - {v_shared}))
                v_far_v = next(iter(common_v - {v_shared}))

                # reflect_edge_frames wrote two candidate positions for
                # v_shared (one from each predecessor) under the same key,
                # the second overwriting the first — recover both by
                # re-deriving them individually before they collide.
                edge_u = reflect_edge_frames(
                    mesh, f_normals, {k: v for k, v in frames.items()
                                     if k[0] == prev_u}, prev_u, face_id)
                edge_v = reflect_edge_frames(
                    mesh, f_normals, {k: v for k, v in frames.items()
                                     if k[0] == prev_v}, prev_v, face_id)

                pt_shared_u = edge_u[(face_id, v_shared)]["pt"]
                pt_shared_v = edge_v[(face_id, v_shared)]["pt"]
                pt_far_u = edge_u[(face_id, v_far_u)]["pt"]
                pt_far_v = edge_v[(face_id, v_far_v)]["pt"]

                drift = rectangle_from_two_edges(
                    mesh, f_normals, frames, face_id,
                    ((v_shared, pt_shared_u), (v_far_u, pt_far_u)),
                    ((v_shared, pt_shared_v), (v_far_v, pt_far_v)))
                stats["interior_drift"].append((face_id, drift))
            except DupinPropagationError as e:
                stats["interior_skipped"].append((face_id, prev_u, prev_v, str(e)))
                continue

            placed.add(face_id)

    return frames, placed, stats


def build_edge_aligned_rectangle(mesh, f_normals, face_id,
                                 theta=0.0, offset_u=0.0, offset_v=0.0,
                                 width_ref=0.5, height_ref=0.5):
    """
    Build a true rectangle inside a face, aligned with the face's own edges.

    The base frame is edge-aligned: u runs along the projection of the face's
    first edge onto the face plane, v = n x u. `theta` then rotates (u, v)
    about the normal away from that alignment. The rectangle is centered at
    the face centroid, shifted by (offset_u, offset_v) along (u, v) — each a
    fraction of the corresponding face edge length — and sized by
    (width_ref, height_ref), also fractions of the face's edge lengths.

    Unlike create_rectangle_inside_face (which clips rulings to the face
    boundary and can end up as a general, non-rectangular quad on skewed
    faces), this always produces an exact rectangle.

    The offset is clamped so the rectangle stays inside the face: after
    shifting, any corner lying outside a face edge pushes the whole rectangle
    back inward, preserving size, shape and orientation. If the rectangle is
    larger than the face no shift can contain it, and this settles at the
    best achievable centering instead of diverging.

    Args:
        mesh: Mesh object with .vertices and .faces.
        f_normals: Per-face normals (nF, 3).
        face_id: Face to seed in.
        theta: Rotation of (u, v) about the normal, in radians (default: 0).
        offset_u: Center shift along u, as a fraction of the u-edge length.
        offset_v: Center shift along v, as a fraction of the v-edge length.
        width_ref: Rectangle side length along u, as a fraction of the u-edge
            length (default: 0.5).
        height_ref: Rectangle side length along v, as a fraction of the
            v-edge length (default: 0.5).

    Returns:
        The frames dict {(face_id, vertex_id): frame}, one entry per face
        corner, built fresh (does not mutate an existing frames dict — pass
        the result to the caller's own dict via `.update()` if needed).
    """
    face_loop = [int(w) for w in mesh.faces[face_id]]
    fv = mesh.vertices[face_loop]
    n = unit(np.asarray(f_normals[face_id], dtype=float))

    edge_u = fv[1] - fv[0]
    edge_v = fv[-1] - fv[0]
    len_u = np.linalg.norm(orth_proj(edge_u, n))
    len_v = np.linalg.norm(orth_proj(edge_v, n))

    u_base = unit(orth_proj(edge_u, n))
    rot = rotation_matrix(n, theta)
    u = rot @ u_base
    v = np.cross(n, u)

    center = fv.mean(axis=0) + offset_u * len_u * u + offset_v * len_v * v

    half_w = 0.5 * width_ref * len_u
    half_h = 0.5 * height_ref * len_v

    signs = ((-1, -1), (1, -1), (1, 1), (-1, 1))
    offsets = [su * half_w * u + sv * half_h * v for su, sv in signs]

    center = center + _inward_shift(fv, n, [center + o for o in offsets])

    # Assign each corner to the face vertex it actually lies nearest. Walking
    # face_loop in step with `signs` would instead assume the rectangle winds
    # the same way as the face; v = n x u need not follow the loop's own
    # orientation, and when it doesn't the two diagonal pairs come out
    # transposed — every corner keyed to the vertex across from it.
    corner_pts = [center + o for o in offsets]
    order = _match_corners_to_vertices(corner_pts, fv)

    corners = {
        face_loop[order[k]]: (corner_pts[k], su * u, sv * v)
        for k, (su, sv) in enumerate(signs)
    }

    frames = {}
    for v_id, (pt, cu, cv) in corners.items():
        frames[(face_id, v_id)] = {"pt": pt, "u": cu, "v": cv, "n": n,
                                    "f": int(face_id)}
    return frames


def create_rectangle_inside_face(mesh, frames, frame_id,
                                 width_ref_edge=0.5, height_ref_edge=0.5):
    """
    Complete a rectangle inside a face from one corner frame.

    The u and v rulings are clipped to the face, and the far corners are placed
    at a fraction of the larger clip parameter, so the rectangle always stays
    inside the face.

    Args:
        mesh: Mesh object.
        frames: Frame dict keyed by (face_id, vertex_id); mutated in place.
        frame_id: Key (face_id, vertex_id) of the seed corner.
        width_ref_edge: Fraction along v in (0, 1] (default: 0.5).
        height_ref_edge: Fraction along u in (0, 1] (default: 0.5).

    Returns:
        The frames dict, mutated in place.
    """
    face_id, _ = frame_id
    frame = frames[frame_id]
    point, u, v, n = frame["pt"], frame["u"], frame["v"], frame["n"]

    _, lu_dir = plane_plane_intersection(n, point, u, point)
    _, lv_dir = plane_plane_intersection(n, point, v, point)

    face_pts = mesh.vertices[mesh.faces[face_id]]
    tu_i, tu_e = clip_line_to_convex_face(point, lu_dir, face_pts)
    tv_i, tv_e = clip_line_to_convex_face(point, lv_dir, face_pts)

    # Extend toward the farther face edge so the rectangle stays inside.
    tw = height_ref_edge * (tu_e if abs(tu_i) < abs(tu_e) else tu_i)
    th = width_ref_edge * (tv_e if abs(tv_i) < abs(tv_e) else tv_i)

    point_w = point + tw * lu_dir
    point_h = point + th * lv_dir
    point_hw = point + tw * lu_dir + th * lv_dir

    # Assign the three remaining corners by walking the face loop from the seed
    # vertex, not by nearest-vertex: on a skewed quad two of the new points can
    # share a nearest vertex, which would leave the fourth corner unassigned.
    face_loop = [int(w) for w in mesh.faces[face_id]]
    seed_vertex = frame_id[1]
    k = face_loop.index(seed_vertex)
    nxt = face_loop[(k + 1) % 4]
    opp = face_loop[(k + 2) % 4]
    prv = face_loop[(k + 3) % 4]

    # Which of the two face edges leaving the seed runs along +u decides whether
    # "next" is the u-corner or the v-corner.
    edge_next = mesh.vertices[nxt] - mesh.vertices[seed_vertex]
    if abs(edge_next @ unit(lu_dir)) >= abs(edge_next @ unit(lv_dir)):
        close_w, close_h = nxt, prv
    else:
        close_w, close_h = prv, nxt
    close_hw = opp

    frames[(face_id, close_h)] = {"pt": point_h, "u": -u, "v": v,
                                  "n": n, "f": int(face_id)}
    frames[(face_id, close_w)] = {"pt": point_w, "u": u, "v": -v,
                                  "n": n, "f": int(face_id)}
    frames[(face_id, close_hw)] = {"pt": point_hw, "u": -u, "v": -v,
                                   "n": n, "f": int(face_id)}
    return frames


def get_rectangle(mesh, frames, face_id):
    """
    Corner points and side lengths of the rectangle stored on a face.

    Opposite sides are averaged, so the returned width and height describe the
    rectangle even when reflection has left it slightly non-planar.

    Args:
        mesh: Mesh object.
        frames: Frame dict keyed by (face_id, vertex_id).
        face_id: Face index.

    Returns:
        tuple: (corner points (4, 3) in face vertex order, width, height).

    Raises:
        DupinPropagationError: If the face lacks a frame at any of its corners.
    """
    frame_ids = [(face_id, int(v_i)) for v_i in mesh.faces[face_id]]

    missing = [fid for fid in frame_ids if fid not in frames]
    if missing:
        raise DupinPropagationError(
            f"Face {face_id} has no complete rectangle: missing frames at "
            f"{missing}."
        )

    pts = np.array([frames[fid]["pt"] for fid in frame_ids])
    u_dir = np.asarray(frames[frame_ids[0]]["u"], float)

    # Identify which pair of sides runs along u.
    if np.isclose(unit(pts[1] - pts[0]) @ u_dir, 0):
        w = np.linalg.norm(pts[3] - pts[0])
        w_2 = np.linalg.norm(pts[2] - pts[1])
        h = np.linalg.norm(pts[1] - pts[0])
        h_2 = np.linalg.norm(pts[3] - pts[2])
    else:
        w = np.linalg.norm(pts[1] - pts[0])
        w_2 = np.linalg.norm(pts[2] - pts[3])
        h = np.linalg.norm(pts[3] - pts[0])
        h_2 = np.linalg.norm(pts[2] - pts[1])

    return pts, (w + w_2) * 0.5, (h + h_2) * 0.5


def set_rectangle_edge(mesh, frames, face_id, dist):
    """
    Extend the two frames already on a face into a full rectangle.

    After reflecting an edge, a face carries exactly two frames. This walks them
    perpendicular to that shared edge by *dist* to place the opposite corners.

    Args:
        mesh: Mesh object.
        frames: Frame dict keyed by (face_id, vertex_id); mutated in place.
        face_id: Face index carrying exactly two frames.
        dist: Distance to extend, in world units.

    Returns:
        The frames dict, mutated in place.

    Raises:
        DupinPropagationError: If the face does not carry exactly two frames.
    """
    frames_at_face = [fid for fid in frames if fid[0] == face_id]
    if len(frames_at_face) != 2:
        raise DupinPropagationError(
            f"Cannot set rectangle on face {face_id}: expected exactly 2 frames "
            f"after the edge reflection, found {len(frames_at_face)}."
        )

    (_, vertex_id_1), (_, vertex_id_2) = frames_at_face
    frame_1 = frames[frames_at_face[0]]
    frame_2 = frames[frames_at_face[1]]

    point_1, point_2 = frame_1["pt"], frame_2["pt"]
    dir_edge = unit(point_2 - point_1)

    # Step perpendicular to the shared edge: along v if the edge runs along u.
    along_u = np.isclose(abs(dir_edge @ np.asarray(frame_1["u"], float)), 1)
    key = "v" if along_u else "u"

    point_1_new = point_1 - dist * np.asarray(frame_1[key], float)
    point_2_new = point_2 - dist * np.asarray(frame_2[key], float)

    options_ids = [v_id for v_id in mesh.faces[face_id]
                   if v_id not in (vertex_id_1, vertex_id_2)]
    d0 = np.linalg.norm(point_1_new - mesh.vertices[options_ids[0]])
    d1 = np.linalg.norm(point_1_new - mesh.vertices[options_ids[1]])
    point_1_id, point_2_id = ((options_ids[0], options_ids[1]) if d0 < d1
                              else (options_ids[1], options_ids[0]))

    for v_id, src, pt_new in ((point_1_id, frame_1, point_1_new),
                              (point_2_id, frame_2, point_2_new)):
        new_frame = {"pt": pt_new, "u": src["u"], "v": src["v"],
                     "n": src["n"], "f": int(face_id)}
        new_frame[key] = -np.asarray(src[key], float)   # flip the stepped axis
        frames[(face_id, v_id)] = new_frame

    return frames


def project_rectangle_to_face(mesh, frames, face_id, f_normals=None):
    """
    Flatten a face's rectangle onto the face plane and re-square it.

    A rectangle built from a Gram-Schmidt-averaged frame is only approximately
    planar and only approximately right-angled: the averaged frame is a blend
    of two reflections and need not lie in the face plane. This projects the
    four corners onto the plane of face_id, then rebuilds them as an exact
    rectangle about their common centre, keeping the side lengths and the
    dominant in-plane orientation.

    Args:
        mesh: Mesh object.
        frames: Frame dict keyed by (face_id, vertex_id); mutated in place.
        face_id: Face index carrying a complete rectangle (4 frames).
        f_normals: Per-face normals (nF, 3); defaults to mesh.face_normals.

    Returns:
        The frames dict, mutated in place.

    Raises:
        DupinPropagationError: If the face lacks a frame at any of its corners.
    """
    frame_ids = [(face_id, int(v_i)) for v_i in mesh.faces[face_id]]
    missing = [fid for fid in frame_ids if fid not in frames]
    if missing:
        raise DupinPropagationError(
            f"Cannot project face {face_id}: missing frames at {missing}."
        )

    n = unit(np.asarray(f_normals[face_id] if f_normals is not None
                        else mesh.face_normals[face_id], float))

    pts = np.array([frames[fid]["pt"] for fid in frame_ids], dtype=float)
    centre = pts.mean(axis=0)

    # Drop the out-of-plane component about the centre.
    flat = pts - np.outer((pts - centre) @ n, n)

    # Recover an orthonormal in-plane frame from the flattened corners: average
    # the two edge directions that should run along u, then square v off it.
    d0 = (flat[1] - flat[0]) + (flat[2] - flat[3])
    if np.linalg.norm(d0) < 1e-14:
        d0 = flat[1] - flat[0]
    u = unit(orth_proj(d0, n))
    v = np.cross(n, u)

    # Half-extents in that frame, averaged over the four corners.
    local = flat - centre
    half_w = np.abs(local @ u).mean()
    half_h = np.abs(local @ v).mean()

    for fid, p in zip(frame_ids, local):
        su = 1.0 if p @ u >= 0 else -1.0
        sv = 1.0 if p @ v >= 0 else -1.0
        frames[fid] = {"pt": centre + su * half_w * u + sv * half_h * v,
                       "u": su * u, "v": sv * v, "n": n, "f": int(face_id)}

    return frames


def reflect_rectangle_rigid(mesh, f_normals, frames, face_i, face_k):
    """
    Carry a whole rectangle to an edge-adjacent face by pure reflection.

    Every corner of the rectangle on face_i is reflected across the bisecting
    plane of (face_i, face_k). Reflection is a rigid motion, so the image is
    congruent to the source: *both* side lengths are preserved and it is still
    an exact rectangle.

    This is the difference from reflect_rectangle, which reflects only the two
    shared-edge corners and rebuilds the far pair by stepping a caller-supplied
    distance — that step replaces the perpendicular side length, so the
    rectangle changes size as it propagates.

    Each reflected corner is keyed to the vertex of face_k it actually lands
    nearest: reflection carries a corner across the shared edge, so it does
    not stay near the vertex it came from.

    Args:
        mesh: Mesh object.
        f_normals: Per-face normals (nF, 3).
        frames: Frame dict keyed by (face_id, vertex_id); mutated in place.
        face_i: Source face index, carrying a complete rectangle (4 frames).
        face_k: Target face index, sharing an edge with face_i.

    Returns:
        The frames dict, mutated in place.

    Raises:
        DupinPropagationError: If face_i lacks a complete rectangle, the faces
            are coplanar, or two corners land on the same vertex.
    """
    frame_ids = [(face_i, int(v_i)) for v_i in mesh.faces[face_i]]
    missing = [fid for fid in frame_ids if fid not in frames]
    if missing:
        raise DupinPropagationError(
            f"Cannot reflect the rectangle from face {face_i} to {face_k}: "
            f"missing frames at {missing}."
        )

    normal_i = np.asarray(f_normals[face_i], float)
    normal_k = np.asarray(f_normals[face_k], float)
    if np.isclose(normal_i @ normal_k, 1.0):
        raise DupinPropagationError(
            f"Cannot reflect the rectangle from face {face_i} to face {face_k}: "
            f"the faces are coplanar (n_i · n_k = {normal_i @ normal_k:.6f}), "
            f"so their bisecting plane is undefined."
        )

    normal_bisec, offset_bisec, _ = bisecting_plane(
        mesh, f_normals, face_i, face_k)

    reflected = [
        reflect_frame_across_plane(frames[fid], normal_bisec, offset_bisec,
                                   face=face_k)
        for fid in frame_ids
    ]

    face_k_verts = [int(v) for v in mesh.faces[face_k]]
    order = _match_corners_to_vertices([fr["pt"] for fr in reflected],
                                       mesh.vertices[face_k_verts])
    if len(set(order)) != len(order):
        raise DupinPropagationError(
            f"Cannot reflect the rectangle from face {face_i} to {face_k}: "
            f"two corners map to the same vertex."
        )

    for fr, k in zip(reflected, order):
        frames[(face_k, face_k_verts[k])] = fr

    return frames


def reflect_rectangle(mesh, f_normals, frames, face_i, face_k, dist):
    """
    Carry a rectangle from one face to an edge-adjacent face.

    Reflects the shared-edge frames onto face_k, then extends them by *dist* to
    complete the rectangle.

    Args:
        mesh: Mesh object.
        f_normals: Per-face normals (nF, 3).
        frames: Frame dict keyed by (face_id, vertex_id); mutated in place.
        face_i: Source face index.
        face_k: Target face index.
        dist: Distance to extend on the target face, in world units.

    Returns:
        The frames dict, mutated in place.

    Raises:
        DupinPropagationError: If the reflection or the extension fails.
    """
    reflect_edge_frames(mesh, f_normals, frames, face_i, face_k)
    set_rectangle_edge(mesh, frames, face_k, dist)
    return frames
