"""Research geometry moved out of hanan: LMesh.utils.laguerre and LMesh.utils.dupin."""

import os

import numpy as np
import pytest

from hanan.geometry.construction import normalize_vertices
from hanan.geometry.io import read_obj
from hanan.geometry.mesh import Mesh
from hanan.geometry.conical import compute_cone_axes, conical_mesh_smallest_curvature_sphere

from LMesh.utils.laguerre import compute_discrete_laguerre_channel_surface
from LMesh.utils.dupin import (
    DupinPropagationError,
    find_corner_faces,
    grid_coordinates,
    seed_direction_towards,
)

DATA = os.path.join(os.path.dirname(__file__), "..", "data")


def _grid_mesh(n=5):
    xs, ys = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
    V = np.column_stack([xs.ravel(), ys.ravel(), 0.1 * (xs.ravel() ** 2 + ys.ravel() ** 2)])
    F = [[r * n + c, r * n + c + 1, (r + 1) * n + c + 1, (r + 1) * n + c]
         for r in range(n - 1) for c in range(n - 1)]
    mesh = Mesh()
    mesh.make_mesh(V, F)
    return mesh


@pytest.fixture(scope="module")
def conical_spheres():
    """A sphere congruence on the conical test mesh, as the lab notebooks build it."""
    V, F = read_obj(os.path.join(DATA, "cone_mesh_small.obj"))
    V = normalize_vertices(V)
    mesh = Mesh()
    mesh.make_mesh(V, F)
    axes, angles = compute_cone_axes(mesh)
    _, rad = conical_mesh_smallest_curvature_sphere(mesh.vertices, axes, angles,
                                                    mesh.vertex_adjacency_list())
    radii = 0.5 * rad
    centers = V + radii[:, None] * (axes / np.sin(angles[:, None]))
    return mesh, centers, radii


def test_channel_surface_is_finite(conical_spheres):
    _, centers, radii = conical_spheres
    spheres, cones, rest = compute_discrete_laguerre_channel_surface(centers[:6], radii[:6])
    for V in (spheres[0], cones[0]):
        V = np.asarray(V)
        assert V.ndim == 2 and V.shape[1] == 3 and np.isfinite(V).all()


def test_spherical_panels_lie_on_their_spheres(conical_spheres):
    from LMesh.L_mesh import Lmesh
    mesh, centers, radii = conical_spheres
    lm = Lmesh()
    lm.init_L_mesh(mesh, orientation=1, combinatorics="primal", mode="SphCongruence",
                   radii=radii, centers=centers)
    lm.improve_contact()
    lm.compute_spherical_panels()
    lm.compute_panel_meshes()
    panels = [(i, p) for i, p in enumerate(lm.sv_panels) if p is not None and p[1] is not None]
    assert len(panels) > len(radii) // 2
    for i, (PV, PF) in panels:
        d = np.linalg.norm(np.asarray(PV) - lm.sv_centers[i], axis=1)
        np.testing.assert_allclose(d, abs(lm.sv_radii[i]), rtol=1e-6)


def test_dupin_grid_structure():
    mesh = _grid_mesh(5)
    corners = find_corner_faces(mesh)
    assert sorted(np.asarray(corners).tolist()) == [0, 3, 12, 15]
    grid = grid_coordinates(mesh.faces, seed_face=0)
    assert grid.shape == (4, 4)
    assert sorted(grid.ravel().tolist()) == list(range(16))


def test_dupin_seed_direction_is_tangent():
    mesh = _grid_mesh(5)
    n = mesh.face_normals
    u = seed_direction_towards(mesh, n, 0, 1)
    assert abs(np.linalg.norm(u) - 1) < 1e-9
    assert abs(u @ n[0]) < 1e-9                       # lies in the seed face plane
    assert issubclass(DupinPropagationError, Exception)


def _conical_lmesh(conical_spheres):
    from LMesh.L_mesh import Lmesh
    mesh, centers, radii = conical_spheres
    lm = Lmesh()
    lm.init_L_mesh(mesh, orientation=1, combinatorics="primal", mode="SphCongruence",
                   radii=radii, centers=centers)
    lm.improve_contact()
    return lm


def test_cone_strip_fallback_rebuilds_missing_arc(conical_spheres):
    """_cone_strip rebuilds a missing contact arc by intersecting cone rulings with
    the contact plane (line_plane_inter, planes n · x + h = 0); it must match the real arc."""
    lm = _conical_lmesh(conical_spheres)
    pairs = [(i, j) for (i, j) in lm.HCE if i < j and (j, i) in lm.HCE][:3]
    assert pairs
    for i, j in pairs:
        V_ref, F_ref = lm._cone_strip(i, j, lm.HCE)
        for drop in [(j, i), (i, j)]:
            rec = lm.HCE[drop]
            saved, rec["c_arc"] = rec["c_arc"], None
            try:
                V_fb, F_fb = lm._cone_strip(i, j, lm.HCE)
            finally:
                rec["c_arc"] = saved
            assert F_fb == F_ref
            np.testing.assert_allclose(V_fb, V_ref, atol=1e-10)


def test_extract_mesh_keys_and_wireframe(conical_spheres):
    lm = _conical_lmesh(conical_spheres)
    mesh = lm.extract_mesh()
    assert {"spherical_panels", "conical_panels", "flat_panels"} <= set(mesh)
    assert "spherical_wireframe" not in mesh
    for key in ("spherical_panels", "conical_panels", "flat_panels"):
        V, F = mesh[key]
        assert len(V) and len(F) and max(max(f) for f in F) < len(V)
    wf = lm.extract_mesh(wireframe=True)
    for key in ("spherical_wireframe", "flat_wireframe"):
        P, E = wf[key]
        assert P.shape[1] == 3 and E.shape[1] == 2 and E.max() < len(P)


def test_extract_mesh_degenerate_elements(conical_spheres, tmp_path):
    """Spheres of radius ~0 come back as points; a cone between two of them as a segment."""
    lm = _conical_lmesh(conical_spheres)
    inner = set(int(v) for v in lm.top.inner_vertices())
    adj = lm.top.vertex_adjacency_list()
    i = next(v for v in sorted(inner) if any(int(w) in inner for w in adj[v]))
    j = next(int(w) for w in adj[i] if int(w) in inner)
    radii = 0.01 * lm.sv_radii       # small spheres, so zeroing two cannot nest them
    radii[[i, j]] = 0.0
    lm.update_spheres_planes(lm.sv_centers.copy(), radii, lm.PF[:, :3].copy(), lm.PF[:, 3].copy())

    mesh = lm.extract_mesh()
    np.testing.assert_allclose(np.sort(mesh["degenerated_spheres"], axis=0),
                               np.sort(lm.sv_centers[[i, j]], axis=0))
    V, E = mesh["degenerated_cones"]
    segments = [tuple(sorted(map(tuple, np.round(V[e], 9)))) for e in E]
    assert tuple(sorted(map(tuple, np.round(lm.sv_centers[[i, j]], 9)))) in segments

    lm.export_extended_lmesh(str(tmp_path), "lm")
    assert (tmp_path / "lm_degenerated_spheres.obj").exists()
    assert (tmp_path / "lm_degenerated_cones.obj").exists()


@pytest.mark.parametrize("kind", ["trivial", "planar", "second_envelope"])
def test_support_structures_swapped_convention(conical_spheres, kind):
    """Support structures use spheres on vertices / planes on faces (branch swapSP)."""
    lm = _conical_lmesh(conical_spheres)
    V, F = lm.extract_support_structure(kind, 0.02)
    V = np.asarray(V)
    assert len(V) and len(F) and np.all(np.isfinite(V))
    assert max(max(f) for f in F) < len(V)


def test_planar_support_annuli_start_at_their_contact_points(conical_spheres):
    """Each annulus lies in its contact circle's plane and both contact points it is
    built from lie on that circle — fails if a contact point is taken from the wrong
    sphere/plane pair."""
    import LMesh.L_mesh as LM
    lm = _conical_lmesh(conical_spheres)
    calls = []
    orig = LM.Lmesh._annulus_geometry

    def spy(self, cp_i, cp_j, center, axis, nc_i, nc_j, thickness):
        out = orig(self, cp_i, cp_j, center, axis, nc_i, nc_j, thickness)
        calls.append((cp_i, cp_j, center, axis, np.asarray(out[0])))
        return out

    LM.Lmesh._annulus_geometry = spy
    try:
        lm.extract_support_structure("planar", 0.02)
    finally:
        LM.Lmesh._annulus_geometry = orig
    assert calls
    for cp_i, cp_j, c, a, AV in calls:
        assert np.all(np.abs((AV - c) @ a) < 1e-8)
        for cp in (cp_i, cp_j):
            assert abs((cp - c) @ a) < 1e-6
