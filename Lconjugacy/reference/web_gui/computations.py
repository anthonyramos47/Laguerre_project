# Code/LconjugacyGUI/computations.py
"""Shared computation functions that operate on AppState."""

import os
import pickle
import subprocess
import numpy as np
import igl

from pathlib import Path

from LconjugacyGUI.state import AppState

from LconjugacyGUI.utils.bsplines_functions import (
    compute_rij_transport_vectorized,
    control_points_grid,
    sample_bspline_surface,
    bspline_curvatures,
    read_bspline_json,
    save_bspline_to_json,
    principal_frame,
    export_crossfield,
    radius_spline_fitting,
    sample_radius,
    project_points_to_bspline,
    exchange_lconjugate_rulings_curve_tangents,
    fit_bspline_surface,
    principal_frame_at_point,
    check_lconjugate_rulings
)

from hanan.geometry.utils import (
    write_obj,
    triangulate_quads,
    unit,
    read_obj,
    interpolate_triangle_mesh,
    mathematica_v,
)
from LMesh.utils.laguerre import save_discrete_laguerre_channel_surface
from hanan.geometry.mesh import Mesh

from LMesh.L_mesh import Lmesh

from kayviz import serializers as ser
from hanan import glyphs
import lmesh_serializers as lser


# ---------- Radius & reference ----------

def radius_r_at_point(state: AppState, u, v):
    """Compute radius at a given point on the B-spline surface."""
    _, _, k1, k2, n = principal_frame_at_point(state.bspline, u, v)

    if state.radiusFunctionType == "safe":
        r = 1 / np.maximum(np.abs(k1), np.abs(k2)) * state.percentageRadius
    elif state.radiusFunctionType == "mid":
        r = (1 / k1 + 1 / k2) * 0.5
    elif state.radiusFunctionType == "offset":
        r = state.offsetValue
    return r


def init_radius_function(state: AppState) -> None:
    """Fit a B-spline to the radius values at sampled points."""
    sphereCenters = lambda u, v: [u, v, radius_r_at_point(state, u, v)]
    radiusSpline = fit_bspline_surface(sphereCenters, (0, 1), (0, 1),
                                       n_u=100, n_v=100, order=3)
    state.radiusFunction = radiusSpline


def compute_radius_function(state: AppState) -> list:
    """Compute radius values at current sample points. Returns serializer objects."""
    _, _, k1, k2, _ = principal_frame(state.bspline, state.u_sample_pts, state.v_sample_pts)
    Vsampled, Fsampled = sample_bspline_surface(state.bspline,
                                                state.u_sample_pts, state.v_sample_pts)

    if state.radiusFunctionType == "safe":
        radiusValues = 1 / np.maximum(np.abs(k1), np.abs(k2)) * state.percentageRadius
    elif state.radiusFunctionType == "offset":
        radiusValues = state.offsetValue * np.ones_like(k1)
    elif state.radiusFunctionType == "mid":
        radiusValues = (1 / k1 + 1 / k2) * 0.5

    sphereCenters = (Vsampled +
                     state.bspline.normal(state.u_sample_pts, state.v_sample_pts).reshape(-1, 3)
                     * radiusValues.reshape(-1, 1))
    return [ser.surface_mesh("Sphere Centers Direct", sphereCenters, Fsampled,
                             color=(0.0, 0.0, 1.0))]


def compute_reference_structures(state: AppState) -> None:
    """Compute a dense quad mesh from the B-spline surface for reference."""
    init_radius_function(state)

    denseUsample = np.linspace(state.interval_u[0], state.interval_u[1], 250)
    denseVsample = np.linspace(state.interval_v[0], state.interval_v[1], 250)
    V, F = sample_bspline_surface(state.bspline, denseUsample, denseVsample)
    F = np.array(triangulate_quads(F))
    n = state.bspline.normal(denseUsample, denseVsample).reshape(-1, 3)
    Cv = V + n * sample_radius(state.radiusFunction, denseUsample, denseVsample).reshape(-1, 1)

    _, _, k1, k2, _ = principal_frame(state.bspline, denseUsample, denseVsample)
    R1 = 1 / k1
    R2 = 1 / k2

    state.focalSurface1 = [V + n * R1.reshape(-1, 1), F]
    state.focalSurface2 = [V + n * R2.reshape(-1, 1), F]
    state.reference = [V, F, n, Cv]


def compute_radius(state: AppState) -> np.ndarray:
    """Compute radius values at current sample points."""
    _, _, k1, k2, _ = principal_frame(state.bspline, state.u_sample_pts, state.v_sample_pts)
    state.percentageRadius = 0.75
    return (1 / k1 + 1 / k2) * 0.5


# ---------- Sampling & projection ----------

def sample_selection(state: AppState, interval_u: list, interval_v: list) -> None:
    """Update sample points from intervals."""
    state.u_sample_pts = np.linspace(interval_u[0], interval_u[1], state.bspline_u_sample_num)
    state.v_sample_pts = np.linspace(interval_v[0], interval_v[1], state.bspline_v_sample_num)


def project_to_reference_surface(state: AppState, V: np.ndarray):
    """Project points V onto the reference surface, interpolate sphere centers."""
    V_ref, F_ref, n_ref, Cv_ref = state.reference
    _, _, V_projected = igl.point_mesh_squared_distance(V, V_ref, F_ref)
    sphereCenters = interpolate_triangle_mesh(V_projected, V_ref, F_ref, Cv_ref)
    return V_projected, sphereCenters


# ---------- Scene snapshot helpers ----------

def bspline_scene(state: AppState) -> list:
    """Return serializer objects for the B-spline surface visualization."""
    if state.bspline is None or state.u_sample_pts is None:
        return []

    objects = []
    V, F = sample_bspline_surface(state.bspline, state.u_sample_pts, state.v_sample_pts)

    mesh_result = ser.surface_mesh("B-spline Surface", V, F,
                                   color=(0.5, 0.5, 0.5), show_edges=True)
    if isinstance(mesh_result, list):
        objects.extend(mesh_result)
    else:
        objects.append(mesh_result)

    K, _, n = bspline_curvatures(state.bspline, state.u_sample_pts, state.v_sample_pts)
    objects.append(ser.vector_quantity("B-spline Surface", "Normals",
                                       n.reshape(-1, 3), defined_on="vertices"))
    objects.append(ser.scalar_quantity("B-spline Surface", "Gaussian Curvature",
                                       K.reshape(-1), defined_on="vertices"))

    if state.theta_angles is not None:
        objects.append(ser.scalar_quantity("B-spline Surface", "theta_initial",
                                           state.theta_angles, defined_on="vertices"))

    try:
        normalsOpt = unit(state.bSplineOptimizer.unpack("normals").reshape(-1, 3))
        K_opt = state.bSplineOptimizer.unpack("K")
        objects.append(ser.vector_quantity("B-spline Surface", "Optimized Normals",
                                           normalsOpt, defined_on="vertices"))
        objects.append(ser.scalar_quantity("B-spline Surface", "Optimized Gaussian Curvature",
                                           K_opt, defined_on="vertices"))
    except Exception:
        pass

    controlPoints, controlEdges, _ = control_points_grid(state.bspline)
    objects.append(ser.curve_network("Control Mesh",
                                     controlPoints.reshape(-1, 3),
                                     controlEdges, color=(1.0, 0.0, 0.0)))
    objects.append(ser.point_cloud("Control Points",
                                   controlPoints.reshape(-1, 3),
                                   color=(0.0, 0.0, 0.0), radius=0.003))
    return objects


def reference_scene(state: AppState) -> list:
    """Return serializer objects for the reference surface."""
    if state.reference is None:
        return []

    V, F, n, Cv = state.reference
    objects = []

    mesh_result = ser.surface_mesh("Reference Surface", V, F, color=(0.8, 0.8, 0.8))
    if isinstance(mesh_result, list):
        objects.extend(mesh_result)
    else:
        objects.append(mesh_result)

    objects.append(ser.vector_quantity("Reference Surface", "Normals",
                                       n.reshape(-1, 3), defined_on="vertices"))
    objects.append(ser.surface_mesh("Sphere Centers Mesh", Cv, F, color=(0.6, 0.02, 0.8)))
    objects.append(ser.surface_mesh("Focal Surface 1",
                                    state.focalSurface1[0], state.focalSurface1[1],
                                    color=(0.8, 0.0, 0.0)))
    objects.append(ser.set_enabled("Focal Surface 1", False))
    objects.append(ser.surface_mesh("Focal Surface 2",
                                    state.focalSurface2[0], state.focalSurface2[1],
                                    color=(0.0, 0.8, 0.0)))
    objects.append(ser.set_enabled("Focal Surface 2", False))
    return objects


def cross_field_scene(state: AppState) -> list:
    """Return serializer objects for the direction field cross-field visualization."""
    if state.theta_angles is None or state.bspline is None:
        return []

    e1, e2, k1, k2, _ = principal_frame(state.bspline,
                                         state.u_sample_pts, state.v_sample_pts)
    t1 = (np.cos(state.theta_angles)[:, None] * e1
          + np.sin(state.theta_angles)[:, None] * e2)
    t2 = (np.cos(state.phi_angles)[:, None] * e1
          + np.sin(state.phi_angles)[:, None] * e2)
    lc1, lc2 = exchange_lconjugate_rulings_curve_tangents(e1, e2, k1, k2, t1, t2)

    sampledPoints = state.bspline(state.u_sample_pts, state.v_sample_pts).reshape(-1, 3)
    length = state.post_cross_field_len

    objects = []
    for name, d1, d2, colors in (
        ("L-conjugate Tangents_", t1, t2, [[1, 0, 0], [0, 0, 1]]),
        ("L-conjugate Rulings_", lc1, lc2, [[0, 0, 0], [1, 1, 1]]),
    ):
        arms = glyphs.cross_field(sampledPoints, d1.reshape(-1, 3), d2.reshape(-1, 3), length)
        for suffix, (P, E), color in zip(("_D1", "_D2"), arms, colors):
            objects.append(ser.curve_network(name + suffix, P, E, color=color))
    return objects


def remeshed_scene(state: AppState) -> list:
    """Return serializer objects for the remeshed quad surfaces."""
    objects = []
    if state.quadRemeshed_t is not None:
        objects.append(ser.surface_mesh(
            "Quad ConjugateTangents",
            state.quadRemeshed_t.vertices, state.quadRemeshed_t.faces,
            color=(0.8, 0.8, 0.0),
        ))
    if state.quadRemeshed_lc is not None:
        objects.append(ser.surface_mesh(
            "Quad Rulings",
            state.quadRemeshed_lc.vertices, state.quadRemeshed_lc.faces,
            color=(0.0, 0.8, 0.8),
        ))
        objects.append(ser.set_enabled("Quad Rulings", False))
    return objects


def lmesh_scene(state: AppState) -> list:
    """Return serializer objects for the L-mesh."""
    if state.lmesh is None:
        return []
    return lser.lmesh_scene(state.lmesh)


# ---------- Remeshing & L-mesh ----------

def load_remeshed_surfaces(state: AppState) -> list:
    """Load remeshed surfaces from output directory. Returns serializer objects."""
    quadRemeshed_t_path  = os.path.join(state.save_path, "t_Remeshed.obj")
    quadRemeshed_lc_path = os.path.join(state.save_path, "lc_Remeshed.obj")

    if not os.path.exists(quadRemeshed_t_path):
        print("Remeshed surface not found at:", quadRemeshed_t_path)
        return []
    if not os.path.exists(quadRemeshed_lc_path):
        print("Remeshed surface not found at:", quadRemeshed_lc_path)
        return []

    compute_reference_structures(state)
    V_ref, F_ref, n_ref, _ = state.reference

    V_t, F_t   = read_obj(quadRemeshed_t_path)
    V_lc, F_lc = read_obj(quadRemeshed_lc_path)
    V_t_projected, sphereCenters_t = project_to_reference_surface(state, V_t)
    V_lc_projected, _ = project_to_reference_surface(state, V_lc)

    quadRemeshed_t = Mesh()
    quadRemeshed_t.make_mesh(V_t_projected, F_t)

    midpointQuads = quadRemeshed_t.face_barycenters()
    _, _, midPoints_projected = igl.point_mesh_squared_distance(midpointQuads, V_ref, F_ref)
    faceNormals_t = interpolate_triangle_mesh(midPoints_projected, V_ref, F_ref,
                                               n_ref.reshape(-1, 3))

    quadRemeshed_lc = Mesh()
    quadRemeshed_lc.make_mesh(V_lc_projected, F_lc)

    state.quadRemeshed_t  = quadRemeshed_t
    state.quadRemeshed_lc = quadRemeshed_lc
    state.sphereCenters_t = sphereCenters_t
    state.faceNormals_t   = faceNormals_t

    return [
        ser.surface_mesh("Quad ConjugateTangents", V_t_projected, F_t,
                         color=(0.8, 0.8, 0.0)),
        ser.surface_mesh("Quad Rulings", V_lc_projected, F_lc,
                         color=(0.0, 0.8, 0.8)),
        ser.set_enabled("Quad Rulings", False),
    ]


def initialize_lmesh(state: AppState) -> list:
    """Initialize L-mesh structure from remeshed data. Returns serializer objects."""
    if state.quadRemeshed_t is None:
        print("L-mesh initialization failed: No remeshed quad mesh available.")
        return []

    baseMesh = state.quadRemeshed_t
    V_ref, F_ref, n_ref, SC_ref = state.reference

    midQuads = baseMesh.face_barycenters()
    projectedBaricenters = interpolate_triangle_mesh(midQuads, V_ref, F_ref, V_ref)
    centers  = interpolate_triangle_mesh(midQuads, V_ref, F_ref, SC_ref)
    radii    = np.linalg.norm(centers - projectedBaricenters, axis=1)
    normalsVertices   = interpolate_triangle_mesh(baseMesh.vertices, V_ref, F_ref, n_ref)
    distancePlanes    = -np.einsum('ij,ij->i', normalsVertices, baseMesh.vertices)
    planes            = np.column_stack((normalsVertices, distancePlanes))

    lmeshPlanes = Lmesh()
    lmeshPlanes.init_L_mesh(
        baseMesh, orientation=1, combinatorics="primal", mode="OrPlanes",
        planes=planes, centers=centers, radii=radii,
    )
    state.lmesh = lmeshPlanes
    return lser.lmesh_scene(state.lmesh)


# ---------- Cross-field export & remeshing ----------

def save_cross_field(state: AppState) -> None:
    """Save L-conjugate directions to file."""
    e1, e2, k1, k2, _ = principal_frame(state.bspline,
                                         state.u_sample_pts, state.v_sample_pts)
    t1 = (np.cos(state.theta_angles)[:, None] * e1
          + np.sin(state.theta_angles)[:, None] * e2)
    t2 = (np.cos(state.phi_angles)[:, None] * e1
          + np.sin(state.phi_angles)[:, None] * e2)
    lc1, lc2 = exchange_lconjugate_rulings_curve_tangents(e1, e2, k1, k2, t1, t2)

    folder = Path(state.save_path)
    if not folder.exists():
        folder.mkdir(parents=True, exist_ok=True)

    pts = state.bspline(state.u_sample_pts, state.v_sample_pts).reshape(-1, 3)
    export_crossfield(state.save_path, "t",  pts, t1.reshape(-1, 3), t2.reshape(-1, 3))
    export_crossfield(state.save_path, "lc", pts, lc1.reshape(-1, 3), lc2.reshape(-1, 3))

    V, F = sample_bspline_surface(state.bspline, state.u_sample_pts, state.v_sample_pts)
    F = triangulate_quads(F)
    write_obj(os.path.join(state.save_path, "mesh.obj"), V, F)


def save_pre_remesh_state(state: AppState) -> None:
    """Save bspline, sampling, directions, and radius function after remeshing."""
    save_dir = Path(state.save_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    data = {
        "bspline":           state.bspline,
        "normalsFlipped":    state.normalsFlipped,
        "u_sample_pts":      state.u_sample_pts,
        "v_sample_pts":      state.v_sample_pts,
        "theta_angles":      state.theta_angles,
        "phi_angles":        state.phi_angles,
        "radiusFunction":    state.radiusFunction,
        "percentageRadius":  state.percentageRadius,
        "radiusFunctionType": state.radiusFunctionType,
    }
    path = os.path.join(state.save_path, "pre_remesh_state.pkl")
    with open(path, "wb") as f:
        pickle.dump(data, f)
    print("Pre-remesh state saved to:", path)


def load_pre_remesh_state(state: AppState) -> list:
    """Load bspline, sampling, directions, and radius function from pickle.

    Returns serializer objects list (empty if file not found).
    """
    path = os.path.join(state.save_path, "pre_remesh_state.pkl")
    if not os.path.exists(path):
        print("Pre-remesh state not found at:", path)
        return []

    with open(path, "rb") as f:
        data = pickle.load(f)

    state.bspline            = data["bspline"]
    state.normalsFlipped     = data.get("normalsFlipped", False)
    state.u_sample_pts       = data["u_sample_pts"]
    state.v_sample_pts       = data["v_sample_pts"]
    state.theta_angles       = data["theta_angles"]
    state.phi_angles         = data["phi_angles"]
    state.radiusFunction     = data["radiusFunction"]
    state.percentageRadius   = data["percentageRadius"]
    state.radiusFunctionType = data["radiusFunctionType"]

    state.bspline_u_sample_num = len(state.u_sample_pts)
    state.bspline_v_sample_num = len(state.v_sample_pts)
    state.interval_u = [state.u_sample_pts[0], state.u_sample_pts[-1]]
    state.interval_v = [state.v_sample_pts[0], state.v_sample_pts[-1]]

    init_radius_function(state)
    compute_reference_structures(state)

    objects = bspline_scene(state)
    objects.extend(cross_field_scene(state))
    print("Pre-remesh state loaded from:", path)
    return objects


def run_remesher(state: AppState, integrated: bool = True) -> None:
    """Run external quad remesher on the current B-spline surface."""
    folder = Path(state.save_path)
    assert folder.exists(), f"Output directory {state.save_path} does not exist."

    files = os.listdir(state.save_path)
    required_files = {"mesh.obj", "t_D1.dat", "t_D2.dat", "lc_D1.dat", "lc_D2.dat"}
    assert required_files.issubset(files), \
        f"Required files {required_files} not found in {state.save_path}."

    mesh_path = Path(os.path.join(state.save_path, "mesh.obj")).resolve()
    D1_path   = Path(os.path.join(state.save_path, "t_D1.dat")).resolve()
    D2_path   = Path(os.path.join(state.save_path, "t_D2.dat")).resolve()
    lcD1_path = Path(os.path.join(state.save_path, "lc_D1.dat")).resolve()
    lcD2_path = Path(os.path.join(state.save_path, "lc_D2.dat")).resolve()

    mode = "integration" if integrated else "direct"
    commandt  = (f"{state.path_remesher} {mesh_path} {D1_path} {D2_path}"
                 f" {state.save_path} t {mode}")
    comandlc  = (f"{state.path_remesher} {mesh_path} {lcD1_path} {lcD2_path}"
                 f" {state.save_path} lc {mode}")

    print("Remeshing...")
    try:
        result = subprocess.run([commandt], shell=True,
                                capture_output=True, text=True, check=True)
        print(result.stdout)
        result = subprocess.run([comandlc], shell=True,
                                capture_output=True, text=True, check=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr}")


# ---------- Save all data ----------

def save_all(state: AppState) -> None:
    """Export all data of the current session in a pickle file."""
    save_dir = state.save_path
    folder = Path(save_dir)
    if not folder.exists():
        folder.mkdir(parents=True, exist_ok=True)

    compute_reference_structures(state)
    V_ref, F_ref, n_ref, Cv_ref = state.reference

    data_to_save = {
        "bspline":             state.bspline,
        "u_sample_pts":        state.u_sample_pts,
        "v_sample_pts":        state.v_sample_pts,
        "initial_RadiusPercentage": state.percentageRadius,
        "theta_angles":        state.theta_angles,
        "phi_angles":          state.phi_angles,
        "radiusFunction":      state.radiusFunction,
        "quadRemeshed_t":      state.quadRemeshed_t,
        "quadRemeshed_lc":     state.quadRemeshed_lc,
        "sphereCenters_t":     state.sphereCenters_t,
        "sphereCenters_lc":    state.sphereCenters_lc,
        "faceNormals_t":       state.faceNormals_t,
        "faceNormals_lc":      state.faceNormals_lc,
        "v_ref":               V_ref,
        "f_ref":               F_ref,
        "n_ref":               n_ref,
        "sphereCenters_ref":   Cv_ref,
    }

    if state.bSplineOptimizer.it is not None:
        data_to_save["Bspline_Optimization"] = {
            "report": state.bSplineOptimizer.get_report(),
            "data":   state.bSplineOptimizer.get_data_frame_optimization(),
            "log":    state.bSplineOptimizer.log,
        }
    if state.directionFieldOptimizer.it is not None:
        data_to_save["Lconjugacy_Optimization"] = {
            "report": state.directionFieldOptimizer.get_report(),
            "data":   state.directionFieldOptimizer.get_data_frame_optimization(),
            "log":    state.directionFieldOptimizer.log,
        }
    if state.postProcessingOptimizer.it is not None:
        data_to_save["Lmesh_Optimization"] = {
            "report": state.postProcessingOptimizer.get_report(),
            "data":   state.postProcessingOptimizer.get_data_frame_optimization(),
            "log":    state.postProcessingOptimizer.log,
        }
    if state.lmesh is not None:
        data_to_save["Lmesh"] = state.lmesh
    else:
        print("Warning: L-mesh not initialized, skipping L-mesh data saving.")

    out_path = os.path.join(save_dir, "lconjugacy_data.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(data_to_save, f)
    print("L-conjugacy construction data saved to:", out_path)


# ---------- Angle initialization helpers ----------

def linear_angle_distribution(state: AppState, startAngle=0, endAngle=75):
    """Initialize L-conjugate directions with a linear angle distribution."""
    sampledPoints = state.bspline(state.u_sample_pts, state.v_sample_pts).reshape(-1, 3)
    return np.linspace(np.deg2rad(startAngle), np.deg2rad(endAngle), len(sampledPoints))


def linear_angle_distribution_dir_v(state: AppState, startAngle=0, endAngle=85):
    """Linear angle distribution along V direction."""
    return np.tile(
        np.linspace(np.deg2rad(startAngle), np.deg2rad(endAngle), len(state.v_sample_pts)),
        len(state.u_sample_pts),
    ).flatten()


def linear_angle_distribution_dir_u(state: AppState, startAngle=0, endAngle=85):
    """Linear angle distribution along U direction."""
    return np.linspace(
        np.deg2rad(startAngle), np.deg2rad(endAngle), len(state.u_sample_pts)
    ).repeat(len(state.v_sample_pts), axis=0).flatten()


def cosine_angle_distribution_v(state: AppState, minAngle=0, maxAngle=90):
    """Cosine angle distribution along V direction."""
    numU = len(state.u_sample_pts)
    numV = len(state.v_sample_pts)
    t = np.linspace(0, 1, numV).repeat(numU, axis=0).flatten()
    return ((np.deg2rad(maxAngle) + np.deg2rad(minAngle)) * 0.5
            + (np.deg2rad(maxAngle) - np.deg2rad(minAngle)) * 0.5
            * np.cos(t * 2 * np.pi))


def cosine_angle_distribution_u(state: AppState, minAngle=0, maxAngle=90):
    """Cosine angle distribution along U direction."""
    numU = len(state.u_sample_pts)
    numV = len(state.v_sample_pts)
    t = np.tile(np.linspace(0, 1, numU), numV)
    return ((np.deg2rad(maxAngle) + np.deg2rad(minAngle)) * 0.5
            + (np.deg2rad(maxAngle) - np.deg2rad(minAngle)) * 0.5
            * np.cos(t * 2 * np.pi))
