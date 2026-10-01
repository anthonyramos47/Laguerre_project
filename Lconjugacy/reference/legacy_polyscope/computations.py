# Code/gui/computations.py
"""Shared computation functions that operate on AppState."""

import os
import pickle
import subprocess
import numpy as np
import igl

from pathlib import Path
from scipy.interpolate import bisplev

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
    visualize_frame,
    visualize_cross_field,
    write_obj,
    triangulate_quads,
    unit,
    read_obj,
    interpolate_triangle_mesh,
    mathematica_v,
    save_discrete_laguerre_channel_surface,
)
from hanan.geometry.mesh import Mesh

from LMesh.L_mesh import Lmesh

import polyscope as ps


# ---------- Radius & reference ----------


def radius_r_at_point(state:AppState, u, v):
    """
    Compute the sphere center at a given point on the B-spline surface, given the radius percentage.
    """
    _, _, k1, k2, n = principal_frame_at_point(state.bspline, u, v)

    if state.radiusFunctionType == "safe":
        r = 1/np.maximum(np.abs(k1), np.abs(k2)) * state.percentageRadius
    elif state.radiusFunctionType == "mid":
        r = (1 / k1 + 1 / k2) * 0.5
    elif state.radiusFunctionType == "offset":
        r = state.offsetValue

    return r


def init_radius_function(state: AppState) -> None:
    """Fit a B-spline to the radius values at sampled points."""

    sphereCenters = lambda u, v: [u,v, radius_r_at_point(state, u, v)]
    radiusSpline = fit_bspline_surface(sphereCenters,(0,1), (0,1), n_u=100, n_v=100, order=3)
    state.radiusFunction = radiusSpline

def compute_radius_function(state: AppState) -> None:
    """Compute radius values at current sample points and update the radius function."""
    _, _, k1, k2, _ = principal_frame(state.bspline, state.u_sample_pts, state.v_sample_pts)

    Vsampled, Fsampled = sample_bspline_surface(state.bspline, state.u_sample_pts, state.v_sample_pts)

    if state.radiusFunctionType == "safe":
        radiusValues = 1/np.maximum(np.abs(k1), np.abs(k2)) * state.percentageRadius
    elif state.radiusFunctionType == "offset":
        radiusValues = state.offsetValue * np.ones_like(k1)
    elif state.radiusFunctionType == "mid":
        radiusValues = (1 / k1 + 1 / k2) * 0.5
    
    sphereCenters = Vsampled + state.bspline.normal(state.u_sample_pts, state.v_sample_pts).reshape(-1, 3) * radiusValues.reshape(-1, 1)

    ps.register_surface_mesh("Sphere Centers Direct", sphereCenters, Fsampled, color=(0.0, 0.0, 1.0))
    

def compute_reference_structures(state: AppState) -> None:
    """Compute a dense quad mesh from the B-spline surface for reference."""
    init_radius_function(state)

    denseUsample =np.linspace(state.interval_u[0], state.interval_u[1], 250)
    denseVsample =np.linspace(state.interval_v[0], state.interval_v[1], 250)
    V, F = sample_bspline_surface(state.bspline, denseUsample, denseVsample)
    F = np.array(triangulate_quads(F))
    n = state.bspline.normal(denseUsample, denseVsample).reshape(-1, 3)
    Cv = V + n * sample_radius(state.radiusFunction, denseUsample, denseVsample).reshape(-1, 1)

    _, _, k1, k2, _ = principal_frame(state.bspline, denseUsample, denseVsample)

    R1 = 1/k1 
    R2 = 1/k2

    focalVertices1 = V + n * R1.reshape(-1, 1)
    focalVertices2 = V + n * R2.reshape(-1, 1)

    state.focalSurface1 = [focalVertices1, F]
    state.focalSurface2 = [focalVertices2, F]

    state.reference = [V, F, n, Cv]


def compute_radius(state: AppState) -> np.ndarray:
    """Compute radius values at current sample points."""
    _, _, k1, k2, _ = principal_frame(state.bspline, state.u_sample_pts, state.v_sample_pts)
    state.percentageRadius = 0.75
    radiusValues = (1 / k1 + 1 / k2) * 0.5
    return radiusValues


# ---------- Sampling & projection ----------

def sample_selection(state: AppState, interval_u: list, interval_v: list) -> None:
    """Update sample points from intervals."""
    state.u_sample_pts = np.linspace(interval_u[0], interval_u[1], state.u_sample_num)
    state.v_sample_pts = np.linspace(interval_v[0], interval_v[1], state.v_sample_num)


def project_to_reference_surface(state: AppState, V: np.ndarray):
    """Project points V onto the reference surface, interpolate sphere centers."""
    V_ref, F_ref, n_ref, Cv_ref = state.reference
    _, _, V_projected = igl.point_mesh_squared_distance(V, V_ref, F_ref)
    sphereCenters = interpolate_triangle_mesh(V_projected, V_ref, F_ref, Cv_ref)
    return V_projected, sphereCenters


# ---------- Visualization helpers ----------

def update_visualization(state: AppState) -> None:
    """Visualize B-spline surface with Polyscope."""
    if state.v_sample_num < 2 or state.u_sample_num < 2:
        return

    V, F = sample_bspline_surface(state.bspline, state.u_sample_pts, state.v_sample_pts)
    ps_mesh = ps.register_surface_mesh("B-spline Surface", V, F, color=(0.5, 0.5, 0.5), edge_color=(0, 0, 0), edge_width=1.1)
    if state.theta_angles is not None:
        ps_mesh.add_scalar_quantity("theta_initial", state.theta_angles, enabled=True)
    K, _, n = bspline_curvatures(state.bspline, state.u_sample_pts, state.v_sample_pts)
    try:
        normalsOpt = unit(state.bSplineOptimizer.unpack("normals").reshape(-1, 3))
        K_opt = state.bSplineOptimizer.unpack("K")
        ps_mesh.add_vector_quantity("Optimized Normals",
                                     normalsOpt, 
                                     defined_on="vertices", 
                                     enabled=True, 
                                     length=0.8)
        ps_mesh.add_scalar_quantity("Optimized Gaussian Curvature",
                                     K_opt, 
                                     defined_on="vertices", 
                                     cmap='coolwarm')
    except:
        pass


    ps_mesh.add_vector_quantity("Normals", 
                                n.reshape(-1, 3), 
                                defined_on="vertices", 
                                enabled=True, 
                                length=0.3)
    ps_mesh.add_scalar_quantity("Gaussian Curvature", 
                                K.reshape(-1),
                                defined_on="vertices",
                                cmap='coolwarm')
    
    controlPoints, controlEdges, _ = control_points_grid(state.bspline)
    ps.register_curve_network("Control Mesh",
                                controlPoints.reshape(-1, 3),
                                controlEdges, 
                                radius=0.001, 
                                color=(1.0, 0.0, 0.0))
    
    ps.register_point_cloud("Control Points", 
                            controlPoints.reshape(-1, 3),
                            radius=0.003, 
                            color=(0.0, 0.0, 0.0))


def visualize_reference_surface(state: AppState) -> None:
    """Visualize reference surface in Polyscope."""
    V, F, n, Cv = state.reference

    ps_mesh = ps.register_surface_mesh("Reference Surface", V, F, color=(0.8, 0.8, 0.8), edge_color=(0, 0, 0), edge_width=0.3)
    ps_mesh.add_vector_quantity("Normals", n.reshape(-1, 3), defined_on="vertices")
    ps.register_surface_mesh("Sphere Centers Mesh", Cv, F, color=(0.6, 0.02, 0.8), edge_color=(0, 0, 0), edge_width=1.5)
    ps.register_surface_mesh("Focal Surface 1", state.focalSurface1[0], state.focalSurface1[1], color=(0.8, 0.0, 0.0), edge_color=(0, 0, 0), edge_width=0.5, enabled=False)
    ps.register_surface_mesh("Focal Surface 2", state.focalSurface2[0], state.focalSurface2[1], color=(0.0, 0.8, 0.0), edge_color=(0, 0, 0), edge_width=0.5, enabled=False)

# ---------- Remeshing & L-mesh ----------

def load_remeshed_surfaces(state: AppState) -> None:
    """Load remeshed surfaces from output directory."""
    quadRemeshed_t_path  = os.path.join(state.save_path, "t_Remeshed.obj")
    quadRemeshed_lc_path = os.path.join(state.save_path, "lc_Remeshed.obj")

    if not os.path.exists(quadRemeshed_t_path):
        print("Remeshed surface not found at:", quadRemeshed_t_path)
        return
    if not os.path.exists(quadRemeshed_lc_path):
        print("Remeshed surface not found at:", quadRemeshed_lc_path)
        return

    # Get the reference surface ready for projection and interpolation
    compute_reference_structures(state)
    V_ref, F_ref, n_ref, _ = state.reference

    # Read remeshed surfaces and project them onto the reference surface, also interpolate sphere centers for later use
    V_t, F_t   = read_obj(quadRemeshed_t_path)
    V_lc, F_lc = read_obj(quadRemeshed_lc_path)
    V_t_projected, sphereCenters_t = project_to_reference_surface(state, V_t)
    V_lc_projected, _ = project_to_reference_surface(state, V_lc)


    quadRemeshed_t = Mesh()
    quadRemeshed_t.make_mesh(V_t_projected, F_t)

    midpointQuads = quadRemeshed_t.face_barycenters()
    _, _, midPoints_projected = igl.point_mesh_squared_distance(midpointQuads, V_ref, F_ref)
    normalsSplines = n_ref.reshape(-1, 3)
    faceNormals_t = interpolate_triangle_mesh(midPoints_projected, V_ref, F_ref, normalsSplines)

    quadRemeshed_lc = Mesh()
    quadRemeshed_lc.make_mesh(V_lc_projected, F_lc)
    midpointQuads = quadRemeshed_lc.face_barycenters()
    
    state.quadRemeshed_t = quadRemeshed_t
    state.quadRemeshed_lc = quadRemeshed_lc
    state.sphereCenters_t = sphereCenters_t
    #state.sphereCenters_lc = sphereCenters_lc
    state.faceNormals_t = faceNormals_t
    #state.faceNormals_lc = faceNormals_lc

    ps.register_surface_mesh("Quad ConjugateTangets", V_t_projected, F_t, color=(0.8, 0.8, 0.0), enabled=True)
    ps.register_surface_mesh("Quad Rulings", V_lc_projected, F_lc, color=(0.0, 0.8, 0.8), enabled=False)

    # # Load Radius function from pickle file
    # radius_function_path = os.path.join(state.save_path, "radius_function.pkl")
    # if os.path.exists(radius_function_path):
    #     with open(radius_function_path, "rb") as f:
    #         state.radiusFunction = pickle.load(f)
    # else:        
    #     print("Radius function file not found at:", radius_function_path)
    #     print("Using existing radius function in state, if available.")
        



def initialize_lmesh(state: AppState) -> None:
    """Initialize L-mesh structure from remeshed data."""
    if state.quadRemeshed_t is None:
        print("L-mesh initialization failed: No remeshed quad mesh available.")
        return

    baseMesh = state.quadRemeshed_t

    quadVertices = baseMesh.vertices
    quadFaces = baseMesh.faces
    #sphereCenters = state.sphereCenters_t
    V_ref, F_ref, n_ref, SC_ref = state.reference

    midQuads = baseMesh.face_barycenters()
    projectedBaricenters = interpolate_triangle_mesh(midQuads, V_ref, F_ref, V_ref)  
    # Project midpoints of quads onto reference surface for better interpolation of sphere centers
    centers = interpolate_triangle_mesh(midQuads, V_ref, F_ref, SC_ref)
    radii = np.linalg.norm(centers - projectedBaricenters, axis=1)
    normalsVertices = interpolate_triangle_mesh(baseMesh.vertices, V_ref, F_ref, n_ref)
    distancePlanes = -np.einsum('ij,ij->i', normalsVertices, baseMesh.vertices)
    planes = np.column_stack((normalsVertices, distancePlanes))
    lmeshPlanes = Lmesh()
    lmeshPlanes.init_L_mesh(
        baseMesh, orientation=1, combinatorics="primal", mode="OrPlanes",
        planes=planes, centers=centers, radii=radii,
    )
    state.lmesh = lmeshPlanes
    #Radii computed as the mean distance from centers to quad vertices
    #radii =  np.mean(np.linalg.norm(quadVertices[quadFaces] - centers[:,None,:], axis=2), axis=1)

    # #Initialization using spheres passing through the vertices of the quads and with center in the sphere centers mesh
    # lmeshSpheres = Lmesh()
    # lmeshSpheres.init_L_mesh(baseMesh, orientation=1, combinatorics="primal", mode="SphCongruence", centers=centers, radii=radii, planes=planes)
    # state.lmesh = lmeshSpheres

    # visQuadVertices = ps.register_point_cloud("QuadVerts", baseMesh.vertices)
    # visQuadVertices.add_vector_quantity("normals", normalsVertices, length=0.5, enabled=True)
    # sphereCenters = ps.register_point_cloud("phereCenter", centers, color=(1,0,0))
    # normalSpheres= unit( midQuads - centers)
    # sphereCenters.add_vector_quantity("normals", normalSpheres, length=0.5, enabled=True)

    #Initializaiton with spheres at vertices
    # inn_v = baseMesh.inner_vertices()
    # # uvVertices = project_points_to_bspline(state.bspline, baseMesh.vertices)
    # sphereCenters = interpolate_triangle_mesh(baseMesh.vertices, V_ref, F_ref, SC_ref)   
    # radii = np.linalg.norm( sphereCenters - baseMesh.vertices, axis=1)
    # # Init L-mesh with spheres
    # lmeshSpheres = Lmesh()
    # lmeshSpheres.init_L_mesh(baseMesh, orientation=-1, mode="SphCongruence", centers=sphereCenters[inn_v], radii=radii[inn_v])
    # state.lmesh = lmeshSpheres


# ---------- Cross-field export & remeshing ----------

def save_cross_field(state: AppState) -> None:
    """Save L-conjugate directions to file."""
    e1, e2, k1, k2, _ = principal_frame(state.bspline, state.u_sample_pts, state.v_sample_pts)

    t1 = np.cos(state.theta_angles)[:, None] * e1 + np.sin(state.theta_angles)[:, None] * e2
    t2 = np.cos(state.phi_angles)[:, None] * e1 + np.sin(state.phi_angles)[:, None] * e2

    lc1, lc2 = exchange_lconjugate_rulings_curve_tangents(e1, e2, k1, k2, t1, t2)

    folder = Path(state.save_path)
    if not folder.exists():
        folder.mkdir(parents=True, exist_ok=True)

    print("Saving L-conjugate directions to:", state.save_path)

    pts = state.bspline(state.u_sample_pts, state.v_sample_pts).reshape(-1, 3)
    export_crossfield(state.save_path, "t", pts, t1.reshape(-1, 3), t2.reshape(-1, 3))
    export_crossfield(state.save_path, "lc", pts, lc1.reshape(-1, 3), lc2.reshape(-1, 3))

    V, F = sample_bspline_surface(state.bspline, state.u_sample_pts, state.v_sample_pts)
    F = triangulate_quads(F)
    write_obj(os.path.join(state.save_path, "mesh.obj"), V, F)


def save_pre_remesh_state(state: AppState) -> None:
    """Save bspline, sampling, directions, and radius function after remeshing."""
    save_dir = Path(state.save_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    data = {
        "bspline": state.bspline,
        "normalsFlipped": state.normalsFlipped,
        "u_sample_pts": state.u_sample_pts,
        "v_sample_pts": state.v_sample_pts,
        "theta_angles": state.theta_angles,
        "phi_angles": state.phi_angles,
        "radiusFunction": state.radiusFunction,
        "percentageRadius": state.percentageRadius,
        "radiusFunctionType": state.radiusFunctionType,
    }

    path = os.path.join(state.save_path, "pre_remesh_state.pkl")
    with open(path, "wb") as f:
        pickle.dump(data, f)
    print("Pre-remesh state saved to:", path)


def load_pre_remesh_state(state: AppState) -> bool:
    """Load bspline, sampling, directions, and radius function from pickle.

    Returns True if loaded successfully, False if file not found.
    """
    path = os.path.join(state.save_path, "pre_remesh_state.pkl")
    if not os.path.exists(path):
        print("Pre-remesh state not found at:", path)
        return False

    with open(path, "rb") as f:
        data = pickle.load(f)

    state.bspline            = data["bspline"]
    state.u_sample_pts       = data["u_sample_pts"]
    state.v_sample_pts       = data["v_sample_pts"]
    state.theta_angles       = data["theta_angles"]
    state.phi_angles         = data["phi_angles"]
    state.radiusFunction     = data["radiusFunction"]
    state.percentageRadius   = data["percentageRadius"]
    state.radiusFunctionType = data["radiusFunctionType"]

    state.u_sample_num = len(state.u_sample_pts)
    state.v_sample_num = len(state.v_sample_pts)

    state.interval_u = [state.u_sample_pts[0], state.u_sample_pts[-1]]
    state.interval_v = [state.v_sample_pts[0], state.v_sample_pts[-1]]

    update_visualization(state)

    e1, e2, k1, k2, _ = principal_frame(state.bspline, state.u_sample_pts, state.v_sample_pts)
    
    t1 = np.cos(state.theta_angles)[:, None] * e1 + np.sin(state.theta_angles)[:, None] * e2
    t2 = np.cos(state.phi_angles)[:, None]   * e1 + np.sin(state.phi_angles)[:, None]   * e2

    lc1, lc2 = exchange_lconjugate_rulings_curve_tangents(e1, e2, k1, k2, t1, t2)
    sampledPoints = state.bspline(state.u_sample_pts, state.v_sample_pts).reshape(-1, 3)
   
    visualize_cross_field("L-conjugate Tangents_", sampledPoints, t1.reshape(-1, 3), t2.reshape(-1, 3), state.lenghtCrossField, colors=[[1, 0, 0], [0, 0, 1]], radius=state.radiusCrossField)
    visualize_cross_field("L-conjugate Rulings_", sampledPoints, lc1.reshape(-1, 3), lc2.reshape(-1, 3), state.lenghtCrossField, colors=[[0, 0, 0], [1, 1, 1]], radius=state.radiusCrossField)

    # Recompute reference structures with the loaded bspline and radius
    init_radius_function(state)
    compute_reference_structures(state)

    print("Pre-remesh state loaded from:", path)
    return True


def run_remesher(state: AppState, integrated: bool = True) -> None:
    """Run external quad remesher on the current B-spline surface."""
    folder = Path(state.save_path)
    assert folder.exists(), f"Output directory {state.save_path} does not exist."

    files = os.listdir(state.save_path)
    required_files = {"mesh.obj", "t_D1.dat", "t_D2.dat", "lc_D1.dat", "lc_D2.dat"}
    assert required_files.issubset(files), f"Required files {required_files} not found in {state.save_path}."

    mesh_path = Path(os.path.join(state.save_path, "mesh.obj")).resolve()
    D1_path = Path(os.path.join(state.save_path, "t_D1.dat")).resolve()
    D2_path = Path(os.path.join(state.save_path, "t_D2.dat")).resolve()
    lcD1_path = Path(os.path.join(state.save_path, "lc_D1.dat")).resolve()
    lcD2_path = Path(os.path.join(state.save_path, "lc_D2.dat")).resolve()

    mode = "integration" if integrated else "direct"
    commandt = f"{state.path_remesher} {mesh_path} {D1_path} {D2_path} {state.save_path} t {mode}"
    comandlc = f"{state.path_remesher} {mesh_path} {lcD1_path} {lcD2_path} {state.save_path} lc {mode}"

    print("Remeshing...")
    try:
        result = subprocess.run([commandt], shell=True, capture_output=True, text=True, check=True)
        print(result.stdout)
        result = subprocess.run([comandlc], shell=True, capture_output=True, text=True, check=True)
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
        "bspline": state.bspline,
        "u_sample_pts": state.u_sample_pts,
        "v_sample_pts": state.v_sample_pts,
        "initial_RadiusPercentage": state.percentageRadius,
        "theta_angles": state.theta_angles,
        "phi_angles": state.phi_angles,
        "radiusFunction": state.radiusFunction,
        "quadRemeshed_t": state.quadRemeshed_t,
        "quadRemeshed_lc": state.quadRemeshed_lc,
        "sphereCenters_t": state.sphereCenters_t,
        "sphereCenters_lc": state.sphereCenters_lc,
        "faceNormals_t": state.faceNormals_t,
        "faceNormals_lc": state.faceNormals_lc,
        "v_ref": V_ref,
        "f_ref": F_ref,
        "n_ref": n_ref,
        "sphereCenters_ref": Cv_ref,
    }

    if state.bSplineOptimizer.it is not None:
        data_to_save["Bspline_Optimization"] = {
            "report": state.bSplineOptimizer.get_report(),
            "data": state.bSplineOptimizer.get_data_frame_optimization(),
            "log": state.bSplineOptimizer.log
        }

    if state.directionFieldOptimizer.it is not None:
        data_to_save["Lconjugacy_Optimization"] = {
            "report": state.directionFieldOptimizer.get_report(),
            "data": state.directionFieldOptimizer.get_data_frame_optimization(),
            "log": state.directionFieldOptimizer.log
        }

    if state.postProcessingOptimizer.it is not None:
        data_to_save["Lmesh_Optimization"] = {
            "report": state.postProcessingOptimizer.get_report(),
            "data": state.postProcessingOptimizer.get_data_frame_optimization(),
            "log": state.postProcessingOptimizer.log
        }

    if state.lmesh is not None:
        data_to_save["Lmesh"] = state.lmesh
    else:
        print("Warning: L-mesh not initialized, skipping L-mesh data saving.")

    with open(os.path.join(save_dir, "lconjugacy_data.pkl"), "wb") as f:
        pickle.dump(data_to_save, f)

    print("L-conjugacy construction data saved to:", os.path.join(save_dir, "lconjugacy_data.pkl"))


# ---------- Angles Initialization ----------

def linear_angle_distribution(state: AppState, startAngle=0, endAngle=75) -> None:
    """Initialize L-conjugate directions with a linear angle distribution."""
    sampledPoints = state.bspline(state.u_sample_pts, state.v_sample_pts).reshape(-1, 3)
    angle = np.linspace(np.deg2rad(startAngle), np.deg2rad(endAngle), len(sampledPoints))
    return angle

def linear_angle_distribution_dir_v(state: AppState, startAngle=0, endAngle=85) -> None:
    """Initialize L-conjugate directions with a linear angle distribution along the U directions, where startin angle is at V=0 and ending angle is at V=1."""
    
    angles = np.tile(np.linspace(np.deg2rad(startAngle), np.deg2rad(endAngle), len(state.v_sample_pts)), len(state.u_sample_pts)).flatten()

    return angles

def linear_angle_distribution_dir_u(state: AppState, startAngle=0, endAngle=85) -> None:
    """Initialize L-conjugate directions with a linear angle distribution along the V directions, where starting angle is at U=0 and ending angle is at U=1."""
    angles = np.linspace(np.deg2rad(startAngle), np.deg2rad(endAngle), len(state.u_sample_pts)).repeat(len(state.v_sample_pts), axis=0).flatten()

    return angles

def cosine_angle_distribution_v(state: AppState, minAngle=0, maxAngle=90) -> None:
    """Initialize L-conjugate directions with a cosine angle distribution."""
    
    numU = len(state.u_sample_pts)
    numV = len(state.v_sample_pts)
    t = np.linspace(0, 1, numV).repeat(numU, axis=0).flatten()  # Parameter t goes from 0 to 1 along the V direction
    angle = (np.deg2rad(maxAngle) + np.deg2rad(minAngle)) * 0.5 + (np.deg2rad(maxAngle) - np.deg2rad(minAngle)) * 0.5 * (np.cos(t * 2 * np.pi))  # Cosine distribution between minAngle and maxAngle
    return angle

def cosine_angle_distribution_u(state: AppState, minAngle=0, maxAngle=90) -> None:
    """Initialize L-conjugate directions with a cosine angle distribution."""
    
    numU = len(state.u_sample_pts)
    numV = len(state.v_sample_pts)
    t = np.tile(np.linspace(0, 1, numU), numV)  # Parameter t goes from 0 to 1 along the V direction
    angle = (np.deg2rad(maxAngle) + np.deg2rad(minAngle)) * 0.5 + (np.deg2rad(maxAngle) - np.deg2rad(minAngle)) * 0.5 * (np.cos(t * 2 * np.pi))  # Cosine distribution between minAngle and maxAngle
    return angle

    