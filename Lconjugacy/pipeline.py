"""
L-conjugacy pipeline — the original Polyscope app's logic, without a GUI.

Ported from the original app (commit 9624f23, kept read-only in ``legacy_polyscope/``):
``computations.py`` and the logic of ``bspline_panel.py`` / ``lconjugacy_panel.py``.
Numerics are unchanged; every Polyscope call is gone — functions return arrays and the
caller (``notebooks/lconjugacy_lab.ipynb``) decides what to show. Lines marked
``ORIGINAL:`` keep behaviour of the original app that we still have to review.

Stages
    0. load_surface          surface JSON, sampling, radius function, dense reference
    1. B-spline              flip_normals, set_sampling, optimize_bspline, export_bspline
    2. L-conjugate field     initial_angles, init_directions
                             (setup_df_optimizer / run_df_optimizer: kept but not used for
                             the project — slow, and the radius write-back is unresolved)
    3. Remeshing             save_cross_field, run_remesher (external), save_pre_remesh_state,
                             load_remeshed_surfaces
    4. L-mesh                initialize_lmesh, setup_lmesh_optimizer, run_lmesh_optimizer,
                             finalize_lmesh, export_lmesh (ported to the swapped Lmesh:
                             spheres on top.vertices, planes on top.faces)
"""

from __future__ import annotations

import os
import pickle
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import igl
import numpy as np

from hanan.geometry.construction import interpolate_triangle_mesh
from hanan.geometry.io import read_obj, triangulate_quads, write_obj
from hanan.geometry.mesh import Mesh
from hanan.optimization import Optimizer, PrevIterationRegularizer

from Lconjugacy.utils.bsplines_functions import (
    bspline_curvatures,
    compute_crossfield_singularities,
    compute_rij_transport_vectorized,
    control_points_grid,
    exchange_lconjugate_rulings_curve_tangents,
    export_crossfield,
    fit_bspline_surface,
    init_lconjugate_boundary,
    init_lconjugate_directions,
    principal_frame,
    principal_frame_at_point,
    read_bspline_json,
    sample_bspline_surface,
    sample_radius,
    save_bspline_to_json,
)
from Lconjugacy.utils.directionsOpt.angleBoundary import AngleBoundary
from Lconjugacy.utils.directionsOpt.angleLconjugates import AngleLconjugate
from Lconjugacy.utils.directionsOpt.DirectionFieldFairness import DirectionFieldSmoothness
from Lconjugacy.utils.directionsOpt.LconjugateOpt import Lconjugacy
from LMesh.L_mesh import Lmesh
from LMesh.Optimization.ContactPointsGlide import ContactPointsGlide
from LMesh.Optimization.ContactPointsProximity import ContactPointsProximity
from LMesh.Optimization.Or_contactFree import Or_contactFree
from Lconjugacy.utils.lmeshOpt.QLmesh_reg_faces_lconj import QLmesh_reg_faces_lconj
from Lconjugacy.utils.lmeshOpt.QLmesh_reg_strips_lconj import QLmesh_reg_strips_lconj
from LMesh.Optimization.Tangential_Dist import Tangential_Dist

HERE = Path(__file__).resolve().parent
SURFACE_DIR = HERE.parent / "data" / "bsplines"
OUT_DIR = HERE.parent / "notebooks" / "out" / "lconjugacy"   # git-ignored, local only
# Quad remesher binary: set QUAD_REMESHER in the environment, or st.path_remesher in the notebook.
DEFAULT_REMESHER = os.environ.get("QUAD_REMESHER", "/path/to/quadRemesher")

# Weights of the original post-optimisation panel ("Run Optimization" / "Last Opt").
# Keys are term names; "cv_*" were "cf_*" before the swap (sphere centres cf -> cv).
LMESH_WEIGHTS = {
    "or_contact": 1.0, "tangentialDist": 1e-1, "quadFair": 1e-3, "quadFairSph": 1e-3,
    "contactProximity": 1e-4, "contactGliding": 1e-4, "cv_proximity": 0.0, "cv_gliding": 0.0,
}
LMESH_FINAL_WEIGHTS = {
    "or_contact": 1.0, "tangentialDist": 0.0, "quadFair": 0.0, "quadFairSph": 0.0,
    "contactProximity": 0.0, "contactGliding": 0.0, "cv_proximity": 0.0, "cv_gliding": 0.0,
    "previous_iteration_regularizer": 0.0,
}
# Paper: both fairness energies are multiplied by 0.1 every ten steps.
FAIRNESS_TERMS = ("quadFair", "quadFairSph")
FAIRNESS_DECAY = 0.1
FAIRNESS_DECAY_EVERY = 10

INIT_METHODS = ("constant", "linear", "linearU", "linearV", "cosineU", "cosineV", "boundary")


@dataclass
class LconjugacyState:
    """The fields of the original ``AppState`` used by stages 0–3 (same defaults)."""
    # B-spline
    name: str = ""
    bspline: object = None
    bsplineOriginal: object = None
    normalsFlipped: bool = False          # flip_normals (reverse u), as in the original app
    uvSwapped: bool = False               # swap_uv (u <-> v roles; also flips the normals)
    u_sample_pts: np.ndarray = None
    v_sample_pts: np.ndarray = None
    u_sample_num: int = 20
    v_sample_num: int = 20
    interval_u: list = field(default_factory=lambda: [0.0, 1.0])
    interval_v: list = field(default_factory=lambda: [0.0, 1.0])
    radiusFunction: object = None
    percentageRadius: float = 0.75
    offsetValue: float = 0.2
    radiusFunctionType: str = "safe"      # "safe" | "mid" | "offset"
    reference: list = None                # [V, F, n, Cv] dense (250x250) reference
    focalSurface1: list = None
    focalSurface2: list = None

    # Direction field
    theta_angles: np.ndarray = None
    phi_angles: np.ndarray = None
    initial_theta: float = 45.0           # degrees
    target_angle: float = 30.0            # degrees (AngleLconjugate term)
    init_method: str = "constant"
    boundary_data: tuple = None           # from init_lconjugate_boundary
    weight_DF: dict = field(default_factory=lambda: {
        "lconjugacy": 1.1, "smooth": 0.0001, "angle": 0.0, "angleBoundary": 0.5})
    directionFieldOptimizer: Optimizer = None
    rij: np.ndarray = None                # optimised radius control values (see update_variables_from_optimizer)
    singularity_vertices: np.ndarray = None
    singularity_indices: np.ndarray = None

    # Remeshed data
    quadRemeshed_t: object = None
    quadRemeshed_lc: object = None
    sphereCenters_t: np.ndarray = None
    faceNormals_t: np.ndarray = None

    # L-mesh (stage 4)
    lmesh: object = None
    lmeshOptimizer: Optimizer = None
    weight_lmesh: dict = field(default_factory=lambda: dict(LMESH_WEIGHTS))

    # Paths
    experiment_name: str = "experiment0"
    surface_path: str = str(SURFACE_DIR)
    output_path: str = str(OUT_DIR)
    save_path: str = ""
    path_remesher: str = DEFAULT_REMESHER


# ── stage 0: setup ───────────────────────────────────────────────────────────

def load_surface(name: str, experiment: str = "experiment0", surface_dir=SURFACE_DIR,
                 out_dir=OUT_DIR, fit: bool = True, **settings) -> LconjugacyState:
    """Read ``<surface_dir>/<name>.json`` and sample it.

    fit=True also runs the original app's start-up fitting (radius function + dense
    reference, see set_reference); fit=False only loads and samples, so the (u, v)
    range can be chosen first.
    ``settings`` override state fields before sampling (e.g. ``u_sample_num=30``,
    ``interval_u=[0.1, 0.9]``, ``radiusFunctionType="mid"``).
    """
    state = LconjugacyState(name=name, experiment_name=experiment,
                            surface_path=str(surface_dir), output_path=str(out_dir))
    for key, value in settings.items():
        if not hasattr(state, key):
            raise AttributeError(f"unknown setting '{key}'")
        setattr(state, key, value)
    state.save_path = os.path.join(state.output_path, state.name, state.experiment_name)
    state.bspline = read_bspline_json(os.path.join(state.surface_path, name + ".json"))
    # A copy: the original kept the same object, so in-place flips/swaps also changed
    # the "original" and reset_bspline could not undo them.
    state.bsplineOriginal = state.bspline.clone()
    sample_selection(state, state.interval_u, state.interval_v)
    if fit:
        # The original app called init_radius_function() here too, but
        # compute_reference_structures() refits it from the same input straight away.
        compute_reference_structures(state)
    return state


def sample_selection(state, interval_u, interval_v) -> None:
    """Update sample points from intervals."""
    state.u_sample_pts = np.linspace(interval_u[0], interval_u[1], state.u_sample_num)
    state.v_sample_pts = np.linspace(interval_v[0], interval_v[1], state.v_sample_num)


def radius_r_at_point(state, u, v):
    """Sphere radius at (u, v) for the chosen radius function type."""
    _, _, k1, k2, n = principal_frame_at_point(state.bspline, u, v)
    if state.radiusFunctionType == "safe":
        r = 1 / np.maximum(np.abs(k1), np.abs(k2)) * state.percentageRadius
    elif state.radiusFunctionType == "mid":
        r = (1 / k1 + 1 / k2) * 0.5
    elif state.radiusFunctionType == "offset":
        r = state.offsetValue
    return r


RADIUS_FIT = dict(n_u=100, n_v=100, order=3)
EXACT_RADIUS_FIT = False   # True: the original point-by-point evaluation (~4.5 s, bit-identical)


def _radius_from_curvatures(state, k1, k2):
    if state.radiusFunctionType == "safe":
        return 1 / np.maximum(np.abs(k1), np.abs(k2)) * state.percentageRadius
    if state.radiusFunctionType == "mid":
        return (1 / k1 + 1 / k2) * 0.5
    if state.radiusFunctionType == "offset":
        return np.full(np.shape(k1), float(state.offsetValue))
    raise ValueError(f"unknown radius function type '{state.radiusFunctionType}'")


def init_radius_function(state, exact=None) -> None:
    """Fit a B-spline (u, v, r(u, v)) to the radius function over [0,1]².

    Interpolates r at the Greville points of a 100x100 cubic basis, exactly like the
    original (fit_bspline_surface). By default the curvatures on that grid are computed
    in one vectorised call (~0.1 s); ``exact=True`` (or EXACT_RADIUS_FIT) evaluates them
    point by point as the original did (~4.5 s) — the results agree to ~1e-15.
    """
    exact = EXACT_RADIUS_FIT if exact is None else exact
    if exact:
        sphereCenters = lambda u, v: [u, v, radius_r_at_point(state, u, v)]
        state.radiusFunction = fit_bspline_surface(sphereCenters, (0, 1), (0, 1), **RADIUS_FIT)
        return
    import splipy as sp
    import splipy.surface_factory as surface_factory
    n_u, n_v, order = RADIUS_FIT["n_u"], RADIUS_FIT["n_v"], RADIUS_FIT["order"]

    def knots(n):   # same open uniform knot vector as fit_bspline_surface
        return [0.0] * order + list(np.linspace(0.0, 1.0, n - order + 2)[1:-1]) + [1.0] * order

    basis_u = sp.BSplineBasis(order=order, knots=knots(n_u))
    basis_v = sp.BSplineBasis(order=order, knots=knots(n_v))
    u_pts, v_pts = np.asarray(basis_u.greville(), float), np.asarray(basis_v.greville(), float)
    _, _, k1, k2, _ = principal_frame(state.bspline, u_pts, v_pts)
    r = _radius_from_curvatures(state, np.asarray(k1).reshape(n_u, n_v), np.asarray(k2).reshape(n_u, n_v))
    U, V = np.meshgrid(u_pts, v_pts, indexing="ij")
    pts = np.stack([U, V, r], axis=-1)
    state.radiusFunction = surface_factory.interpolate(pts, [basis_u, basis_v], [u_pts, v_pts])


def compute_reference_structures(state) -> None:
    """Dense (250x250) triangle mesh of the surface, its sphere centres and focal surfaces."""
    init_radius_function(state)
    denseUsample = np.linspace(state.interval_u[0], state.interval_u[1], 250)
    denseVsample = np.linspace(state.interval_v[0], state.interval_v[1], 250)
    V, F = sample_bspline_surface(state.bspline, denseUsample, denseVsample)
    F = np.array(triangulate_quads(F))
    n = state.bspline.normal(denseUsample, denseVsample).reshape(-1, 3)
    Cv = V + n * sample_radius(state.radiusFunction, denseUsample, denseVsample).reshape(-1, 1)
    _, _, k1, k2, _ = principal_frame(state.bspline, denseUsample, denseVsample)
    state.focalSurface1 = [V + n * (1 / k1).reshape(-1, 1), F]
    state.focalSurface2 = [V + n * (1 / k2).reshape(-1, 1), F]
    state.reference = [V, F, n, Cv]


def reference_preview(state, n=60):
    """Coarse n×n version of the reference, for display only.

    Same formulas as compute_reference_structures (surface, sphere centres from the
    radius function, focal surfaces) on an n×n grid over the current (u, v) range; the
    computations keep using the full 250×250 state.reference.
    Returns dict: name → (V, F).
    """
    us = np.linspace(state.interval_u[0], state.interval_u[1], n)
    vs = np.linspace(state.interval_v[0], state.interval_v[1], n)
    V, F = sample_bspline_surface(state.bspline, us, vs)
    nrm = state.bspline.normal(us, vs).reshape(-1, 3)
    Cv = V + nrm * sample_radius(state.radiusFunction, us, vs).reshape(-1, 1)
    _, _, k1, k2, _ = principal_frame(state.bspline, us, vs)
    return {"reference": (V, F), "sphere_centers": (Cv, F),
            "focal_1": (V + nrm * (1 / k1).reshape(-1, 1), F),
            "focal_2": (V + nrm * (1 / k2).reshape(-1, 1), F)}


# ── stage 1: B-spline ────────────────────────────────────────────────────────

def set_sampling(state, interval_u=None, interval_v=None, u_sample_num=None, v_sample_num=None) -> None:
    """Change the (u, v) range and/or sample counts (the sampling sliders)."""
    if interval_u is not None: state.interval_u = list(interval_u)
    if interval_v is not None: state.interval_v = list(interval_v)
    if u_sample_num is not None: state.u_sample_num = u_sample_num
    if v_sample_num is not None: state.v_sample_num = v_sample_num
    sample_selection(state, state.interval_u, state.interval_v)


def set_radius_function(state, kind=None, percentage=None, offset=None) -> None:
    """Radius function type / parameters (the radio buttons), then fit the spheres.

    Fits the radius function and rebuilds the dense reference for the current surface
    and (u, v) range (one fit; see set_reference).
    """
    if kind is not None: state.radiusFunctionType = kind
    if percentage is not None: state.percentageRadius = percentage
    if offset is not None: state.offsetValue = offset
    compute_reference_structures(state)


def flip_normals(state) -> None:
    """Reverse the surface's u direction (flips the normals) and refit the radius function.

    The original app's "Flip Normals" button. swap_uv is the alternative that keeps the
    parametrisation orientation of each direction and just exchanges their roles.
    """
    state.bspline.reverse(direction=0)
    state.normalsFlipped = not state.normalsFlipped
    init_radius_function(state)


def swap_uv(state) -> None:
    """Exchange the roles of u and v; flips the normals (Sv × Su = −Su × Sv).

    The (u, v) range and sample counts are swapped with the directions, so the same
    region of the surface stays selected (its sample grid is transposed). The radius
    function and reference refer to the old parametrisation and are cleared — fit
    them again with set_radius_function / set_reference.
    """
    state.bspline.swap()
    state.uvSwapped = not state.uvSwapped
    state.interval_u, state.interval_v = list(state.interval_v), list(state.interval_u)
    state.u_sample_num, state.v_sample_num = state.v_sample_num, state.u_sample_num
    sample_selection(state, state.interval_u, state.interval_v)
    state.radiusFunction = None
    state.reference = state.focalSurface1 = state.focalSurface2 = None


def set_reference(state) -> None:
    """Fit the radius function and the dense reference for the current surface and range
    ("Set Reference")."""
    compute_reference_structures(state)


def surface_preview(state, n=60):
    """The whole surface ([0,1]², n×n samples) for display, e.g. behind the selected range."""
    us = np.linspace(0.0, 1.0, n)
    return sample_bspline_surface(state.bspline, us, us)


def optimize_bspline(state, w_curv=0.5, w_mean=0.001, w_lap=1.0, max_iter=40,
                     epsilon_K=0.5, epsilon_H=0.001) -> None:
    """B-spline optimisation of the original panel (bsplineOptimization_jax.optimize)."""
    from Lconjugacy.utils.LConjugacy.bsplineOptimization_jax import optimize
    state.bspline = optimize(state.bspline, state.u_sample_pts, state.v_sample_pts,
                             epsilon_K=epsilon_K, epsilon_H=epsilon_H, w_curv=w_curv,
                             w_mean=w_mean, w_lap=w_lap, max_iter=max_iter)


def reset_bspline(state) -> None:
    state.bspline = state.bsplineOriginal.clone()


def export_bspline(state, output_name: str) -> str:
    """Save the current surface as ``<surface_path>/<output_name>.json``."""
    path = os.path.join(state.surface_path, output_name + ".json")
    save_bspline_to_json(state.bspline, path)
    return path


def surface_mesh(state):
    """Sampled surface (V, F), Gaussian curvature K and normals at the samples."""
    V, F = sample_bspline_surface(state.bspline, state.u_sample_pts, state.v_sample_pts)
    K, _, n = bspline_curvatures(state.bspline, state.u_sample_pts, state.v_sample_pts)
    return V, F, np.asarray(K).reshape(-1), np.asarray(n).reshape(-1, 3)


def control_net(state):
    """Control points (N, 3) and control-net edges."""
    P, E, _ = control_points_grid(state.bspline)
    return np.asarray(P).reshape(-1, 3), E


def radius_centers(state):
    """Sphere centres at the samples from the current radius type (``compute_radius_function``)."""
    _, _, k1, k2, _ = principal_frame(state.bspline, state.u_sample_pts, state.v_sample_pts)
    V, F = sample_bspline_surface(state.bspline, state.u_sample_pts, state.v_sample_pts)
    if state.radiusFunctionType == "safe":
        r = 1 / np.maximum(np.abs(k1), np.abs(k2)) * state.percentageRadius
    elif state.radiusFunctionType == "offset":
        r = state.offsetValue * np.ones_like(k1)
    elif state.radiusFunctionType == "mid":
        r = (1 / k1 + 1 / k2) * 0.5
    n = state.bspline.normal(state.u_sample_pts, state.v_sample_pts).reshape(-1, 3)
    return V + n * r.reshape(-1, 1), F


# ── stage 2: L-conjugate directions ──────────────────────────────────────────

def linear_angle_distribution(state, startAngle=0, endAngle=75):
    sampledPoints = state.bspline(state.u_sample_pts, state.v_sample_pts).reshape(-1, 3)
    return np.linspace(np.deg2rad(startAngle), np.deg2rad(endAngle), len(sampledPoints))


def linear_angle_distribution_dir_v(state, startAngle=0, endAngle=85):
    return np.tile(np.linspace(np.deg2rad(startAngle), np.deg2rad(endAngle), len(state.v_sample_pts)),
                   len(state.u_sample_pts)).flatten()


def linear_angle_distribution_dir_u(state, startAngle=0, endAngle=85):
    return np.linspace(np.deg2rad(startAngle), np.deg2rad(endAngle),
                       len(state.u_sample_pts)).repeat(len(state.v_sample_pts), axis=0).flatten()


def cosine_angle_distribution_v(state, minAngle=0, maxAngle=90):
    numU, numV = len(state.u_sample_pts), len(state.v_sample_pts)
    t = np.linspace(0, 1, numV).repeat(numU, axis=0).flatten()
    return (np.deg2rad(maxAngle) + np.deg2rad(minAngle)) * 0.5 + \
        (np.deg2rad(maxAngle) - np.deg2rad(minAngle)) * 0.5 * np.cos(t * 2 * np.pi)


def cosine_angle_distribution_u(state, minAngle=0, maxAngle=90):
    numU, numV = len(state.u_sample_pts), len(state.v_sample_pts)
    t = np.tile(np.linspace(0, 1, numU), numV)
    return (np.deg2rad(maxAngle) + np.deg2rad(minAngle)) * 0.5 + \
        (np.deg2rad(maxAngle) - np.deg2rad(minAngle)) * 0.5 * np.cos(t * 2 * np.pi)


def initial_angles(state, method="constant"):
    """First-direction angles θ for an initialisation method (the panel's radio buttons).

    Same ranges as the original panel: linear 0–90°, linearU / linearV 0–60°,
    constant = ``state.initial_theta``. For "boundary" θ comes from
    ``init_lconjugate_boundary`` inside :func:`init_directions`.
    """
    if method == "linear":
        return linear_angle_distribution(state, 0, 90)
    if method == "linearU":
        return linear_angle_distribution_dir_u(state, 0, 60)
    if method == "linearV":
        return linear_angle_distribution_dir_v(state, 0, 60)
    if method == "constant":
        return np.deg2rad(state.initial_theta) * np.ones(len(state.u_sample_pts) * len(state.v_sample_pts))
    if method == "cosineV":
        return cosine_angle_distribution_v(state)
    if method == "cosineU":
        return cosine_angle_distribution_u(state)
    if method == "boundary":
        return state.theta_angles
    raise ValueError(f"unknown initialisation method '{method}' (one of {INIT_METHODS})")


def init_directions(state, method="constant", build_optimizer=False, legacy_regularizer=True) -> None:
    """Initialise θ/φ ("Initialize L-conjugate Directions").

    build_optimizer: also set up the direction-field optimizer (setup_df_optimizer).
        Off by default — the direction-field optimisation is not used for the project
        (it is slow and its radius write-back is broken, see update_variables_from_optimizer).
    legacy_regularizer: for the optimizer — the pre-2026-09 previous-iterate regulariser
        (residual X_k − X_{k−1}) the original app used; False uses the proximal version.
    """
    if state.radiusFunction is None:
        raise RuntimeError("no radius function: fit the spheres first "
                           "(set_radius_function / set_reference, e.g. after swap_uv)")
    state.init_method = method
    state.theta_angles = initial_angles(state, method)
    radiusValues = sample_radius(state.radiusFunction, state.u_sample_pts, state.v_sample_pts)
    if method != "boundary":
        _, _, _, _, theta, phi = init_lconjugate_directions(
            state.bspline, state.u_sample_pts, state.v_sample_pts, radiusValues, state.theta_angles)
        state.boundary_data = None
    else:
        _, _, _, _, theta, phi, *boundary = init_lconjugate_boundary(
            state.bspline, state.u_sample_pts, state.v_sample_pts, radiusValues, state.initial_theta)
        state.boundary_data = tuple(boundary)
    state.theta_angles, state.phi_angles = theta, phi
    state.directionFieldOptimizer = None
    if build_optimizer:
        setup_df_optimizer(state, legacy_regularizer=legacy_regularizer)


def setup_df_optimizer(state, legacy_regularizer=True) -> None:
    """Direction-field optimizer of the original panel (``LConjugacyPanel.setup_optimizer``).

    Unlike the original, a fresh Optimizer is created each time; the original reused one
    object, so a second "Initialize" appended a second set of variables to it.
    """
    opt = Optimizer()
    state.directionFieldOptimizer = opt
    radiusFunctionCtrlPts = state.radiusFunction.controlpoints.reshape(-1, 3)[:, 2]
    beta = state.phi_angles - state.theta_angles

    opt.add_variable("th", state.theta_angles)
    opt.add_variable("bet", beta)
    opt.add_variable("rij", radiusFunctionCtrlPts + 0.1)   # small offset avoids zero-radius issues

    t1, _, k1, k2, _ = principal_frame(state.bspline, state.u_sample_pts, state.v_sample_pts)
    V, F = sample_bspline_surface(state.bspline, state.u_sample_pts, state.v_sample_pts)
    mesh = Mesh()
    mesh.make_mesh(V, F)
    ei, ej = mesh.edge_vertices()
    n = state.bspline.normal(state.u_sample_pts, state.v_sample_pts).reshape(-1, 3)
    alpha_ij = compute_rij_transport_vectorized(
        t1[ei].reshape(-1, 3), t1[ej].reshape(-1, 3), n[ei].reshape(-1, 3), n[ej].reshape(-1, 3))

    term = Lconjugacy(); term.name = "lconjugacy"
    opt.add_objective_term(term, args=(k1, k2, state.radiusFunction, state.u_sample_pts, state.v_sample_pts),
                           w=state.weight_DF["lconjugacy"], ce=1)
    term = DirectionFieldSmoothness(); term.name = "smooth"
    opt.add_objective_term(term, args=(mesh.edge_vertices(), alpha_ij), w=state.weight_DF["smooth"], ce=1)
    term = AngleLconjugate(); term.name = "angle"
    opt.add_objective_term(term, args=([np.deg2rad(state.target_angle)]), w=state.weight_DF["angle"], ce=1)
    if state.init_method == "boundary":
        term = AngleBoundary(); term.name = "angleBoundary"
        opt.add_objective_term(term, args=state.boundary_data, w=state.weight_DF["angleBoundary"], ce=1)

    reg = PrevIterationRegularizer(lagged=legacy_regularizer)
    reg.name = "previous_iteration_regularizer"
    opt.add_objective_term(reg, (), w=0.1, ce=False)

    opt.initialize_optimizer(verbose=True, adaptive_mu=False)


def update_variables_from_optimizer(state) -> None:
    theta, beth, rij = state.directionFieldOptimizer.unpack("th", "bet", "rij")
    # ORIGINAL: `state.radiusFunction[2] = rij` — assigns the whole rij vector to control
    # point #2 of the radius spline; with splipy 1.10 this raises on the first iteration
    # (the original app crashes there too). Pending decision on the intended write-back
    # (rij was initialised as control-point radii + 0.1), the optimised radii are kept
    # in state.rij and the radius function is left unchanged.
    state.rij = np.array(rij)
    state.theta_angles = theta
    state.phi_angles = theta + beth


def run_df_optimizer(state, max_its=40, callback=None) -> None:
    """Run ``max_its`` optimisation steps ("Run Optimization"); callback(state, it) after each."""
    opt = state.directionFieldOptimizer
    opt.set_term_weights(state.weight_DF)
    for it in range(max_its):
        opt.get_gradients()
        opt.optimize_step()
        update_variables_from_optimizer(state)
        if callback is not None:
            callback(state, it)
    opt.print_report()


def cross_field(state):
    """Sample points, the two L-conjugate curve tangents t1, t2 and the rulings lc1, lc2."""
    e1, e2, k1, k2, _ = principal_frame(state.bspline, state.u_sample_pts, state.v_sample_pts)
    t1 = np.cos(state.theta_angles)[:, None] * e1 + np.sin(state.theta_angles)[:, None] * e2
    t2 = np.cos(state.phi_angles)[:, None] * e1 + np.sin(state.phi_angles)[:, None] * e2
    lc1, lc2 = exchange_lconjugate_rulings_curve_tangents(e1, e2, k1, k2, t1, t2)
    pts = state.bspline(state.u_sample_pts, state.v_sample_pts).reshape(-1, 3)
    return pts, t1.reshape(-1, 3), t2.reshape(-1, 3), lc1.reshape(-1, 3), lc2.reshape(-1, 3)


def sphere_centers(state):
    """Sphere centres from the (optimised) radius function at the samples, and faces."""
    radiusValues = sample_radius(state.radiusFunction, state.u_sample_pts, state.v_sample_pts)
    _, F = sample_bspline_surface(state.bspline, state.u_sample_pts, state.v_sample_pts)
    pts = state.bspline(state.u_sample_pts, state.v_sample_pts).reshape(-1, 3)
    n = state.bspline.normal(state.u_sample_pts, state.v_sample_pts).reshape(-1, 3)
    return pts + radiusValues[:, None] * n, F


def principal_directions(state):
    e1, e2, _, _, n = principal_frame(state.bspline, state.u_sample_pts, state.v_sample_pts)
    pts = state.bspline(state.u_sample_pts, state.v_sample_pts).reshape(-1, 3)
    return pts, e1.reshape(-1, 3), e2.reshape(-1, 3), n.reshape(-1, 3)


def compute_singularities(state):
    """Cross-field singularities (4-symmetry): vertex indices and their indices (±1/4)."""
    V, F = sample_bspline_surface(state.bspline, state.u_sample_pts, state.v_sample_pts)
    mesh = Mesh()
    mesh.make_mesh(V, F)
    t1, _, _, _, n = principal_frame(state.bspline, state.u_sample_pts, state.v_sample_pts)
    _, sing_verts, sing_indices = compute_crossfield_singularities(mesh, state.theta_angles, t1, n, symmetry=4)
    state.singularity_vertices, state.singularity_indices = sing_verts, sing_indices
    return V, sing_verts, sing_indices


# ── stage 3: remeshing (external) ────────────────────────────────────────────

def save_cross_field(state) -> None:
    """Write mesh.obj and the t / lc direction fields (remesher input) to ``save_path``."""
    pts, t1, t2, lc1, lc2 = cross_field(state)
    Path(state.save_path).mkdir(parents=True, exist_ok=True)
    export_crossfield(state.save_path, "t", pts, t1, t2)
    export_crossfield(state.save_path, "lc", pts, lc1, lc2)
    V, F = sample_bspline_surface(state.bspline, state.u_sample_pts, state.v_sample_pts)
    write_obj(os.path.join(state.save_path, "mesh.obj"), V, triangulate_quads(F))


def run_remesher(state, integrated: bool = True) -> bool:
    """Run the external quad remesher on the saved cross fields (t and lc).

    The remesher (quadRemesher) is interactive: after computing the parametrisation it
    opens a viewer window, and the quad mesh is only written when you press **S** in
    that window; the call returns when the window is closed. It runs twice — first for
    the curve tangents (t), then for the rulings (lc).

    Returns True when t_Remeshed.obj was produced; then the pre-remesh state is saved,
    as in the original "Run Remesher" buttons.
    """
    required = {"mesh.obj", "t_D1.dat", "t_D2.dat", "lc_D1.dat", "lc_D2.dat"}
    missing = required - set(os.listdir(state.save_path)) if os.path.isdir(state.save_path) else required
    if missing:
        raise FileNotFoundError(f"run save_cross_field first; missing {sorted(missing)} in {state.save_path}")
    if not os.path.isfile(state.path_remesher):
        raise FileNotFoundError(f"remesher not found: {state.path_remesher} (set state.path_remesher)")
    p = lambda f: str(Path(state.save_path, f).resolve())
    # The remesher writes <output_folder><name>_Remeshed.obj with no separator in
    # between, so the folder must end with "/" (without it the file lands next to the
    # experiment folder as "<experiment>t_Remeshed.obj").
    out_folder = str(Path(state.save_path).resolve()) + os.sep
    mode = "integration" if integrated else "direct"
    labels = {"t": "L-conjugate curve tangents (t)", "lc": "rulings (lc)"}
    for tag in ("t", "lc"):
        print(f"Remesher window for the {labels[tag]}: press S to save the quad mesh, "
              f"then close the window to continue.", flush=True)
        cmd = [state.path_remesher, p("mesh.obj"), p(f"{tag}_D1.dat"), p(f"{tag}_D2.dat"),
               out_folder, tag, mode]
        result = subprocess.run(cmd, capture_output=True, text=True)
        saved = os.path.exists(os.path.join(state.save_path, f"{tag}_Remeshed.obj"))
        print(f"  {tag}: {'saved ' + tag + '_Remeshed.obj' if saved else 'no quad mesh saved (S not pressed?)'}",
              flush=True)
        if result.returncode != 0:
            print("Remesher error:", result.stderr)
    ok = os.path.exists(os.path.join(state.save_path, "t_Remeshed.obj"))
    if ok:
        save_pre_remesh_state(state)
    return ok


def save_pre_remesh_state(state) -> str:
    """Pickle bspline, sampling, directions and radius function (same keys as the original)."""
    Path(state.save_path).mkdir(parents=True, exist_ok=True)
    data = {
        "bspline": state.bspline,
        "normalsFlipped": state.normalsFlipped,
        "uvSwapped": state.uvSwapped,
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
    return path


def load_pre_remesh_state(state) -> bool:
    """Restore the pickle written by save_pre_remesh_state; False if it does not exist."""
    path = os.path.join(state.save_path, "pre_remesh_state.pkl")
    if not os.path.exists(path):
        return False
    with open(path, "rb") as f:
        data = pickle.load(f)
    for key in ("bspline", "u_sample_pts", "v_sample_pts", "theta_angles", "phi_angles",
                "radiusFunction", "percentageRadius", "radiusFunctionType"):
        setattr(state, key, data[key])
    state.u_sample_num = len(state.u_sample_pts)
    state.v_sample_num = len(state.v_sample_pts)
    state.interval_u = [state.u_sample_pts[0], state.u_sample_pts[-1]]
    state.interval_v = [state.v_sample_pts[0], state.v_sample_pts[-1]]
    # ORIGINAL: refits the radius function (discarding the loaded one) before the reference.
    init_radius_function(state)
    compute_reference_structures(state)
    return True


def load_remeshed_surfaces(state) -> bool:
    """Read t_Remeshed.obj / lc_Remeshed.obj, project them onto the reference surface."""
    t_path = os.path.join(state.save_path, "t_Remeshed.obj")
    lc_path = os.path.join(state.save_path, "lc_Remeshed.obj")
    if not (os.path.exists(t_path) and os.path.exists(lc_path)):
        return False
    compute_reference_structures(state)
    V_ref, F_ref, n_ref, Cv_ref = state.reference

    def project(V):
        _, _, V_proj = igl.point_mesh_squared_distance(V, V_ref, F_ref)
        return V_proj, interpolate_triangle_mesh(V_proj, V_ref, F_ref, Cv_ref)

    V_t, F_t = read_obj(t_path)
    V_lc, F_lc = read_obj(lc_path)
    V_t_proj, sphereCenters_t = project(V_t)
    V_lc_proj, _ = project(V_lc)

    quad_t = Mesh(); quad_t.make_mesh(V_t_proj, F_t)
    _, _, mid_proj = igl.point_mesh_squared_distance(quad_t.face_barycenters(), V_ref, F_ref)
    state.faceNormals_t = interpolate_triangle_mesh(mid_proj, V_ref, F_ref, n_ref.reshape(-1, 3))
    quad_lc = Mesh(); quad_lc.make_mesh(V_lc_proj, F_lc)

    state.quadRemeshed_t, state.quadRemeshed_lc = quad_t, quad_lc
    state.sphereCenters_t = sphereCenters_t
    return True


# ── stage 4: L-mesh on the remeshed quads ────────────────────────────────────
#
# The original (pre-swap Lmesh) used combinatorics="primal": spheres on the quads' FACES
# (lmesh.SF) and planes on ALL quad VERTICES (top.vertices). In the swapped Lmesh that is
# combinatorics="dual": top.vertices = quad faces (sphere i <-> face i), top.faces = one
# plane per INNER quad vertex, in inner_vertices() order. Boundary-vertex planes of the
# original have no place in the dual top and are dropped.

def initialize_lmesh(state) -> Lmesh:
    """L-mesh on quadRemeshed_t (original ``computations.initialize_lmesh``).

    Spheres: centre = reference sphere-centre surface at each quad barycentre, radius =
    distance to the reference surface there. Planes: tangent planes of the reference at
    each (inner) quad vertex, n·x + h = 0.
    """
    if state.quadRemeshed_t is None:
        raise RuntimeError("no remeshed quad mesh: run load_remeshed_surfaces first")
    base = state.quadRemeshed_t
    V_ref, F_ref, n_ref, SC_ref = state.reference
    n_ref = np.asarray(n_ref).reshape(-1, 3)

    mid = base.face_barycenters()
    projected = interpolate_triangle_mesh(mid, V_ref, F_ref, V_ref)
    centers = interpolate_triangle_mesh(mid, V_ref, F_ref, SC_ref)
    radii = np.linalg.norm(centers - projected, axis=1)

    normals = interpolate_triangle_mesh(base.vertices, V_ref, F_ref, n_ref)
    planes = np.column_stack((normals, -np.einsum("ij,ij->i", normals, base.vertices)))
    inner = base.inner_vertices()

    lmesh = Lmesh()
    lmesh.init_L_mesh(base, orientation=1, combinatorics="dual", mode="OrPlanes",
                      planes=planes[inner], centers=centers, radii=radii)
    state.lmesh = lmesh
    return lmesh


def setup_lmesh_optimizer(state, legacy_regularizer=True) -> Optimizer:
    """Post-optimisation of the original panel (``PostOptimizationPanel.setup_optimization``).

    Term names are the original ones, so LMESH_WEIGHTS / LMESH_FINAL_WEIGHTS apply.
    Mapping to the swapped terms:
        Or_contactFree          -> Or_contactFree(top)
        QuadLmeshFairnessPoly   -> QLmesh_reg_strips_lconj "quadFair"     (planes around a sphere)
        QuadLmeshFairnessSpheres-> QLmesh_reg_faces_lconj  "quadFairSph"  (spheres around a plane)
        Tangential_Dist, ContactPointsProximity/Glide: same terms, swapped adjacency.
    Both fairness terms build their stencils from ``top`` (spheres or planes next to the
    boundary with an incomplete stencil are skipped).
    Not ported: QuadLmeshFairness ("quadFairPrincipal"), whose weight is 0 in both
    weight sets. ``legacy_regularizer=True`` keeps the original prev-iteration term
    (r = X_k - X_{k-1}); False uses the proximal one.
    """
    lmesh = state.lmesh
    top = lmesh.top
    V_ref, F_ref, _, SC_ref = state.reference

    opt = Optimizer()
    state.lmeshOptimizer = opt
    opt.add_variable("cv", top.vertices[:, :3].flatten())   # sphere centres (per quad face)
    opt.add_variable("rv", top.vertices[:, 3].copy())       # sphere radii
    opt.add_variable("npf", lmesh.PF[:, :3].flatten())      # plane normals (per inner quad vertex)
    opt.add_variable("hpf", lmesh.PF[:, 3].copy())          # plane offsets

    sph_planes = top.vertex_face_adjacency_list()

    term = Or_contactFree(); term.name = "or_contact"
    opt.add_objective_term(term, args=(top,), w=10, ce=1)
    term = QLmesh_reg_strips_lconj(); term.name = "quadFair"
    opt.add_objective_term(term, args=(top,), w=0.01, ce=1)
    term = QLmesh_reg_faces_lconj(); term.name = "quadFairSph"
    opt.add_objective_term(term, args=(top,), w=0.01, ce=1)
    term = Tangential_Dist(); term.name = "tangentialDist"
    opt.add_objective_term(term, args=(top.edge_vertices(),), w=1e-4, ce=1)
    term = ContactPointsProximity(); term.name = "contactProximity"
    opt.add_objective_term(term, args=(sph_planes, V_ref, F_ref), w=1e-4, ce=1)
    term = ContactPointsGlide(); term.name = "contactGliding"
    opt.add_objective_term(term, args=(sph_planes, V_ref, F_ref), w=1e-5, ce=1)

    reg = PrevIterationRegularizer(lagged=legacy_regularizer)
    reg.name = "previous_iteration_regularizer"
    opt.add_objective_term(reg, (), w=1e-4, ce=False)
    opt.proximity_reference("cv", SC_ref, F_ref, w_proximity=0.001, w_gliding=0.01)
    opt.unitize_variable("npf", 3, 10)

    opt.initialize_optimizer(verbose=True, adaptive_mu=False)
    return opt


def update_lmesh_from_optimizer(state) -> None:
    cv, rv, npf, hpf = state.lmeshOptimizer.unpack("cv", "rv", "npf", "hpf")
    state.lmesh.update_spheres_planes(cv.reshape(-1, 3), rv, npf.reshape(-1, 3), hpf)


def run_lmesh_optimizer(state, max_its=40, weights=None, callback=None,
                        fairness_decay=FAIRNESS_DECAY, decay_every=FAIRNESS_DECAY_EVERY) -> None:
    """"Run Optimization": ``max_its`` steps with ``weights`` (default state.weight_lmesh).

    The fairness weights (FAIRNESS_TERMS) are multiplied by ``fairness_decay`` every
    ``decay_every`` steps, as in the paper: full weight for steps 0-9, x0.1 from step 10,
    x0.01 from step 20, ... Each call starts again from the full weights (like pressing
    "Run Optimization" again in the original). ``fairness_decay=1`` switches it off.
    The original code decayed inside the terms' grad() and already at step 0 (see
    legacy_polyscope); the L-conjugacy copies QLmesh_reg_*_lconj (Lconjugacy/utils/lmeshOpt/) do not decay
    themselves (unlike LMesh.Optimization.QLmesh_reg_*).

    callback(state, it) after each step sees the optimizer; the L-mesh itself is updated
    once at the end (rebuilding it every step is slow).
    """
    opt = state.lmeshOptimizer
    weights = dict(state.weight_lmesh if weights is None else weights)
    opt.set_term_weights(weights)
    for it in range(max_its):
        if it > 0 and it % decay_every == 0:
            factor = fairness_decay ** (it // decay_every)
            opt.set_term_weights({k: weights[k] * factor for k in FAIRNESS_TERMS if k in weights})
        opt.get_gradients()
        opt.optimize_step()
        if callback is not None:
            callback(state, it)
    update_lmesh_from_optimizer(state)
    opt.print_report()


def finalize_lmesh(state, its=20) -> None:
    """"Last Opt": ``its`` steps of oriented contact only (LMESH_FINAL_WEIGHTS)."""
    run_lmesh_optimizer(state, max_its=its, weights=LMESH_FINAL_WEIGHTS)


def lmesh_centers(state):
    """Current sphere centres (from the optimizer if there is one), connected by the planes."""
    if state.lmeshOptimizer is not None:
        C = state.lmeshOptimizer.unpack("cv").reshape(-1, 3)
    else:
        C = state.lmesh.top.vertices[:, :3]
    return C, [list(f) for f in state.lmesh.top.faces]


def export_lmesh(state, name="lmesh") -> str:
    """Original "Export L-mesh": Lmesh.export_data plus the two sphere-centre meshes."""
    lmesh = state.lmesh
    lmesh.export_data(state.save_path, name)
    write_obj(Path(state.save_path) / "sphereCenterQuadMesh_t.obj",
              state.sphereCenters_t, state.quadRemeshed_t.faces)
    # sphere centres connected like the quads (one centre per quad face, one face per plane)
    write_obj(Path(state.save_path) / "sphereCentersLmesh.obj",
              lmesh.top.vertices[:, :3], [list(f) for f in lmesh.top.faces])
    return state.save_path
