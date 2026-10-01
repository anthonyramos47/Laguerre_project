# Code/LconjugacyGUI/state.py
from dataclasses import dataclass, field
import os
import numpy as np
from hanan.optimization.optimizer import Optimizer
from kayviz.webscope import (
    gui_state, slider, int_slider, dropdown, checkbox, text_field,
)


@gui_state
@dataclass
class AppState:

    # ── Exposed to the frontend AutoPanel ────────────────────────────────────

    # BSpline section
    bspline_u_sample_num:  int_slider(2, 100, 1,    "U Samples")       = 20
    bspline_v_sample_num:  int_slider(2, 100, 1,    "V Samples")       = 20
    bspline_percentage_r:  slider(0.0, 1.0,  0.01,  "% Radius")        = 0.75
    bspline_offset_value:  slider(0.0, 2.0,  0.01,  "Offset")          = 0.2
    bspline_radius_type:   dropdown(["safe", "mid", "offset"], "Radius Mode") = "safe"
    bspline_flip_normals:  checkbox("Flip Normals")                     = False

    # Lconj section
    lconj_theta_initial:    slider(0.0,  90.0, 0.5,   "Initial θ (deg)")  = 45.0
    lconj_w_lconjugacy:     slider(0.0,  5.0,  0.01,  "w Lconj")          = 1.1
    lconj_w_smooth:         slider(0.0,  0.01, 0.0001,"w Smooth")          = 0.0001
    lconj_w_angle_boundary: slider(0.0,  5.0,  0.01,  "w Boundary")        = 0.5

    # Post section
    post_cross_field_len: slider(0.001, 0.1, 0.001,   "CF Length")  = 0.02
    post_cross_field_rad: slider(0.00001, 0.01, 0.00001, "CF Radius") = 0.0001

    # Paths / config
    experiment_name: text_field("Experiment Name") = "experiment0"
    path_remesher:   text_field("Remesher Path") = os.environ.get(
        "QUAD_REMESHER_PATH",
        "/path/to/quadRemesher",
    )

    # ── Internal state (not exposed to the panel) ─────────────────────────────

    bspline: object         = None
    bsplineOriginal: object = None
    normalsFlipped: bool    = False

    u_sample_pts: np.ndarray = None
    v_sample_pts: np.ndarray = None
    interval_u: list = field(default_factory=lambda: [0.0, 1.0])
    interval_v: list = field(default_factory=lambda: [0.0, 1.0])

    radiusFunction: object     = None
    percentageRadius: float    = 0.75
    offsetValue: float         = 0.2
    radiusFunctionType: str    = "safe"
    reference: list            = None
    focalSurface1: list        = None
    focalSurface2: list        = None

    bSplineOptimizer:       Optimizer = field(default_factory=Optimizer)
    directionFieldOptimizer: Optimizer = field(default_factory=Optimizer)
    postProcessingOptimizer: Optimizer = field(default_factory=Optimizer)

    theta_angles: np.ndarray      = None
    phi_angles: np.ndarray        = None
    singularity_vertices: np.ndarray = None
    singularity_indices: np.ndarray  = None

    quadRemeshed_t: object      = None
    quadRemeshed_lc: object     = None
    sphereCenters_t: np.ndarray = None
    sphereCenters_lc: np.ndarray = None
    faceNormals_t: np.ndarray   = None
    faceNormals_lc: np.ndarray  = None

    lmesh: object = None

    selected_vertex: object   = None
    selected_face: object     = None
    vertex_neighbors: object  = None
    selected_neighbor: object = None
    selectFaceMode: bool      = False

    name: str           = ""
    save_path: str      = ""
    output_path: str    = ""
    surface_path: str   = ""
    current_dir: str    = ""

    is_running: bool    = False
    log: str            = ""
