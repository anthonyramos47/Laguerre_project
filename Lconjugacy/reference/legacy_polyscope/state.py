# Code/gui/state.py
from dataclasses import dataclass, field
import numpy as np
from hanan.optimization.Optimizer import Optimizer

@dataclass
class AppState:
    # B-spline
    bspline: object = None
    bsplineOriginal: object = None
    normalsFlipped: bool = False
    u_sample_pts: np.ndarray = None
    v_sample_pts: np.ndarray = None
    u_sample_num: int = 20
    v_sample_num: int = 20
    interval_u: list = field(default_factory=lambda: [0.0, 1.0])
    interval_v: list = field(default_factory=lambda: [0.0, 1.0])
    radiusFunction: object = None
    percentageRadius: float = 0.75
    offsetValue: float = 0.2
    radiusFunctionType: str = "safe"  # "absolute" or "relative"
    reference: list = None  # [V, F, n, Cv]
    focalSurface1: list = None  # [V, F]
    focalSurface2: list = None  # [V, F]

    # Optimizers
    bSplineOptimizer: Optimizer = field(default_factory=Optimizer)
    directionFieldOptimizer: Optimizer = field(default_factory=Optimizer)
    postProcessingOptimizer: Optimizer = field(default_factory=Optimizer)

    # Direction field
    theta_angles: np.ndarray = None
    phi_angles: np.ndarray = None
    initial_theta: float = 45.0
    weight_DF: dict = field(default_factory=lambda: {"lconjugacy": 1.1, "smooth": 0.0001, "angle": 0.0, "angleBoundary": 0.5})
    singularity_vertices: np.ndarray = None
    singularity_indices: np.ndarray = None

    # Remeshed data
    quadRemeshed_t: object = None
    quadRemeshed_lc: object = None
    sphereCenters_t: np.ndarray = None
    sphereCenters_lc: np.ndarray = None
    faceNormals_t: np.ndarray = None
    faceNormals_lc: np.ndarray = None

    # L-mesh
    lmesh: object = None

    # Mouse / selection state
    selected_vertex: object = None
    selected_face: object = None
    vertex_neighbors: object = None
    selected_neighbor: object = None
    selectFaceMode: bool = False

    # Paths
    name: str = ""
    save_path: str = ""
    output_path: str = ""
    experiment_name: str = ""
    surface_path: str = ""
    path_remesher: str = "/path/to/quadRemesher"
    current_dir: str = ""

    # Visualization settings
    lenghtCrossField: float = 0.02
    radiusCrossField: float = 0.0001
    palleteColors = {"spheres": "#59053C", 
                     "cones"  : "#a9d6fd", 
                     "planes" : "#16105c", 
                     "remeshed": "#6C47CB",
                     "sphereCenters": "#11b888",
                     }



    # Logging
    log: str = ""
