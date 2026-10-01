# Code/gui/post_main.py
"""Standalone entry point for the post-optimization stage.

Usage:
    python -m LconjugacyGUI.post_main <surface_name> <experiment_name>

Example:
    python -m LconjugacyGUI.post_main nautilus experiment0

This loads the pre-remesh state pickle (bspline, sampling, directions,
radius function) from gui/out/<surface_name>/<experiment_name>/ and opens
only the PostOptimizationPanel. If the pickle is not found, it falls back
to loading the B-spline JSON from gui/surfaces/<surface_name>.json.
"""

import argparse
import os
import numpy as np
import polyscope as ps

from LconjugacyGUI.state import AppState
import LconjugacyGUI.computations as comp
from LconjugacyGUI.post_optimization_panel import PostOptimizationPanel
from LconjugacyGUI.utils.bsplines_functions import read_bspline_json


class PostOptimizationApp:
    def __init__(self, project_path: str, name: str, experiment_name: str):
        self.state = AppState()

        # Shared surfaces folder at Code/surfaces/
        bspline_path = os.path.join(os.path.dirname(project_path), "surfaces", name + ".json")

        self.state.name = name
        self.state.surface_path = os.path.join(os.path.dirname(project_path), "surfaces")
        self.state.output_path = os.path.join(project_path, "out") + "/"
        self.state.experiment_name = experiment_name
        self.state.save_path = os.path.join(self.state.output_path, self.state.name, self.state.experiment_name)
        self.state.current_dir = os.getcwd()

        # Verify the experiment folder and required files exist
        save_path = self.state.save_path
        if not os.path.exists(save_path):
            raise FileNotFoundError(f"Experiment folder not found: {os.path.abspath(save_path)}")

        required = ["t_Remeshed.obj", "lc_Remeshed.obj"]
        missing = [f for f in required if not os.path.exists(os.path.join(save_path, f))]
        if missing:
            raise FileNotFoundError(
                f"Missing required files in {os.path.abspath(save_path)}: {missing}"
            )

        # Try loading the pre-remesh state pickle first
        # This restores: bspline, u/v samples, theta/phi angles, radius function
        loaded = comp.load_pre_remesh_state(self.state)

        if not loaded:
            # Fallback: load B-spline from JSON and compute from scratch
            print("Falling back to B-spline JSON:", bspline_path)
            self.state.bspline = read_bspline_json(bspline_path)
            self.state.bsplineOriginal = self.state.bspline
            comp.sample_selection(self.state, self.state.interval_u, self.state.interval_v)
            comp.init_radius_function(self.state)
            comp.compute_reference_structures(self.state)

        # Create only the post-optimization panel
        self.post_optimization_panel = PostOptimizationPanel(self.state)

    def init(self):
        ps.set_automatically_compute_scene_extents(False)
        ps.set_length_scale(0.5)
        low = np.array((-2, -2., -2.))
        high = np.array((2., 2., 2.))
        ps.set_bounding_box(low, high)

        ps.set_user_callback(self.main_loop)

    def main_loop(self):
        import polyscope.imgui as psim

        psim.Text(f"Surface: {self.state.name}")
        psim.Text(f"Experiment: {self.state.experiment_name}")
        psim.Separator()

        self.post_optimization_panel.draw()
        self.post_optimization_panel.update()


def main():
    parser = argparse.ArgumentParser(
        description="Standalone L-mesh post-optimization GUI"
    )
    parser.add_argument(
        "name",
        type=str,
        help="Name of the surface file (without .json extension)",
    )
    parser.add_argument(
        "experiment",
        type=str,
        help="Experiment name (subfolder inside output/<name>/)",
    )
    args = parser.parse_args()

    project_path = os.path.join(os.path.dirname(os.path.abspath(__file__)))

    # Shared surfaces folder at Code/surfaces/
    bspline_path = os.path.join(os.path.dirname(project_path), "surfaces", args.name + ".json")
    if not os.path.exists(bspline_path):
        raise FileNotFoundError(f"Surface file not found: {bspline_path}")

    ps.init()
    app = PostOptimizationApp(project_path, args.name, experiment_name=args.experiment)
    app.init()
    comp.update_visualization(app.state)
    ps.show()


if __name__ == "__main__":
    main()
