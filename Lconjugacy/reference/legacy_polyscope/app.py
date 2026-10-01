# Code/gui/app.py
"""Main application orchestrator that initializes state and panels."""

import os
import numpy as np
import polyscope as ps

from LconjugacyGUI.state import AppState
import LconjugacyGUI.computations as comp

from LconjugacyGUI.bspline_panel import BSplinePanel
from LconjugacyGUI.lconjugacy_panel import LConjugacyPanel
from LconjugacyGUI.post_optimization_panel import PostOptimizationPanel
from LconjugacyGUI.integration_panel import IntegrationPanel
from LconjugacyGUI.local_analysis_panel import LocalAnalysisPanel

from LconjugacyGUI.utils.bsplines_functions import read_bspline_json
from pathlib import Path


class App:
    def __init__(self, project_path: str | Path, name: str, experiment_name: str):
        # Build shared state
        self.state = AppState()


        # Shared surfaces folder at Code/surfaces/
        path_surfaces = os.path.join(os.path.dirname(project_path), "surfaces")
        bspline_path  = os.path.join(path_surfaces, name + ".json")

        self.state.name = os.path.splitext(os.path.basename(bspline_path))[0]
        self.state.surface_path = os.path.join(os.path.dirname(project_path), "surfaces")
        self.state.output_path =  os.path.join(project_path, "out") 
        self.state.experiment_name = experiment_name
        self.state.save_path = os.path.join(self.state.output_path, self.state.name, self.state.experiment_name)
        self.state.current_dir = os.getcwd()

        self.state.bspline = read_bspline_json(bspline_path)
        self.state.bsplineOriginal = self.state.bspline

        # Initial sampling
        comp.sample_selection(self.state, self.state.interval_u, self.state.interval_v)

        # Initialize radius function and reference structures
        comp.init_radius_function(self.state)
        comp.compute_reference_structures(self.state)

        # Create panels
        self.bspline_panel = BSplinePanel(self.state)
        self.lconjugacy_panel = LConjugacyPanel(self.state)
        self.post_optimization_panel = PostOptimizationPanel(self.state)
        self.integration_panel = IntegrationPanel(self.state)
        self.local_analysis_panel = LocalAnalysisPanel(self.state)

    def init(self):
        """Initialize Polyscope and scene."""
        ps.set_automatically_compute_scene_extents(False)
        ps.set_length_scale(0.5)
        low = np.array((-2, -2., -2.))
        high = np.array((2., 2., 2.))
        ps.set_bounding_box(low, high)

        ps.init()
        ps.set_user_callback(self.main_loop)

    def main_loop(self):
        """Polyscope user callback — called every frame."""
        
        self.bspline_panel._draw()
        self.lconjugacy_panel._draw()
        self.post_optimization_panel._draw()
        self.integration_panel._draw()
        # self.local_analysis_panel._draw()

        # psim.End()

        # Update all panels (optimization stepping)
        self.bspline_panel.update()
        self.lconjugacy_panel.update()
        self.post_optimization_panel.update()
        
        #self.integration_panel.update()
        self.local_analysis_panel.update()
