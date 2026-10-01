# Code/gui/bspline_panel.py
from LconjugacyGUI.state import AppState
import LconjugacyGUI.computations as comp

import polyscope as ps
import polyscope.imgui as psim
import os

from LconjugacyGUI.utils.bsplines_functions import (
    control_points_grid,
    save_bspline_to_json,
    sphere_center_spline_fitting,
    sample_bspline_surface
)
from LconjugacyGUI.utils.LConjugacy.bsplineOptimization_jax import optimize

# from LconjugacyGUI.utils.bsplineOpt.GaussCurvatureBspline import GaussCurvatureBspline
from hanan.geometry.mesh import Mesh


class BSplinePanel:
    def __init__(self, state: AppState):
        self.state = state
        # Panel-local UI state
        self.opt_on = False
        self.it = 0
        self.max_its = 40
        self.output_name = "bsplineLab_output"

        # Weights
        self.w_curv = 0.5
        self.w_mean = 0.001
        self.w_lap = 1.0

        # Button States
        self.oldFunctionType = self.state.radiusFunctionType
        self.oldPercentageRadius = self.state.percentageRadius
        self.oldOffsetValue = self.state.offsetValue

    # def setup(self):
    #     """Called once to set up the B-spline optimizer."""
    #     print("B-spline control points shape:", self.state.bspline.controlpoints.shape)
    #     numControlPoints = self.state.bspline.controlpoints.shape[0] * self.state.bspline.controlpoints.shape[1]

    #     self.state.bSplineOptimizer.add_variable("control_points", numControlPoints * 3)
    #     self.state.bSplineOptimizer.initialize_optimizer(True)
    #     self.state.bSplineOptimizer.init_variable("control_points", self.state.bspline.controlpoints.flatten())

    #     gaussCurvatureTerm = GaussCurvatureBspline()
    #     self.state.bSplineOptimizer.add_objective_term(
    #         gaussCurvatureTerm,
    #         args=(self.state.u_sample_pts, self.state.v_sample_pts, self.state.bspline, 2),
    #         w=2, ce=1,
    #     )

    #     controlPoints, _, controlFaces = control_points_grid(self.state.bspline)
    #     controlMesh = Mesh()
    #     controlMesh.make_mesh(controlPoints, controlFaces)

    #     v_adj = controlMesh.vertex_adjacency_list()

    #     self.state.bSplineOptimizer.prev_iteration_regularizer(w=0.01)
    #     self.state.bSplineOptimizer.set_lap_smooth("control_points", v_adj, dim=3, w=0.001)

    def draw_sampling(self):
        """Draw sampling controls — always visible."""

        psim.Text("Experiment")

        changed, self.state.experiment_name = psim.InputText("Experiment name", self.state.experiment_name)
        if changed:
            self.state.save_path = os.path.join(self.state.output_path, self.state.name, self.state.experiment_name)

        psim.Text("Sampling")
        any_changed = False

        # Half-width for side-by-side sliders (8px accounts for item spacing)
        half_width = (psim.GetContentRegionAvail()[0] - 8) * 0.35

        psim.SetNextItemWidth(half_width)
        changed, self.state.u_sample_num = psim.SliderInt("U samples", self.state.u_sample_num, 10, 200)
        any_changed = changed or any_changed

        psim.SameLine()

        psim.SetNextItemWidth(half_width)
        changed, self.state.v_sample_num = psim.SliderInt("V samples", self.state.v_sample_num, 10, 200)
        any_changed = changed or any_changed

        psim.SetNextItemWidth(half_width)
        changed, self.state.interval_u[0] = psim.SliderFloat("U min", self.state.interval_u[0], 0, self.state.interval_u[1] - 0.01)
        any_changed = changed or any_changed

        psim.SameLine()

        psim.SetNextItemWidth(half_width)
        changed, self.state.interval_u[1] = psim.SliderFloat("U max", self.state.interval_u[1], self.state.interval_u[0] + 0.01, 1)
        any_changed = changed or any_changed

        psim.SetNextItemWidth(half_width)
        changed, self.state.interval_v[0] = psim.SliderFloat("V min", self.state.interval_v[0], 0, self.state.interval_v[1] - 0.01)
        any_changed = changed or any_changed

        psim.SameLine()

        psim.SetNextItemWidth(half_width)
        changed, self.state.interval_v[1] = psim.SliderFloat("V max", self.state.interval_v[1], self.state.interval_v[0] + 0.01, 1)
        any_changed = changed or any_changed

        if any_changed:
            comp.sample_selection(self.state, self.state.interval_u, self.state.interval_v)
            comp.update_visualization(self.state)

        if psim.Button("Flip Normals"):
            self.state.bspline.reverse(direction=0)
            self.state.normalsFlipped = not self.state.normalsFlipped
            comp.init_radius_function(self.state)
            comp.update_visualization(self.state)

        psim.SameLine()

        if psim.Button("Set Reference"):
            comp.compute_reference_structures(self.state)
            comp.visualize_reference_surface(self.state)

        psim.Text("Radius Function Type selection")

        if psim.RadioButton("Safe Spheres", self.state.radiusFunctionType == "safe"):
            self.state.radiusFunctionType = "safe"
        
        if psim.RadioButton("Offset Spheres", self.state.radiusFunctionType == "offset"):
            self.state.radiusFunctionType = "offset"

        psim.SameLine()
        if psim.RadioButton("Mid Spheres", self.state.radiusFunctionType == "mid"):
            self.state.radiusFunctionType = "mid"
        
        if self.state.radiusFunctionType != self.oldFunctionType or self.oldPercentageRadius != self.state.percentageRadius or self.oldOffsetValue != self.state.offsetValue:
            self.oldFunctionType = self.state.radiusFunctionType
            self.oldPercentageRadius = self.state.percentageRadius
            self.oldOffsetValue = self.state.offsetValue
            comp.init_radius_function(self.state)
            comp.compute_radius_function(self.state)
            comp.update_visualization(self.state)


            # sphereCentersSpline = sphere_center_spline_fitting(self.state.bspline, self.state.u_sample_pts, self.state.v_sample_pts)
            
            # sphereCentersV, sphereCentersF = sample_bspline_surface(sphereCentersSpline, self.state.u_sample_pts, self.state.v_sample_pts)
            # ps.register_surface_mesh("Sphere Centers Spline", sphereCentersV, sphereCentersF, color=(0.6, 0.02, 0.8), edge_color=(0, 0, 0), edge_width=1.5) 


        if self.state.radiusFunctionType == "safe":
            pct = self.state.percentageRadius * 100
            _, pct = psim.SliderFloat("Safe radius", pct, 0, 99, "%.0f%%")
            self.state.percentageRadius = pct / 100
            

        if self.state.radiusFunctionType == "offset":
            _, self.state.offsetValue = psim.SliderFloat("Offset value", self.state.offsetValue, 0.0, 1.0)
            
        psim.Separator()

    def draw(self):
        """Draw the optimization UI elements (inside collapsible header)."""

        changed, self.max_its = psim.SliderInt("Max Optimization Iterations", self.max_its, 1, 500)
        _, self.w_curv = psim.SliderFloat("Gaussian Curvature weight", self.w_curv, 0.0, 10.0)
        _, self.w_mean = psim.SliderFloat("Mean Curvature weight", self.w_mean, 0.0, 10.0)
        _, self.w_lap = psim.SliderFloat("Laplacian Smoothness weight", self.w_lap, 0.0, 10.0)

        if psim.Button("Run Optimization"):
            opt_bsp = optimize(
                self.state.bspline, self.state.u_sample_pts, self.state.v_sample_pts,
                epsilon_K=0.5,
                epsilon_H=0.001,
                w_curv=self.w_curv,
                w_mean=self.w_mean,
                w_lap=self.w_lap,
                max_iter=self.max_its
            )
            self.state.bspline = opt_bsp
            comp.update_visualization(self.state)
       

        psim.Text("Export Bspline surface filename:")
        changed, self.output_name = psim.InputText("Name", self.output_name)

        if psim.Button("Export"):
            save_bspline_to_json(self.state.bspline, str(self.state.surface_path) +"/"+ self.output_name + ".json")

        psim.Separator()

        if psim.Button("Reset B-spline"):
            self.state.bspline = self.state.bsplineOriginal.clone()
            comp.update_visualization(self.state)

    def _draw(self):
        """Called every frame. Sampling always visible, optimization collapsible."""
        self.draw_sampling()
        if psim.CollapsingHeader("B-spline Optimization"):
            self.draw()

    def update(self):
        """Called every frame. Optimization is event-driven, nothing to step."""
        pass