# Code/gui/lconjugacy_panel.py
from LconjugacyGUI.state import AppState
import LconjugacyGUI.computations as comp

import os
import numpy as np
import polyscope as ps
import polyscope.imgui as psim

from LconjugacyGUI.utils.bsplines_functions import (
    compute_rij_transport_vectorized,
    compute_lconjugate_at_point,
    sample_bspline_surface,
    principal_frame,
    compute_lconjugate_directions,
    init_lconjugate_directions,
    radius_spline_fitting,
    sample_radius,
    exchange_lconjugate_rulings_curve_tangents,
    compute_crossfield_singularities,
    visualize_singularities,
    init_lconjugate_boundary
)

from LconjugacyGUI.utils.directionsOpt.DirectionFieldFairness import DirectionFieldSmoothness
from LconjugacyGUI.utils.directionsOpt.LconjugateOpt import Lconjugacy
from LconjugacyGUI.utils.directionsOpt.angleLconjugates import AngleLconjugate
from LconjugacyGUI.utils.directionsOpt.angleBoundary import AngleBoundary

from hanan.geometry.mesh import Mesh
from hanan.geometry.utils import (
    visualize_frame,
    visualize_cross_field,
    deg_to_rad,
    draw_polyline
)


class LConjugacyPanel:
    def __init__(self, state: AppState):
        self.state = state
        # Panel-local UI state
        self.optDF_on = False
        self.itDF = 0
        self.maxitsDF = 40
        self.targetAngle = 30.0
        

        # Button state Initialization
        self.initializationMethod = "constant"
        # ["constant", "linear","linearU", "linearV", "boundary"]

        # Boundary initialization data
        self.indicesBoundaryTheta, self.valuesBoundaryTheta, self.indicesBoundaryPhi, self.valuesBoundaryPhi = None, None, None, None

    def setup_optimizer(self):
        """Set up the direction field optimizer."""
        state = self.state
        num_sampledPts = len(state.u_sample_pts) * len(state.v_sample_pts)
        radiusFunctionCtrlPts = state.radiusFunction.controlpoints.reshape(-1, 3)[:, 2]  # Get only the radius component of the control points

        beta = state.phi_angles - state.theta_angles

        # Declare variables and set initial values
        state.directionFieldOptimizer.add_variable("th",  state.theta_angles)
        state.directionFieldOptimizer.add_variable("bet", beta)
        state.directionFieldOptimizer.add_variable("rij", radiusFunctionCtrlPts + 0.1)  # small offset avoids zero-radius issues

        t1, _, k1, k2, _ = principal_frame(state.bspline, state.u_sample_pts, state.v_sample_pts)

        V, F = sample_bspline_surface(state.bspline, state.u_sample_pts, state.v_sample_pts)
        mesh = Mesh()
        mesh.make_mesh(V, F)
        ei, ej = mesh.edge_vertices()

        n = state.bspline.normal(state.u_sample_pts, state.v_sample_pts).reshape(-1, 3)

        alpha_ij = compute_rij_transport_vectorized(
            t1[ei].reshape(-1, 3), t1[ej].reshape(-1, 3),
            n[ei].reshape(-1, 3), n[ej].reshape(-1, 3),
        )

        lConjugacyTerm = Lconjugacy()
        lConjugacyTerm.name = "lconjugacy"
        state.directionFieldOptimizer.add_objective_term(
            lConjugacyTerm,
            args=(k1, k2, state.radiusFunction, state.u_sample_pts, state.v_sample_pts),
            w=state.weight_DF["lconjugacy"], ce=1,
        )

        smoothnessTerm = DirectionFieldSmoothness()
        smoothnessTerm.name = "smooth"
        state.directionFieldOptimizer.add_objective_term(
            smoothnessTerm,
            args=(mesh.edge_vertices(), alpha_ij),
            w=state.weight_DF["smooth"], ce=1,
        )

        targetAngle = deg_to_rad(self.targetAngle)
        angleTerm = AngleLconjugate()
        angleTerm.name = "angle"
        state.directionFieldOptimizer.add_objective_term(
            angleTerm, args=([targetAngle]), w=state.weight_DF["angle"], ce=1,
        )

        if self.initializationMethod == "boundary":
            angleBoundaryTerm = AngleBoundary()
            angleBoundaryTerm.name = "angleBoundary"
            state.directionFieldOptimizer.add_objective_term(
                angleBoundaryTerm, args=(self.indicesBoundaryTheta, self.valuesBoundaryTheta, self.indicesBoundaryPhi, self.valuesBoundaryPhi), w=state.weight_DF["angleBoundary"], ce=1,
            )

        state.directionFieldOptimizer.prev_iteration_regularizer(w=0.1)

        # Reset iteration bookkeeping — call after all variables and terms are set up
        state.directionFieldOptimizer.initialize_optimizer(verbose=True, adaptive_mu=False)

    def update_cross_field(self):
        """Update cross field visualization after optimization step."""
        state = self.state
        e1, e2, k1, k2, _ = principal_frame(state.bspline, state.u_sample_pts, state.v_sample_pts)
        
        t1 = np.cos(state.theta_angles)[:, None] * e1 + np.sin(state.theta_angles)[:, None] * e2
        t2 = np.cos(state.phi_angles)[:, None] * e1 + np.sin(state.phi_angles)[:, None] * e2

        lc1, lc2 = exchange_lconjugate_rulings_curve_tangents(e1, e2, k1, k2, t1, t2)

        sampledPoints = state.bspline(state.u_sample_pts, state.v_sample_pts).reshape(-1, 3)

        sampleVertices, sampleFaces = sample_bspline_surface(state.bspline, state.u_sample_pts, state.v_sample_pts)
        n = state.bspline.normal(state.u_sample_pts, state.v_sample_pts).reshape(-1, 3)

        visualize_cross_field("L-conjugate Tangents_", sampledPoints, t1.reshape(-1, 3), t2.reshape(-1, 3), self.state.lenghtCrossField, colors=[[1, 0, 0], [0, 0, 1]], radius=self.state.radiusCrossField)
        visualize_cross_field("L-conjugate Rulings_", sampledPoints, lc1.reshape(-1, 3), lc2.reshape(-1, 3), self.state.lenghtCrossField, colors=[[0, 0, 0], [1, 1, 1]], radius=self.state.radiusCrossField)

        # Visualize Sphere centers
        radiusValues = comp.sample_radius(state.radiusFunction, state.u_sample_pts, state.v_sample_pts)
        sphereCenters = sampledPoints + radiusValues[:, None] * n 
        ps.register_surface_mesh("Sphere Centers", sphereCenters, sampleFaces, enabled=True, color=[0.0, 1.0, 0.0], edge_color= (0,0,0), edge_width=0.01)
        
    def compute_singularities(self):
        """Compute and visualize cross-field singularities."""
        state = self.state
        if state.theta_angles is None:
            print("Please initialize L-conjugate directions first.")
            return

        V, F = sample_bspline_surface(state.bspline, state.u_sample_pts, state.v_sample_pts)
        mesh = Mesh()
        mesh.make_mesh(V, F)

        t1, _, _, _, n = principal_frame(state.bspline, state.u_sample_pts, state.v_sample_pts)

        _, sing_verts, sing_indices = compute_crossfield_singularities(
            mesh, state.theta_angles, t1, n, symmetry=4,
        )

        visualize_singularities("Cross-field Singularities", sing_verts, sing_indices, V, radius=0.02)

        state.singularity_vertices = sing_verts
        state.singularity_indices = sing_indices

  
    def init_directions(self):
        """Initialize L-conjugate directions with a given angles for the first L-conjugate direction."""
        state = self.state

        radiusValues = comp.sample_radius(state.radiusFunction, state.u_sample_pts, state.v_sample_pts)
        sampledPoints = state.bspline(state.u_sample_pts, state.v_sample_pts).reshape(-1, 3)

        if self.initializationMethod != "boundary":
            t1, t2, lc1, lc2, theta, phi = init_lconjugate_directions(
                state.bspline,
                state.u_sample_pts,
                state.v_sample_pts,
                radiusValues,
                state.theta_angles,
            )

            # Update angles 
            state.theta_angles = theta
            state.phi_angles = phi
        else: 
            t1, t2, lc1, lc2, theta, phi, self.indicesBoundaryTheta, self.valuesBoundaryTheta, self.indicesBoundaryPhi, self.valuesBoundaryPhi = init_lconjugate_boundary(
                state.bspline,
                state.u_sample_pts,
                state.v_sample_pts,
                radiusValues,
                state.initial_theta,
            )

            # Update angles 
            state.theta_angles = theta
            state.phi_angles = phi

        self.setup_optimizer()
        comp.update_visualization(state)
        self.update_cross_field()

    def update_variables_from_optimizer(self):
        """Update direction field variables from optimizer after optimization step."""
        state = self.state

        theta, beth, rij = state.directionFieldOptimizer.unpack("th", "bet", "rij")
        phi = theta + beth

        state.radiusFunction[2] = rij
        state.theta_angles = theta
        state.phi_angles = phi
        
        


    def draw(self):
        """Draw the L-conjugacy panel UI elements."""
        state = self.state

        _, self.state.lenghtCrossField = psim.SliderFloat("Cross field length", self.state.lenghtCrossField, 0.01, 0.1)
        _, self.state.radiusCrossField = psim.SliderFloat("Cross field radius", self.state.radiusCrossField, 0.0001, 0.02, format="%.4f")


        psim.Separator()

        if psim.Button("Compute Principal Directions"):
            e1, e2, _, _, n = principal_frame(state.bspline, state.u_sample_pts, state.v_sample_pts)
            pts = state.bspline(state.u_sample_pts, state.v_sample_pts).reshape(-1, 3)
            visualize_frame("Principal Directions", pts, e1.reshape(-1, 3), e2.reshape(-1, 3), n.reshape(-1, 3))
        psim.Separator()


        psim.Text("L-conjugate Optimization")
        changedInit, self.state.initial_theta = psim.SliderFloat("Initial Angle (deg)", self.state.initial_theta, 0.0, 90.0)
        changedTarget, self.targetAngle = psim.SliderFloat("Target Angle (deg)", self.targetAngle, 0.0, 90.0)
        if psim.RadioButton("Constant Angle Initialization", self.initializationMethod == "constant"):
            self.initializationMethod = "constant"
        if psim.RadioButton("Linear Angle Initialization", self.initializationMethod == "linear"):
            self.initializationMethod = "linear"
        if psim.RadioButton("Linear U Angle Initialization", self.initializationMethod == "linearU"):
            self.initializationMethod = "linearU"
        if psim.RadioButton("Linear V Angle Initialization", self.initializationMethod == "linearV"):
            self.initializationMethod = "linearV"
        if psim.RadioButton("Cosine V Angle Initialization", self.initializationMethod == "cosineV"):
            self.initializationMethod = "cosineV"
        if psim.RadioButton("Cosine U Angle Initialization", self.initializationMethod == "cosineU"):
            self.initializationMethod = "cosineU"
        if psim.RadioButton("Boundary Angle Initialization", self.initializationMethod == "boundary"):
            self.initializationMethod = "boundary"
     
        if psim.Button("Initialize L-conjugate Directions") or changedInit or changedTarget:
            if self.initializationMethod == "linear":
                state.theta_angles = comp.linear_angle_distribution(state,0, 90)
            elif self.initializationMethod == "linearU":
                state.theta_angles = comp.linear_angle_distribution_dir_u(state,0, 60)
            elif self.initializationMethod == "linearV":
                state.theta_angles = comp.linear_angle_distribution_dir_v(state,0, 60)
            elif self.initializationMethod == "constant":
                state.theta_angles= deg_to_rad(self.state.initial_theta) * np.ones(len(state.u_sample_pts) * len(state.v_sample_pts))
            elif self.initializationMethod == "cosineV":
                state.theta_angles = comp.cosine_angle_distribution_v(state)
            elif self.initializationMethod == "cosineU":
                state.theta_angles = comp.cosine_angle_distribution_u(state)
            
            self.init_directions()        


        psim.Separator()
        psim.Text("Optimization:")
        psim.Text("Weights for L-conjugate direction optimization:")

        _, state.weight_DF["lconjugacy"] = psim.SliderFloat(
                                            "L-conjugacy weight",  # Label
                                            state.weight_DF["lconjugacy"],  # Value
                                            0.0,  # Minimum
                                            10.0, # Maximum
                                            format="%.4f" 
                                            )
        
        _, state.weight_DF["smooth"] = psim.SliderFloat(
                                            "Smoothness weight", 
                                            state.weight_DF["smooth"],
                                            0.0, 
                                            0.01, 
                                            format="%.4f"
                                            )
        
        _, state.weight_DF["angle"] = psim.SliderFloat(
                                        "Angle weight", 
                                        state.weight_DF["angle"], 
                                        0.0, 
                                        0.01,
                                        format="%.4f"                                        
                                        )
        _, state.weight_DF["angleBoundary"] = psim.SliderFloat(
                                        "Boundary Angle weight", 
                                        state.weight_DF["angleBoundary"], 
                                        0.0, 
                                        10.0,
                                        format="%.4f"                                        
                                        )

        _, self.maxitsDF = psim.SliderInt("Max Optimization Iterations", self.maxitsDF, 1, 100)

        if psim.Button("Run Optimization"):
            state.directionFieldOptimizer.set_term_weights(state.weight_DF)
            self.optDF_on = True
            self.itDF = 0


        psim.Separator()
        psim.Separator()

        psim.Text("Quad Remeshing:")

        if psim.Button("Run Remesher (Integration-based)"):
            comp.save_cross_field(state)
            comp.run_remesher(state)
            if os.path.exists(os.path.join(state.save_path, "t_Remeshed.obj")):
                comp.save_pre_remesh_state(state)

        if psim.Button("Run Remesher (Direct)"):
            comp.save_cross_field(state)
            comp.run_remesher(state, integrated=False)
            if os.path.exists(os.path.join(state.save_path, "t_Remeshed.obj")):
                comp.save_pre_remesh_state(state)

    def _draw(self):
        """Called every frame. Collapsible section inside the main window."""
        if psim.CollapsingHeader("L-conjugacy Directions"):
            self.draw()

    def update(self):
        """Called every frame for optimization stepping."""
        if self.itDF >= self.maxitsDF:
            self.optDF_on = False
            self.itDF = 0
            self.state.directionFieldOptimizer.print_report()

        if self.optDF_on:
            self.state.directionFieldOptimizer.get_gradients()
            self.state.directionFieldOptimizer.optimize_step()
            self.update_variables_from_optimizer()
            self.update_cross_field()
            self.itDF += 1
