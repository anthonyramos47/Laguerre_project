# Code/gui/integration_panel.py
from LconjugacyGUI.state import AppState

import numpy as np
import polyscope as ps
import polyscope.imgui as psim

from LconjugacyGUI.utils.bsplines_functions import (
    principal_frame_at_point,
    compute_lconjugate_curve_tangents_at_point,
    exchange_lconjugate_rulings_curve_tangents_at_point,
    integrate_lconjugate_dir,
    sample_radius,
    char_circle_direction
)

from hanan.geometry.utils import (
    unit,
    deg_to_rad,
    mathematica_v,
    save_discrete_laguerre_channel_surface,
    draw_cone_contact
)


class IntegrationPanel:
    def __init__(self, state: AppState):
        self.state = state
        # Panel-local UI state
        self.u0 = 0.5
        self.v0 = 0.5
        self.integration_step = 0.001
        self.numSteps = 200
        self.uvPoint = None
        self.uvIntegratedCurve = None
        self.curve = None
        self.pointOnCurve = None
        self.maxCurveRange = None

    

    def show_discrete_channel_surface(self, uv_integratedCurve):
        """Visualize discrete channel surface in Polyscope."""
        state = self.state
        sampling = len(uv_integratedCurve) // 10
        uv_valuesToUse = uv_integratedCurve[::sampling]

        pointSpline = np.array([state.bspline(uv[0], uv[1]) for uv in uv_valuesToUse])
        radiusValues = np.array([state.radiusFunction(uv[0], uv[1])[2] for uv in uv_valuesToUse])
        n = np.array([state.bspline.normal(uv[0], uv[1]) for uv in uv_valuesToUse])
        centersSpheres = pointSpline + radiusValues[:, np.newaxis] * n

        for i in range(len(centersSpheres) - 1):
            ci = centersSpheres[i]
            ri = radiusValues[i]
            cj = centersSpheres[i + 1]
            rj = radiusValues[i + 1]
            draw_cone_contact(ci, ri, cj, rj, n=400, name="cone_strip_face_" + str(i))

        ps.register_curve_network(
            "Discrete Centers", centersSpheres,
            np.array([[i, i + 1] for i in range(len(centersSpheres) - 1)]),
            radius=0.002, color=(0.0, 0.5, 0.5),
        )

        centers = np.array([c.tolist() for c in centersSpheres])
        points = np.array([p.tolist() for p in pointSpline])
        radius = np.array(radiusValues)

        print("centers = ", mathematica_v(centers))
        print("points = ", mathematica_v(points))
        print("radius = ", mathematica_v(radius))

        save_discrete_laguerre_channel_surface("discreteCh", points, centers)

        NormalLines = np.column_stack((pointSpline, centersSpheres)).reshape(-1, 3)
        edges = np.array([[i, i + 1] for i in range(0, len(NormalLines), 2)])
        ps.register_curve_network("Normal lines", NormalLines, edges, radius=0.003, color=(0, 0, 0))

    def show_lconjugacy_structure(self, angle):
        """Visualize L-conjugacy structure at a point."""
        state = self.state
        if self.uvPoint is None:
            return

        uP, vP = self.uvPoint
        pointOnCurve = state.bspline(uP, vP)

        e1, e2, k1, k2, n = principal_frame_at_point(state.bspline, uP, vP)
        r = state.radiusFunction(uP, vP)[2]  # Get radius value at current (u,v)

        theta = angle
        t1, t2, phi = compute_lconjugate_curve_tangents_at_point(e1, e2, k1, k2, r, theta)
        ruling1, ruling2 = exchange_lconjugate_rulings_curve_tangents_at_point(e1, e2, k1, k2, t1, t2)
    
        point = ps.register_point_cloud("Point on Curve", np.array([pointOnCurve]), radius=0.005, color=(0.0, 0.0, 0.0))
        point.add_vector_quantity("e1", np.array([e1]), length=0.05, color=(0.0, 1.0, 1.0), enabled=False)
        point.add_vector_quantity("e2", np.array([e2]), length=0.05, color=(0.0, 1.0, 0.0), enabled=False)
        point.add_vector_quantity("Ruling 1", np.array([ruling1]), length=0.05, color=(0, 0, 0), enabled=True)
        point.add_vector_quantity("Ruling 2", np.array([ruling2]), length=0.05, color=(1, 1, 1), enabled=True)
        point.add_vector_quantity("Tangent 1", np.array([t1]), length=0.05, color=(1, 0, 0), enabled=True)
        point.add_vector_quantity("Tangent 2", np.array([t2]), length=0.05, color=(0, 0, 1), enabled=True)


        orthogonalCircleDir1 = char_circle_direction(e1, e2, k1, k2, r, t1)
        orthogonalCircleDir2 = char_circle_direction(e1, e2, k1, k2, r, t2)

        point.add_vector_quantity("charCirc1", np.array([orthogonalCircleDir1]), length=0.05, color=(0.8, 0.8, 0), enabled=True)
        point.add_vector_quantity("charCirc2", np.array([orthogonalCircleDir2]), length=0.05, color=(0.8, 0.0, 0.8), enabled=True)

    def draw(self):
        """Draw the integration panel UI elements."""
        state = self.state

        psim.Text("Initial Point (u0, v0):")
        _, self.u0 = psim.SliderFloat("u0", self.u0, 0.0, 1.0)
        _, self.v0 = psim.SliderFloat("v0", self.v0, 0.0, 1.0)

        init_point = ps.register_point_cloud("Initial Point", np.array([state.bspline(self.u0, self.v0)]), radius=0.007, color=(0.5, 0.0, 0.7))
        t1, t2, _, _, _ = principal_frame_at_point(state.bspline, self.u0, self.v0)

        _, self.integration_step = psim.SliderFloat("Integration Step Size", self.integration_step, 0.0001, 0.01, format="%.2e")
        _, self.numSteps = psim.SliderInt("Number of Integration Steps", self.numSteps, 100, 5000)

        initial_angle = deg_to_rad(self.state.initial_theta)

        if psim.Button("Integrate L-conjugate Directions"):
            curve, centers, uv_integratedCurve = integrate_lconjugate_dir(
                state.bspline, state.radiusFunction,
                initial_angle, self.u0, self.v0, 
                step_size=self.integration_step, num_steps=self.numSteps,
            )

            ps.register_curve_network(
                "Integrated pseudoLconj curve", curve,
                np.array([[i, i + 1] for i in range(len(curve) - 1)]),
                radius=0.001, color=(0.0, 0.05, 0.8),
            )
            ps.register_curve_network(
                "Integrated centers", centers,
                np.array([[i, i + 1] for i in range(len(centers) - 1)]),
                radius=0.0001, color=(0.0, 1.0, 1.0), enabled=True
            )
            self.show_discrete_channel_surface(uv_integratedCurve)

            self.curve = curve
            
            self.uvIntegratedCurve = uv_integratedCurve
            self.pointOnCurve = 0
            self.maxCurveRange = len(uv_integratedCurve) - 1

            psim.Separator()

        psim.Text("Visualize L-conjugacy Structure at Point on Curve")

        if self.uvIntegratedCurve is not None:
            changed, self.pointOnCurve = psim.SliderFloat("t", self.pointOnCurve, 0, self.maxCurveRange)
            self.uvPoint = self.uvIntegratedCurve[int(self.pointOnCurve)]

            if changed:
                self.show_lconjugacy_structure(initial_angle)

    def _draw(self):
        """Called every frame. Collapsible section inside the main window."""
        if psim.CollapsingHeader("Integration"):
            self.draw()

    def update(self):
        """Called every frame. Integration is event-driven, nothing to step."""
        pass
