# Code/gui/visualizer.py
"""Standalone visualizer for saved lconjugacy_data.pkl files.

Usage:
    python -m gui.visualizer <path_to_lconjugacy_data.pkl>

Example:
    python -m gui.visualizer gui/out/testSimpleOptimized/ilustration4/lconjugacy_data.pkl
"""

import argparse
import os
import pickle
import sys
import numpy as np
import polyscope as ps
import polyscope.imgui as psim

from LconjugacyGUI.utils.bsplines_functions import (
    sample_bspline_surface,
    bspline_curvatures,
    control_points_grid,
    principal_frame,
    sample_radius,
)
from hanan.geometry.utils import (
    triangulate_quads,
    visualize_cross_field,
    unit,
    write_obj,
    draw_polygon,
    save_discrete_laguerre_channel_surface_spheres,
    draw_cone_contact,
)





class Visualizer:
    def __init__(self, pkl_path: str):
        with open(pkl_path, "rb") as f:
            self.data = pickle.load(f)

        self.save_path = os.path.dirname(pkl_path)

        self.channelSurfacePath = "/path/to/Laguerre_project/Code/gui/channelSurfaces/"

        # ----- Unpack saved data -----
        self.bspline = self.data["bspline"]
        self.u_sample_pts = self.data["u_sample_pts"]
        self.v_sample_pts = self.data["v_sample_pts"]
        self.theta_angles = self.data["theta_angles"]
        self.phi_angles = self.data.get("phi_angles")
        self.radiusFunction = self.data["radiusFunction"]
        self.initial_radiusPercentage = self.data.get("initial_RadiusPercentage", None)

        # Quad remeshed meshes (Mesh objects)
        self.quadRemeshed_t = self.data.get("quadRemeshed_t")
        self.quadRemeshed_lc = self.data.get("quadRemeshed_lc")

        # Sphere centers for remeshed meshes
        self.sphereCenters_t = self.data.get("sphereCenters_t")
        self.sphereCenters_lc = self.data.get("sphereCenters_lc")

        # Reference surface
        self.v_ref = self.data.get("v_ref")
        self.f_ref = self.data.get("f_ref")
        self.n_ref = self.data.get("n_ref")
        self.sphereCenters_ref = self.data.get("sphereCenters_ref")

        # L-mesh (key is "Lmesh" in save_all)
        self.lmesh = self.data.get("Lmesh")

        centers = [sf.center for sf in self.lmesh.SF]
        radii = [sf.radius for sf in self.lmesh.SF]
        normals = [vp[:3] for vp in self.lmesh.top.vertices]
        offsets = [vp[3] for vp in self.lmesh.top.vertices]

        self.lmesh.update_spheres_planes(centers, radii, normals, offsets)

        # Optimization reports
        self.bspline_opt = self.data.get("Bspline_Optimization")
        self.lconjugacy_opt = self.data.get("Lconjugacy_Optimization")
        self.lmesh_opt = self.data.get("Lmesh_Optimization")

        # Derive intervals from sample points
        self.interval_u = [float(self.u_sample_pts[0]), float(self.u_sample_pts[-1])]
        self.interval_v = [float(self.v_sample_pts[0]), float(self.v_sample_pts[-1])]

        self.selectedFace = 0
        self.MaxFaces = len(self.quadRemeshed_t.faces) if self.quadRemeshed_t is not None else 0

        # UI toggles
        self.show_bspline = True
        self.show_control_mesh = False
        self.show_lmesh = True
        self.show_reference = False
        self.show_sphere_centers_ref = False
        self.show_crossfield = False
        self.show_focal1 = False
        self.show_focal2 = False

    # ------------------------------------------------------------------ #
    # Registration helpers
    # ------------------------------------------------------------------ #

    def register_bspline(self):
        V, F = sample_bspline_surface(self.bspline, self.u_sample_pts, self.v_sample_pts)
        K, _, n = bspline_curvatures(self.bspline, self.u_sample_pts, self.v_sample_pts)
        mesh = ps.register_surface_mesh("B-spline Surface", V, F,
                                        color=(0.5, 0.5, 0.5),
                                        edge_color=(0, 0, 0), edge_width=1.1,
                                        enabled=self.show_bspline)
        mesh.add_vector_quantity("Normals", n.reshape(-1, 3),
                                defined_on="vertices", length=0.3)
        mesh.add_scalar_quantity("Gaussian Curvature", K.reshape(-1),
                                defined_on="vertices", cmap="coolwarm")
        # mesh.add_scalar_quantity("Mean Curvature", H.reshape(-1),
        #                         defined_on="vertices", cmap="coolwarm")
        
        mesh.add_scalar_quantity("theta", self.theta_angles.reshape(-1),
                                defined_on="vertices", enabled=True)
        
        # U=0 and V=0 isolines
        u_isoline = self.bspline(self.u_sample_pts[0], self.v_sample_pts).reshape(-1, 3)
        v_isoline = self.bspline(self.u_sample_pts, self.v_sample_pts[0]).reshape(-1, 3)

        Uedges = np.array([[i, i + 1] for i in range(len(u_isoline) - 1)])
        Vedges = np.array([[i, i + 1] for i in range(len(v_isoline) - 1)])

        ps.register_curve_network("U=0 Isoline", u_isoline, Uedges, radius=0.002, color=(1.0, 0.0, 1.0))
        ps.register_curve_network("V=0 Isoline", v_isoline, Vedges, radius=0.002, color=(1.0, 1.0, 0.0))

        ps.register_surface_mesh("Quad Mesh", self.quadRemeshed_t.vertices, self.quadRemeshed_t.faces, color=(0.0, 0.5, 0.5), edge_color=(0, 0, 0), edge_width=1.0, enabled=True)

        # Control mesh
        cp, ce, _ = control_points_grid(self.bspline)
        ps.register_curve_network("Control Mesh", cp.reshape(-1, 3), ce,
                                  radius=0.001, color=(1, 0, 0),
                                  enabled=self.show_control_mesh)
        ps.register_point_cloud("Control Points", cp.reshape(-1, 3),
                                radius=0.003, color=(0, 0, 0),
                                enabled=self.show_control_mesh)

    def register_lmesh(self):
        if self.lmesh is None:
            return
        self.lmesh.visualize()

    def register_reference(self):
        if self.v_ref is None:
            return
        ps.register_surface_mesh("Reference Surface", self.v_ref, self.f_ref,
                                 color=(0.8, 0.8, 0.8),
                                 edge_color=(0, 0, 0), edge_width=1.5,
                                 enabled=self.show_reference)

    def register_sphere_centers_ref(self):
        if self.sphereCenters_ref is None or self.f_ref is None:
            return
        ps.register_surface_mesh("Sphere Centers Ref", self.sphereCenters_ref, self.f_ref,
                                 color=(0.6, 0.02, 0.8),
                                 edge_color=(0, 0, 0), edge_width=1.5,
                                 enabled=self.show_sphere_centers_ref)

    def register_focal_surfaces(self):
        if self.v_ref is None:
            return
        denseU = np.linspace(self.interval_u[0], self.interval_u[1], 250)
        denseV = np.linspace(self.interval_v[0], self.interval_v[1], 250)
        V, F = sample_bspline_surface(self.bspline, denseU, denseV)
        F_tri = np.array(triangulate_quads(F))
        n = self.bspline.normal(denseU, denseV).reshape(-1, 3)
        _, _, k1, k2, _ = principal_frame(self.bspline, denseU, denseV)

        R1 = 1 / k1
        R2 = 1 / k2
        focal1 = V + n * R1.reshape(-1, 1)
        focal2 = V + n * R2.reshape(-1, 1)

        ps.register_surface_mesh("Focal Surface 1", focal1, F_tri,
                                 color=(0.8, 0.0, 0.0),
                                 edge_color=(0, 0, 0), edge_width=0.5,
                                 enabled=self.show_focal1)
        ps.register_surface_mesh("Focal Surface 2", focal2, F_tri,
                                 color=(0.0, 0.8, 0.0),
                                 edge_color=(0, 0, 0), edge_width=0.5,
                                 enabled=self.show_focal2)

    def register_crossfield(self):
        e1, e2, k1, k2, _ = principal_frame(self.bspline, self.u_sample_pts, self.v_sample_pts)
        t1 = np.cos(self.theta_angles)[:, None] * e1 + np.sin(self.theta_angles)[:, None] * e2
        if self.phi_angles is not None:
            t2 = np.cos(self.phi_angles)[:, None] * e1 + np.sin(self.phi_angles)[:, None] * e2
        else:
            t2 = -np.sin(self.theta_angles)[:, None] * e1 + np.cos(self.theta_angles)[:, None] * e2

        pts = self.bspline(self.u_sample_pts, self.v_sample_pts).reshape(-1, 3)
        visualize_cross_field("L-conjugate Tangents_", pts,
                            t1.reshape(-1, 3), t2.reshape(-1, 3),
                            0.02, colors=[[1, 0, 0], [0, 0, 1]], radius=0.002)

    # ------------------------------------------------------------------ #
    # Scene setup
    # ------------------------------------------------------------------ #

    def build_scene(self):
        self.register_bspline()
        self.register_lmesh()
        self.register_reference()
        self.register_sphere_centers_ref()

    # ------------------------------------------------------------------ #
    # imgui callback
    # ------------------------------------------------------------------ #

    def callback(self):
        psim.TextUnformatted("=== Saved Data Visualizer ===")
        psim.Separator()

        # ---- Info section ----
        if psim.CollapsingHeader("Info"):
            psim.Text(f"U samples: {len(self.u_sample_pts)}")
            psim.Text(f"V samples: {len(self.v_sample_pts)}")
            psim.Text(f"U range: [{self.interval_u[0]:.4f}, {self.interval_u[1]:.4f}]")
            psim.Text(f"V range: [{self.interval_v[0]:.4f}, {self.interval_v[1]:.4f}]")

            if self.initial_radiusPercentage is not None:
                psim.Text(f"Initial Radius Percentage: {self.initial_radiusPercentage:.4f}")

            if self.theta_angles is not None:
                psim.Text(f"Initial theta range: [{np.rad2deg(self.theta_angles.min()):.2f}, {np.rad2deg(self.theta_angles.max()):.2f}] deg")

            psim.Separator()

            # Oriented contact deviation
            if self.lmesh is not None:
                try:
                    dev = self.lmesh.compute_oriented_contact_dev()
                    psim.Text(f"Oriented Contact Deviation: {dev:.3e}")
                except Exception as e:
                    psim.Text(f"Contact deviation error: {e}")
            else:
                psim.Text("L-mesh: not available")

        # ---- Visibility toggles ----
        if psim.CollapsingHeader("Visibility"):
            changed, self.show_bspline = psim.Checkbox("B-spline Surface", self.show_bspline)
            if changed:
                ps.get_surface_mesh("B-spline Surface").set_enabled(self.show_bspline)

            changed, self.show_control_mesh = psim.Checkbox("Control Mesh", self.show_control_mesh)
            if changed:
                ps.get_curve_network("Control Mesh").set_enabled(self.show_control_mesh)
                ps.get_point_cloud("Control Points").set_enabled(self.show_control_mesh)

            changed, self.show_reference = psim.Checkbox("Reference Surface", self.show_reference)
            if changed:
                try:
                    ps.get_surface_mesh("Reference Surface").set_enabled(self.show_reference)
                except:
                    pass

            changed, self.show_sphere_centers_ref = psim.Checkbox("Sphere Centers Ref Surface", self.show_sphere_centers_ref)
            if changed:
                try:
                    ps.get_surface_mesh("Sphere Centers Ref").set_enabled(self.show_sphere_centers_ref)
                except:
                    pass

        # ---- Compute on-demand ----
        if psim.CollapsingHeader("On-Demand"):
            if psim.Button("Show Cross Field"):
                self.register_crossfield()

            if psim.Button("Compute Focal Surfaces"):
                self.register_focal_surfaces()

            if self.lmesh is not None and psim.Button("Re-export L-mesh"):
                self.lmesh.export_data(self.save_path, 'lmesh')

            if psim.Button("print Optimizer Logs"):
                print(self.lmesh_opt["data"])
            

            changed, self.selectedFace = psim.SliderInt("Selected Face", self.selectedFace, 0, self.MaxFaces - 1)

            if psim.Button("Show Channel Surfaces"):
                u_faces, v_faces = self.quadRemeshed_t.quad_face_strips(self.selectedFace)

                u_contactPoints = []
                u_sphereCenters = []
                u_radius = []

                quadMeshBarycenters = self.quadRemeshed_t.face_barycenters()

                for uf in u_faces:
                    pts = self.quadRemeshed_t.vertices[self.quadRemeshed_t.faces[uf]]
                    draw_polygon(pts, name="u_strip_face_" + str(uf), color=[1, 0, 0])
                    u_contactPoints.append(quadMeshBarycenters[uf])
                    u_sphereCenters.append(self.lmesh.SF[uf].center)
                    u_radius.append(self.lmesh.SF[uf].radius)

                save_discrete_laguerre_channel_surface_spheres(self.channelSurfacePath+"discreteUChannel", np.array(u_sphereCenters), np.array(u_radius), np.array(u_contactPoints))

                v_contactPoints = []
                v_sphereCenters = []
                v_radius = []

                for vf in v_faces:
                    pts = self.quadRemeshed_t.vertices[self.quadRemeshed_t.faces[vf]]
                    draw_polygon(pts, name="v_strip_face_" + str(vf), color=[0, 1, 0])
                    v_contactPoints.append(quadMeshBarycenters[vf])
                    v_sphereCenters.append(self.lmesh.SF[vf].center)
                    v_radius.append(self.lmesh.SF[vf].radius)

                save_discrete_laguerre_channel_surface_spheres(self.channelSurfacePath+"discreteVChannel", np.array(v_sphereCenters), np.array(v_radius), np.array(v_contactPoints))

                for i in range(len(u_faces) - 1):
                    s1 = self.lmesh.SF[u_faces[i]]
                    s2 = self.lmesh.SF[u_faces[i + 1]]
                    draw_cone_contact(s1.center, s1.radius, s2.center, s2.radius, n=300, name="cone_stripUface_" + str(i))

                for i in range(len(v_faces) - 1):
                    s1 = self.lmesh.SF[v_faces[i]]
                    s2 = self.lmesh.SF[v_faces[i + 1]]
                    draw_cone_contact(s1.center, s1.radius, s2.center, s2.radius, n=300, name="cone_stripVface_" + str(i))
        
        # ---- Optimization reports ----
        if psim.CollapsingHeader("Optimization Reports"):
            if self.bspline_opt is not None:
                psim.Text("--- B-spline Optimization ---")
                for line in str(self.bspline_opt["report"]).split("\n"):
                    psim.Text(line)
                psim.Separator()

            if self.lconjugacy_opt is not None:
                psim.Text("--- L-conjugacy Optimization ---")
                for line in str(self.lconjugacy_opt["report"]).split("\n"):
                    psim.Text(line)
                psim.Separator()

            if self.lmesh_opt is not None:
                psim.Text("--- L-mesh Optimization ---")
                for line in str(self.lmesh_opt["report"]).split("\n"):
                    psim.Text(line)

            if self.bspline_opt is None and self.lconjugacy_opt is None and self.lmesh_opt is None:
                psim.Text("No optimization reports saved.")


def main():
    parser = argparse.ArgumentParser(description="Visualize saved lconjugacy_data.pkl")
    parser.add_argument("surface", type=str,
                        help="Surface name ")
    parser.add_argument("experiment", type=str,
                        help="Experiment name")
    # parser.add_argument("pkl_path", type=str,
    #                     help="Path to lconjugacy_data.pkl file")
    args = parser.parse_args()

    surface_name = args.surface
    experiment_name = args.experiment

    pkl_path = f"gui/out/{surface_name}/{experiment_name}/lconjugacy_data.pkl"

    vis = Visualizer(pkl_path)

    # Polyscope init
    ps.set_automatically_compute_scene_extents(False)
    ps.set_length_scale(0.5)
    ps.set_bounding_box(np.array((-2., -2., -2.)), np.array((2., 2., 2.)))
    ps.init()

    vis.build_scene()
    ps.set_user_callback(vis.callback)
    ps.show()


if __name__ == "__main__":
    main()
