# Code/gui/local_analysis_panel.py
from LconjugacyGUI.state import AppState

import numpy as np
import polyscope as ps
import polyscope.imgui as psim

from LconjugacyGUI.utils.bsplines_functions import draw_cone_contact

from hanan.geometry.utils import (
    unit,
    draw_polysphere,
    draw_line_pp,
    draw_polygon,
    save_discrete_laguerre_channel_surface_spheres,
)


class LocalAnalysisPanel:
    def __init__(self, state: AppState):
        self.state = state

    def handle_input(self):
        """Mouse and keyboard callback for interaction."""
        state = self.state
        io = psim.GetIO()

        if io.MouseClicked[1]:  # Right mouse button
            screen_pos = io.MousePos
            pick_result = ps.pick(screen_coords=screen_pos)

            if (pick_result.is_hit
                    and "element_type" in pick_result.structure_data
                    and pick_result.structure_data["element_type"] == "vertex"):
                vertex_id = pick_result.structure_data["index"]
                if vertex_id in range(len(state.quadRemeshed_t.vertices)):
                    state.selected_vertex = vertex_id
                    state.vertex_neighbors = state.quadRemeshed_t.vertex_adjacency_list()[state.selected_vertex]

        if psim.IsKeyPressed(psim.ImGuiKey_F):
            state.selectFaceMode = not state.selectFaceMode

        if state.selectFaceMode and io.MouseClicked[0]:  # Left mouse button
            screen_pos = io.MousePos
            pick_result = ps.pick(screen_coords=screen_pos)

            if (pick_result.is_hit
                    and "element_type" in pick_result.structure_data
                    and pick_result.structure_data["element_type"] == "face"):
                face_id = pick_result.structure_data["index"]
                if face_id in range(len(state.quadRemeshed_t.faces)):
                    state.selected_face = face_id

    def draw(self):
        """Draw the local analysis panel UI elements."""
        state = self.state

        psim.Text("L-mesh Local Analysis GUI")

        if state.selected_vertex is not None:
            psim.Text(f"Selected vertex: {state.selected_vertex}")
            ps.register_point_cloud("selected_vertex", np.array([state.quadRemeshed_t.vertices[state.selected_vertex]]), radius=0.005, color=[1, 0, 0])
            state.selected_neighbor = None
        else:
            psim.Text("No vertex selected")

        if state.vertex_neighbors is not None:
            psim.Text(f"Neighbors :{[int(n) for n in state.vertex_neighbors]}")
            for i, n in enumerate(state.vertex_neighbors):
                ps.register_point_cloud(f"neighbor_{i}", np.array([state.quadRemeshed_t.vertices[n]]), radius=0.004, color=[0, 0, 0])

        try:
            for i, neighbor in enumerate(state.vertex_neighbors):
                if psim.Button(f"Select Neighbor {neighbor}"):
                    state.selected_neighbor = neighbor
                    psim.Text(f"Selected Neighbor: {state.selected_neighbor}")
        except:
            pass

        psim.Separator()

        if state.selected_neighbor is not None:
            psim.Text(f"Selected Neighbor: {state.selected_neighbor}")
            ps.register_point_cloud("selected_neighbor", np.array([state.quadRemeshed_t.vertices[state.selected_neighbor]]), radius=0.005, color=[0, 1, 0])

        if state.selected_neighbor is not None and state.selected_vertex is not None:
            ci = state.sphereCenters_t[state.selected_vertex]
            ri = np.linalg.norm(state.quadRemeshed_t.vertices[state.selected_vertex] - ci)
            cj = state.sphereCenters_t[state.selected_neighbor]
            rj = np.linalg.norm(state.quadRemeshed_t.vertices[state.selected_neighbor] - cj)

            p = state.quadRemeshed_t.vertices[state.selected_vertex]

            draw_line_pp(p, p + 0.1 * unit(cj - ci), name="Dir centers", color=[1, 1, 0])
            draw_cone_contact(ci, ri, cj, rj, n=200, name="cone_strip" + str(state.selected_vertex) + "_" + str(state.selected_neighbor))

            ps.register_point_cloud("center_i", np.array([ci]), radius=0.005, color=[1, 0, 0])
            ps.register_point_cloud("center_j", np.array([cj]), radius=0.005, color=[0, 1, 0])
            draw_polysphere(ci, ri, sub_lvl=4, name="sphere_i", color=[0.1, 0.1, 0.1], transparency=0.6)

        if state.selected_face is not None:
            self._draw_face_analysis(state)

    def _draw_face_analysis(self, state):
        """Handle face selection analysis and visualization."""
        spherical_Face = state.lmesh.SF[state.selected_face]
        quadMeshBarycenters = state.quadRemeshed_t.face_barycenters()
        baryCentralFace = quadMeshBarycenters[state.selected_face]

        center = spherical_Face.center
        radius = spherical_Face.radius

        draw_polysphere(center, radius, sub_lvl=5, name="spherical_face_selected", color=[0.8, 0.1, 0.1], transparency=0.5)

        neighbor_faces = state.lmesh.top.face_face_adjacency_list()[state.selected_face]

        centers = [center]
        radii = [radius]
        contactPoints = [baryCentralFace]

        ps.register_point_cloud("selectedContactPoint", np.array([baryCentralFace]), radius=0.005, color=[0, 0, 0])
        ps.register_point_cloud("selectedCenter", np.array([center]), radius=0.005, color=[0, 0, 0])
        ps.register_curve_network("lineSelected", np.array([baryCentralFace, center]), np.array([[0, 1]]), radius=0.002, color=[0, 0, 0])

        u_faces, v_faces = state.quadRemeshed_t.quad_face_strips(state.selected_face)

        u_contactPoints = []
        u_sphereCenters = []
        u_radius = []

        for uf in u_faces:
            pts = state.quadRemeshed_t.vertices[state.quadRemeshed_t.faces[uf]]
            draw_polygon(pts, name="u_strip_face_" + str(uf), color=[1, 0, 0])
            u_contactPoints.append(quadMeshBarycenters[uf])
            u_sphereCenters.append(state.lmesh.SF[uf].center)
            u_radius.append(state.lmesh.SF[uf].radius)

        save_discrete_laguerre_channel_surface_spheres("discreteUChannel", np.array(u_sphereCenters), np.array(u_radius), np.array(u_contactPoints))

        v_contactPoints = []
        v_sphereCenters = []
        v_radius = []

        for vf in v_faces:
            pts = state.quadRemeshed_t.vertices[state.quadRemeshed_t.faces[vf]]
            draw_polygon(pts, name="v_strip_face_" + str(vf), color=[0, 1, 0])
            v_contactPoints.append(quadMeshBarycenters[vf])
            v_sphereCenters.append(state.lmesh.SF[vf].center)
            v_radius.append(state.lmesh.SF[vf].radius)

        save_discrete_laguerre_channel_surface_spheres("discreteVChannel", np.array(v_sphereCenters), np.array(v_radius), np.array(v_contactPoints))

        for i in range(len(u_faces) - 1):
            s1 = state.lmesh.SF[u_faces[i]]
            s2 = state.lmesh.SF[u_faces[i + 1]]
            draw_cone_contact(s1.center, s1.radius, s2.center, s2.radius, n=300, name="cone_stripUface_" + str(i))

        for i in range(len(v_faces) - 1):
            s1 = state.lmesh.SF[v_faces[i]]
            s2 = state.lmesh.SF[v_faces[i + 1]]
            draw_cone_contact(s1.center, s1.radius, s2.center, s2.radius, n=300, name="cone_stripVface_" + str(i))

        for i, nf in enumerate(neighbor_faces):
            spherical_NeighborFace = state.lmesh.SF[nf]
            centerN = spherical_NeighborFace.center
            radiusN = spherical_NeighborFace.radius

            color = [0, 0, 1] if i % 2 == 0 else [0, 1, 1]

            ps.register_point_cloud("contactPoint" + str(i), np.array([quadMeshBarycenters[nf]]), radius=0.005, color=color)
            ps.register_point_cloud("center_face_" + str(i), np.array([centerN]), radius=0.005, color=color)
            ps.register_curve_network("line_center_face_" + str(i), np.array([quadMeshBarycenters[nf], centerN]), np.array([[0, 1]]), radius=0.002, color=color)

            centers.append(centerN)
            radii.append(radiusN)
            contactPoints.append(quadMeshBarycenters[nf])

            draw_cone_contact(center, radius, centerN, radiusN, n=100, name="cone_strip_face_" + str(i))

        state.selected_face = None

    def _draw(self):
        """Called every frame. Collapsible section inside the main window."""
        if psim.CollapsingHeader("L-mesh Local Analysis"):
            self.draw()

    def update(self):
        """Called every frame. Local analysis is event-driven."""
        pass
