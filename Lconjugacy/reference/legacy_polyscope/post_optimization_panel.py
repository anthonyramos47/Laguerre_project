# Code/gui/post_optimization_panel.py
from LconjugacyGUI.state import AppState
import LconjugacyGUI.computations as comp

import os
import numpy as np
import polyscope as ps
import polyscope.imgui as psim
from pathlib import Path

from hanan.geometry.utils import write_obj

from LMesh.Optimization.Or_contactFree import Or_contactFree
from LMesh.Optimization.Tangential_Dist import Tangential_Dist
from LMesh.Optimization.ContactPointsProximity import ContactPointsProximity
from LMesh.Optimization.ContactPointsGlide import ContactPointsGlide
from LMesh.Optimization.QuadLmeshFairnessPoly import QuadLmeshFairnessPoly
from LMesh.Optimization.QuadLmeshFairnessPrincipal import QuadLmeshFairnessPrincipal
from LMesh.Optimization.QuadLmeshFairness import QuadLmeshFairness
from LMesh.Optimization.QuadLmeshFairnessSpheres import QuadLmeshFairnessSpheres


class PostOptimizationPanel:
    def __init__(self, state: AppState):
        self.state = state
        # Panel-local UI state
        self.postOptimizationOn = False
        self.itLmesh = 0
        self.maxItsLmesh = 40
        self.weights = {
            "cf_proximity": 0.0,
            "cf_gliding": 0.0,
            "tangentialDist": 1e-1,
            "or_contact": 1.0,
            "quadFair": 1e-3,
            "quadFairSph": 1e-3,
            "contactProximity": 1e-4,
            "contactGliding": 1e-4,
            "quadFairPrincipal": 0
        }

        self.finalWeights = {
            "cf_proximity": 0.0,
            "cf_gliding": 0.0,
            "tangentialDist": 0,
            "or_contact": 1.0,
            "quadFair": 0,
            "quadFairSph": 0,
            "contactProximity": 0,
            "contactGliding": 0,
            "quadFairPrincipal": 0,
            "previous_iteration_regularizer": 0,
            "Lap_Fairness_cf":0,
            "Lap_Fairness_rf":0
        }

    def setup_optimization(self):
        """Set up the post-optimization for L-mesh."""
        state = self.state
        assert state.lmesh is not None, "L-mesh not initialized."

        optimizer = state.postProcessingOptimizer

        cf  = np.array([state.lmesh.SF[i].center for i in range(len(state.lmesh.SF))])
        rf  = np.array([state.lmesh.SF[i].radius for i in range(len(state.lmesh.SF))])
        nvp = state.lmesh.top.vertices[:, :3]
        hvp = state.lmesh.top.vertices[:, 3]

        # Declare variables and set initial values
        optimizer.add_variable("cf",  cf.flatten())   # Center of spherical faces
        optimizer.add_variable("rf",  rf)             # Radius of spherical faces
        optimizer.add_variable("nvp", nvp.flatten())  # Normals of vertex planes
        optimizer.add_variable("hvp", hvp)            # Distances of vertex planes

        vertexPlane_adjacentFaces = state.lmesh.top.vertex_face_adjacency_list()

        Or_contact = Or_contactFree()
        Or_contact.name = "or_contact"
        optimizer.add_objective_term(Or_contact, args=([vertexPlane_adjacentFaces]), w=10, ce=1)

        sphericalFace_adjacentFaces = state.lmesh.top.face_face_adjacency_list()

        orPlanesPerFaceOrdered = []
        for f, sfAdjacentList in enumerate(sphericalFace_adjacentFaces):
            if len(sfAdjacentList) == 4:
                s0, _, s2, _ = sfAdjacentList
                vp0, vp1 = state.lmesh.top.shared_edge_vertices(f, s0)
                vp2, vp3 = state.lmesh.top.shared_edge_vertices(f, s2)
                orPlanesPerFaceOrdered.append([vp0, vp1, vp2, vp3])
            else:
                orPlanesPerFaceOrdered.append(state.lmesh.top.vertices[f])

        Quad_Fair = QuadLmeshFairnessPoly()
        Quad_Fair.name = "quadFair"
        optimizer.add_objective_term(Quad_Fair, args=([sphericalFace_adjacentFaces, orPlanesPerFaceOrdered]), w=0.01, ce=0)

        vp_sf_adjList = state.lmesh.top.vertex_face_adjacency_list()
        vp_vp_adjList = state.lmesh.top.vertex_adjacency_list()

        Quad_FairSph = QuadLmeshFairnessSpheres()
        Quad_FairSph.name = "quadFairSph"
        optimizer.add_objective_term(Quad_FairSph, args=([vp_sf_adjList, vp_vp_adjList]), w=0.01, ce=0)

        quadFairnessPrincipal = QuadLmeshFairness()
        quadFairnessPrincipal.name = "quadFairPrincipal"
        optimizer.add_objective_term(quadFairnessPrincipal, args=([sphericalFace_adjacentFaces, orPlanesPerFaceOrdered]), w=0.0001, ce=0)

        sph_sph_edges = state.lmesh.top.edge_faces()

        tangentialDist_term = Tangential_Dist()
        tangentialDist_term.name = "tangentialDist"
        optimizer.add_objective_term(tangentialDist_term, args=([sph_sph_edges]), w=0.0001, ce=1)

        V_ref, F_ref, n_ref, SC_ref = state.reference

        ContactPointsProximity_term = ContactPointsProximity()
        ContactPointsProximity_term.name = "contactProximity"
        optimizer.add_objective_term(ContactPointsProximity_term, args=(state.lmesh.top.faces, V_ref, F_ref), w=1e-4, ce=1)

        ContactPointsGlide_term = ContactPointsGlide()
        ContactPointsGlide_term.name = "contactGliding"
        optimizer.add_objective_term(ContactPointsGlide_term, args=(state.lmesh.top.faces, V_ref, F_ref), w=1e-5, ce=1)

        # Get adjacency for Laplacian smoothing. Neighbor spheres to a given sphere 
        face_adj = state.lmesh.top.face_face_adjacency_list()

        # # Get adjacency for Laplacian smoothing. Neighbor vertex planes to a given vertex plane
        # #vert_adj = self.lmesh.top.vertex_adjacency_list()

        # optimizer.set_lap_smooth("cf", face_adj,  3, 0.0001)
        # si, sj = state.lmesh.top.edge_faces()

        # mask = (si != -1) & (sj != -1)
        # si, sj = si[mask], sj[mask]
        # optimizer.edge_length("cf", [si,sj], target_length=None, w=0.1) 
        optimizer.prev_iteration_regularizer(w=1e-4)
        # optimizer.set_lap_smooth("cf", face_adj, 3, w=1e-3)
        # optimizer.set_lap_smooth("rf", face_adj, 1, w=1e-3)
        optimizer.proximity_reference("cf", SC_ref, F_ref, w_proximity=0.001, w_gliding=0.01)
        optimizer.unitize_variable("nvp", 3, 10)

        # Reset iteration bookkeeping — call after all variables and terms are set up
        optimizer.initialize_optimizer(verbose=True, adaptive_mu=False)

    def draw(self):
        """Draw the post-optimization panel UI elements."""
        state = self.state

        if psim.Button("Load Remeshed Surface"):
            comp.load_pre_remesh_state(state)
            comp.load_remeshed_surfaces(state)
            comp.initialize_lmesh(state)
            state.lmesh.visualize()
            state.lmesh.export_data(state.save_path, 'initial_lmesh')
            self.setup_optimization()

        psim.Text("L-mesh Optimization GUI")

        half_width = (psim.GetContentRegionAvail()[0] - 8) * 0.35

        psim.SetNextItemWidth(half_width)

        _, self.weights["or_contact"] = psim.SliderFloat("Oriented Contact Deviation Weight", self.weights["or_contact"], 0.0, 20.0)
        psim.SetNextItemWidth(half_width)
        _, self.weights["cf_proximity"] = psim.SliderFloat("Proximity Weight", self.weights["cf_proximity"], 0.0, 2.0, format="%.2e")
        psim.SetNextItemWidth(half_width)
        _, self.weights["cf_gliding"] = psim.SliderFloat("Gliding Weight", self.weights["cf_gliding"], 0.0, 2.0 , format="%.2e")
        psim.SetNextItemWidth(half_width)
        _, self.weights["tangentialDist"] = psim.SliderFloat("Tangential Distance Weight", self.weights["tangentialDist"], 0.0, 1.0, format="%.2e")
        psim.SetNextItemWidth(half_width)
        _, self.weights["quadFair"] = psim.SliderFloat("Planes Fairness Weight", self.weights["quadFair"], 0.0, 0.1, format="%.2e")
        psim.SetNextItemWidth(half_width)
        _, self.weights["quadFairSph"] = psim.SliderFloat("Spheres Fairness Weight", self.weights["quadFairSph"], 0.0, 0.1, format="%.2e")
        psim.SetNextItemWidth(half_width)
        _, self.weights["quadFairPrincipal"] = psim.SliderFloat("Principal Curvature Fairness Weight", self.weights["quadFairPrincipal"], 0.0, 0.1, format="%.2e")
        psim.SetNextItemWidth(half_width)
        _, self.weights["contactProximity"] = psim.SliderFloat("Proximity Control Points Weight", self.weights["contactProximity"], 0.0, 1.0, format="%.2e")
        psim.SetNextItemWidth(half_width)
        _, self.weights["contactGliding"] = psim.SliderFloat("Gliding Control Points Weight", self.weights["contactGliding"], 0.0, 1.0, format="%.2e")
        

        _, self.maxItsLmesh = psim.SliderInt("MaxIterations", self.maxItsLmesh, 1, 1000)

        if psim.Button("Run Optimization"):
            optimizer = state.postProcessingOptimizer
            self.postOptimizationOn = True
            optimizer.set_term_weights(self.weights)
            self.itLmesh = 0
        
        if psim.Button("Last Opt"):
            optimizer = state.postProcessingOptimizer
            self.postOptimizationOn = False
            optimizer.set_term_weights(self.finalWeights)
            for _ in range(20):
                optimizer.get_gradients()
                optimizer.optimize_step()
            
            cf_opt = optimizer.unpack("cf").reshape(-1, 3)
            rf_opt = optimizer.unpack("rf")
            nvp_opt = optimizer.unpack("nvp").reshape(-1, 3)
            hvp_opt = optimizer.unpack("hvp")

            state.lmesh.update_spheres_planes(cf_opt, rf_opt, nvp_opt, hvp_opt)
            state.lmesh.visualize()
            optimizer.print_report()
            print("L-mesh optimization finished.")
            

        try:
            psim.Text(f"Oriented Contact Deviation: {state.lmesh.compute_oriented_contact_dev()}")
        except:
            pass

        if psim.Button("Export L-mesh"):
            state.lmesh.export_data(state.save_path, 'lmesh')

            centers = np.array([state.lmesh.SF[f].center for f in range(len(state.lmesh.SF))])
            plane_sf_adj = state.lmesh.top.vertex_face_adjacency_list()

            write_obj(Path(os.path.join(state.save_path, "sphereCenterQuadMesh_t.obj")), state.sphereCenters_t, state.quadRemeshed_t.faces)
            write_obj(Path(os.path.join(state.save_path, "sphereCentersLmesh.obj")), centers, plane_sf_adj)

        psim.Separator()

        if psim.Button("Save All Data"):
            comp.save_all(state)

    def _draw(self):
        """Called every frame. Collapsible section inside the main window."""
        if psim.CollapsingHeader("Post Optimization"):
            self.draw()

    def update(self):
        """Called every frame for optimization stepping."""
        state = self.state
        optimizer = state.postProcessingOptimizer

        if self.postOptimizationOn:
            if self.itLmesh < self.maxItsLmesh:
                optimizer.get_gradients()
                optimizer.optimize_step()

                sphereCenters = optimizer.unpack("cf").reshape(-1, 3)
                _, sphereCenterFaces = state.lmesh.sphere_centers_mesh()

                ps.register_point_cloud("centers", sphereCenters, radius=0.008, color=(0.8, 0.0, 0.1))
                ps.register_surface_mesh("sphere centers Mesg",  sphereCenters, sphereCenterFaces, color=(0.8, 0.8, 0.8), edge_color=(0.0, 0.0, 0.0), edge_width=2.8)

                # cf_opt = optimizer.unpack("cf").reshape(-1, 3)
                # rf_opt = optimizer.unpack("rf")
                # nvp_opt = optimizer.unpack("nvp").reshape(-1, 3)
                # hvp_opt = optimizer.unpack("hvp")
                # state.lmesh.update_spheres_planes(cf_opt, rf_opt, nvp_opt, hvp_opt)
                # state.lmesh.visualize()
                
                self.itLmesh += 1
            else:
                self.postOptimizationOn = False
                self.itLmesh = 0

                cf_opt = optimizer.unpack("cf").reshape(-1, 3)
                rf_opt = optimizer.unpack("rf")
                nvp_opt = optimizer.unpack("nvp").reshape(-1, 3)
                hvp_opt = optimizer.unpack("hvp")

                state.lmesh.update_spheres_planes(cf_opt, rf_opt, nvp_opt, hvp_opt)
                state.lmesh.visualize()
                optimizer.print_report()
                print("L-mesh optimization finished.")
