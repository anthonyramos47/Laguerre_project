"""
LconjugacyApp — web backend for the L-conjugacy direction field + L-mesh GUI.

Register in server/main.py:
    from Code.LconjugacyGUI.web_app import LconjugacyApp
    from kayviz.webscope import register_app
    register_app(LconjugacyApp())

Navigate to: http://localhost:8000?app=lconjugacy&mesh=<surface_name>
"""

from __future__ import annotations
import os
import numpy as np

from fastapi import WebSocket

from kayviz.webscope import GUIApp
from kayviz import serializers as ser

from LconjugacyGUI.state import AppState
import LconjugacyGUI.computations as comp

from LconjugacyGUI.utils.bsplines_functions import (
    read_bspline_json,
    sample_bspline_surface,
    principal_frame,
    compute_rij_transport_vectorized,
    init_lconjugate_directions,
    sample_radius,
    exchange_lconjugate_rulings_curve_tangents,
)
from hanan.geometry.mesh import Mesh
from hanan.geometry.utils import unit

from LconjugacyGUI.utils.directionsOpt.DirectionFieldFairness import DirectionFieldSmoothness
from LconjugacyGUI.utils.directionsOpt.LconjugateOpt import Lconjugacy
from LconjugacyGUI.utils.directionsOpt.angleLconjugates import AngleLconjugate
from LconjugacyGUI.utils.directionsOpt.angleBoundary import AngleBoundary


class LconjugacyApp(GUIApp):
    name      = "lconjugacy"
    state_cls = AppState
    out_dir   = os.path.join("Code", "LconjugacyGUI", "out")
    panel_js  = os.path.join(os.path.dirname(__file__), "lconjugacy_panel.js")

    _SURFACE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "bsplines")

    # ── Initial scene ─────────────────────────────────────────────────────────

    def initial_scene(self, mesh_name: str = "ex1") -> dict:
        surface_path = os.path.join(self._SURFACE_DIR, mesh_name + ".json")
        if not os.path.exists(surface_path):
            return {"error": f"Surface not found: {surface_path}"}

        state = self.state
        state.name         = mesh_name
        state.surface_path = surface_path
        state.output_path  = self.out_dir
        state.save_path    = os.path.join(self.out_dir, mesh_name,
                                          state.experiment_name)

        state.bspline = read_bspline_json(surface_path)
        state.bsplineOriginal = state.bspline

        state.u_sample_pts = np.linspace(
            state.interval_u[0], state.interval_u[1], state.bspline_u_sample_num
        )
        state.v_sample_pts = np.linspace(
            state.interval_v[0], state.interval_v[1], state.bspline_v_sample_num
        )

        # Sync state fields from bspline config
        state.percentageRadius   = state.bspline_percentage_r
        state.radiusFunctionType = state.bspline_radius_type
        state.offsetValue        = state.bspline_offset_value

        objects = comp.bspline_scene(state)
        return {
            "surface_name":    state.name,
            "experiment_name": state.experiment_name,
            "objects":         objects,
        }

    # ── Action registration ───────────────────────────────────────────────────

    def register_actions(self):
        # BSpline panel
        self.action("resample_bspline",       self._resample_bspline)
        self.action("compute_reference",      self._compute_reference)
        self.action("visualize_reference",    self._visualize_reference)
        self.action("load_pre_remesh_state",  self._load_pre_remesh)

        # LConjugacy panel
        self.action("init_direction_field",   self._init_df)
        self.action("setup_df_optimizer",     self._setup_df_opt)
        self.action("run_df_optimizer",       self._run_df_opt)
        self.action("stop_df_optimizer",      self._stop)
        self.action("visualize_cross_field",  self._viz_cross_field)
        self.action("save_cross_field",       self._save_cross_field)

        # Post-optimization / remeshing
        self.action("save_pre_remesh_state",  self._save_pre_remesh)
        self.action("run_remesher",           self._run_remesher)
        self.action("load_remeshed",          self._load_remeshed)
        self.action("init_lmesh",             self._init_lmesh)

        # Integration / L-mesh optimization
        self.action("run_post_optimizer",     self._run_post_opt)
        self.action("stop_post_optimizer",    self._stop)
        self.action("show_lmesh",             self._show_lmesh)

        # Save / export
        self.action("save_all",               self._save_all)

    # ── BSpline panel actions ─────────────────────────────────────────────────

    def _resample_bspline(self, state: AppState, data: dict) -> dict:
        state.u_sample_pts = np.linspace(
            state.interval_u[0], state.interval_u[1], state.bspline_u_sample_num
        )
        state.v_sample_pts = np.linspace(
            state.interval_v[0], state.interval_v[1], state.bspline_v_sample_num
        )
        state.percentageRadius   = state.bspline_percentage_r
        state.radiusFunctionType = state.bspline_radius_type
        state.offsetValue        = state.bspline_offset_value
        objects = comp.bspline_scene(state)
        return {"action": "scene_update", "objects": objects}

    def _compute_reference(self, state: AppState, data: dict) -> dict:
        comp.compute_reference_structures(state)
        objects = comp.reference_scene(state)
        return {"action": "scene_update", "objects": objects}

    def _visualize_reference(self, state: AppState, data: dict) -> dict:
        objects = comp.reference_scene(state)
        return {"action": "scene_update", "objects": objects}

    def _load_pre_remesh(self, state: AppState, data: dict) -> dict:
        objects = comp.load_pre_remesh_state(state)
        return {"action": "scene_update", "objects": objects}

    # ── Direction field panel actions ─────────────────────────────────────────

    def _init_df(self, state: AppState, data: dict) -> dict:
        """Initialize direction field angles uniformly."""
        method = data.get("method", "constant")
        if method == "constant":
            n_pts = len(state.u_sample_pts) * len(state.v_sample_pts)
            angle = np.deg2rad(state.lconj_theta_initial)
            state.theta_angles = np.full(n_pts, angle)
            state.phi_angles   = np.full(n_pts, angle)
        elif method == "linear":
            state.theta_angles = comp.linear_angle_distribution(state)
            state.phi_angles   = comp.linear_angle_distribution(state)
        elif method == "linearU":
            state.theta_angles = comp.linear_angle_distribution_dir_u(state)
            state.phi_angles   = comp.linear_angle_distribution_dir_u(state)
        elif method == "linearV":
            state.theta_angles = comp.linear_angle_distribution_dir_v(state)
            state.phi_angles   = comp.linear_angle_distribution_dir_v(state)

        objects = comp.cross_field_scene(state)
        return {"action": "scene_update", "objects": objects}

    def _setup_df_opt(self, state: AppState, data: dict) -> dict:
        """Set up the direction field Lconjugacy optimizer."""
        from hanan.optimization.optimizer import Optimizer

        opt = Optimizer()
        radiusFunctionCtrlPts = (state.radiusFunction.controlpoints.reshape(-1, 3)[:, 2])
        beta = state.phi_angles - state.theta_angles

        opt.add_variable("th",  state.theta_angles)
        opt.add_variable("bet", beta)
        opt.add_variable("rij", radiusFunctionCtrlPts + 0.1)

        e1, _, k1, k2, _ = principal_frame(state.bspline,
                                             state.u_sample_pts, state.v_sample_pts)
        V, F = sample_bspline_surface(state.bspline, state.u_sample_pts, state.v_sample_pts)
        mesh = Mesh()
        mesh.make_mesh(V, F)
        ei, ej = mesh.edge_vertices()
        n = state.bspline.normal(state.u_sample_pts, state.v_sample_pts).reshape(-1, 3)
        alpha_ij = compute_rij_transport_vectorized(
            e1[ei].reshape(-1, 3), e1[ej].reshape(-1, 3),
            n[ei].reshape(-1, 3), n[ej].reshape(-1, 3),
        )

        term_lconj = Lconjugacy()
        opt.add_objective_term(
            term_lconj,
            args=(state.bspline, state.u_sample_pts, state.v_sample_pts),
            w=state.lconj_w_lconjugacy, ce=True,
        )

        term_smooth = DirectionFieldSmoothness()
        opt.add_objective_term(
            term_smooth,
            args=(ei, ej, alpha_ij),
            w=state.lconj_w_smooth, ce=True,
        )

        term_angle = AngleLconjugate()
        opt.add_objective_term(
            term_angle,
            args=(state.bspline, state.u_sample_pts, state.v_sample_pts),
            w=0.0, ce=False,
        )

        inner_v = mesh.inner_vertices()
        bound_v = mesh.boundary_vertices()
        term_boundary = AngleBoundary()
        opt.add_objective_term(
            term_boundary,
            args=(bound_v,),
            w=state.lconj_w_angle_boundary, ce=True,
        )

        opt.initialize_optimizer(verbose=True)
        state.directionFieldOptimizer = opt
        return self._done("df_optimizer_ready")

    async def _run_df_opt(self, state: AppState, data: dict, ws: WebSocket) -> None:
        """Stream direction field optimization."""
        state.is_running = True
        opt = state.directionFieldOptimizer
        max_its = int(data.get("max_its", 40))

        for _ in range(max_its):
            if not state.is_running:
                break
            opt.get_gradients()
            opt.optimize_step()

            state.theta_angles = opt.unpack("th")
            state.phi_angles   = state.theta_angles + opt.unpack("bet")

            objects = comp.cross_field_scene(state)
            await ws.send_json({
                "action":    "scene_update",
                "iteration": opt.it,
                "energy":    float(opt.energy[-1]) if opt.energy else None,
                "objects":   objects,
            })

        state.is_running = False
        await ws.send_json(self._done("df_optimization_complete"))

    def _viz_cross_field(self, state: AppState, data: dict) -> dict:
        objects = comp.cross_field_scene(state)
        return {"action": "scene_update", "objects": objects}

    def _save_cross_field(self, state: AppState, data: dict) -> dict:
        os.makedirs(state.save_path, exist_ok=True)
        comp.save_cross_field(state)
        return self._done("cross_field_saved")

    # ── Remeshing panel actions ───────────────────────────────────────────────

    def _save_pre_remesh(self, state: AppState, data: dict) -> dict:
        os.makedirs(state.save_path, exist_ok=True)
        comp.save_pre_remesh_state(state)
        return self._done("pre_remesh_saved")

    def _run_remesher(self, state: AppState, data: dict) -> dict:
        integrated = bool(data.get("integrated", True))
        comp.run_remesher(state, integrated=integrated)
        return self._done("remesher_done")

    def _load_remeshed(self, state: AppState, data: dict) -> dict:
        objects = comp.load_remeshed_surfaces(state)
        return {"action": "scene_update", "objects": objects}

    def _init_lmesh(self, state: AppState, data: dict) -> dict:
        objects = comp.initialize_lmesh(state)
        return {"action": "scene_update", "objects": objects}

    # ── Post-optimization panel actions ──────────────────────────────────────

    async def _run_post_opt(self, state: AppState, data: dict, ws: WebSocket) -> None:
        """Stream post-processing L-mesh optimization."""
        state.is_running = True
        opt = state.postProcessingOptimizer
        max_its = int(data.get("max_its", 50))

        for _ in range(max_its):
            if not state.is_running:
                break
            opt.get_gradients()
            opt.optimize_step()

            objects = comp.lmesh_scene(state)
            await ws.send_json({
                "action":    "scene_update",
                "iteration": opt.it,
                "energy":    float(opt.energy[-1]) if opt.energy else None,
                "objects":   objects,
            })

        state.is_running = False
        await ws.send_json(self._done("post_optimization_complete"))

    def _show_lmesh(self, state: AppState, data: dict) -> dict:
        objects = comp.lmesh_scene(state)
        return {"action": "scene_update", "objects": objects}

    # ── Common actions ────────────────────────────────────────────────────────

    def _stop(self, state: AppState, data: dict) -> dict:
        state.is_running = False
        return self._done("stopped")

    def _save_all(self, state: AppState, data: dict) -> dict:
        os.makedirs(state.save_path, exist_ok=True)
        comp.save_all(state)
        return self._done("save_done")


# ── Standalone: PYTHONPATH=Code python -m LconjugacyGUI.web_app ──────────────
if __name__ == "__main__":
    import kayviz as kv
    kv.register_app(LconjugacyApp())
    kv.show()
