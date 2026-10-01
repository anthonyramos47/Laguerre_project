"""
B-spline Surface Optimizer — JAX
Minimises  E = w_curv * sum max(0, epsilon - K)^2  +  w_lap * ||Laplacian||^2
to push Gaussian curvature K above a target epsilon while keeping the surface smooth.
"""

import os
import sys
import numpy as np
import jax
import jax.numpy as jnp
import jaxopt
import splipy as sp
import polyscope as ps
import polyscope.imgui as psim

sys.path.insert(0, os.path.dirname(__file__))
from LconjugacyGUI.utils.bsplines_functions import read_bspline_json, \
                                sample_bspline_surface, \
                                bspline_curvatures

jax.config.update("jax_enable_x64", True)


# ─── B-spline basis evaluation ───────────────────────────────────────────────

def _bspline_basis_derivs(basis_sp, t, n_deriv=2):
    """Evaluate basis functions and derivatives using splipy.

    Parameters
    ----------
    basis_sp : splipy.BSplineBasis
    t        : 1-D numpy array of parameter values
    n_deriv  : number of derivatives (default 2 -> N, dN, d2N)

    Returns
    -------
    list of jnp arrays: [N, dN, d2N, ...]  each shape (len(t), n_basis)
    """
    t_np = np.asarray(t, dtype=np.float64)
    results = []
    for d in range(n_deriv + 1):
        M = basis_sp.evaluate(t_np, d=d)
        if hasattr(M, 'toarray'):
            M = M.toarray()
        results.append(jnp.array(M, dtype=jnp.float64))
    return results


# ─── Surface evaluation & curvature ──────────────────────────────────────────

def evaluate_surface(ctrl_pts, basis_u, basis_v):
    tmp = jnp.einsum('ui,ijk->ujk', basis_u, ctrl_pts)
    return jnp.einsum('ujk,vj->uvk', tmp, basis_v)


def curvatures(ctrl_pts, bases_u, bases_v):
    """Return (K, H) — Gaussian and Mean curvature, both flattened."""
    Nu, dNu, d2Nu = bases_u
    Nv, dNv, d2Nv = bases_v

    S_u  = evaluate_surface(ctrl_pts, dNu,  Nv)
    S_v  = evaluate_surface(ctrl_pts, Nu,   dNv)
    S_uu = evaluate_surface(ctrl_pts, d2Nu, Nv)
    S_vv = evaluate_surface(ctrl_pts, Nu,   d2Nv)
    S_uv = evaluate_surface(ctrl_pts, dNu,  dNv)

    E = (S_u * S_u).sum(-1)
    F = (S_u * S_v).sum(-1)
    G = (S_v * S_v).sum(-1)

    n = jnp.cross(S_u, S_v)
    det = E * G - F ** 2
    sqrt_det = jnp.sqrt(jnp.maximum(det, 1e-12))

    L = (S_uu * n).sum(-1) / sqrt_det
    M = (S_uv * n).sum(-1) / sqrt_det
    N = (S_vv * n).sum(-1) / sqrt_det

    K = ((L * N - M ** 2) / jnp.maximum(det, 1e-12)).ravel()
    H = ((E * N + G * L - 2 * F * M) / (2 * jnp.maximum(det, 1e-12))).ravel()
    return K, H


# ─── Optimizer ────────────────────────────────────────────────────────────────

def optimize(bspline, u_vals, v_vals,
             epsilon_K=0.0, epsilon_H=0.0,
             w_curv=1.0, w_mean=0.0, w_lap=1e-2,
             max_iter=100):
    """Optimize B-spline control points so that K > epsilon_K and H > epsilon_H.

    Uses L-BFGS (quasi-Newton) via jaxopt.

    Energy = w_curv * sum max(0, epsilon_K - K_i)^2
           + w_mean * sum max(0, epsilon_H - H_i)^2
           + w_lap  * ||Laplacian(ctrl)||^2

    Parameters
    ----------
    bspline      : splipy Surface
    u_vals, v_vals : 1-D arrays of sample parameters
    epsilon_K    : target minimum Gaussian curvature
    epsilon_H    : target minimum Mean curvature
    w_curv       : Gaussian curvature violation weight
    w_mean       : Mean curvature violation weight
    w_lap        : Laplacian smoothing weight
    max_iter     : number of L-BFGS steps
    """
    # ── Basis setup ───────────────────────────────────────────────────────
    basis_u_sp = bspline.bases[0]
    basis_v_sp = bspline.bases[1]

    bu = _bspline_basis_derivs(basis_u_sp, u_vals, n_deriv=2)
    bv = _bspline_basis_derivs(basis_v_sp, v_vals, n_deriv=2)
    bases_u = (bu[0], bu[1], bu[2])
    bases_v = (bv[0], bv[1], bv[2])

    ctrl = jnp.array(bspline.controlpoints.copy(), dtype=jnp.float64)

    # ── Energy function ───────────────────────────────────────────────────
    def energy(ctrl_pts):
        K, H = curvatures(ctrl_pts, bases_u, bases_v)

        viol_K = jnp.maximum(0.0, epsilon_K - K)
        curv_term = w_curv * (viol_K ** 2).sum()

        viol_H = jnp.maximum(0.0, epsilon_H - H)
        mean_term = w_mean * (viol_H ** 2).sum()

        # Laplacian on the full grid (padded with edge replication)
        pad = jnp.pad(ctrl_pts, ((1, 1), (1, 1), (0, 0)), mode='edge')
        lap = (pad[:-2, 1:-1, :] + pad[2:, 1:-1, :] +
               pad[1:-1, :-2, :] + pad[1:-1, 2:, :] -
               4.0 * pad[1:-1, 1:-1, :])
        lap_term = w_lap * (lap ** 2).sum()

        return curv_term + mean_term + lap_term

    # ── Flat energy for L-BFGS (needs 1-D input) ───────────────────────
    cp_shape = ctrl.shape
    n_vars = ctrl.size

    def energy_flat(x):
        return energy(x.reshape(cp_shape))

    energy_fn = jax.jit(energy_flat)

    # ── Initial report ────────────────────────────────────────────────────
    K0, H0 = [np.asarray(x) for x in curvatures(ctrl, bases_u, bases_v)]
    n_samples = len(K0)
    print(f"Initial K: min={K0.min():.4f}  max={K0.max():.4f}  "
          f"violations(K>{epsilon_K})={np.sum(K0 < epsilon_K)}/{n_samples}")
    print(f"Initial H: min={H0.min():.4f}  max={H0.max():.4f}  "
          f"violations(H>{epsilon_H})={np.sum(H0 < epsilon_H)}/{n_samples}")
    print(f"Variables: {n_vars}")

    # ── L-BFGS optimisation ───────────────────────────────────────────────
    solver = jaxopt.LBFGS(fun=energy_flat, maxiter=max_iter,
                          history_size=20, tol=1e-10)
    x0 = ctrl.ravel()

    # Callback-style logging via manual run_iterator
    state = solver.init_state(x0)
    x = x0
    for it in range(1, max_iter + 1):
        x, state = solver.update(x, state)

        if it % 10 == 0 or it == 1:
            ctrl_cur = x.reshape(cp_shape)
            K_cur, H_cur = [np.asarray(v) for v in curvatures(ctrl_cur, bases_u, bases_v)]
            n_viol_K = int(np.sum(K_cur < epsilon_K))
            n_viol_H = int(np.sum(H_cur < epsilon_H))
            E_val = float(energy_fn(x))
            print(f"  iter {it:4d}  E={E_val:.6f}  "
                  f"K=[{K_cur.min():.4f}, {K_cur.max():.4f}] viol={n_viol_K}  "
                  f"H=[{H_cur.min():.4f}, {H_cur.max():.4f}] viol={n_viol_H}")

            if n_viol_K == 0 and n_viol_H == 0:
                print(f"\n  -> All points satisfy K > {epsilon_K} and H > {epsilon_H}")
                break

            if E_val < 1e-12:
                print("\n  -> Energy ~0, converged.")
                break

    ctrl = x.reshape(cp_shape)

    # ── Final report ──────────────────────────────────────────────────────
    K_final, H_final = [np.asarray(v) for v in curvatures(ctrl, bases_u, bases_v)]
    print(f"\nFinal K: min={K_final.min():.4f}  max={K_final.max():.4f}  "
          f"violations={np.sum(K_final < epsilon_K)}/{n_samples}")
    print(f"Final H: min={H_final.min():.4f}  max={H_final.max():.4f}  "
          f"violations={np.sum(H_final < epsilon_H)}/{n_samples}")

    opt_cp = np.asarray(ctrl)
    opt_bspline = sp.Surface(basis_u_sp, basis_v_sp, opt_cp, raw=True)
    return opt_bspline


# ─── Main (interactive GUI) ───────────────────────────────────────────────────

def main():
    surface_path = "/path/to/Laguerre_project/Code/LConjugacy/surfaces"
    surface_name = "helixDef2"

    bspline_file = os.path.join(surface_path, surface_name + ".json")
    print(f"Loading: {bspline_file}")
    bspline = read_bspline_json(bspline_file)

    # ── GUI state ─────────────────────────────────────────────────────────
    gui = dict(
        # Sampling
        u_min=0.0, u_max=1.0,
        v_min=0.0, v_max=1.0,
        n_u=80, n_v=80,
        # Weights & targets
        epsilon_K=0.5, epsilon_H=0.0,
        w_curv=1.0, w_mean=0.0, w_lap=1e-2,
        # L-BFGS
        max_iter=100,
        # State
        original_bspline=bspline,
        optimized_bspline=bspline.clone(),
        status="Ready",
    )

    # ── Cached sample arrays (recomputed only when range/resolution changes)
    def _build_sample_arrays():
        gui["u_sample"] = np.linspace(gui["u_min"], gui["u_max"], gui["n_u"])
        gui["v_sample"] = np.linspace(gui["v_min"], gui["v_max"], gui["n_v"])

    _build_sample_arrays()

    # ── Helper: register surface with curvatures ────────────────────────
    def show_surface(name, bsp, u_pts, v_pts, enabled_K=True):
        V, F = sample_bspline_surface(bsp, u_pts, v_pts)
        Kv, Hv, _ = bspline_curvatures(bsp, u_pts, v_pts)
        mesh = ps.register_surface_mesh(name, V, F)
        mesh.add_scalar_quantity("K", Kv.flatten(),
                                defined_on='vertices', enabled=enabled_K)
        mesh.add_scalar_quantity("H", Hv.flatten(),
                                defined_on='vertices')
        return Kv, Hv

    # ── Initial visualisation ─────────────────────────────────────────────
    ps.init()
    show_surface("Original Surface", bspline, gui["u_sample"], gui["v_sample"])

    # ── Callback ──────────────────────────────────────────────────────────
    def callback():
        psim.Begin("Curvature Optimizer", True)

        # ── Sampling range ────────────────────────────────────────────
        psim.Text("Sampling Range")
        c1, gui["u_min"] = psim.SliderFloat("u min", gui["u_min"], 0.0, 1.0)
        c2, gui["u_max"] = psim.SliderFloat("u max", gui["u_max"], 0.0, 1.0)
        c3, gui["v_min"] = psim.SliderFloat("v min", gui["v_min"], 0.0, 1.0)
        c4, gui["v_max"] = psim.SliderFloat("v max", gui["v_max"], 0.0, 1.0)

        psim.Separator()
        psim.Text("Samples per direction")
        c5, gui["n_u"] = psim.SliderInt("n_u", gui["n_u"], 10, 200)
        c6, gui["n_v"] = psim.SliderInt("n_v", gui["n_v"], 10, 200)

        # Rebuild cached arrays only when sliders change
        if any([c1, c2, c3, c4, c5, c6]):
            _build_sample_arrays()
            show_surface("Original Surface", bspline,
                         gui["u_sample"], gui["v_sample"])
            if gui["optimized_bspline"] is not None:
                show_surface("Optimized Surface",
                             gui["optimized_bspline"],
                             gui["u_sample"], gui["v_sample"])

        # ── Curvature targets ─────────────────────────────────────────
        psim.Separator()
        psim.Text("Curvature Targets")
        _, gui["epsilon_K"] = psim.InputFloat("epsilon K", gui["epsilon_K"])
        _, gui["epsilon_H"] = psim.InputFloat("epsilon H", gui["epsilon_H"])

        # ── Weights ───────────────────────────────────────────────────
        psim.Separator()
        psim.Text("Weights")
        _, gui["w_curv"] = psim.SliderFloat("w_curv (K)", gui["w_curv"],
                                            0.0, 10.0)
        _, gui["w_mean"] = psim.SliderFloat("w_mean (H)", gui["w_mean"],
                                            0.0, 10.0)
        _, gui["w_lap"]  = psim.SliderFloat("w_lap",  gui["w_lap"],
                                            0.0, 1.0)

        # ── L-BFGS iterations ─────────────────────────────────────────
        psim.Separator()
        _, gui["max_iter"] = psim.SliderInt("L-BFGS iters",
                                           gui["max_iter"], 100, 5000)

        # ── Run button ────────────────────────────────────────────────
        psim.Separator()
        if psim.Button("Run Optimization"):
            gui["status"] = "Running..."

            opt_bsp = optimize(
                gui["optimized_bspline"], gui["u_sample"], gui["v_sample"],
                epsilon_K=gui["epsilon_K"],
                epsilon_H=gui["epsilon_H"],
                w_curv=gui["w_curv"],
                w_mean=gui["w_mean"],
                w_lap=gui["w_lap"],
                max_iter=gui["max_iter"],
            )
            gui["optimized_bspline"] = opt_bsp

            Kopt, Hopt = show_surface("Optimized Surface", opt_bsp,
                                      gui["u_sample"], gui["v_sample"])

            gui["status"] = (
                f"Done  K=[{Kopt.min():.3f}, {Kopt.max():.3f}]  "
                f"H=[{Hopt.min():.3f}, {Hopt.max():.3f}]"
            )

        # ── Reset button ──────────────────────────────────────────────
        if psim.Button("Reset to Original"):
            gui["optimized_bspline"] = gui["original_bspline"].clone() 
            gui["status"] = "Reset"
            ps.remove_surface_mesh("Optimized Surface", error_if_absent=False)

        psim.Separator()
        psim.Text(gui["status"])
        psim.End()

    ps.set_user_callback(callback)
    ps.show()


if __name__ == "__main__":
    main()
