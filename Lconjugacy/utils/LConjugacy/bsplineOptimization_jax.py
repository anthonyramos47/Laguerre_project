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

sys.path.insert(0, os.path.dirname(__file__))
from Lconjugacy.utils.bsplines_functions import read_bspline_json, \
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
