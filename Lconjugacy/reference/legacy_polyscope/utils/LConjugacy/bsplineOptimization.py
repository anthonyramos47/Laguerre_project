"""
B-spline Surface Optimizer for Positive Gaussian Curvature (K > epsilon).

Uses scipy.optimize.minimize with inequality constraints to ensure K > epsilon
at sampled points. Objective: Laplacian smoothing of control points.

Assumes a splipy Surface object as input.
"""

import os
import sys
import numpy as np
import splipy as sp
from scipy.optimize import minimize
import polyscope as ps

sys.path.insert(0, os.path.dirname(__file__))
from bsplines_functions import read_bspline_json, sample_grid, bspline_to_mesh


def bspline_gaussian_curvature(bsp, u_pts, v_pts):
    """Compute Gaussian curvature K at grid (u_pts x v_pts).

    Returns:
        K: (len(u_pts)*len(v_pts),) flattened Gaussian curvature values
    """
    du  = bsp.derivative(u_pts, v_pts, d=(1, 0))
    dv  = bsp.derivative(u_pts, v_pts, d=(0, 1))
    duu = bsp.derivative(u_pts, v_pts, d=(2, 0))
    dvv = bsp.derivative(u_pts, v_pts, d=(0, 2))
    duv = bsp.derivative(u_pts, v_pts, d=(1, 1))

    E = np.sum(du * du, axis=2)
    F = np.sum(du * dv, axis=2)
    G = np.sum(dv * dv, axis=2)

    n   = np.cross(du, dv)
    det = E * G - F**2

    L = np.sum(duu * n, axis=2) / np.sqrt(det)
    M = np.sum(duv * n, axis=2) / np.sqrt(det)
    N = np.sum(dvv * n, axis=2) / np.sqrt(det)

    K = (L * N - M**2) / (E * G - F**2)
    return K.flatten()


def laplacian_energy(ctrl_pts):
    """Laplacian smoothing energy on control point grid.

    Lap(P_{i,j}) = P_{i-1,j} + P_{i+1,j} + P_{i,j-1} + P_{i,j+1} - 4*P_{i,j}

    Returns:
        scalar energy = sum ||Lap(P)||^2 over interior points
    """
    lap = (ctrl_pts[:-2, 1:-1, :] + ctrl_pts[2:, 1:-1, :] +
           ctrl_pts[1:-1, :-2, :] + ctrl_pts[1:-1, 2:, :] -
           4.0 * ctrl_pts[1:-1, 1:-1, :])
    return np.sum(lap**2)


def optimize_gaussian_curvature(bspline, u_vals, v_vals, epsilon=0.2, w_lap=1e-3):
    """Optimize control points so that K > epsilon everywhere.

    Objective:  w_lap * laplacian_energy(control_points)
    Constraint: K(u,v) - dummy^2 - epsilon >= 0  (ensures K > epsilon)

    Args:
        bspline: splipy Surface object
        u_vals:  (n,) sample parameter values in u
        v_vals:  (m,) sample parameter values in v
        epsilon: minimum Gaussian curvature target
        w_lap:   weight for Laplacian regularizer

    Returns:
        optimized_bspline: new splipy Surface with optimized control points
        result: scipy OptimizeResult
    """
    basis_u = bspline.bases[0]
    basis_v = bspline.bases[1]
    cp_shape = bspline.controlpoints.shape
    cp_flat  = bspline.controlpoints.flatten()
    n_cp     = len(cp_flat)

    # Compute initial K to size the dummy variables
    K0 = bspline_gaussian_curvature(bspline, u_vals, v_vals)
    n_samples = len(K0)

    print(f"Initial K: min={K0.min():.4f}, max={K0.max():.4f}, "
          f"violating={np.sum(K0 < epsilon)}/{n_samples}")

    # x = [control_points_flat, dummy_vars]
    dummy0 = 0.1 * np.ones(n_samples)
    x0 = np.hstack((cp_flat, dummy0))

    def objective(x):
        cp = x[:n_cp].reshape(cp_shape)
        return w_lap * laplacian_energy(cp)

    def constraint(x):
        cp    = x[:n_cp].reshape(cp_shape)
        dummy = x[n_cp:]

        bsp_new = sp.Surface(basis_u, basis_v, cp)
        K = bspline_gaussian_curvature(bsp_new, u_vals, v_vals)

        # K - 0.2 > 0 <=>  K - 0.2 - |K - 0.2| >= 0
        return K - 0.2 - np.abs(K - 0.2)

    constraints = {'type': 'ineq', 'fun': constraint}

    print("Running optimization...")
    result = minimize(objective, x0, constraints=constraints,
                      method='SLSQP',
                      options={'maxiter': 500, 'disp': True, 'ftol': 1e-9})

    if result.success:
        print("Optimization successful.")
    else:
        print(f"Optimization finished: {result.message}")

    optimized_cp = result.x[:n_cp].reshape(cp_shape)
    optimized_bspline = sp.Surface(basis_u, basis_v, optimized_cp)

    K_final = bspline_gaussian_curvature(optimized_bspline, u_vals, v_vals)
    print(f"Final K: min={K_final.min():.4f}, max={K_final.max():.4f}, "
          f"violating={np.sum(K_final < epsilon)}/{n_samples}")

    return optimized_bspline, result


def main():
    # ---- Fill these ----
    surface_path = "/path/to/Laguerre_project/Code/LConjugacy/surfaces"   # e.g. "/path/to/Laguerre_project/Code/LConjugacy/surfaces"
    surface_name = "helixDef"   # e.g. "liliumNurb"
    # --------------------

    bspline_file = os.path.join(surface_path, surface_name + ".json")
    print(f"Loading: {bspline_file}")

    bspline = read_bspline_json(bspline_file)

    

    sample = 40
    oV, oF, _ = bspline_to_mesh(bspline, sample, sample)
    u_vals, v_vals = sample_grid(sample, sample,
                                 delta_u_Min=0.01, delta_u_Max=0.01,
                                 delta_v_Min=0.01, delta_v_Max=0.01)

    optimized_bspline, result = optimize_gaussian_curvature(
        bspline, u_vals, v_vals, epsilon=0.2, w_lap=1e-3
    )

    V, F, _ = bspline_to_mesh(optimized_bspline, sample, sample)

    ps.init()

    ps.register_surface_mesh("Optimized B-spline Surface", V, F)
    ps.register_surface_mesh("Original B-spline Surface", oV, oF)
    ps.show()


if __name__ == "__main__":
    main()
