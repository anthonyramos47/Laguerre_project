from hanan.optimization.objective_term import ObjectiveTerm
import numpy as np

from ..bsplines_functions import bspline_curvatures

class GaussCurvatureBspline(ObjectiveTerm):

    def __init__(self) -> None:
        """ Gaussian curvature constraint for B-spline surfaces
        Energy for Gaussian curvature:
        E_{G} = \sum_{p in surface} ( K(p) - K_target )^2
        where K_target is the target Gaussian curvature
        """
        super().__init__()
        self.name = "GaussCurvatureBspline" # Name of the constraint
        self.Kprev = None
        self.jacobianMethod = "FD" # Options: "jax" (fastest), "FD_parallel", "FD", "analytical"

    def initialize_objective(self, X, var_idx, u_pts, v_pts, bspline, K_target) -> None:
        """
        Initialize the Gaussian curvature objective term.
        Input:
            X          : Variables
            var_idx    : dictionary of indices of variables
            u_pts      : Sampled U parameter points
            v_pts      : Sampled V parameter points
            bspline    : B-spline surface object
            K_target   : Target Gaussian curvature value
        """

        self.u_pts = u_pts
        self.v_pts = v_pts
        self.bspline = bspline
        self.cp_idx = var_idx["control_points"]
        self.K_target = K_target

        K, _, _ = bspline_curvatures(bspline, u_pts, v_pts)

        self.Ktarget = min(np.abs(K.flatten()))
        self.num_residuals = len(K.flatten())

    def func(self, X):
        # Update control points in B-spline
        cp_flat = X[self.cp_idx].reshape(-1, 3)
        self.bspline.controlpoints = cp_flat.reshape(self.bspline.controlpoints.shape[0], self.bspline.controlpoints.shape[1], 3)

        # Compute Gaussian curvature at sampled points
        K, _, _ = bspline_curvatures(self.bspline, self.u_pts, self.v_pts)

        K_flat = K.flatten()

        neg_indx = np.where(K_flat < 0)[0]
        pos_indx = np.where(K_flat >= 0)[0]

        K_flat[pos_indx]  = 0 

        K_flat[neg_indx] -= self.Ktarget 

        f = K_flat 
        
        return f
    