from hanan.optimization.objective_term import ObjectiveTerm
import numpy as np



class GaussPositive(ObjectiveTerm):

    def __init__(self) -> None:
        """ Objective term to modify Gaussian curvature of a B-spline surface.
        Objective K > 0
        Energy term to solve: K - |K|
        """
        super().__init__()
        self.name = "Gaussi_Positive" # Name of the constraint
        self.absPrevK = None
        self.jacobianMethod = "analytical" # Options: "jax" (fastest), "FD_parallel", "FD", "analytical"

    

    def initialize_objective(self, X, var_idx) -> None:
        """
        Initialize the Gaussian curvature objective term.
        Input:
            X          : Variables
            var_idx    : dictionary of indices of variables
            u_pts      : Sampled U parameter points
            v_pts      : Sampled V parameter points
            bspline    : B-spline surface object
        """

        # Indices of control points in X
        self.K_idx = var_idx['K']  
        
        # Number of residuals
        self.num_residuals = len(self.K_idx)

        # Set rows and columns
        self._rows = np.arange(self.num_residuals)
        self._cols = self.K_idx

        # Previous absolute K values
        self.absPrevK = np.abs(X[self.K_idx] - 0.2)


    def compute(self, X) -> None:
        """
        Compute the residuals of the B-spline auxiliary constraints.
        Input:
            X          : Variables
        Output:
            values, res : Residuals values and residuals
        """

        K = X[self.K_idx]   # K values at sampled points

        #DK => 1
        ones = np.ones_like(K)

        res = np.zeros_like(K)

        #res = K - 0.2 - self.absPrevK

        negative_idx = K < 0.2

        res[negative_idx]  = (K[negative_idx] - self.absPrevK[negative_idx]).flatten()
    
        # Update previous Delta
        self.absPrevK = np.abs(K - 0.2)
        
        return ones, res


    def func(self, X):
        # Compute the residuals of the B-spline auxiliary constraints.
        K = X[self.K_idx]   # K values at sampled points


        res = np.zeros_like(K)

        #res = K - 0.2 - self.absPrevK

        negative_idx = K < 0.2

        res[negative_idx]  = (K[negative_idx] - self.absPrevK[negative_idx]).flatten()
    
        # Update previous Delta
        
        return res