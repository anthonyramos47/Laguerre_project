import numpy as np
from hanan.optimization.objective_term import ObjectiveTerm


class AngleLconjugate(ObjectiveTerm):
    def __init__(self) -> None:
        """ L-conjugacy constraint for the angle between them
        E_{Angle} = \sum_{p in surface} ( cos theta * cos(the + beta) + Sin(theta)Sin(theta+beta) )^2
        """
        super().__init__()
        self.name = "AngleLconjugate" # Name of the constraint
        self.jacobianMethod = "analytical" # Options: "jax" (fastest), "FD_parallel", "FD", "analytical"
    
    def initialize_objective(self, X, var_idx, targetBeta) -> None:
        """
        Initialize the L-conjugacy objective term.
        Input:
            X          : Variables
            var_idx    : dictionary of indices of variables
            k1         : Principal curvature 1 at sampled points
            k2         : Principal curvature 2 at sampled points
            r          : Ratio of principal curvatures for L-conjugacy
        """

        self.targetCos = np.cos(targetBeta)
        self.beta_idx  = var_idx["bet"]
        self.theta_idx = var_idx["th"]
      
       
        self._rows = np.arange(len(self.beta_idx))
        self._cols = self.beta_idx
    
        self.num_residuals = len(self.beta_idx)
    
    

    def compute(self, X):
        """
        Compute residuals.
        """

        beta  = X[self.beta_idx]
        theta = X[self.theta_idx]

        # d th => 0

        dbet = np.cos(theta+beta) * np.sin(theta) - np.sin(theta+beta) * np.cos(theta)

        # Residuals
        res = self.func(X)

        return dbet, res
    
    def func(self, X):
        beta = X[self.beta_idx]
        theta = X[self.theta_idx]

        res = np.cos(theta) * np.cos(theta + beta) + np.sin(theta) * np.sin(theta + beta) - self.targetCos

        return res
    
