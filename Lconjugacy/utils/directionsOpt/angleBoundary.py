import numpy as np
from hanan.optimization.objective_term import ObjectiveTerm


class AngleBoundary(ObjectiveTerm):
    def __init__(self) -> None:
        """ Boundary angle constraint to keep the angles at the boundary close to a target angle
        E_{Boundary} = \sum_{p in boundaryU} (  theta - theta_u )^2 + \sum_{p in boundaryV} ( theta - theta_v )^2
        where theta_u and theta_v are the target angles at the boundaries in U and V directions
        """
        super().__init__()
        self.name = "AngleBoundary" # Name of the constraint
        self.jacobianMethod = "analytical" # Options: "jax" (fastest), "FD_parallel", "FD", "analytical"
    
    def initialize_objective(self, X, var_idx, indicesTheta, thetaValues, indicesPhi, phiValues) -> None:
        """
        Initialize the L-conjugacy objective term.
        Input:
            X          : Variables
            var_idx    : dictionary of indices of variables
            indicesTheta: Indices of theta variables at the boundary
            thetaValues : Target theta values at the boundary
            indicesPhi  : Indices of phi variables at the boundary
            phiValues   : Target phi values at the boundary
        """

        self.targetTheta = thetaValues
        self.targetPhi = phiValues
        self.theta_idx = var_idx["th"][indicesTheta]

        # Beta = phi - theta, so we can compute the target beta at the boundary as well
        self.beta_idx   = var_idx["bet"][indicesPhi]
        self.thetaAtPhi = var_idx["th"][indicesPhi]

        numTheta = len(self.theta_idx)
        numPhi = len(self.beta_idx)

        self._rows   = np.concatenate( ( np.arange(numTheta), np.arange(numTheta, numTheta + numPhi).repeat(2).flatten()  ) )
        self._cols   = np.concatenate((self.theta_idx, np.column_stack((self.thetaAtPhi, self.beta_idx)).flatten()))


    
        self.num_residuals = len(self.beta_idx) + len(self.theta_idx)
   
    

    def compute(self, X):
        """
        Compute residuals.
        """


        theta = X[self.theta_idx]
        beta  = X[self.beta_idx]
        phi   = X[self.thetaAtPhi] + beta

        resTheta = np.sin( 2 * (theta - self.targetTheta) )/2  # Sin to avoid issues with angle periodicity
        resPhi   = np.sin( 2 * (phi - self.targetPhi) )/2

        dtheta       = np.cos( 2 * (theta - self.targetTheta) )  # Derivative of the residual w.r.t. theta
        dbeta  = np.cos( 2 * (phi - self.targetPhi) )
        

        values = np.concatenate((dtheta, dbeta.repeat(2) )).flatten()

       

        # Residuals
        res = np.concatenate((resTheta, resPhi)).flatten()

        return values, res
    
    def func(self, X):
     

        theta = X[self.theta_idx]
        beta  = X[self.beta_idx]
        phi   = X[self.thetaAtPhi] + beta

        resTheta = np.sin( 2 * (theta - self.targetTheta) )/2  # Sin to avoid issues with angle periodicity
        resPhi   = np.sin( 2 * (phi - self.targetPhi) )/2

        # Residuals
        res = np.concatenate((resTheta, resPhi)).flatten()

        return res
    
