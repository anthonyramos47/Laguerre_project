
import numpy as np
from scipy.interpolate import bisplev, bisplrep
from hanan.optimization.objective_term import ObjectiveTerm
from Lconjugacy.utils.bsplines_functions import sample_radius

class Lconjugacy(ObjectiveTerm):
    def __init__(self) -> None:
        """ L-conjugacy constraint for B-spline surfaces
        Energy for L-conjugacy: 
        E_{L} = \sum_{p in surface} ( (k1**2 * r - k1)cos th cos (th + bet) + (k2**2 * r - k2)sin th sin (th + bet) )^2
        th, phi : angles between principal directions and L-conjugate directions
         l1 = cos th e1 + sin th e2, where e1, e2 are principal directions
         l2 = cos phi e1 + sin phi e2; phi = th + bet
        """
        super().__init__()
        self.name = "Lconjugacy" # Name of the constraint
        self.jacobianMethod = "analytical" # Options: "jax" (fastest), "FD_parallel", "FD", "analytical"
    
    def initialize_objective(self, X, var_idx, k1, k2, RadiusFunction, u_pts, v_pts) -> None:
        """
        Initialize the L-conjugacy objective term.
        Input:
            X          : Variables
            var_idx    : dictionary of indices of variables
            k1         : Principal curvature 1 at sampled points
            k2         : Principal curvature 2 at sampled points
            r          : Ratio of principal curvatures for L-conjugacy
        """

        self.k1 = k1
        self.k2 = k2
        #self.r = r
        self.th_idx  = var_idx["th"]
        self.bet_idx = var_idx["bet"]
        self.rij_idx = var_idx["rij"]
        self.radiusFunction = RadiusFunction

        self.u_pts = u_pts
        self.v_pts = v_pts

        self.auxiliarRadiusFunction = RadiusFunction.clone()

        auxiliarBspline = self.auxiliarRadiusFunction
        controlPts = auxiliarBspline.controlpoints.reshape(-1, 3)[:, 2]  # Get only the radius component of the control points

        basisU = auxiliarBspline.bases[0]
        basisV = auxiliarBspline.bases[1]

        # Evaluate bases functions
        Nu = basisU.evaluate(u_pts)
        Nv = basisV.evaluate(v_pts)

        NuNv = np.einsum('iu,jv->ijuv', Nu, Nv).reshape(RadiusFunction.controlpoints.shape[0] * RadiusFunction.controlpoints.shape[1], -1)  # (numControlPts x numSampledPts)

        numSampledPts = len(u_pts) * len(v_pts)

        # # Identity matrix representin derivative of each control point and coordinate
        # controlPtsDerivatives = np.eye(len(controlPts))
        # NuNv = np.zeros((len(controlPts), numSampledPts))

        # # Compute the values of the J 
        # for i in range(len(controlPts)):
        #     # Get corresponding row 
        #     dControlPt = controlPtsDerivatives[i]
        #     # Modify control points of auxiliar bspline
        #     auxiliarBspline.controlpoints.reshape(-1, 3)[:, 2] = dControlPt
        #     # Compute the derivative of the control points
        #     NuNv[i] = bisplev(u_pts, v_pts, auxiliarBspline).flatten()

        # Store Basis multiplications (number of control points x number of sampled pts)
        self.NuNv = NuNv.T  # (numSampledPts x numControlPts)

        self.rij_idx = var_idx["rij"]

        self._rows = np.arange(len(k1)).repeat(2 + len(self.rij_idx))
        self._cols = np.column_stack((self.th_idx, self.bet_idx, np.tile([self.rij_idx], (len(self.th_idx), 1)))).flatten()

        self.num_residuals = len(k1)

    def compute(self, X):
        th  = X[self.th_idx]
        bet = X[self.bet_idx]

        self.auxiliarRadiusFunction.controlpoints.reshape(-1, 3)[:, 2] = X[self.rij_idx]


        r = self.auxiliarRadiusFunction(self.u_pts, self.v_pts).reshape(-1,3)[:, 2]  # Evaluate the radius function at the given points, get only the radius component (z)


        k1r = (self.k1**2 * r - self.k1)
        k2r = (self.k2**2 * r - self.k2)

        sinth = np.sin(th)
        costh = np.cos(th)
        sinth_bet = np.sin(th + bet)
        costh_bet = np.cos(th + bet)

        # D th => k1(k1r - 1) (-sin th) cos (th + bet) + ( k2(k2r - 1) cos th (-sin (th + bet)) + ( k1(k1r - 1) cos th sin (th + bet) + (( k2(k2r - 1) sin th cos (th + bet )
        dth = k1r *(-sinth)*costh_bet - k1r * costh*sinth_bet + k2r * costh*sinth_bet + k2r * sinth*costh_bet
        
        # D bet => ( k1(k1r - 1) cos th (-sin (th + bet)) + ( k2(k2r - 1) sin th cos (th + bet)
        dbet = k1r * costh*(-sinth_bet) + k2r * sinth*costh_bet

        # print("NuNv shape:", self.NuNv.shape)
        # print("k1 shape:", self.k1.shape)
        # print("k2 shape:", self.k2.shape)
        # print("k2 * NuNv shape:", ((self.k2**2)[:, None] * self.NuNv).shape)

        # D rij ->  (k1^2 cos th cos (th + bet) + k2^2 sin th sin (th + bet) )NuNv 
        drij = (self.k1**2 * costh * costh_bet + self.k2**2 * sinth * sinth_bet)[:, None] * self.NuNv 

        values = np.column_stack( [dth, dbet, drij] ).flatten()


        # Residuals
        res = ( k1r * costh * costh_bet + k2r * sinth * sinth_bet )

        return values, res
    
    def func(self, X):
        th  = X[self.th_idx]
        bet = X[self.bet_idx]

        self.auxiliarRadiusFunction.controlpoints.reshape(-1, 3)[:, 2] = X[self.rij_idx]


        r = self.auxiliarRadiusFunction(self.u_pts, self.v_pts).reshape(-1,3)[:, 2]  # Evaluate the radius function at the given points, get only the radius component (z)
        k1r = (self.k1**2 * r - self.k1)
        k2r = (self.k2**2 * r - self.k2)


        # Residuals
        res = ( (self.k1**2 * r - self.k1) * np.cos(th) * np.cos(th + bet) + (self.k2**2 * r - self.k2) * np.sin(th) * np.sin(th + bet) )

        return res
    