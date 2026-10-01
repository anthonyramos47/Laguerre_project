
import numpy as np
from hanan.optimization.objective_term import ObjectiveTerm

class DirectionFieldSmoothness(ObjectiveTerm):

    def __init__(self) -> None:
        """ Smoothness constraint for direction fields
        Energy for smoothness:
        E_{Smoothness} = \sum_{(i,j) in E} (th_i - (th_j + alpha_ij)^2 + ( th_i + bet_i - (th_j + bet_j + alpha_ij)^2 )
        where d_i, d_j are direction vectors at vertices i and j
        """
        super().__init__()
        self.name = "DirectionFieldSmoothness" # Name of the constraint
        self.jacobianMethod = "analytical" # Options: "jax" (fastest), "FD_parallel", "FD", "analytical"

    def initialize_objective(self, X, var_idx, edge_vertices, alpha_ij) -> None:
        """
        Initialize the smoothness objective term.
        Input:
            X              : Variables
            var_idx        : dictionary of indices of variables
            edge_vertices  : list of edges between vertices (i, j)
        """

        ei, ej = edge_vertices
        self.th_i = var_idx["th"][ei]
        self.th_j = var_idx["th"][ej]

        self.bet_i = var_idx["bet"][ei]
        self.bet_j = var_idx["bet"][ej]

        self.alpha_ij = alpha_ij

        self.num_residuals = len(ei) 

        # # Rows for derivatives
        self._rows = np.arange(self.num_residuals).repeat(4)

        # # Columns for derivatives
        self._cols = np.column_stack((self.th_i, self.th_j, self.bet_i, self.bet_j)).flatten()

    def compute(self, X):
        """
        Compute the residuals and their derivatives.
        """
        th_i = X[self.th_i]
        th_j = X[self.th_j]

        bet_i = X[self.bet_i]
        bet_j = X[self.bet_j]

        thi_bei_thj_alpha = th_i + bet_i - th_j - self.alpha_ij

        thj_bej_thi_alpha = th_j + bet_j - th_i + self.alpha_ij

        thj_thi_alpha     = th_j - th_i + self.alpha_ij

        thbj_alpha_thbi   = th_j + bet_j + self.alpha_ij - th_i - bet_i

        thb2i_thb2j_alpha = th_i + bet_i*0.5 - th_j - bet_j*0.5 - self.alpha_ij


        # D th_i
        dth_i = 2 * np.sin( 2 * (thi_bei_thj_alpha) ) \
              - 2 * np.sin( 2 * (thj_bej_thi_alpha) ) \
              - 2 * np.sin( 2 * (thj_thi_alpha    ) ) \
              - 2 * np.sin( 2 * (thbj_alpha_thbi  ) ) \
              + 8 * np.sin( 4 * (thb2i_thb2j_alpha) )
    
        # D th_j
        dth_j = - 2 * np.sin( 2 * ( thi_bei_thj_alpha ) ) \
            + 2 * np.sin( 2 * ( thj_bej_thi_alpha ) )  \
            + 2 * np.sin( 2 * (thj_thi_alpha) ) \
            + 2 * np.sin( 2 * (thbj_alpha_thbi) )\
            - 8 * np.sin( 4 * (thb2i_thb2j_alpha) )
        
        # D bet_i
        dbet_i = - 2 * np.sin( 2 * bet_i)  \
            + 2 * np.sin( 2 * (thi_bei_thj_alpha ) ) \
            - 2 * np.sin( 2 * (thbj_alpha_thbi) ) \
            + 4 * np.sin( 4 * (thb2i_thb2j_alpha) )
        
        # D bet_j
        dbet_j = - 2 * np.sin( 2 * bet_j)  \
            + 2 * np.sin( 2 * ( thj_bej_thi_alpha ) ) \
            + 2 * np.sin( 2 * ( thbj_alpha_thbi) )  \
            - 4 * np.sin( 4 * (thb2i_thb2j_alpha) )
        
        values = np.column_stack( [dth_i, dth_j, dbet_i, 
                                   dbet_j] ).flatten()

        res =  4 \
            + np.cos( 2 * bet_i)  \
            + np.cos( 2 * bet_j)  \
            - np.cos( 2 * ( thi_bei_thj_alpha ) )  \
            - np.cos( 2 * ( thj_bej_thi_alpha ) )  \
            - np.cos( 2 * ( thj_thi_alpha     ) )  \
            - np.cos( 2 * (thbj_alpha_thbi) )  \
            - 2 * np.cos(4 *(thb2i_thb2j_alpha) )
        
        return values, res 


    def func(self, X):
        th_i = X[self.th_i]
        th_j = X[self.th_j]

        bet_i = X[self.bet_i]
        bet_j = X[self.bet_j]

        # Residuals
        res = 4  \
            + np.cos( 2 * bet_i)  \
            + np.cos( 2 * bet_j)  \
            - np.cos( 2 * ( th_i + bet_i - th_j - self.alpha_ij ) )  \
            - np.cos( 2 * ( th_j + bet_j - th_i + self.alpha_ij ) )  \
            - np.cos( 2 * (th_j - th_i + self.alpha_ij) )  \
            - np.cos( 2 * (th_j + bet_j + self.alpha_ij - th_i - bet_i) )  \
            - 2 * np.cos(4 *(th_i + bet_i*0.5 - th_j - bet_j*0.5 - self.alpha_ij) )

        return res
