import numpy as np
from hanan.optimization.objective_term import ObjectiveTerm

from hanan.optimization.indexing import interleave_indices_dim
from hanan.geometry.utils import mathematica_v

cols = [(1,0,0), (0,1,0), (0,0,1), (1,1,0), (1,0,1), (0,1,1)]


class Sphere_center(ObjectiveTerm):

    def __init__(self) -> None:
        """ 
        Energy to define the sphere center auxiliary var for a 
        optimization where the variable is the radius
        E_{c} = \sum_{f} || c - v_i  - rv/sin(phi_i) n_i   ||^2

        The only variables are rv and c
        """
        super().__init__()
        self.name = "Sphere_centers" # Name of the constraint
        
        # Cont for weight decrease
        self.cont = 0

    

      
    def initialize_objective(self, X, var_idx, vertices, axes, half_angles) -> None:
        """ 
        We assume knots are normalized
        Input:
            X : Variables
            var_idx       : dictionary of indices of variables
            lmesh_topology : L-Mesh topology information
        """

        # Get rv, npf
        self.rv_idx  = var_idx["rv"]
        self.cv_idx  = var_idx["c_aux"].reshape(-1, 3) 

        # Axes scaled by sin of half angles
        self.scaled_axes = axes / np.sin(half_angles)[:, None]

        self.vertices = vertices
        
        # Init rows and cols lists
        self._rows_const = []
        self._cols_const = []    

        # Valid rows 
        number_of_rows = len(self.cv_idx) * 3

        self._rows_const = np.arange(number_of_rows).repeat(2).tolist()

        # Interleave: cv[i,0], rv[i], cv[i,1], rv[i], cv[i,2], rv[i], ...
        # np.stack([cv_idx, rf3], axis=2) → (N,3,2): each row is [cf_j, rf_i]
        # flatten gives: cv[0,0],rv[0], cv[0,1],rv[0], cv[0,2],rv[0], cv[1,0],rv[1], ...
        rf3 = self.rv_idx.repeat(3).reshape(-1, 3)            # (N,3)
        self._cols_const = np.stack([self.cv_idx, rf3], axis=2).reshape(-1).tolist()

        # Interleave: 1, -scaled_axes[i,0], 1, -scaled_axes[i,1], 1, -scaled_axes[i,2], ...
        # ones_3 (N,3) paired with -scaled_axes (N,3) → stack → (N,3,2) → flatten
        ones_3 = np.ones((len(self.cv_idx), 3))                    # dc  = 1
        neg_axes = -self.scaled_axes                           # drf = -scaled_axes[i,j]
        self._values_const = np.stack([ones_3, neg_axes], axis=2).reshape(-1).tolist()

        

        self.num_residuals = number_of_rows

    def grad(self, X):
        """ Gradient is constant, so we can return the precomputed values directly. """
        return None

    def res(self, X):

        # Get rv, c_aux
        rv = X[self.rv_idx].reshape(-1, 1)  # (N,1)
        c_aux = X[self.cv_idx].reshape(-1, 3)  # (N,3)

        res = c_aux - self.vertices - rv * self.scaled_axes  # (N,3)

    
        return res.flatten()

