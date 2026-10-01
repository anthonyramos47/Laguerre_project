import numpy as np
from hanan.optimization.objective_term import ObjectiveTerm

from hanan.optimization.indexing import interleave_indices_dim


class Tangential_Dist(ObjectiveTerm):

    def __init__(self) -> None:
        """ Template constraint
        Energy minimize tangential distance between spheres.
        E_{Tangential_Dist} =  \sum_{vi, vj in E} ( |(cj-ci)|^2 - (rj - ri)^2 )^2; 
        """
        super().__init__()
        self.name = "Tangential_Dist" # Name of the constraint
        
        # Cont for weight decrease
        self.cont = 0


      
    def initialize_objective(self, X, var_idx, sf_sf_indices ) -> None:
        """ 
        We assume knots are normalized
        Input:
            X : Variables
            var_idx       : dictionary of indices of variables
            sf_sf_indices : list of indices of spherical faces sharing an edge (sfi, sfj)
        """

        sfi_idx, sfj_idx = sf_sf_indices

        # Filter only valid indices (both not -1)
        valid_mask = (sfi_idx != -1) & (sfj_idx != -1)
        sfi_idx = sfi_idx[valid_mask]
        sfj_idx = sfj_idx[valid_mask]
        
        # Get centers and radii indices
        self.ci_idx = var_idx["cv"][interleave_indices_dim(sfi_idx, 3)]
        self.cj_idx = var_idx["cv"][interleave_indices_dim(sfj_idx, 3)]
        self.ri_idx = var_idx["rv"][sfi_idx]
        self.rj_idx = var_idx["rv"][sfj_idx]

        # dci ->  2(cj - ci) # repeat row idx 3 times
        # dcj -> -2(cj - ci) # repeat row idx 3 times
        # dri -> -2(ri - rj) # repeat row idx
        # drj ->  2(ri - rj) # repeat row idx
        # Each row index is repeated 8 times (3 for dci, 3 for dcj, 1 for dri, 1 for drj)
        self._rows =  np.arange(len(sfi_idx)).repeat(8)
        self._cols =  np.column_stack((
                self.ci_idx.reshape(-1, 3),
                self.cj_idx.reshape(-1, 3),
                self.ri_idx,
                self.rj_idx)
                ).flatten()        
        
        # Number of residuals equals number of spherical face pairs
        self.num_residuals = len(sfi_idx)


    def grad(self, X):
        ci = X[self.ci_idx].reshape(-1, 3)
        cj = X[self.cj_idx].reshape(-1, 3)
        ri = X[self.ri_idx]
        rj = X[self.rj_idx]
        vij = cj - ci
        dci = -2 * vij
        dcj =  2 * vij
        dri =  2 * (ri - rj)
        drj = -2 * (ri - rj)
        return np.column_stack((dci, dcj, dri, drj)).flatten()

    def res(self, X):
        ci = X[self.ci_idx].reshape(-1, 3)
        cj = X[self.cj_idx].reshape(-1, 3)
        ri = X[self.ri_idx]
        rj = X[self.rj_idx]
        vij = cj - ci
        return np.einsum('ij,ij->i', vij, vij) - (rj - ri)**2
        
