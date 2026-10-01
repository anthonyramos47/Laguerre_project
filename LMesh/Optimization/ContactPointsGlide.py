import numpy as np
import igl
from hanan.optimization.objective_term import ObjectiveTerm

from hanan.optimization.indexing import interleave_indices_dim


class ContactPointsGlide(ObjectiveTerm):

    def __init__(self) -> None:
        """ Template constraint
        Energy minimize tangential distance between spheres.
        E_{ControlPointsGliding} = ((cp - cp_prox) . n_prox)^2 ; 
        """
        super().__init__()
        self.name = "ControlPointsGliding" # Name of the constraint
        
        # Cont for weight decrease
        self.cont = 0

        self.nProx = None # Normals at projection points

    def initialize_objective(self, X, var_idx, spheresAdjacentVertexPlanes, Vref, Fref ) -> None:
        """ 
        We assume knots are normalized
        Input:
            X : Variables
            var_idx       : dictionary of indices of variables
            spheresAdjacentVertexPlanes : list of indices vertex planes adjacent to spheres si 
            Vref         : Reference positions
            Fref         : Reference faces
        """
        # Set reference values
        self.Vref = Vref
        self.Fref = Fref

        # Get indices of adjacent vertex planes
        adjVP = spheresAdjacentVertexPlanes
        
        # Get centers and radii indices
        self.c_idx = var_idx["cv"].reshape(-1,3) 
        self.n_idx = var_idx["npf"].reshape(-1,3)
        self.r_idx = var_idx["rv"]

        self.ci_idx = []
        self.ri_idx = []
        self.vp_idx = []

        self._rows = []
        self._cols = []

        self.num_residuals = 0

        # Init indices of vertex planes normals and offsets
        for i in range(len(adjVP)):
            vertexPlaneIndices = adjVP[i] 

            self.num_residuals += len(vertexPlaneIndices)

            for j in range(len(vertexPlaneIndices)):

                # Store indices for computations
                self.ci_idx.append( self.c_idx[ i ].tolist() )
                self.ri_idx.append( self.r_idx[ i ] )
                self.vp_idx.append( self.n_idx[ vertexPlaneIndices[j] ].tolist() )

                # Indices for derivatives
                self._cols.extend( np.concatenate((
                    self.c_idx[ i ],
                    self.n_idx[ vertexPlaneIndices[j] ],
                    [self.r_idx[ i ]] )
                ).flatten() )
        
        self._rows = np.arange(self.num_residuals).repeat(7)

        # Get Variables
        c = X[self.ci_idx].reshape(-1, 3)
        n = X[self.vp_idx].reshape(-1, 3)
        r = X[self.ri_idx].reshape(-1)

        # Compute contact points projections and residuals
        contactPoints = c - n * r.reshape(-1,1)

        # Projected points on reference surface
        _, _, cpProx = igl.point_mesh_squared_distance(contactPoints.reshape(-1,3), self.Vref, self.Fref)
    

        self.nProx = contactPoints - cpProx
        self.nProx = self.nProx / (np.linalg.norm(self.nProx, axis=1)[:,np.newaxis] + 1e-8)





    def grad(self, X):
        c = X[self.ci_idx].reshape(-1, 3)
        n = X[self.vp_idx].reshape(-1, 3)
        r = X[self.ri_idx].reshape(-1)
        contactPoints = c - n * r.reshape(-1, 1)
        _, _, cpProx = igl.point_mesh_squared_distance(contactPoints, self.Vref, self.Fref)
        dc = self.nProx
        dn = -r[:, np.newaxis] * self.nProx
        dr = -np.einsum('ij,ij->i', n, self.nProx)
        values = np.column_stack((dc, dn, dr)).flatten()
        # Update normals for next iteration
        self.nProx = contactPoints - cpProx
        self.nProx = self.nProx / (np.linalg.norm(self.nProx, axis=1)[:, np.newaxis] + 1e-8)
        return values

    def res(self, X):
        c = X[self.ci_idx].reshape(-1, 3)
        n = X[self.vp_idx].reshape(-1, 3)
        r = X[self.ri_idx].reshape(-1)
        contactPoints = c - n * r.reshape(-1, 1)
        _, _, cpProx = igl.point_mesh_squared_distance(contactPoints.reshape(-1, 3), self.Vref, self.Fref)
        return np.einsum('ij,ij->i', contactPoints - cpProx.reshape(-1, 3), self.nProx)
        
