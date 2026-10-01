import numpy as np
import igl
from hanan.optimization.objective_term import ObjectiveTerm

from hanan.optimization.indexing import interleave_indices_dim


class ContactPointsProximity(ObjectiveTerm):

    def __init__(self) -> None:
        """ Template constraint
        Energy minimize tangential distance between spheres.
        E_{ContactPointsProximity} =  \sum_{vi, vj in E} ( cp - cp* )^2 ; 
        """
        super().__init__()
        self.name = "ContactPointsProximity" # Name of the constraint
        
        # Cont for weight decrease
        self.cont = 0


      
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
            self.num_residuals += 3 * len(vertexPlaneIndices)
            for j in range(len(vertexPlaneIndices)):

                # Store indices for computations
                self.ci_idx.append( self.c_idx[ i ].tolist() )
                self.ri_idx.append( self.r_idx[ i ] )
                self.vp_idx.append( self.n_idx[ vertexPlaneIndices[j] ].tolist() )

                # Indices for derivatives
                self._cols.extend( np.column_stack((
                    self.c_idx[ i ],
                    self.n_idx[ vertexPlaneIndices[j] ],
                    self.r_idx[ i ].repeat(3) )
                ).flatten() )

        self._rows = np.arange(self.num_residuals).repeat(3)


    def grad(self, X):
        c = X[self.ci_idx].reshape(-1, 3)
        n = X[self.vp_idx].reshape(-1, 3)
        r = X[self.ri_idx].reshape(-1)
        dc = np.ones(len(self.ci_idx) * 3)
        dn = -r.repeat(3).reshape(-1, 3)
        dr = -n
        return np.column_stack((dc, dn.flatten(), dr.flatten())).flatten()

    def res(self, X):
        c = X[self.ci_idx].reshape(-1, 3)
        n = X[self.vp_idx].reshape(-1, 3)
        r = X[self.ri_idx].reshape(-1)
        ContactPoints = c - n * r.reshape(-1, 1)
        _, _, CP_proj = igl.point_mesh_squared_distance(ContactPoints.reshape(-1, 3), self.Vref, self.Fref)
        return (ContactPoints - CP_proj.reshape(-1, 3)).flatten()
        
