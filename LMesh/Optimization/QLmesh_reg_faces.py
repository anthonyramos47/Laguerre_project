import numpy as np
from hanan.optimization.objective_term import ObjectiveTerm

from hanan.optimization.indexing import interleave_indices_dim
from hanan.geometry.utils import mathematica_v


class QLmesh_reg_faces(ObjectiveTerm):

    def __init__(self) -> None:
        """
        Energy for the poly-line regularity structure of the QL-Mesh spherical
        panels.

        Swapped convention (branch swapSP): oriented spheres live on the
        *vertices* of ``lmesh.top`` (variables ``cv``, ``rv``), and oriented
        planes live on the *faces* (variables ``npf``, ``hpf``).  The stencil is
        centred on a plane vp with two opposite neighbouring planes n0, n2 and
        the four spheres s0..s3 on vp's own vertices.  The sphere centres cancel,
        leaving

        A = (a1 - a0 - a3 + a2) = n0 r0 + n2 r1 - np ( r0 + r1 )
        B = (b1 - b0 - b3 + b2) = n0 r3 + n2 r2 - np ( r2 + r3 )

        E_{SR} = \sum_{v in Vp}  (a1 - a0 - a3 + a2)^2 + (b1 - b0 - b3 + b2)^2 + ...
                    

        Diagram of the structure regularity constraint between adjacent spherical faces:
        a0 = c0 - r0 * n0
        a1 = c0 - r0 * np
        a2 = c1 - r1 * np
        a3 = c1 - r1 * n2

        b0 = c3 - r3 * n0
        b1 = c3 - r3 * np
        b2 = c2 - r2 * np
        b3 = c2 - r2 * n2
                            v3
                      .____________.
                      |            |
             s3       |  or-cone   |    s2
                      |            |
         b0----------b1.__________.b2----------b3
         |            |     vp     |            |
   v0    |  or-cone   | ti     ti+1|   or-cone  | v2
         |            |->        <-|            |
         a0----------a1.__________.a2----------a3
                      |            |
             s0       |  or-cone   |    s1
                      |            |
                      .____________.
                            v1 
    

        """
        super().__init__()
        self.name = "QLmesh_fairness_faces" # Name of the constraint
        
        # Cont for weight decrease
        self.cont = 0
        self.prevol = 0


      
    def initialize_objective(self, X, var_idx, lmesh_topology) -> None:
        """
        Input:
            X              : Variables
            var_idx        : dictionary of indices of variables
            lmesh_topology : ``lmesh.top`` (spheres on vertices, planes on faces)

        Under the swapped convention the stencil is centred on a *plane*
        (a face of ``lmesh.top``): ``vp`` is that plane, ``v0``/``v2`` are two
        opposite neighbouring planes, and ``s0..s3`` are the spheres sitting on
        the plane's own vertices.

        The stencil needs v0 to be the plane across the edge (s3, s0) and v2 the
        plane across (s1, s2), i.e. neighbour k across the edge (s_{k-1}, s_k).
        The neighbours are looked up from those edges rather than taken from
        face_face_adjacency_list(), whose order does not follow the face's
        vertices. Planes with a missing neighbour (boundary) are skipped.
        """
        top = lmesh_topology
        vp_sf_adjList = [list(f) for f in top.faces]
        vp_vp_adjList = []
        for vp, sph in enumerate(vp_sf_adjList):
            adj = []
            if len(sph) == 4:
                for k in range(4):
                    across = set(top.faces_from_edge(sph[k - 1], sph[k])) - {-1, vp}
                    if len(across) != 1:
                        break
                    adj.append(across.pop())
            vp_vp_adjList.append(adj)

        # Get cv, rv, npf, hpf 
        cv_idx  = var_idx["cv"].reshape(-1, 3)
        rv_idx  = var_idx["rv"]
        npf_idx = var_idx["npf"].reshape(-1, 3)
        
        # Init rows and cols lists
        self._rows = []
        self._cols = []    

        # Valid rows 
        number_of_rows = 0

        self.indices_vp = []
        self.indices_v0 = []
        self.indices_v2 = []
        self.indices_c0 = []
        self.indices_c1 = []
        self.indices_c2 = []
        self.indices_c3 = []

        self.indices_r0 = []
        self.indices_r1 = []
        self.indices_r2 = []
        self.indices_r3 = []

        count_vp = 0

        # Num of rows per spherical face
        numRowsPerFace = 4 * 3  # 4  adjacent spherical faces  * 3 rows for (vi - ui - ui+1 + vi+1) 3 vector

        # Prepare face 
        for vp, vpAdj in enumerate(vp_vp_adjList):

            numVP  = len(vpAdj) # number of vertex points adjacent to face f
            numSph = len(vp_sf_adjList[vp]) # number of spherical faces adjacent to face f

            # Only valid for vertex planes with 4 adjacent vertex planes
            if numVP == 4 and numSph == 4:
                
                # Add rows
                number_of_rows += numRowsPerFace
        
                # Number of rows 
                self._rows.extend( np.arange(count_vp * numRowsPerFace, (count_vp +1) * numRowsPerFace ).repeat(5).flatten() )
                
                # Get spherical faces adjacent to vertex point
                sphAdj = vp_sf_adjList[vp]
                vpAdj  = vp_vp_adjList[vp]

                # Indices per derivative
                # dnp  # 1
                # dn0  # 1
                # dn2  # 1
                # dr0  # 1
                # dr1  # 1
                # Total indices = 5 per side

                for i in range(2):
                    
                    # Roll to get consistent ordering
                    s0, s1, s2, s3 = np.roll(sphAdj, - i)
                    v0, _, v2, _ = np.roll(vpAdj, - i)

                    n0Idx = npf_idx[v0].tolist()
                    n2Idx = npf_idx[v2].tolist()
                    npIdx = npf_idx[vp].tolist()
                    c0Idx = cv_idx[s0].tolist()
                    c1Idx = cv_idx[s1].tolist()
                    c2Idx = cv_idx[s2].tolist()
                    c3Idx = cv_idx[s3].tolist()
                    r0Idx = rv_idx[s0]
                    r1Idx = rv_idx[s1]
                    r2Idx = rv_idx[s2]
                    r3Idx = rv_idx[s3]

                    # Store indices for computation
                    self.indices_vp.append(npf_idx[vp].tolist() )
                    self.indices_v0.append(n0Idx)
                    self.indices_v2.append(n2Idx)
                    self.indices_c0.append(c0Idx)
                    self.indices_c1.append(c1Idx)
                    self.indices_c2.append(c2Idx)
                    self.indices_c3.append(c3Idx)
                    self.indices_r0.append(r0Idx)
                    self.indices_r1.append(r1Idx)  
                    self.indices_r2.append(r2Idx)
                    self.indices_r3.append(r3Idx)
                    # Define indices for derivatives
                    # Order: dnp, dn0, dn2, dr0, dr1, dnp, dn0, dn2, dr2, dr3

                    # 3 rows (x, y, z) x 5 entries, row-major:
                    # entry order is dnp, dn0, dn2, dr?, dr?
                    cols_arrengementA = np.column_stack([
                        npIdx,
                        n0Idx,
                        n2Idx,
                        np.repeat(r0Idx, 3),
                        np.repeat(r1Idx, 3)
                    ]).flatten()

                    cols_arrengementB = np.column_stack([
                        npIdx,
                        n0Idx,
                        n2Idx,
                        np.repeat(r2Idx, 3),
                        np.repeat(r3Idx, 3)
                    ]).flatten()


                    cols_arrengement = np.concatenate( (cols_arrengementA, cols_arrengementB) )
                    self._cols.extend( cols_arrengement.tolist() )
            
                count_vp += 1

        self.num_residuals = number_of_rows
    

    def grad(self, X):
        # Get variables
        n_p = X[self.indices_vp].reshape(-1, 3)
        n0 = X[self.indices_v0].reshape(-1, 3)
        n2 = X[self.indices_v2].reshape(-1, 3)
        c0 = X[self.indices_c0].reshape(-1, 3)
        c1 = X[self.indices_c1].reshape(-1, 3)
        c2 = X[self.indices_c2].reshape(-1, 3)
        c3 = X[self.indices_c3].reshape(-1, 3)
        r0 = X[self.indices_r0]
        r1 = X[self.indices_r1]
        r2 = X[self.indices_r2]
        r3 = X[self.indices_r3]

        # a0 = c0 - r0 * n0
        # a1 = c0 - r0 * np
        # a2 = c1 - r1 * np
        # a3 = c1 - r1 * n2

        # b0 = c3 - r3 * n0
        # b1 = c3 - r3 * np
        # b2 = c2 - r2 * np
        # b3 = c2 - r2 * n2

        ones = np.ones_like(n_p)

        # A = (a1 - a0 - a3 + a2) = n0 r0 + n2 r1 - np ( r0 + r1 )
        dnpA = (-r0 - r1)[:, None] * ones
        dn0A =  r0[:, None] * ones
        dn2A =  r1[:, None] * ones
        dr0A =  n0 - n_p
        dr1A =  n2 - n_p

        # B = (b1 - b0 - b3 + b2) = n0 r3 + n2 r2 - np ( r2 + r3 )
        dnpB = (-r2 - r3)[:, None] * ones
        dn0B =  r3[:, None] * ones
        dn2B =  r2[:, None] * ones
        dr2B =  n2 - n_p
        dr3B =  n0 - n_p

        # Per stencil, _rows emits the A block (rows 0-2) then the B block
        # (rows 3-5), each as 3 rows (x, y, z) x 5 entries, row-major, in the
        # entry order dnp, dn0, dn2, dr?, dr?.  Stacking on axis=2 builds each
        # block as (nb, 3, 5); concatenating on axis=1 stacks A above B.
        A = np.stack([dnpA, dn0A, dn2A, dr0A, dr1A], axis=2)
        B = np.stack([dnpB, dn0B, dn2B, dr2B, dr3B], axis=2)

        # Weight decay: full weight at step 0, x0.1 at each of steps 1-10, then off.
        if self.cont <= 10 and self.cont != 0:
            self.w *= 0.1
        elif self.cont > 10:
            self.w = 0.0
        self.cont += 1

        return np.concatenate([A, B], axis=1).reshape(-1)

    def res(self, X):
        n_p = X[self.indices_vp].reshape(-1, 3)
        n0 = X[self.indices_v0].reshape(-1, 3)
        n2 = X[self.indices_v2].reshape(-1, 3)
        c0 = X[self.indices_c0].reshape(-1, 3)
        c1 = X[self.indices_c1].reshape(-1, 3)
        c2 = X[self.indices_c2].reshape(-1, 3)
        c3 = X[self.indices_c3].reshape(-1, 3)
        r0 = X[self.indices_r0]
        r1 = X[self.indices_r1]
        r2 = X[self.indices_r2]
        r3 = X[self.indices_r3]

        a0 = c0 - r0[:, np.newaxis] * n0
        a1 = c0 - r0[:, np.newaxis] * n_p
        a2 = c1 - r1[:, np.newaxis] * n_p
        a3 = c1 - r1[:, np.newaxis] * n2

        b0 = c3 - r3[:, np.newaxis] * n0
        b1 = c3 - r3[:, np.newaxis] * n_p
        b2 = c2 - r2[:, np.newaxis] * n_p
        b3 = c2 - r2[:, np.newaxis] * n2

        resA = (a1 - a0 - a3 + a2)
        resB = (b1 - b0 - b3 + b2)

        # rows 0-2 are the A residual, rows 3-5 the B residual, per stencil
        return np.concatenate([resA, resB], axis=1).reshape(-1)