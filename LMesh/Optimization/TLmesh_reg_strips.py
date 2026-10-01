import numpy as np
from hanan.optimization.objective_term import ObjectiveTerm

from hanan.optimization.indexing import interleave_indices_dim
from hanan.geometry.utils import mathematica_v


class TLmesh_reg_strips(ObjectiveTerm):

    def __init__(self) -> None:
        """
        Energy for the poly-line regularity structure of the TL-Mesh strips.

        Swapped convention (branch swapSP): oriented spheres live on the
        *vertices* of ``lmesh.top`` (variables ``cv``, ``rv``), and oriented
        planes live on the *faces* (variables ``npf``, ``hpf``).

        A strip is a run of three consecutive sphere *vertices* s1 - s - s2,
        taken as opposite pairs in the 6-ring around s.  The two faces shared by
        each consecutive sphere pair supply the plane normals: nl1/nr1 on edge
        (s1, s) and nl2/nr2 on edge (s, s2) -- "l" and "r" being the two sides
        of the edge.

        E_{SR} = \sum_{s} \sum_{(s1,s2) opposite pair in ring(s)}
                    ( a0 - a1 - a2 + a3 )^2 + ( b0 - b1 - b2 + b3 )^2

            s1         s        s2
           /   \ nl1 /   \ nl2 /   \
               b0   b1   b2   b3
          |     |===|     |===|     |
               a0   a1   a2   a3
           \   / nr1 \   / nr2 \   /

        Each a/b is the contact point of a sphere on one of those planes:

        a0 = c1 - r1 * nl1        b0 = c1 - r1 * nr1
        a1 = c  - r  * nl1        b1 = c  - r  * nr1
        a2 = c  - r  * nl2        b2 = c  - r  * nr2
        a3 = c2 - r2 * nl2        b3 = c2 - r2 * nr2

        so, expanding the "a" row (the "b" row is identical with nr in place of
        nl), the centres do NOT cancel here:

        A = (a0 - a1 - a2 + a3)
          = c1 + c2 - 2 c + ( nl1 + nl2 ) r - nl1 r1 - nl2 r2

        Variables touched: cv, rv (per vertex), npf (per face).
        Spheres are skipped unless they have exactly 6 neighbours.
        """
        super().__init__()
        self.name = "TLmesh_fairness_strips" # Name of the constraint
        
        # Cont for weight decrease
        self.cont = 0


      
    def initialize_objective(self, X, var_idx, lmesh_topology) -> None:
        """ 
        We assume knots are normalized
        Input:
            X : Variables
            var_idx       : dictionary of indices of variables
            lmesh_topology : L-Mesh topology information
        """

        # Get cv, rv, npf, hpf 
        cv_idx  = var_idx["cv"].reshape(-1, 3)
        rv_idx  = var_idx["rv"]
        npf_idx = var_idx["npf"].reshape(-1, 3)

        adj_spheres = lmesh_topology.vertex_adjacency_list()
        # inner_faces = lmesh_topology.inner_faces
        # number_faces = len(adj_spheres)
        
        # Init rows and cols lists
        self._rows = []
        self._cols = []    

        # Valid rows 
        number_of_rows = 0

        self.c1_idx = []
        self.r1_idx = []
        self.cv_idx = []
        self.rv_idx = []
        self.c2_idx = []
        self.r2_idx = []
        self.nl1_idx = []
        self.nr1_idx = []
        self.nl2_idx = []
        self.nr2_idx = []

        for sk, adj in enumerate(adj_spheres):

            if len(adj) == 6:
                pairs = [ (adj[i], adj[(i+3) % 6]) for i in range(3) ]
            else:
                continue

            for s1, s2 in pairs:
                nl_1, nr_1 = lmesh_topology.faces_from_edge(s1, sk)
                nl_2, nr_2 = lmesh_topology.faces_from_edge(sk, s2)
                
                self.c1_idx.append(cv_idx[s1].tolist())
                self.r1_idx.append(rv_idx[s1])
                self.cv_idx.append(cv_idx[sk].tolist())
                self.rv_idx.append(rv_idx[sk])
                self.c2_idx.append(cv_idx[s2].tolist())
                self.r2_idx.append(rv_idx[s2])

                self.nl1_idx.append(npf_idx[nl_1].tolist())
                self.nr1_idx.append(npf_idx[nr_1].tolist())
                self.nl2_idx.append(npf_idx[nl_2].tolist())
                self.nr2_idx.append(npf_idx[nr_2].tolist())

                self._rows.extend(np.arange(number_of_rows, number_of_rows+6).repeat(8).tolist())
                
                self._cols.extend( np.vstack((
                     cv_idx[s1].tolist(), 
                    [rv_idx[s1]]*3, 
                     cv_idx[sk].tolist(), 
                    [rv_idx[sk]]*3, 
                     cv_idx[s2].tolist(), 
                    [rv_idx[s2]]*3, 
                     npf_idx[nl_1].tolist(), 
                     npf_idx[nl_2].tolist())).transpose().flatten() )

                self._cols.extend( np.vstack((
                     cv_idx[s1].tolist(), 
                    [rv_idx[s1]]*3, 
                     cv_idx[sk].tolist(), 
                    [rv_idx[sk]]*3, 
                     cv_idx[s2].tolist(), 
                    [rv_idx[s2]]*3, 
                    npf_idx[nr_1].tolist(), 
                    npf_idx[nr_2].tolist())).transpose().flatten()  )

                number_of_rows += 6

        self.num_residuals = number_of_rows


    def grad(self, X):
        c1 = X[self.c1_idx].reshape(-1, 3)
        r1 = X[self.r1_idx]
        c  = X[self.cv_idx].reshape(-1, 3)
        r  = X[self.rv_idx]
        c2 = X[self.c2_idx].reshape(-1, 3)
        r2 = X[self.r2_idx]
        nl1 = X[self.nl1_idx].reshape(-1, 3)
        nr1 = X[self.nr1_idx].reshape(-1, 3)
        nl2 = X[self.nl2_idx].reshape(-1, 3)
        nr2 = X[self.nr2_idx].reshape(-1, 3)

        ones = np.ones_like(c1)

        # A = c1 + c2 - 2 c + (nl1 + nl2) r - nl1 r1 - nl2 r2
        dc1 = dc2 = ones
        dc        = -2 * ones
        dr1_a = - nl1
        dr_a  = nl1 + nl2
        dr2_a = - nl2
        # d/dnl1 = r - r1 ; d/dnl2 = r - r2 (same scalars for the b row)
        dnl1 = dnr1 = (r - r1)[:, None] * ones
        dnl2 = dnr2 = (r - r2)[:, None] * ones

        # B = c1 + c2 - 2 c + (nr1 + nr2) r - nr1 r1 - nr2 r2
        dr1_b = - nr1
        dr_b  = nr1 + nr2
        dr2_b = - nr2

        # Per stencil, _rows emits the A block (rows 0-2) then the B block
        # (rows 3-5), each as 3 rows (x, y, z) x 8 entries, row-major, in the
        # order c1, r1, c, r, c2, r2, n?1, n?2.  Stacking on axis=2 builds each
        # block as (nb, 3, 8); concatenating on axis=1 stacks A above B.
        A = np.stack([dc1, dr1_a, dc, dr_a, dc2, dr2_a, dnl1, dnl2], axis=2)
        B = np.stack([dc1, dr1_b, dc, dr_b, dc2, dr2_b, dnr1, dnr2], axis=2)

        # Weight decay: full weight at step 0, x0.1 at each of steps 1-10, then off.
        if self.cont <= 10 and self.cont != 0:
            self.w *= 0.1
        elif self.cont > 10:
            self.w = 0.0
        self.cont += 1

        return np.concatenate([A, B], axis=1).reshape(-1)

    def res(self, X):
        c1 = X[self.c1_idx].reshape(-1, 3)
        r1 = X[self.r1_idx]
        c  = X[self.cv_idx].reshape(-1, 3)
        r  = X[self.rv_idx]
        c2 = X[self.c2_idx].reshape(-1, 3)
        r2 = X[self.r2_idx]
        nl1 = X[self.nl1_idx].reshape(-1, 3)
        nr1 = X[self.nr1_idx].reshape(-1, 3)
        nl2 = X[self.nl2_idx].reshape(-1, 3)
        nr2 = X[self.nr2_idx].reshape(-1, 3)
        a0 = c1 - r1[:, np.newaxis] * nl1
        a1 = c  - r[:, np.newaxis] * nl1
        a2 = c  - r[:, np.newaxis] * nl2
        a3 = c2 - r2[:, np.newaxis] * nl2

        b0 = c1 - r1[:, np.newaxis] * nr1
        b1 = c  - r[:, np.newaxis] * nr1
        b2 = c  - r[:, np.newaxis] * nr2
        b3 = c2 - r2[:, np.newaxis] * nr2


        res_a = (a0 - a1 - a2 + a3) 
        res_b = (b0 - b1 - b2 + b3)

        residuals = np.column_stack([res_a, res_b]).flatten()

        return residuals