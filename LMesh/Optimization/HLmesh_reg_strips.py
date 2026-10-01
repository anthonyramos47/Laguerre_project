import numpy as np
from hanan.optimization.objective_term import ObjectiveTerm

from hanan.optimization.indexing import interleave_indices_dim
from hanan.geometry.utils import mathematica_v


class HLmesh_reg_strips(ObjectiveTerm):

    def __init__(self) -> None:
        """
        Energy for the poly-line regularity structure of the HL-Mesh strips.

        Swapped convention (branch swapSP): oriented spheres live on the
        *vertices* of ``lmesh.top`` (variables ``cv``, ``rv``), and oriented
        planes live on the *faces* (variables ``npf``, ``hpf``).  A strip is a
        run of three consecutive hexagonal faces pf1 - pf - pf2 sharing the two
        opposite sphere pairs of the middle face.

        E_{SR} = \sum_{pf1, pf, pf2} ( a0 - a1 - a2 + a3 )^2
                                    + ( b0 - b1 - b2 + b3 )^2

           pf1        pf        pf2
           /   \ s1  /   \ s5  /   \
               b0   b1   b2   b3
          |     |===|     |===|     |
               a0   a1   a2   a3
           \   / s2 \    / s4 \   /

        s1, s2, s4, s5 are sphere *vertices* (centre c_i, radius r_i);
        pf1, pf, pf2 are the face *planes* with unit normals n1, n, n2.
        Each a/b is the contact point of a sphere on the plane it touches:

        a0 = c2  - r2 * n1      b0 = c1  - r1 * n1
        a1 = c2  - r2 * n       b1 = c1  - r1 * n
        a2 = c4  - r4 * n       b2 = c5  - r5 * n
        a3 = c4  - r4 * n2      b3 = c5  - r5 * n2

        The sphere centres cancel pairwise, so the residual depends only on the
        radii and the plane normals:

        A = (a0 - a1 - a2 + a3) = r2 ( n1 - n ) + r4 ( n2 - n )
        B = (b0 - b1 - b2 + b3) = r1 ( n1 - n ) + r5 ( n2 - n )

        Variables touched: rv (per vertex), npf (per face).
        """
        super().__init__()
        self.name = "HLmesh_fairness_strips" # Name of the constraint
        
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

        spheres_face = lmesh_topology.faces
        adj_faces    = lmesh_topology.face_face_adjacency_list()

        # Inner vertices
        inner_faces = lmesh_topology.inner_faces

        
        # Init rows and cols lists
        self._rows = []
        self._cols = []    

        # Valid rows 
        number_of_rows = 0


        # List to store indices
        self.r1_idx = []
        self.r2_idx = []
        self.r4_idx = []
        self.r5_idx = []
        
        
        self.n1_idx = []
        self.n_idx  = []
        self.n2_idx = []
        

        for k, ip in enumerate(inner_faces):

            sph_pf = spheres_face[ip]
            adj_pf = adj_faces[ip]

            if len(sph_pf) == 6 and len(adj_pf) == 6:
                p_pairs = [ (adj_pf[i], adj_pf[(i+3) % 6]) for i in range(3) ]
                s_pairs = [ (sph_pf[(i-1)% 6], sph_pf[i], sph_pf[(i+3) % 6] ,  sph_pf[(i+3-1) % 6]  ) for i in range(3) ]
            else:
                continue

            for (p1,p2), (s1, s2, s5, s4) in zip(p_pairs, s_pairs):
                
                # Assign indices for the current segment 
                self.r1_idx.append(rv_idx[s1])
                self.r2_idx.append(rv_idx[s2])
                self.r4_idx.append(rv_idx[s4])
                self.r5_idx.append(rv_idx[s5])

                self.n1_idx.append(npf_idx[p1].tolist())
                self.n_idx.append(npf_idx[ip].tolist())
                self.n2_idx.append(npf_idx[p2].tolist())

                # a's rows + cols: derivatives dr2 dr4 dn dn1 dn2  (rows 0..2)
                self._rows.extend(np.arange(number_of_rows,   number_of_rows+3).repeat(5).tolist())
                self._cols.extend( np.vstack((
                    [rv_idx[s2]]*3,
                    [rv_idx[s4]]*3,
                    npf_idx[ip].tolist(),
                    npf_idx[p1].tolist(),
                    npf_idx[p2].tolist()
                )).transpose().flatten() )

                # b's rows + cols: derivatives dr1 dr5 dn dn1 dn2  (rows 3..5)
                self._rows.extend(np.arange(number_of_rows+3, number_of_rows+6).repeat(5).tolist())
                self._cols.extend( np.vstack((
                    [rv_idx[s1]]*3,
                    [rv_idx[s5]]*3,
                    npf_idx[ip].tolist(),
                    npf_idx[p1].tolist(),
                    npf_idx[p2].tolist()
                )).transpose().flatten() )

                number_of_rows += 6

        self.num_residuals = number_of_rows


    def grad(self, X):
        r1 = X[self.r1_idx]
        r2 = X[self.r2_idx]
        r4 = X[self.r4_idx]
        r5 = X[self.r5_idx]
        n1 = X[self.n1_idx].reshape(-1, 3)
        n  = X[self.n_idx].reshape(-1, 3)
        n2 = X[self.n2_idx].reshape(-1, 3)

        
        ones = np.ones_like(n)

        # Derivatives of A w.r.t. r2, r4, n, n1, n2
        dr2_a = (n1 - n)
        dr4_a = (n2 - n)
        dn_a  = (- r2 - r4)[:, None] * ones
        dn1_a = r2[:, None] * ones
        dn2_a = r4[:, None] * ones

        # Derivatives of B w.r.t. r1, r5, n, n1, n2
        dr1_b = ( n1 - n )
        dr5_b = ( n2 - n )
        dn_b  = ( - r1 - r5)[:, None] * ones
        dn1_b = r1[:, None] * ones
        dn2_b = r5[:, None] * ones

        # Per stencil, _rows emits the A block (rows 0-2) then the B block
        # (rows 3-5), each as 3 rows (x, y, z) x 5 entries, row-major.
        # Stacking on axis=2 builds each block as (nb, 3, 5); concatenating on
        # axis=1 stacks A above B to give (nb, 6, 5), matching (_rows, _cols).
        A = np.stack([dr2_a, dr4_a, dn_a, dn1_a, dn2_a], axis=2)
        B = np.stack([dr1_b, dr5_b, dn_b, dn1_b, dn2_b], axis=2)

        # Weight decay: full weight at step 0, x0.1 at each of steps 1-10, then off.
        if self.cont <= 10 and self.cont != 0:
            self.w *= 0.1
        elif self.cont > 10:
            self.w = 0.0
        self.cont += 1

        return np.concatenate([A, B], axis=1).reshape(-1)

    def res(self, X):
        r1 = X[self.r1_idx]
        r2 = X[self.r2_idx]
        r4 = X[self.r4_idx]
        r5 = X[self.r5_idx]
        n1 = X[self.n1_idx].reshape(-1, 3)
        n  = X[self.n_idx].reshape(-1, 3)
        n2 = X[self.n2_idx].reshape(-1, 3)

        # A = (a0 - a1 - a2 + a3) = r2 ( n1 - n ) - r4 ( n2 - n )
        A  = r2[:,None] * ( n1 - n ) + r4[:,None] * ( n2 - n )
        # B = (b0 - b1 - b2 + b3) = r1 ( n1 - n ) - r5 ( n2 - n )
        B  = r1[:,None] * ( n1 - n ) + r5[:,None] * ( n2 - n )
        # self.cont += 1
        # if self.cont % 5 == 0:
        #     self.w *= 0.01

        return np.column_stack([A, B]).reshape(-1, 3).flatten()