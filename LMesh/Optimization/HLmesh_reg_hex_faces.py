import numpy as np
from hanan.optimization.objective_term import ObjectiveTerm

from hanan.optimization.indexing import interleave_indices_dim
from hanan.geometry.utils import mathematica_v

cols = [(1,0,0), (0,1,0), (0,0,1), (1,1,0), (1,0,1), (0,1,1)]


class HLmesh_reg_hex_faces(ObjectiveTerm):

    def __init__(self) -> None:
        """
        Energy for the poly-line regularity structure of the HL-Mesh hexagonal
        planar faces.

        Swapped convention (branch swapSP): oriented spheres live on the
        *vertices* of ``lmesh.top`` (variables ``cv``, ``rv``), and oriented
        planes live on the *faces* (variables ``npf``, ``hpf``).

        For every interior edge (pfi, pfj) of the L-mesh -- i.e. a pair of
        adjacent hexagonal *faces*, each carrying a plane with unit normal ni
        resp. nj -- the two hexagons are rotated into correspondence by
        ``reorder_neighborig_vertices`` so that side n of pfi matches side n of
        pfj.  The energy asks matching sides to stay parallel and equal:

        E_{SR} = \sum_{(pfi,pfj) edge}
          \sum_{n in {0,1,3,4}} ( a1 - a0 - b1 + b0 )^2

                pfi   pfj
                / \   / \
               /   \ /   \
            a0|  b0 |     |
               \   / \   /
                \ /   \ /
                a1    b1

        Sides 2 and 5 are skipped: they are the shared edge itself and its
        opposite, which are already tied by the adjacency.

        sa0, sa1 are consecutive sphere *vertices* along side n of pfi, and
        sb0, sb1 the corresponding pair on pfj.  Each a/b is the contact point
        of that sphere on its face plane:

        a0 = ca0  - ra0  * ni       b0 = cb0  - rb0  * nj
        a1 = ca1  - ra1  * ni       b1 = cb1  - rb1  * nj

        (a1 - a0) = ca1 - ca0 - (ra1 - ra0) * ni
        (b1 - b0) = cb1 - cb0 - (rb1 - rb0) * nj

        residual = (a1 - a0) - (b1 - b0)
                 = (ca1 - ca0) - (cb1 - cb0)
                   - (ra1 - ra0) * ni + (rb1 - rb0) * nj

        Variables touched: cv, rv (per vertex), npf (per face).
        Edges whose two faces are not both hexagons are skipped.
        """
        super().__init__()
        self.name = "HLmesh_fairness_hex_faces" # Name of the constraint
        
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

        # Get rv, npf
        rv_idx  = var_idx["rv"]
        cv_idx  = var_idx["cv"].reshape(-1, 3) 
        npf_idx = var_idx["npf"].reshape(-1, 3)

        #v_adj = lmesh_topology.vertex_adjacency_list()
        planar_faces = lmesh_topology.faces
        ei, ej = lmesh_topology.edge_faces()

        # Filter -1 edges 
        valid_edges = (ei != -1) & (ej != -1)
        ei = ei[valid_edges]
        ej = ej[valid_edges]
        
        # Init rows and cols lists
        self._rows = []
        self._cols = []    

        # Valid rows 
        number_of_rows = 0


        self.ca1_idx = []
        self.ca0_idx = []
        self.cb1_idx = []
        self.cb0_idx = []
        self.ra1_idx = []
        self.ra0_idx = []
        self.rb1_idx = []
        self.rb0_idx = []
        self.ni_idx = []
        self.nj_idx = []
  


        
        for k in range(len(ei)):

            pfi = ei[k]
            pfj = ej[k]

            sph_r, sph_l = lmesh_topology.shared_edge_vertices(pfi, pfj)

            # Get corresponding neighboring veritces
            num_adj_vertices_i = len(planar_faces[pfi])
            num_adj_vertices_j = len(planar_faces[pfj])

            if num_adj_vertices_i != num_adj_vertices_j or num_adj_vertices_i != 6:
                continue  # Skip

            adj_si, adj_sj = reorder_neighborig_vertices(planar_faces[pfi], planar_faces[pfj], sph_r, sph_l)

            # Loop per each corrsponding side of the hexagon except the one corresponding to the edge (pfi, pfj) and its opposite side 
            for n in [0, 1, 3, 4]:

                # Get neighbor for vi and vj corresponding to the current side n
                sa0 = adj_si[n]
                sa1 = adj_si[(n+1) % 6]

                sb0 = adj_sj[n]
                sb1 = adj_sj[(n+1) % 6]

                self._rows.extend(np.arange(number_of_rows, number_of_rows + 3).repeat(10).tolist())


                # Get indices for ca0, ca1, cb0, cb1, ra0, ra1, rb0, rb1       
                self.ca0_idx.append(cv_idx[sa0].tolist())
                self.ca1_idx.append(cv_idx[sa1].tolist())
                self.cb0_idx.append(cv_idx[sb0].tolist())
                self.cb1_idx.append(cv_idx[sb1].tolist())
                self.ra0_idx.append(rv_idx[sa0])
                self.ra1_idx.append(rv_idx[sa1])
                self.rb0_idx.append(rv_idx[sb0])
                self.rb1_idx.append(rv_idx[sb1])
                self.ni_idx.append(npf_idx[pfi].tolist())
                self.nj_idx.append(npf_idx[pfj].tolist())


                # Derivatives in order dci1, dci0, dri1, dri0, dni, dcj1, dcj0, drj1, drj0,  dnj
                derivative_indices = np.vstack((
                    (
                        cv_idx[sa1].tolist(), 
                        cv_idx[sa0].tolist(), 
                        [rv_idx[sa1]]*3, 
                        [rv_idx[sa0]]*3, 
                        npf_idx[pfi].tolist(),
                        cv_idx[sb1].tolist(), 
                        cv_idx[sb0].tolist(), 
                        [rv_idx[sb1]]*3, 
                        [rv_idx[sb0]]*3, 
                        npf_idx[pfj].tolist()
                    )
                    )).transpose().flatten()

    
                self._cols.extend(derivative_indices)
                number_of_rows += 3  # 3 residuals per edge (x, y, z)
        
        # self.print_duplicates()

        self.num_residuals = number_of_rows

    def grad(self, X):
        ri1 = X[self.ra1_idx]
        ri0 = X[self.ra0_idx]
        rj1 = X[self.rb1_idx]
        rj0 = X[self.rb0_idx]
        ni  = X[self.ni_idx].reshape(-1, 3)
        nj  = X[self.nj_idx].reshape(-1, 3)
        
        ones = np.ones_like(ni)
        dci1 = dcj0 =  ones
        dci0 = dcj1 = -ones
        dri1 = - ni
        dri0 = ni
        dni  = (ri0 - ri1)[:, None] * ones
        drj1 = nj
        drj0 = - nj
        dnj  = (rj1 - rj0)[:, None] * ones

        # _cols is laid out per stencil as 3 rows (x, y, z) x 10 entries,
        # row-major. Stacking on axis=2 gives exactly that (nb, 3, 10) layout,
        # so the values line up with (_rows, _cols) without any transpose.

        # Weight decay: full weight at step 0, x0.1 at each of steps 1-10, then off.
        if self.cont <= 10 and self.cont != 0:
            self.w *= 0.1
        elif self.cont > 10:
            self.w = 0.0
        self.cont += 1


        return np.stack(
            [dci1, dci0, dri1, dri0, dni, dcj1, dcj0, drj1, drj0, dnj],
            axis=2,
        ).reshape(-1)

    def res(self, X):
        ca1 = X[self.ca1_idx].reshape(-1, 3)
        ca0 = X[self.ca0_idx].reshape(-1, 3)
        cb1 = X[self.cb1_idx].reshape(-1, 3)
        cb0 = X[self.cb0_idx].reshape(-1, 3)
        ra1 = X[self.ra1_idx]
        ra0 = X[self.ra0_idx]
        rb1 = X[self.rb1_idx]
        rb0 = X[self.rb0_idx]
        ni  = X[self.ni_idx].reshape(-1, 3)
        nj  = X[self.nj_idx].reshape(-1, 3)

        a10 = ca1 - ca0 - (ra1 - ra0)[:,None] * ni
        b10 = cb1 - cb0 - (rb1 - rb0)[:,None] * nj

        res = ( a10 - b10 ).flatten()

        return res


def rotate_to(lst, i):
        lst = list(lst)
        i %= len(lst)  # handle out-of-bounds and negative i
        return lst[i:] + lst[:i]


def reorder_neighborig_vertices(face_i, face_j, sph_init, sph_aux):
    """Reorder neighboring vertices of vi and vj to ensure consistent ordering for the fairness constraint. We consider only the case of 6 neighboring vertices for both vi and vj, otherwise we skip the edge."""

    # Find position of sph_init in face_i and face_j (faces may be numpy arrays)
    face_i = list(face_i)
    face_j = list(face_j)
    pos_face_i = face_i.index(sph_init)
    pos_face_j = face_j.index(sph_init)
 
    # Rotate the face_i list to start from the vertex opposite to sph_init
    new_face_i = rotate_to(face_i, pos_face_i - 2)

    # Rotate the face_j list to start from the sph_init
    new_face_j = rotate_to(face_j, pos_face_j)
    return new_face_i, new_face_j
