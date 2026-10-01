import numpy as np
from hanan.optimization.objective_term import ObjectiveTerm

from hanan.optimization.indexing import interleave_indices_dim
from hanan.geometry.utils import mathematica_v

cols = [(1,0,0), (0,1,0), (0,0,1), (1,1,0), (1,0,1), (0,1,1)]


class TLmesh_reg_hex_faces(ObjectiveTerm):

    def __init__(self) -> None:
        """
        Energy for the poly-line regularity structure of the TL-Mesh spherical
        polygons (the sphere "panels" of a triangular L-mesh).

        Swapped convention (branch swapSP): oriented spheres live on the
        *vertices* of ``lmesh.top`` (variables ``cv``, ``rv``), and oriented
        planes live on the *faces* (variables ``npf``, ``hpf``).

        This is the vertex-side dual of ``HLmesh_reg_hex_faces``: instead of
        walking the sides of two adjacent hexagonal faces, it walks the ring of
        face-planes around two adjacent sphere *vertices* si, sj.  Both rings
        are rotated into correspondence by ``reorder_neighborig_vertices`` using
        the shared face as the anchor, so side n of si matches side n of sj.

        E_{SR} = \sum_{(si, sj) edge}
          \sum_{n in {0,1,3,4}} ( a1 - a0 - b1 + b0 )^2

                si        sj
                / \      / \
               /   \    /   \
           na0|     |==|nb0   |
               \   /    \   /
                \ /      \ /
                na1       nb1

        si, sj are adjacent sphere vertices (centres ci, cj; radii ri, rj);
        na0, na1 are consecutive face-plane normals around si, and nb0, nb1 the
        corresponding pair around sj.  Each a/b is a contact point:

        a0 = ci - ri * na0        b0 = cj - rj * nb0
        a1 = ci - ri * na1        b1 = cj - rj * nb1

        The centres cancel within each pair, leaving

        residual = (a1 - a0) - (b1 - b0)
                 = ri ( na0 - na1 ) + rj ( nb1 - nb0 )

        Variables touched: rv (per vertex), npf (per face).
        Edges are skipped unless both endpoints have exactly 6 incident faces.
        """
        super().__init__()
        self.name = "TLmesh_fairness_hex_faces" # Name of the constraint
        
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

        si_idx, sj_idx = lmesh_topology.edge_vertices()
        inn_sv = lmesh_topology.inner_vertices()
        sph_adj_planes = lmesh_topology.vertex_face_adjacency_list()

        # Filter only edges between inner faces
        valid_edges = np.isin(si_idx, inn_sv) & np.isin(sj_idx, inn_sv)
        si_idx = si_idx[valid_edges]
        sj_idx = sj_idx[valid_edges]

        # Init rows and cols lists
        self._rows = []
        self._cols = []    

        # Valid rows 
        number_of_rows = 0

        self.ri_idx = []
        self.rj_idx = []
        self.na0_idx = []
        self.na1_idx = []
        self.nb0_idx = []
        self.nb1_idx = []
       
        for k, (i, j) in enumerate(zip(si_idx, sj_idx)):

            # Get corresponding vertices for face i and j
            num_face_planes_si = len(sph_adj_planes[i])
            num_face_planes_sj = len(sph_adj_planes[j])

            if num_face_planes_si != num_face_planes_sj or num_face_planes_si != 6:
                continue  # Skip if the number of vertex planes doesn't match
            
            # Get the corresponding faces for the edge (i, j)
            _, plane_r = lmesh_topology.faces_from_edge(i, j)

            adj_si, adj_sj = reorder_neighborig_vertices(sph_adj_planes[i], sph_adj_planes[j], plane_r)

            # Loop per each corrsponding side of the hexagon except the one corresponding to the edge (pfi, pfj) and its opposite side 
            for n in [0, 1, 3, 4]:


                na0 = adj_si[n]
                na1 = adj_si[(n + 1) % 6]
                nb0 = adj_sj[n]
                nb1 = adj_sj[(n + 1) % 6]
            
                self._rows.extend(np.arange(number_of_rows, number_of_rows + 3).repeat(6).tolist())

                na0_indices = npf_idx[na0].tolist()
                na1_indices = npf_idx[na1].tolist()
                nb0_indices = npf_idx[nb0].tolist()
                nb1_indices = npf_idx[nb1].tolist()

                # Get relevant indices for the variables
                self.ri_idx.append(rv_idx[i])
                self.rj_idx.append(rv_idx[j])
                self.na0_idx.extend(na0_indices)
                self.na1_idx.extend(na1_indices)
                self.nb0_idx.extend(nb0_indices)
                self.nb1_idx.extend(nb1_indices)

                derivative_indices = np.vstack(([rv_idx[i]]*3, [rv_idx[j]]*3, na0_indices, na1_indices, nb0_indices, nb1_indices)).transpose().flatten()

                self._cols.extend(derivative_indices)
                number_of_rows += 3  # 3 residuals per edge (x, y, z)
    
        #self.print_duplicate_entries()
        self.num_residuals = number_of_rows

    def grad(self, X):
        ri = X[self.ri_idx]
        rj = X[self.rj_idx]
        ni0 = X[self.na0_idx].reshape(-1, 3)
        ni1 = X[self.na1_idx].reshape(-1, 3)
        nj0 = X[self.nb0_idx].reshape(-1, 3)
        nj1 = X[self.nb1_idx].reshape(-1, 3)

        # E = ri (ni0 - ni1) + rj (nj1 - nj0)
        ones = np.ones_like(ni0)
        dri  =  (ni0 - ni1)
        drj  =  (nj1 - nj0)
        dni0 =  ri[:, None] * ones
        dni1 = -ri[:, None] * ones
        dnj0 = -rj[:, None] * ones
        dnj1 =  rj[:, None] * ones

        # _cols is laid out per stencil as 3 rows (x, y, z) x 6 entries,
        # row-major. Stacking on axis=2 gives exactly that (nb, 3, 6) layout,
        # so the values line up with (_rows, _cols) without any transpose.
        # Weight decay: full weight at step 0, x0.1 at each of steps 1-10, then off.
        if self.cont <= 10 and self.cont != 0:
            self.w *= 0.1
        elif self.cont > 10:
            self.w = 0.0
        self.cont += 1

        return np.stack([dri, drj, dni0, dni1, dnj0, dnj1], axis=2).reshape(-1)

    def res(self, X):
        ri = X[self.ri_idx]
        rj = X[self.rj_idx]
        ni0 = X[self.na0_idx].reshape(-1, 3)
        ni1 = X[self.na1_idx].reshape(-1, 3)
        nj0 = X[self.nb0_idx].reshape(-1, 3)
        nj1 = X[self.nb1_idx].reshape(-1, 3)

        res = (ri[:, None] * (ni0 - ni1) + rj[:, None] * (nj1 - nj0)).flatten()
        return res

    def print_duplicate_entries(self):
        """Print (row, col) tuples that appear more than once in self._rows / self._cols."""
        pairs = list(zip(self._rows, self._cols))
        seen = {}
        for pair in pairs:
            seen[pair] = seen.get(pair, 0) + 1
        duplicates = {pair: count for pair, count in seen.items() if count > 1}
        if duplicates:
            print(f"Found {len(duplicates)} duplicate (row, col) pairs:")
            for pair, count in sorted(duplicates.items()):
                print(f"  row={pair[0]}, col={pair[1]}  —  appears {count} times")
        else:
            print("No duplicate (row, col) pairs found.")

def printable_list(lst):
    return [int(i) for i in lst]

def rotate_to(lst, i):
    i %= len(lst)  # handle out-of-bounds and negative i
    return lst[i:] + lst[:i]


def reorder_neighborig_vertices(adj_faces_i, adj_faces_j, plane_init):
    """Reorder neighboring vertices of vi and vj to ensure consistent ordering for the fairness constraint. We consider only the case of 6 neighboring vertices for both vi and vj, otherwise we skip the edge."""

    
    # Find position of sph_init in face_i and face_j
    pos_face_i = (adj_faces_i.index(plane_init))
    pos_face_j = (adj_faces_j.index(plane_init))


    # Rotate the face_i list to start from the vertex opposite to sph_init
    new_face_i = rotate_to(adj_faces_i, pos_face_i - 2)

    
    # Rotate the face_j list to start from the sph_init
    new_face_j = rotate_to(adj_faces_j, pos_face_j)

    
    return new_face_i, new_face_j
