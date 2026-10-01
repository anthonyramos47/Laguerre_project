import numpy as np
from hanan.optimization.objective_term import ObjectiveTerm

from hanan.optimization.indexing import interleave_indices_dim


class Or_Contact_Radius(ObjectiveTerm):

    def __init__(self) -> None:
        """ Template constraint
        Energy to constraint the oriented contact of or-spheres with incident or-planes.
        With centers constrained along an axis and contact points i.e. cv = q + r*nq, where q is the contact point, nf the normal at cp and r the radius of the or-sphere. Vpi is a vertex plane with normal np and distance to origin hp.

        Energy:
            E_{Or_Contact} = \sum_{vpi} \sum_{sf ~ vpi} ( (q + r nq).np - rv + hp )^2
        """
        super().__init__()
        self.name = "Or Contact" # Name of the constraint
        
        # Cont for weight decrease
        self.cont = 0

        # Contact points
        self.q = None

        # Axex for the centers of the or-spheres
        self.n_q = None


      
    def initialize_objective(self, X, var_idx, sphere_planes_adj_list, contact_points, normals_at_contact_points) -> None:
        """
        We assume knots are normalized

        NEW CONVENTION (spheres on vertices, planes on faces):

        Input:
            X : Variables
            var_idx                : dictionary of indices of variables
            sphere_planes_adj_list : list of plane (face) indices incident to each
                                     sphere (vertex) — top.vertex_face_adjacency_list()
            contact_points         : list of contact points per sphere (vertex)
            normals_at_contact_points : list of normals at contact points per sphere (vertex)
        """

        var_r_indices  = var_idx["rv"]            # sphere radii, per vertex
        npf_idx = var_idx["npf"].reshape(-1, 3)    # plane normals, per face
        hpf_idx = var_idx["hpf"]                   # plane offsets, per face

        self.r_idx = []

        # Initialize lists to store indices for Jacobian computation
        self.rv_idx   = []
        self.npf_idx   = []
        self.hpf_idx   = []

        self.q = [] # contact points
        self.n_q = [] # normals at contact points, which are the axes for the centers of the or-spheres

        # Store contact points and normals

        rows = 0

        self._rows = []
        self._cols = []

        # Loop per sphere (vertex)
        for i in range(len(sphere_planes_adj_list)):
            # Loop per incident plane (face) to sphere i
            for j in range(len(sphere_planes_adj_list[i])):

                # Index plane face incident to sphere-vertex i
                idx_pf = sphere_planes_adj_list[i][j]

                # np indices — plane normal (per face)
                np_indices = npf_idx[idx_pf]

                # hv index — plane offset (per face)
                hv_index   = hpf_idx[idx_pf]

                # r index — sphere radius (per vertex)
                r_index     = var_r_indices[i]

                # Add rows and cols for Jacobian
                self._rows.extend([rows]*5)
                self._cols.extend(np.concatenate(([r_index], [hv_index], np_indices)))

                # Store indices for Jacobian computation
                self.npf_idx.extend(np_indices)
                self.hpf_idx.append(hv_index)
                self.rv_idx.append(r_index)
                self.q.append(contact_points[i])
                self.n_q.append(normals_at_contact_points[i])

                rows += 1


        self.q = np.array(self.q)
        self.n_q = np.array(self.n_q)

        self.num_residuals = rows
    
        
    def grad(self, X):
        rv  = X[self.rv_idx]
        npf = X[self.npf_idx].reshape(-1, 3)
        cv  = self.q + rv[:, None] * self.n_q
        drf = np.einsum('ij,ij->i', self.n_q, npf) - 1.0
        dhi = np.ones(len(self.hpf_idx), dtype=np.float64)
        dnp = cv
        return np.column_stack((drf, dhi, dnp)).flatten()

    def res(self, X):
        rv  = X[self.rv_idx]
        npf = X[self.npf_idx].reshape(-1, 3)
        hpf = X[self.hpf_idx]
        cv  = self.q + rv[:, None] * self.n_q
        return np.einsum('ij,ij->i', cv, npf) - rv + hpf
