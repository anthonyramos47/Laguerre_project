import numpy as np
from hanan.optimization.objective_term import ObjectiveTerm
import time 


class Or_contactFree(ObjectiveTerm):

    def __init__(self) -> None:
        """ Template constraint
        Energy to constraint the oriented contact of spheres with face or. planes.
        E_{Or_Contact} = \sum_{vpi} \sum_{sf ~ vpi} ( cv.ni - rv + hi )^2
        """
        super().__init__()
        self.name = "Or Contact" # Name of the constraint

        # Cont for weight decrease
        self.cont = 0


    def initialize_objective(self, X, var_idx, top) -> None:
        """
        We assume knots are normalized

        NEW CONVENTION (spheres on vertices, planes on faces):

        Input:
            X : Variables
            var_idx                : dictionary of indices of variables
            top                    : Combinatorial structure of the mesh
            sphere_planes_adj_list : adjacency list of planes (faces) per sphere
                                     (vertex). Entry i lists the plane-face indices
                                     incident to sphere-vertex i. This is exactly
                                     ``top.vertex_face_adjacency_list()``.
        """
        # Initialize indices
        cv_idx = var_idx["cv"]   # sphere centers, per vertex
        r_idx  = var_idx["rv"]   # sphere radii,   per vertex
        n_idx  = var_idx["npf"]  # plane normals,  per face
        h_idx  = var_idx["hpf"]  # plane offsets,  per face



        rows = 0

        # Initialize lists to store indices for Jacobian computation
        self.cv_idx   = []
        self.rv_idx   = []
        self.npf_idx   = []
        self.hpf_idx   = []

        self._rows = []
        self._cols = []

        # Index values
        self._values_idx = []

        sphere_planes_adj_list = top.vertex_face_adjacency_list()

        # Outer loop: spheres (vertices). Inner loop: incident planes (faces).
        for i in range(len(sphere_planes_adj_list)):
            for j in range(len(sphere_planes_adj_list[i])):

                # Index plane face incident to sphere-vertex i
                idx_pf = sphere_planes_adj_list[i][j]

                # np indices — plane normal (per face idx_pf)
                np_indices = n_idx[3*idx_pf:3*(idx_pf+1)]
                # hv index — plane offset (per face idx_pf)
                hv_index   = h_idx[idx_pf]

                # cv indices — sphere center (per vertex i)
                cf_indices = cv_idx[3*i:3*(i+1)]
                # rv index — sphere radius (per vertex i)
                rf_index   = r_idx[i]

                # # Add indices for constant part of Jacobian
                # self._rows_const.extend([rows, rows])
                # self._cols_const.extend([rf_index, hv_index])
                #self._values_const.extend([-1.0, 1.0])
                #self._values_const.extend([-1.0, 1.0])

                # Add indices for variable part of Jacobian
                self._rows.extend([rows]*8)  # 6 derivatives that change per residual
                # Derivatives in order dr, dh, dcf(3), dnp(3)
                self._cols.extend([rf_index, hv_index] + cf_indices.tolist() + np_indices.tolist() ) 

                
                # Derivatives wrt npf 
                self.npf_idx.extend(np_indices.tolist())

                # Derivatives wrt hpf
                self.hpf_idx.append(hv_index)

                # Derivatives wrt cv
                self.cv_idx.extend(cf_indices.tolist())

                # Derivatives wrt rv
                self.rv_idx.append(rf_index)

                rows += 1

        # Convert to numpy arrays for faster access
        # self.dcf_idx = np.array(self.dcf_idx, dtype=np.int32)
        # self.drf_idx = np.array(self.drf_idx, dtype=np.int32)
        # self.dnp_idx = np.array(self.dnp_idx, dtype=np.int32)
        # self.dhp_idx = np.array(self.dhp_idx, dtype=np.int32)

        # Derivative structure
        # dcf    => npf # repeat row idx 3 times
        # drf(C) => -1  # repeat row idx 1 time  
        # dnp    => cv  # repeat row idx 3 times
        # dhp(C) => 1   # repeat row idx 1 time

        # Number of residuals        
        self.num_residuals = rows

        # # Allocate rows and cols for variable derivatives
        # self._rows = np.arange(self.num_residuals).repeat(6)  # 6 derivatives that change per residual
        # self._cols = np.concatenate([self.dcf_idx, self.dnp_idx])  # Only variable derivative indices

        # # Allocate constant part of Jacobian 
        # self._rows_const = np.arange(rows).repeat(2) # 2 derivatives that do not change per residual
        # self._cols_const = np.concatenate([self.drf_idx, self.dhp_idx])
        # self._values_const = np.concatenate([-1.0 * np.ones(rows), 1.0 * np.ones(rows)])
        

    def grad(self, X):
        cv  = X[self.cv_idx].reshape(-1, 3)
        rv  = X[self.rv_idx]
        npf = X[self.npf_idx].reshape(-1, 3)
        hpf = X[self.hpf_idx]
        drf = -np.ones(len(rv), dtype=np.float64)
        dhp =  np.ones(len(hpf), dtype=np.float64)
        dcf = npf
        dnp = cv
        return np.column_stack([drf, dhp, dcf, dnp]).flatten()

    def res(self, X):
        cv  = X[self.cv_idx].reshape(-1, 3)
        rv  = X[self.rv_idx]
        npf = X[self.npf_idx].reshape(-1, 3)
        hpf = X[self.hpf_idx]
        return np.einsum('ij,ij->i', cv, npf) - rv + hpf
