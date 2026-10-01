import numpy as np
from hanan.optimization.objective_term import ObjectiveTerm

from hanan.optimization.indexing import interleave_indices_dim
from hanan.geometry.utils import mathematica_v


class QuadLmeshFairness(ObjectiveTerm):

    def __init__(self) -> None:
        """ Template constraint
        Energy for poly-line regularity structure of the L-Mesh.
        E_{SR} = \sum_{f in sf} \sum_{fi in adj(f)} 
                      ((p_i,i+1 - p_i )/Td(f,fi+1) . cni )^2
                    + ((p_i,i-1 - p_i )/Td(f,fi-1) . cni )^2     

                = \sum_{f in sf} \sum_{fi in adj(f)}
                      ( (cf_{i+1} - cv)/Td(f,fi+1) . cni - (rfi+1 - rv)* cni. n_{i+1} )^2
                    + ( (cf_{i-1} - cv)/Td(f,fi-1) . cni - (rfi-1 - rv)* cni. n_{i} )^2    

        cni = ci - cv/||ci - cv|| 

        Diagram of the structure regularity constraint between adjacent spherical faces:
        f0,f1,f2,f3 are neighboring spherical faces
        vi  are left or-plane to spherical face fi
        pi  are contact points at face f;   pi = c - r * n_i where n_i is the normal of or-plane vi
        pij are contact points at face fj; pij = c_j - r_j * n_i

                           f2
                    p23._________.p22
                      |            |
             v3       |  or-cone   |     v2
                      |            |
       p33----------p3.___________.p2----------p21
         |            |            |            |
     f3  |  or-cone   |    f       |   or-cone  | f1
         |            |            |            |
        p03----------p0.___________.p1----------p11
                      |            |
            v0        |  or-cone   |    v1
                      |            |
                    p00._________.p01
                           f0  
    

        """
        super().__init__()
        self.name = "StructureReg" # Name of the constraint
        
        # Cont for weight decrease
        self.cont = 0


      
    def initialize_objective(self, X, var_idx, sf_adjList, orPlanesIdx) -> None:
        """ 
        We assume knots are normalized
        Input:
            X : Variables
            var_idx       : dictionary of indices of variables
            sf_adjList    : Spherical face adjacency list
            orPlanesIdx   : Or-planes indices between spherical faces
        """

        # Get cv, rv, npf, hpf 
        cv_idx  = var_idx["cv"].reshape(-1, 3)
        rv_idx  = var_idx["rv"]
        npf_idx = var_idx["npf"].reshape(-1, 3)
        
        # Init rows and cols lists
        self._rows = []
        self._cols = []    

        # Valid rows 
        number_of_rows = 0

        self.indices_cf = []
        self.indices_rf = []
        self.indices_nvi = []
        self.indices_cfi = []
        self.indices_rfi = []
        self.indices_cfi_plus_1 = []
        self.indices_rfi_plus_1 = []
        self.indices_cfi_minus_1 = []
        self.indices_rfi_minus_1 = []
        self.indices_nvi_plus_1 = []

        # Prepare face 
        for f, sfAdj in enumerate(sf_adjList):

            numSF = len(sfAdj)

            # Only valid for spherical faces with 4 adjacent spherical faces
            if numSF == 4:
                number_of_rows += 8  # 2 residuals per adjacent spherical face
                
                
                self.indices_cf.extend(cv_idx[f].tolist())
                self.indices_rf.append(rv_idx[f])

                orPlanes_f = orPlanesIdx[f]

                for i in range(numSF):
                    
                    # Next and previous spherical face in adjacency
                    fi_plus_1  = sfAdj[(i + 1) % numSF]
                    fi_minus_1 = sfAdj[(i - 1) % numSF]
                    fi         = sfAdj[i]

                    # vertex planes indices
                    ni        = orPlanes_f[i]
                    ni_plus_1 = orPlanes_f[(i + 1) % numSF]

                    # Indices for variables cv, rv, nvi, cfi+1, rfi+1, cfi-1, rfi-1, nvi+1
                    self.indices_cfi.extend(cv_idx[fi].tolist())
                    self.indices_rfi.append(rv_idx[fi])

                    self.indices_nvi.extend(npf_idx[ni].tolist())

                    self.indices_cfi_plus_1.extend(cv_idx[fi_plus_1].tolist())
                    self.indices_rfi_plus_1.append(rv_idx[fi_plus_1])

                    self.indices_cfi_minus_1.extend(cv_idx[fi_minus_1].tolist())
                    self.indices_rfi_minus_1.append(rv_idx[fi_minus_1])

                    self.indices_nvi_plus_1.extend(npf_idx[ni_plus_1].tolist())

                    # Indices for derivatives ( (cf_{i+1} - cv)/Td(f,fi+1) . cni - (rfi+1 - rv)* cni. n_{i+1} )^2;
                    # Total indices = 14
                    # dcf indices # 3
                    self._cols.extend( cv_idx[f].tolist() )
                    # drf indices # 1
                    self._cols.append( rv_idx[f] )

                    # dci indices # 3
                    self._cols.extend( cv_idx[fi].tolist() )

                    # dci_plus_1 indices # 3
                    self._cols.extend( cv_idx[fi_plus_1].tolist() )
                    # dri_plus_1 indices # 1
                    self._cols.append( rv_idx[fi_plus_1] )
                 
                    # dni+1 indices # 3
                    self._cols.extend( npf_idx[ni_plus_1].tolist() )

                    # Indices for derivatives ((cf_{i-1} - cv)/Td(f,fi-1) . cni - (rfi-1 - rv)* cni. n_{i} )^2;
                    # total indices = 14                 
                    # dcf indices # 3
                    self._cols.extend( cv_idx[f].tolist() )
                    # drf indices # 1
                    self._cols.append( rv_idx[f] )
 
                    # dci indices # 3
                    self._cols.extend( cv_idx[fi].tolist() )

                    # dci_minus_1 indices # 3
                    self._cols.extend( cv_idx[fi_minus_1].tolist() )
                    # dri_minus_1 indices # 1
                    self._cols.append( rv_idx[fi_minus_1] )
                 
                    # dni indices # 3
                    self._cols.extend( npf_idx[ni].tolist() )
        
        self._rows = np.arange(number_of_rows).repeat(14)  # 14 derivatives per 2 residuals

        # print("Rows ", self._rows[:29])
        # print("Cols ",[int(col) for col in self._cols[:29]])

        self.update_tangential_distances_distance_centers(X)

        self.num_residuals = number_of_rows

    def update_tangential_distances_distance_centers(self, X) -> None:
        # Compute tangetial distance and lenghts 
        cv  = X[self.indices_cf].reshape(-1, 3).repeat(4, axis=0)
        rv  = X[self.indices_rf].repeat(4)

        cfi_plus_1   = X[self.indices_cfi_plus_1].reshape(-1, 3)
        rfi_plus_1   = X[self.indices_rfi_plus_1]

        cfi_minus_1  = X[self.indices_cfi_minus_1].reshape(-1, 3)
        rfi_minus_1  = X[self.indices_rfi_minus_1]

        cfi  = X[self.indices_cfi].reshape(-1, 3)

        ni        = X[self.indices_nvi].reshape(-1, 3)
        ni_plus_1 = X[self.indices_nvi_plus_1].reshape(-1, 3)

        vi_plus_1_fi_plus_1 = cfi_plus_1  - rfi_plus_1[:, np.newaxis]  * ni_plus_1
        vi_fi_minus_1       = cfi_minus_1 - rfi_minus_1[:, np.newaxis] * ni

        vi        = cv - rv[:, np.newaxis] * ni
        vi_plus_1 = cv - rv[:, np.newaxis] * ni_plus_1

        # Store Td(f,fi+1) and Td(f,fi-1)
        self.TDcfi_plus_1  = np.linalg.norm( vi_plus_1_fi_plus_1 - vi_plus_1, axis=1 )
        self.TDcfi_minus_1 = np.linalg.norm( vi_fi_minus_1 - vi, axis=1 )

        # Store distance between cfi and cv
        self.distance_cfi_cf = np.linalg.norm(cfi - cv, axis=1)


    def grad(self, X):
        # Get Variables
        cv  = X[self.indices_cf].reshape(-1, 3).repeat(4, axis=0)
        rv  = X[self.indices_rf].repeat(4)
        nvi = X[self.indices_nvi].reshape(-1, 3)

        cfi = X[self.indices_cfi].reshape(-1, 3)
        # rfi = X[self.indices_rfi]

        # cfi+1 and rfi+1
        cfi_plus_1 = X[self.indices_cfi_plus_1].reshape(-1, 3)
        rfi_plus_1 = X[self.indices_rfi_plus_1]

        # cfi-1 and rfi-1
        cfi_minus_1 = X[self.indices_cfi_minus_1].reshape(-1, 3)
        rfi_minus_1 = X[self.indices_rfi_minus_1]

        nvi_plus_1 = X[self.indices_nvi_plus_1].reshape(-1, 3)

        # Compute normalized direction bewtween cfi and cv
        cfi_cf = (cfi - cv) 
    
        cfip1_cf  = (cfi_plus_1  - cv)  # cfi+1 - cv
        cfim1_cf  = (cfi_minus_1 - cv) # cfi-1 - cv

        rfip1_rf  = (rfi_plus_1  - rv)   # rfi+1 - rv
        rfim1_rf  = (rfi_minus_1 - rv)  # rfi-1 - rv


        TD_plus_cfi  = self.TDcfi_plus_1  * self.distance_cfi_cf + 1e-6
        TD_minus_cfi = self.TDcfi_minus_1 * self.distance_cfi_cf + 1e-6

    
        # Dcf (fi+1) =>  (2*cv - cfi - cfi+1 )/(Td(f,fi+1)*cfi)  + (rfi+1 - rv) * nvi_plus_1 / TD*distance_Cfi
        dcf_fi_plus_1  = (2*cv - cfi - cfi_plus_1)/TD_plus_cfi[:,np.newaxis] + \
                        (rfip1_rf/TD_plus_cfi)[:, np.newaxis] * nvi_plus_1
    
        # Dcf (fi-1) =>  (2*cv - cfi - cfi-1 )/(Td(f,fi-1)*cfi)  + (rfi-1 - rv) * nvi / TD*distance_Cfi
        dcf_fi_minus_1 = (2*cv - cfi - cfi_minus_1)/TD_minus_cfi[:,np.newaxis] + \
                         (rfim1_rf/TD_minus_cfi)[:, np.newaxis] * nvi     
        
        # Drf (fi+1) => (cfi - cv) * nvi+1 / TD*distance_Cfi
        drf_fi_plus_1  = np.einsum( 'ij,ij->i', cfi_cf, nvi_plus_1)/TD_plus_cfi
        
        # Drf (fi-1) => (cfi - cv) * nvi / TD*distance_Cfi
        drf_fi_minus_1 = np.einsum( 'ij,ij->i', cfi_cf, nvi)/TD_minus_cfi

        # Dci (fi+1) =>  (cfi+1 - cv)/(Td(f,fi+1)*cni)
        dci_fi_plus_1  = cfip1_cf /(TD_plus_cfi )[:, np.newaxis]  - (rfip1_rf/ TD_plus_cfi)[:, np.newaxis]  * nvi_plus_1

        # Dci (fi-1) =>  (cfi-1 - cv)/(Td(f,fi-1)*cni)
        dci_fi_minus_1 = cfim1_cf /(TD_minus_cfi )[:, np.newaxis] - (rfim1_rf/ TD_minus_cfi)[:, np.newaxis] * nvi

        # Dni (fi+1) =>  - (rfi+1 - rv) * cfi / TD*distance_Cfi
        dni_fi_plus_1  = - (rfip1_rf/ TD_plus_cfi )[:, np.newaxis] * cfi_cf
        
        # Dni (fi)   =>  - (rfi-1 - rv) * cfi / TD*distance_Cfi
        dni_fi         = - (rfim1_rf/ TD_minus_cfi )[:, np.newaxis] * cfi_cf

        # Dcfi+1 =  (cfi - cv)/(Td(f,fi+1)*dcfi)
        dcfi_plus_1    = cfi_cf /TD_plus_cfi[:, np.newaxis]

        # Dcfi-1 =  (cfi - cv)/(Td(f,fi-1)*dcfi)
        dcfi_minus_1   = cfi_cf /TD_minus_cfi[:, np.newaxis] 
    
        # Drfi+1 = - (cfi - cv) . nvi+1/ TD*distance_Cfi
        drfi_plus_1    = - np.einsum('ij,ij->i', cfi_cf, nvi_plus_1)/TD_plus_cfi

        # Drfi-1 = - (cfi - cv) . nvi/ TD*distance_Cfi
        drfi_minus_1   = - np.einsum('ij,ij->i', cfi_cf, nvi)/TD_minus_cfi
    
        # Order of derivatives: cv, rv, ci, ci+1, ri+1, ni+1, cv, rv, ci, ci-1, ri-1, ni
        # Assemble derivatives
        values = np.column_stack([
            dcf_fi_plus_1,    # Dcf 
            drf_fi_plus_1,    # Drf 
            dci_fi_plus_1,    # Dci 
            dcfi_plus_1,      # Dci+1  
            drfi_plus_1,      # Dri+1
            dni_fi_plus_1,    # Dni+1

            dcf_fi_minus_1,   # Dcf
            drf_fi_minus_1,   # Drf
            dci_fi_minus_1,   # Dci            
            dcfi_minus_1,     # Dci-1
            drfi_minus_1,     # Dri-1
            dni_fi            # Dni
        ]).flatten()

        # Compute residuals for fi+1 and fi-1
        res_fi_plus_1  = np.einsum('ij,ij->i', cfip1_cf, cfi_cf) / TD_plus_cfi - \
                            (rfip1_rf/TD_plus_cfi) * np.einsum('ij,ij->i', cfi_cf, nvi_plus_1 ) 
        
            
        res_fi_minus_1 = np.einsum('ij,ij->i', cfim1_cf, cfi_cf) / TD_minus_cfi - \
                            (rfim1_rf/TD_minus_cfi) * np.einsum('ij,ij->i', cfi_cf, nvi )
        
    
        residuals = np.column_stack([res_fi_plus_1, res_fi_minus_1]).flatten()

       
        self.update_tangential_distances_distance_centers(X)
        self.cont += 1
        if self.cont % 5 == 0:
            self.w *= 0.1

        return values

    def res(self, X):
        cv  = X[self.indices_cf].reshape(-1, 3).repeat(4, axis=0)
        rv  = X[self.indices_rf].repeat(4)
        nvi = X[self.indices_nvi].reshape(-1, 3)
        cfi = X[self.indices_cfi].reshape(-1, 3)
        cfi_plus_1  = X[self.indices_cfi_plus_1].reshape(-1, 3)
        rfi_plus_1  = X[self.indices_rfi_plus_1]
        cfi_minus_1 = X[self.indices_cfi_minus_1].reshape(-1, 3)
        rfi_minus_1 = X[self.indices_rfi_minus_1]
        nvi_plus_1  = X[self.indices_nvi_plus_1].reshape(-1, 3)
        cfi_cf    = cfi - cv
        cfip1_cf  = cfi_plus_1  - cv
        cfim1_cf  = cfi_minus_1 - cv
        rfip1_rf  = rfi_plus_1  - rv
        rfim1_rf  = rfi_minus_1 - rv
        TD_plus  = self.TDcfi_plus_1  * self.distance_cfi_cf
        TD_minus = self.TDcfi_minus_1 * self.distance_cfi_cf
        res_plus  = np.einsum('ij,ij->i', cfip1_cf, cfi_cf) / TD_plus  - \
                    (rfip1_rf / TD_plus)  * np.einsum('ij,ij->i', cfi_cf, nvi_plus_1)
        res_minus = np.einsum('ij,ij->i', cfim1_cf, cfi_cf) / TD_minus - \
                    (rfim1_rf / TD_minus) * np.einsum('ij,ij->i', cfi_cf, nvi)
        return np.column_stack([res_plus, res_minus]).flatten()