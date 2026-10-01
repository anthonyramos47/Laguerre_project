import numpy as np
from hanan.optimization.objective_term import ObjectiveTerm

from hanan.optimization.indexing import interleave_indices_dim
from hanan.geometry.utils import mathematica_v


class QuadLmeshFairnessPrincipal(ObjectiveTerm):

    def __init__(self) -> None:
        """ Template constraint
        Energy for poly-line regularity structure of the L-Mesh.
        E_{SR} = \sum_{f in sf} \sum_{fi in adj(f)} (vi - ui - ui+1 + vi+1)^2 + 
                    

        Diagram of the structure regularity constraint between adjacent spherical faces:
        fi-1  fi           fi+1, spherical faces
        ui   = c_i-1 - r_i-1 * n_i 
        ui+1 = c_i+1 - r_i+1 * n_i+1
        vi   = cv - rv * n_i
        vi+1 = cv - rv * n_i+1
        ti   =  ni x (ci - cv)
        ti+1 =  ni+1 x (ci+1 - cv)
        vol  = ni+1 . ni x (ci - cv)
        w    = vi   - d ti
        w    = vi+1 + d ti+1
        d    = rv/vol * (1 - ni . ni+1)
 
         |            |     f      |            |
   fi-1  |  or-cone   | ti     ti+1|   or-cone  | fi+1
         |            |->        <-|            |
         ui----------vi.____w___.vi+1--------u1+1
                      |            |
            ni->      |  or-cone   |    ni+1 ->
                      |            |
                      .____________.
                            fi  
    

        """
        super().__init__()
        self.name = "StructureReg" # Name of the constraint
        
        # Cont for weight decrease
        self.cont = 0
        self.prevol = 0


   
      
    def initialize_objective(self, X, var_idx, sf_adjList, orPlanesIdx) -> None:
        """ 
        We assume knots are normalized
        Input:
            X : Variables
            var_idx       : dictionary of indices of variables
            sf_adjList    : Spherical face adjacency list
            orPlanesIdx   : Or-planes indices around spherical faces
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
        self.indices_ni = []
        self.indices_ci = []
        self.indices_ri = []
        self.indices_cip1 = []
        self.indices_rip1 = []
        self.indices_cim1 = []
        self.indices_rim1 = []
        self.indices_nip1 = []

        count_f = 0

        # Prepare face 
        for f, sfAdj in enumerate(sf_adjList):

            numSF = len(sfAdj)

            # Only valid for spherical faces with 4 adjacent spherical faces
            if numSF == 4:
                
                number_of_rows += 4*6 # 4 spherical faces 6 rows each 3 for fi, 3 for fi+1
        
                    
                self.indices_cf.extend(cv_idx[f].tolist())
                self.indices_rf.append(rv_idx[f])

                orPlanes_f = orPlanesIdx[f]

                self._rows.extend( np.tile(np.arange(count_f*24, count_f*24 + 24).reshape(-1,3), 17).flatten() ) 

                for i in range(numSF):

                     # 17 derivatives per side each being a 3-vector 

                    # if count_f == 0:
                    #     print("rows:",  np.tile(np.arange(count_f*24, count_f*24 + 24).reshape(-1,3),17).flatten(), "\nLen", len( np.tile(np.arange(count_f*24, count_f*24 + 24).reshape(-1,3), 17).flatten() ) )


                    # dcfx dcfy dcfz       # 3
                    # drf                  # 1
                    # dci-1x dci-1y dci-1z # 3
                    # dri-1                # 1
                    # dci x dci y dci z    # 3
                    # dni x dni y dni z    # 3
                    # dni+1x dni+1y dni+1z # 3
                    # Total indices = 17 per side
                    
                    # Next and previous spherical face in adjacency
                    fip1  = sfAdj[(i + 1) % numSF]
                    fim1  = sfAdj[(i - 1) % numSF]
                    fi    = sfAdj[i]

                    # vertex planes indices
                    ni    = orPlanes_f[i]
                    nip1  = orPlanes_f[(i + 1) % numSF]

                    # Indices for variables ci, ni, cfi+1, rfi+1, cfi-1, rfi-1, nvi+1
                    self.indices_ci.extend(cv_idx[fi].tolist())
                    self.indices_ni.extend(npf_idx[ni].tolist())

                    self.indices_cip1.extend(cv_idx[fip1].tolist())
                    self.indices_rip1.append(rv_idx[fip1])

                    self.indices_cim1.extend(cv_idx[fim1].tolist())
                    self.indices_rim1.append(rv_idx[fim1])
                    self.indices_nip1.extend(npf_idx[nip1].tolist())
                    # Indices for derivatives 2vi - ui - w;
                    # Total indices = 17

                    # dcf indices # 3 indices repeated 3 times
                    self._cols.extend( cv_idx[f].repeat(3).tolist() )

                    # drf indices # 1 index repeated 3 times
                    self._cols.extend( rv_idx[f].repeat(3).tolist() )
 
                    # dci-1 indices # 3 indices repeated 3 times
                    self._cols.extend( cv_idx[fim1].repeat(3) .tolist() )

                    # dri-1 indices # 1 index repeated 3 times
                    self._cols.extend( rv_idx[fim1].repeat(3).tolist() )

                    # dci indices # 3 indices repeated 3 times
                    self._cols.extend( cv_idx[fi].repeat(3).tolist() )
                 
                    # dni indices # 3 indices repeated 3 times
                    self._cols.extend( npf_idx[ni].repeat(3).tolist() )

                    # dni+1 indices # 3 indices repeated 3 times
                    self._cols.extend( npf_idx[nip1].repeat(3).tolist() )
                                   
                    # dcf indices # 3 indices repeated 3 times
                    self._cols.extend( cv_idx[f].repeat(3).tolist() )

                    # drf indices # 1 index repeated 3 times
                    self._cols.extend( rv_idx[f].repeat(3).tolist() )

                    # dci+1 indices # 3 indices repeated 3 times
                    self._cols.extend( cv_idx[fip1].repeat(3) .tolist() )

                    # dri+1 indices # 1 index repeated 3 times
                    self._cols.extend( rv_idx[fip1].repeat(3).tolist() )

                    # dci indices # 3 indices repeated 3 times
                    self._cols.extend( cv_idx[fi].repeat(3).tolist() )
                 
                    # dni indices # 3 indices repeated 3 times
                    self._cols.extend( npf_idx[ni].repeat(3).tolist() )

                    # dni+1 indices # 3 indices repeated 3 times
                    self._cols.extend( npf_idx[nip1].repeat(3).tolist() )

                count_f += 1

        # Compute Ti, Ti+1 ; Ti = || ni x (ci-cv) ||
        cv    = X[self.indices_cf].reshape(-1, 3).repeat(4, axis=0)
        cfi   = X[self.indices_ci].reshape(-1, 3)
        nvi   = X[self.indices_ni].reshape(-1, 3)
        nvip1 = X[self.indices_nip1].reshape(-1, 3)

        # Compute lenghts of tangents 
        self.vol = np.einsum('ij,ij->i', nvip1, np.cross(nvi, cfi - cv))
        self.prevol = self.vol.copy()

        #self.update_tangential_distances_distance_centers(X)
        self.num_residuals = number_of_rows

    def Dcross(self, vectors):
        """
        Compute derivatives of cross product wrt ei (ei is unit vector along x,y,z)
        """
        resx = np.zeros_like(vectors)
        resy = np.zeros_like(vectors)
        resz = np.zeros_like(vectors)

        resx[:,0] = 0 
        resx[:,1] = vectors[:,2]
        resx[:,2] = -vectors[:,1]
    
        resy[:,0] = -vectors[:,2]
        resy[:,1] = 0
        resy[:,2] = vectors[:,0]
    
        resz[:,0] = vectors[:,1]
        resz[:,1] = -vectors[:,0]
        resz[:,2] = 0
        
        return resx, resy, resz
    

    def grad(self, X):
        # Get variables
        cv = X[self.indices_cf].reshape(-1, 3).repeat(4, axis=0)
        rv = X[self.indices_rf].repeat(4)

        ci = X[self.indices_ci].reshape(-1, 3)
        ni = X[self.indices_ni].reshape(-1, 3)

        cip1 = X[self.indices_cip1].reshape(-1, 3)
        rip1 = X[self.indices_rip1]
        nip1 = X[self.indices_nip1].reshape(-1, 3)

        cim1 = X[self.indices_cim1].reshape(-1, 3)
        rim1 = X[self.indices_rim1]

        cicf = (ci - cv)
       
        ti  =  np.cross(ni, cicf)   # ti = ni x (ci - cv)
        tip1 = np.cross(nip1, cicf) # ti+1 = ni+1 x (ci - cv)

        vectorX = np.zeros_like(cv)
        vectorX[:,0] = 1.0

        vectorY = np.zeros_like(cv)
        vectorY[:,1] = 1.0

        vectorZ = np.zeros_like(cv)
        vectorZ[:,2] = 1.0

        dcrossnix, dcrossniy, dcrossniz = self.Dcross(ni)
        dcrossnip1x, dcrossnip1y, dcrossnip1z = self.Dcross(nip1)
        dcrosscicf_x, dcrosscicf_y, dcrosscicf_z = self.Dcross(-cicf)

        nip1_x = nip1[:,0]
        nip1_y = nip1[:,1]
        nip1_z = nip1[:,2]

        ni_x = ni[:,0]
        ni_y = ni[:,1]
        ni_z = ni[:,2]

        d = rv/self.vol 

        nidotnip1 = np.einsum('ij,ij->i', ni, nip1)

        # Derivatives 2vi - ui - w
        # dcfx = (1,0,0) - d( 1 - ni.nip1 ) ni x (1,0,0)
        dcfx =  vectorX - (d * (1 - nidotnip1))[:, np.newaxis] * dcrossnix
        # dcfy = (0,1,0) + rv/(2*Ti^2) * ( -Dydet_c * ti - det * Dcross_ni_y )
        dcfy =  vectorY - (d * (1 - nidotnip1))[:, np.newaxis] * dcrossniy
        # dcfz = (0,0,1) + rv/(2*Ti^2) * ( -Dzdet_c * ti - det * Dcross_ni_z )
        dcfz =  vectorZ - (d * (1 - nidotnip1))[:, np.newaxis] * dcrossniz
        
        # drf = ni + (1- ni.nip1)/vol * ti
        drf  =  - ni + (( 1 - nidotnip1)/self.vol)[:, np.newaxis] * ti

        # dci-1 = -1 
        dcim1x = -1.0 * vectorX
        dcim1y = -1.0 * vectorY
        dcim1z = -1.0 * vectorZ

        # dri-1 = ni 
        drim1 = ni

        # dci = d ni x (1,0,0)
        dci_x = (d * (1 - nidotnip1))[:, np.newaxis] * dcrossnix
        # dci = rv/(2*Ti^2) * (Dydet_ni * ti + det * Dcross_ni_y )
        dci_y = (d * (1 - nidotnip1))[:, np.newaxis] * dcrossniy
        # dci = rv/(2*Ti^2) * (Dzdet_ni * ti + det * Dcross_ni_z )
        dci_z = (d * (1 - nidotnip1))[:, np.newaxis] * dcrossniz

        # dni = -rv (1,0,0) + ri-1 (1,0,0) + d ( -nip1_x * ti + (1 - ni.nip1) * dcrosscicf_x )
        dni_x = -rv[:, np.newaxis] * vectorX + rim1[:, np.newaxis] * vectorX - (d*nip1_x)[:, np.newaxis] * ti + (d*(1- nidotnip1))[:, np.newaxis] * dcrosscicf_x
        # dni = -rv (0,1,0) + ri-1 (0,1,0) + d ( -nip1_y * ti + (1 - ni.nip1) * dcrosscicf_y )
        dni_y = -rv[:, np.newaxis] * vectorY + rim1[:, np.newaxis] * vectorY - (d*nip1_y)[:, np.newaxis] * ti + (d*(1- nidotnip1))[:, np.newaxis] * dcrosscicf_y
        # dni = -rv (0,0,1) + ri-1 (0,0,1) + d ( -nip1_z * ti + (1 - ni.nip1) * dcrosscicf_z )
        dni_z = -rv[:, np.newaxis] * vectorZ + rim1[:, np.newaxis] * vectorZ - (d*nip1_z)[:, np.newaxis] * ti + (d*(1- nidotnip1))[:, np.newaxis] * dcrosscicf_z

        # dni+1 = d ( -ni_x * ti )
        dnip1_x = -(d * ni_x)[:, np.newaxis] * ti
        # dni+1 = d ( -ni_y * ti )
        dnip1_y = -(d * ni_y)[:, np.newaxis] * ti
        # dni+1 = d ( -ni_z * ti )
        dnip1_z = -(d * ni_z)[:, np.newaxis] * ti

        # Dcf
        dcfx2 = vectorX + (d * (1 - nidotnip1))[:, np.newaxis] * dcrossnip1x
        dcfy2 = vectorY + (d * (1 - nidotnip1))[:, np.newaxis] * dcrossnip1y
        dcfz2 = vectorZ + (d * (1 - nidotnip1))[:, np.newaxis] * dcrossnip1z

        # Drf
        drf2  = - nip1 - ((1 - nidotnip1)/self.vol)[:, np.newaxis] * tip1

        # dci+1 
        dcip1x = -1.0 * vectorX
        dcip1y = -1.0 * vectorY
        dcip1z = -1.0 * vectorZ

        # dri+1
        drip1 = nip1  

        # dci
        dci2_x = -(d * (1 - nidotnip1))[:, np.newaxis] * dcrossnip1x
        dci2_y = -(d * (1 - nidotnip1))[:, np.newaxis] * dcrossnip1y
        dci2_z = -(d * (1 - nidotnip1))[:, np.newaxis] * dcrossnip1z

        # dni
        dni2_x = (d * nip1_x)[:, np.newaxis] * tip1
        dni2_y = (d * nip1_y)[:, np.newaxis] * tip1
        dni2_z = (d * nip1_z)[:, np.newaxis] * tip1

        # dni+1
        dnip12_x = -rv[:, np.newaxis] * vectorX + rip1[:, np.newaxis] * vectorX + (d * ni_x)[:, np.newaxis] * tip1 - (d * (1 - nidotnip1))[:, np.newaxis] * dcrosscicf_x
        dnip12_y = -rv[:, np.newaxis] * vectorY + rip1[:, np.newaxis] * vectorY + (d * ni_y)[:, np.newaxis] * tip1 - (d * (1 - nidotnip1))[:, np.newaxis] * dcrosscicf_y
        dnip12_z = -rv[:, np.newaxis] * vectorZ + rip1[:, np.newaxis] * vectorZ + (d * ni_z)[:, np.newaxis] * tip1 - (d * (1 - nidotnip1))[:, np.newaxis] * dcrosscicf_z
    
        # # Order of derivatives: cv, rv, ci-1, ri-1, ci, ni, ni+1, cf2, rf2, ci+1, ri+1, ci2, n2i, n2i+1
        # # Assemble derivatives
        values = np.column_stack([
            dcfx, dcfy, dcfz,               #cv
            drf,                            #rv
            dcim1x, dcim1y, dcim1z,         #ci-1
            drim1,                          #ri-1
            dci_x, dci_y, dci_z,            #ci
            dni_x, dni_y, dni_z,            #ni 
            dnip1_x, dnip1_y, dnip1_z,      #ni+1
            dcfx2, dcfy2, dcfz2,            #cf2
            drf2,                           #rf2
            dcip1x, dcip1y, dcip1z,         #ci+1 
            drip1,                          #ri+1
            dci2_x, dci2_y, dci2_z,         #ci2
            dni2_x, dni2_y, dni2_z,         #ni2
            dnip12_x, dnip12_y, dnip12_z    #ni2+1
        ]).flatten()
        # Set derivatives

        #print("Len vals:", len(values))
        # vi = cv - rv[:, np.newaxis] * ni
        # vip1 = cv - rv[:, np.newaxis] * nip1
        # ui = cim1 - rim1[:, np.newaxis] * ni
        # uip1 = cip1 - rip1[:, np.newaxis] * nip1

        # w = vi - (d * (1 - nidotnip1))[:, np.newaxis] * ti
        # wp1 = vip1 + (d * (1 - nidotnip1))[:, np.newaxis] * tip1

        # residuals_vi_ui_w = 2*vi - ui - w
        # residuals_vip1_uip1_wp1 = 2*vip1 - uip1 - wp1

        residuals_im1 = cv - rv[:, np.newaxis] * ni   - cim1 + rim1[:, np.newaxis] * ni   + (d * (1 - nidotnip1))[:, np.newaxis] * ti
        residuals_ip1 = cv - rv[:, np.newaxis] * nip1 - cip1 + rip1[:, np.newaxis] * nip1 - (d * (1 - nidotnip1))[:, np.newaxis] * tip1

        # Compare residuals
        # print("2vi - ui - w  - res1 =", np.linalg.norm(residuals_vi_ui_w - residuals_im1) )
        # print("2vip1 - uip1 - wp1 - res2 =", np.linalg.norm(residuals_vip1_uip1_wp1 - residuals_ip1) )

        residuals = np.column_stack([residuals_im1, residuals_ip1]).flatten()        

        # wi   = cv - rv[:, np.newaxis] * ni   - (d * (1 - nidotnip1))[:, np.newaxis] * ti
        # wip1 = cv - rv[:, np.newaxis] * nip1 + (d * (1 - nidotnip1))[:, np.newaxis] * tip1

        # print("Avg |wi - wi+1| =", np.mean( np.linalg.norm(wi - wip1, axis=1) ) )
        # print("Vol change:", np.linalg.norm(self.vol - self.prevol) )
        # self.prevol = self.vol.copy()
        

        self.vol = np.einsum('ij,ij->i', nip1, np.cross(ni, cicf)) + 1e-8
        self.cont += 1
        if self.cont % 5 == 0:
            self.w *= 0.1

        return values

    def res(self, X):
        cv = X[self.indices_cf].reshape(-1, 3).repeat(4, axis=0)
        rv = X[self.indices_rf].repeat(4)
        ci = X[self.indices_ci].reshape(-1, 3)
        ni = X[self.indices_ni].reshape(-1, 3)
        cip1 = X[self.indices_cip1].reshape(-1, 3)
        rip1 = X[self.indices_rip1]
        nip1 = X[self.indices_nip1].reshape(-1, 3)
        cim1 = X[self.indices_cim1].reshape(-1, 3)
        rim1 = X[self.indices_rim1]
        cicf = ci - cv
        ti   = np.cross(ni,   cicf)
        tip1 = np.cross(nip1, cicf)
        d = rv / self.vol
        nidotnip1 = np.einsum('ij,ij->i', ni, nip1)
        res_im1 = cv - rv[:, np.newaxis] * ni   - cim1 + rim1[:, np.newaxis] * ni   + (d * (1 - nidotnip1))[:, np.newaxis] * ti
        res_ip1 = cv - rv[:, np.newaxis] * nip1 - cip1 + rip1[:, np.newaxis] * nip1 - (d * (1 - nidotnip1))[:, np.newaxis] * tip1
        return np.column_stack([res_im1, res_ip1]).flatten()