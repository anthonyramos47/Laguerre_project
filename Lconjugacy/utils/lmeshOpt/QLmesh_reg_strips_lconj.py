import numpy as np
from hanan.optimization.objective_term import ObjectiveTerm

from hanan.optimization.indexing import interleave_indices_dim
from hanan.geometry.utils import mathematica_v


class QLmesh_reg_strips_lconj(ObjectiveTerm):

    def __init__(self) -> None:
        """
        L-conjugacy copy of ``LMesh.Optimization.QLmesh_reg_strips``, kept frozen to reproduce
        the paper results: this term does not decay its own weight. The decay is
        applied by ``Lconjugacy.pipeline.run_lmesh_optimizer`` (x0.1 every 10 steps,
        FAIRNESS_DECAY / FAIRNESS_DECAY_EVERY).

        Energy for the poly-line regularity structure of the QL-Mesh strips.

        Swapped convention (branch swapSP): oriented spheres live on the
        *vertices* of ``lmesh.top`` (variables ``cv``, ``rv``), and oriented
        planes live on the *faces* (variables ``npf``, ``hpf``).

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
        self.name = "QLmesh_fairness_strips" # Name of the constraint
        
        # Cont for weight decrease
        self.cont = 0
        self.prevol = 0


      
    def initialize_objective(self, X, var_idx, lmesh_topology) -> None:
        """
        Input:
            X              : Variables
            var_idx        : dictionary of indices of variables
            lmesh_topology : ``lmesh.top`` (spheres on vertices, planes on faces)

        Under the swapped convention a "spherical face" of the original
        derivation is simply a sphere, which now lives on a *vertex* of
        ``lmesh.top``; the or-planes around it are that vertex's incident faces.

        For a sphere f with ring neighbours s0..s3 the stencil needs plane n_i to
        be the plane shared by s_{i-1} and s_i (so that u_i = c_{i-1} - r_{i-1} n_i
        is a contact point). The planes are looked up from the edges (f, s_{i-1})
        and (f, s_i) rather than taken from vertex_face_adjacency_list(), whose
        order is shifted by one. Spheres where a plane is missing (next to the
        boundary) are skipped.
        """
        top = lmesh_topology
        sf_adjList = top.vertex_adjacency_list()
        orPlanesIdx = []
        for f, sfAdj in enumerate(sf_adjList):
            planes = []
            if len(sfAdj) == 4:
                for i in range(4):
                    shared = (set(top.faces_from_edge(f, sfAdj[i - 1]))
                              & set(top.faces_from_edge(f, sfAdj[i]))) - {-1}
                    if len(shared) != 1:
                        break
                    planes.append(shared.pop())
            orPlanesIdx.append(planes)
        sf_adjList = [s if len(p) == 4 else [] for s, p in zip(sf_adjList, orPlanesIdx)]

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

        # Num of rows per spherical face
        numRowsPerFace = 4 * 3  # 4  adjacent spherical faces  * 3 rows for (vi - ui - ui+1 + vi+1) 3 vector

        # Prepare face 
        for f, sfAdj in enumerate(sf_adjList):

            numSF = len(sfAdj)

            # Only valid for spherical faces with 4 adjacent spherical faces
            if numSF == 4:
            
                number_of_rows += numRowsPerFace
        
                # Add indices for cv, rv
                self.indices_cf.extend(cv_idx[f].tolist())
                self.indices_rf.append(rv_idx[f])

                # Get or-planes ~ spherical face f
                orPlanes_f = orPlanesIdx[f]

                
                # Repeat rows for each derivative
                #self._rows.extend( np.arange(count_f * numRowsPerFace, (count_f +1) * numRowsPerFace ).reshape(-1,3).repeat(8, axis=0).flatten() )
                self._rows.extend( np.arange(count_f * numRowsPerFace, (count_f +1) * numRowsPerFace ).repeat(8).flatten() )

                
                # Total derivatives per face = 8

                # dcf    # 1
                # dci-1  # 1
                # dci+1  # 1
                # dni    # 1
                # dni+1  # 1
                # drf    # 1
                # dri-1  # 1
                # dri+1  # 1
                # Total indices = 8 per side
                
                for i in range(numSF):
                    
                    # Next and previous spherical face in adjacency
                    fip1  = sfAdj[(i + 1) % numSF]
                    fim1  = sfAdj[(i - 1) % numSF]
                    
                    # Vertex planes indices
                    ni    = orPlanes_f[i]
                    nip1  = orPlanes_f[(i + 1) % numSF]

                    # Indices for variables ni, ci+1, ri+1, ci-1, ri-1, nvi+1
                    #self.indices_ci.extend(cv_idx[fi].tolist())
                    self.indices_ni.extend(npf_idx[ni].tolist())

                    self.indices_cip1.extend(cv_idx[fip1].tolist())
                    self.indices_rip1.append(rv_idx[fip1])

                    self.indices_cim1.extend(cv_idx[fim1].tolist())
                    self.indices_rim1.append(rv_idx[fim1])
                    self.indices_nip1.extend(npf_idx[nip1].tolist())

                    cols_arrengement = np.column_stack([
                        cv_idx[f].flatten(),
                        cv_idx[fim1].flatten(),
                        cv_idx[fip1].flatten(),
                        npf_idx[ni].flatten(),
                        npf_idx[nip1].flatten(),
                        rv_idx[f].flatten().repeat(3),
                        rv_idx[fim1].flatten().repeat(3),
                        rv_idx[fip1].flatten().repeat(3)
                    ]).flatten()

                    self._cols.extend( cols_arrengement.tolist() )
                                   
                count_f += 1

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

        ni = X[self.indices_ni].reshape(-1, 3)

        cip1 = X[self.indices_cip1].reshape(-1, 3)
        rip1 = X[self.indices_rip1]
        nip1 = X[self.indices_nip1].reshape(-1, 3)

        cim1 = X[self.indices_cim1].reshape(-1, 3)
        rim1 = X[self.indices_rim1]

        # dcf => 2 
        dcf = 2 * np.ones_like(cv).flatten()

        # dci-1 = dci+1 => -1
        dcim1 = dcip1 = - 1 * np.ones_like(cim1).flatten()

        # dni => ri-1 - rv 
        dni = (rim1 - rv).flatten().repeat(3)

        # dni+1 => rip1 - rv
        dnip1 = (rip1 - rv).flatten().repeat(3)

        # drf 
        drf = (- ni - nip1).flatten()

        # dri-1 = ni
        drim1 = ni.flatten()

        # dri+1 = ni+1
        drip1 = nip1.flatten()


        # # Order of derivatives: cv, ci-1, c+1, ni, ni+1, rv, ri-1, ri+1
        # # Assemble derivatives
        values = np.column_stack([
            dcf,
            dcim1,
            dcip1,
            dni,
            dnip1,
            drf,
            drim1,
            drip1
        ]).flatten()


        return values

    def res(self, X):
        cv = X[self.indices_cf].reshape(-1, 3).repeat(4, axis=0)
        rv = X[self.indices_rf].repeat(4)
        ni = X[self.indices_ni].reshape(-1, 3)
        cip1 = X[self.indices_cip1].reshape(-1, 3)
        rip1 = X[self.indices_rip1]
        nip1 = X[self.indices_nip1].reshape(-1, 3)
        cim1 = X[self.indices_cim1].reshape(-1, 3)
        rim1 = X[self.indices_rim1]
        ui   = cim1 - rim1[:, np.newaxis] * ni
        uip1 = cip1 - rip1[:, np.newaxis] * nip1
        vi   = cv - rv[:, np.newaxis] * ni
        vip1 = cv - rv[:, np.newaxis] * nip1
        return (vi - ui - uip1 + vip1).flatten()