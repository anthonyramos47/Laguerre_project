from hanan.optimization.objective_term import ObjectiveTerm
import numpy as np
import splipy as sp
import time

class bsplineAuxiliary(ObjectiveTerm):

    def __init__(self) -> None:
        """ Objective term to set auxiliary constraints on B-spline surfaces 
        to control their curvature properties.
        Based on Computational Design of Weingarten Surfaces 
                DAVIDE PELLIS, EPFL, Switzerland
                MARTIN KILIAN, TU Wien, Austria
                HELMUT POTTMANN, KAUST, Saudi Arabia
                MARK PAULY, EPFL, Switzerland
        Enegy terms defined are 
        1. N 
        2. E, F, G, Delta
        3. L, M, N
        4. H, K
        """
        super().__init__()
        self.name = "bsplineAuxiliary" # Name of the constraint
        self.Kprev = None
        self.jacobianMethod = "analytical" # Options: "jax" (fastest), "FD_parallel", "FD", "analytical"

    

    def compute_derivatives_control_points(self):
        """
        Compute the derivatives of the B-spline to control points.
        Input:
            X          : Variables
            var_idx    : dictionary of indices of variables
        Output:
            dK_dCP     : Derivative w.r.t respect to control points
        """

    def Dcross(self, vectors, shape=None):
        """
        Compute derivatives of cross product wrt ei (ei is unit vector along x,y,z)
        v x (1,0,0) = (0,vz,-vy)
        v x (0,1,0) = (-vz,0,vx)
        v x (0,0,1) = (vy,-vx,0)
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

        if shape is not None:
            resx = resx.reshape(shape)
            resy = resy.reshape(shape)
            resz = resz.reshape(shape)
        
        return resx, resy, resz

    def initialize_objective(self, X, var_idx, u_pts, v_pts, bspline) -> None:
        """
        Initialize the Gaussian curvature objective term.
        Input:
            X          : Variables
            var_idx    : dictionary of indices of variables
            u_pts      : Sampled U parameter points
            v_pts      : Sampled V parameter points
            bspline    : B-spline surface object
        """

        self.cp_idx = var_idx['control_points']  # Indices of control points in X
        self.n_idx  = var_idx['normals']         # Indices of normals in X
        n_vector_idx = self.n_idx.reshape(-1,3)  # Indices in vector form
        self.E_idx  = var_idx['E']               # Indices of E in X
        self.F_idx  = var_idx['F']               # Indices of F in X
        self.G_idx  = var_idx['G']               # Indices of G in X
        self.Delta_idx  = var_idx['Delta']       # Indices of Delta in X
        self.L_idx  = var_idx['L']               # Indices of L in X
        self.M_idx  = var_idx['M']               # Indices of M in X
        self.N_idx  = var_idx['N']               # Indices of N in X
        self.K_idx  = var_idx['K']               # Indices of K in X
        self.u_pts = u_pts                       # Sampled U parameter points
        self.v_pts = v_pts                       # Sampled V parameter points
        self.bspline = bspline.clone()           # B-spline surface object
        
        # Set variable shapes and sizes
        self.shapeControlPoints = bspline.controlpoints.shape
        self.shapeSampledPoints = (len(u_pts), len(v_pts), 3)
        self.numSampledPoints = len(u_pts)*len(v_pts)
        self.numControlPoints = len(bspline.controlpoints.reshape(-1, 3))

        # Get bases functions
        self.N_U = bspline.bases[0]
        self.N_V = bspline.bases[1]

        # Evaluate bases functions
        self.Nu = self.N_U.evaluate(u_pts)
        self.Nv = self.N_V.evaluate(v_pts)
        
        # Get derivative of bases functions
        self.dN_u = self.N_U.evaluate(u_pts, d=1)
        self.dN_v = self.N_V.evaluate(v_pts, d=1)

        # Get second derivative of bases functions
        self.d2N_u = self.N_U.evaluate(u_pts, d=2)
        self.d2N_v = self.N_V.evaluate(v_pts, d=2)


        # # Number of residuals equals number of control points * 11; from ([3]n, E, F, G, L, M, N, K, H)
        self.num_residuals = self.numSampledPoints * (3 + 8)  # n + E + F + G + Delta + L + M + N + K

        # Define rows and columns         
        rows_nEFG = np.arange(self.numSampledPoints * 6).repeat(1 + self.numControlPoints * 3 )  
        # Repeat 1 + numControlPoints times 3 (for x,y,z)
        rows_delta = np.arange(self.numSampledPoints * 6, self.numSampledPoints * 7).repeat(4) 
        # Repeat 4 for Delta E F G
        rows_LMN = np.arange(self.numSampledPoints * 7, self.numSampledPoints * 10).repeat(1 + 3 + self.numControlPoints * 3 )  
        # Repeat 1 L/M/N and 3 n + numControlPoints times 3 (for x,y,z)
        rows_K   = np.arange(self.numSampledPoints * 10, self.numSampledPoints * 11).repeat(4)


        # Columns indices of derivatives 
        cols_n = np.column_stack( (self.n_idx, np.tile( self.cp_idx, (self.numSampledPoints * 3, 1) ))).flatten()
        cols_E = np.column_stack( (self.E_idx, np.tile( self.cp_idx, (self.numSampledPoints , 1) ))).flatten()
        cols_F = np.column_stack( (self.F_idx, np.tile( self.cp_idx, (self.numSampledPoints , 1) ))).flatten()
        cols_G = np.column_stack( (self.G_idx, np.tile( self.cp_idx, (self.numSampledPoints , 1) ))).flatten()
        cols_Delta = np.column_stack( (self.Delta_idx, self.E_idx, self.F_idx, self.G_idx)).flatten()
        cols_L = np.column_stack( (self.L_idx, n_vector_idx, np.tile( self.cp_idx, (self.numSampledPoints , 1) ))).flatten()
        cols_M = np.column_stack( (self.M_idx, n_vector_idx, np.tile( self.cp_idx, (self.numSampledPoints , 1) ))).flatten()
        cols_N = np.column_stack( (self.N_idx, n_vector_idx, np.tile( self.cp_idx, (self.numSampledPoints , 1) ))).flatten()
        cols_K = np.column_stack( (self.K_idx, self.L_idx, self.M_idx, self.N_idx)).flatten()


        # Set rows and columns
        self._rows = np.concatenate( (rows_nEFG, # rows for n, E, F, G
                                      rows_delta, # rows for Delta
                                      rows_LMN,  # rows for L, M, N
                                      rows_K   # rows for K
                                    )
                                    )

        self._cols = np.concatenate( (cols_n, # cols for n
                                      cols_E, # cols for E
                                      cols_F, # cols for F
                                      cols_G, # cols for G
                                      cols_Delta, # cols for Delta
                                      cols_L, # cols for L
                                      cols_M, # cols for M
                                      cols_N, # cols for N
                                      cols_K   # cols for K
                                    )
                                    )
    
        self.prevDelta = X[self.Delta_idx]

    def compute_values_res_k(self, K, L, M, N):
        """
        Compute values and residuals for K - (L*N - M*M)/Delta.
        """

        # dK => 1
        ones = np.ones(len(self.K_idx))

        # dL => - N / Delta
        dK_L = - N / self.prevDelta

        # dM => + 2 M / Delta
        dK_M = + 2 * M / self.prevDelta

        # dN => - L / Delta
        dK_N = - L / self.prevDelta

        # Energy =: (K - (L*N - M*M)/Delta)
        res = (K - (L * N - M * M) / self.prevDelta).flatten()

        values = np.column_stack( (ones, dK_L, dK_M, dK_N) ).flatten()
        return values, res
        

    def compute_values_res_delta(self, Delta, E, F, G):
        """
        Compute values and residuals for Delta = E*G - F*F.
        """

        # dDelta => [G, -2F, E]
        dDelta_E = - G
        dDelta_F = + 2 * F
        dDelta_G = - E

        # Energy =: (Delta - (E*G - F*F))
        res = (Delta - E * G + F * F).flatten()

        ones = np.ones(len(self.Delta_idx))

        values = np.column_stack( (ones,  dDelta_E, dDelta_F, dDelta_G) ).flatten()

        return values, res
        

    
    def compute_values_res_efg(self, E, F, G, su, sv):
        """
        Compute values and residuals for E, F, G
        E = su.su
        F = su.sv
        G = sv.sv
        Input:
            E, F, G : Values of E, F, G at sampled points 
            su, sv   : First derivatives of the B-spline surface at sampled points (num_sampled_pts, 3)
        """

        # Compute residuals
        # Energy =: (E - su.su), (F - su.sv), (G - sv.sv)
        res_E = (E - np.einsum('ij,ij->i', su, su)).flatten()
        res_F = (F - np.einsum('ij,ij->i', su, sv)).flatten()
        res_G = (G - np.einsum('ij,ij->i', sv, sv)).flatten()

        # dE, dF, dG => 1
        ones = np.ones(len(self.E_idx))

        # Reshape su, sv to (u, v, 3)
        su = su.reshape(self.shapeSampledPoints)
        sv = sv.reshape(self.shapeSampledPoints)
        
        # d E => -2 su . dNu * Nv 
        dCpsE  = - 2 * np.einsum('ua,vb,uvk->uvabk', self.dN_u, self.Nv, su).reshape(-1, self.numControlPoints * 3)
        # d F => - sv . dNu * Nv  - su . Nu * dNv 
        dCpsF  = - np.einsum('ua,vb,uvk->uvabk', self.dN_u, self.Nv, sv).reshape(-1, self.numControlPoints * 3)
        dCpsF += - np.einsum('ua,vb,uvk->uvabk', self.Nu, self.dN_v, su).reshape(-1, self.numControlPoints * 3)
        # d G => -2 sv . Nu * dNv 
        dCpsG  = - 2 * np.einsum('ua,vb,uvk->uvabk', self.Nu, self.dN_v, sv).reshape(-1, self.numControlPoints * 3)


        # Assemble Jacobian
        values_E = np.column_stack( (ones, dCpsE) ).flatten()
        values_F = np.column_stack( (ones, dCpsF) ).flatten()
        values_G = np.column_stack( (ones, dCpsG) ).flatten()

        values = np.concatenate( (values_E, values_F, values_G) )
        res = np.concatenate( (res_E, res_F, res_G) )

        return values, res


    def compute_values_res_lmn(self, L, M, N, suu, suv, svv, n):
        """
        Compute values and residuals for L, M, N
        L = suu . n
        M = suv . n
        N = svv . n
        """
        # dL, dM, dN => 1
        dL = np.ones(len(self.L_idx))
        dM = np.ones(len(self.M_idx))
        dN = np.ones(len(self.N_idx))

     
        denom = 1/np.sqrt(self.prevDelta)

        n_normalized = n * denom[:, None]

        suu_denom = suu * denom[:, None]
        suv_denom = suv * denom[:, None]
        svv_denom = svv * denom[:, None]

        # Energy =: (L - suu.n/sqrt(Delta)), (M - suv.n/sqrt(Delta)), (N - svv.n/sqrt(Delta))
        res_L = (L - np.einsum('ij,ij->i', suu, n)*denom).flatten()
        res_M = (M - np.einsum('ij,ij->i', suv, n)*denom).flatten()
        res_N = (N - np.einsum('ij,ij->i', svv, n)*denom).flatten()

        n_normalized = n_normalized.reshape(self.shapeSampledPoints)

        dCpsuu   = - np.einsum('ua,vb,uvk->uvabk', self.d2N_u,    self.Nv,  n_normalized).reshape(-1, self.numControlPoints * 3)
        dCpsuv   = - np.einsum('ua,vb,uvk->uvabk', self.dN_u,   self.dN_v,  n_normalized).reshape(-1, self.numControlPoints * 3)
        dCpsvv   = - np.einsum('ua,vb,uvk->uvabk', self.Nu,    self.d2N_v,  n_normalized).reshape(-1, self.numControlPoints * 3)


        # Assemble Jacobian
        values_L = np.column_stack( (dL, - suu_denom, dCpsuu) ).flatten()
        values_M = np.column_stack( (dM, - suv_denom, dCpsuv) ).flatten()
        values_N = np.column_stack( (dN, - svv_denom, dCpsvv) ).flatten()

        values = np.concatenate( (values_L, values_M, values_N) )
        res = np.concatenate( (res_L, res_M, res_N) )
     
        return values, res

    def compute_values_res_n(self, n, su, sv):
        
        # dn => 1 
        dn = np.ones(len(self.n_idx))

        # Compute cross products (ei x su, ei x sv) for i=1,2,3
        svCrossX, svCrossY, svCrossZ = self.Dcross(sv,shape=self.shapeSampledPoints)
        suCrossX, suCrossY, suCrossZ = self.Dcross(su,shape=self.shapeSampledPoints)

        # d cpx =>  dNu * Nv e1 x sv  - Nu * dNv su x e1
        dcp_x = (np.einsum('ui,vj,uvk->uvijk', self.dN_u, self.Nv, svCrossX)
               - np.einsum('ui,vj,uvk->uvijk', self.Nu, self.dN_v, suCrossX)).reshape(self.numSampledPoints, self.numControlPoints, 3)
        dcp_y = (np.einsum('ui,vj,uvk->uvijk', self.dN_u, self.Nv, svCrossY)
               - np.einsum('ui,vj,uvk->uvijk', self.Nu, self.dN_v, suCrossY)).reshape(self.numSampledPoints, self.numControlPoints, 3)
        dcp_z = (np.einsum('ui,vj,uvk->uvijk', self.dN_u, self.Nv, svCrossZ)
               - np.einsum('ui,vj,uvk->uvijk', self.Nu, self.dN_v, suCrossZ)).reshape(self.numSampledPoints, self.numControlPoints, 3)
        
        # Reshape derivatives to (numSampledPoints, numControlPoints*3, 3)
        dCp = np.stack((dcp_x, dcp_y, dcp_z), axis=2).reshape(self.numSampledPoints, self.numControlPoints * 3, 3)

        # Arrange derivatives as [dcx1 | dcy1 | dcz1 | dcx2 | dcy2 | dcz2 | ...] 
        dCp = dCp.transpose(0,2,1).reshape(self.numSampledPoints*3, -1)

        # Energy =: (n - su x sv)
        res = (n - np.cross(su, sv)).flatten()

        # Assemble values
        values = np.column_stack( (dn, dCp) ).flatten()
        return values, res

    def compute(self, X) -> None:
        """
        Compute the residuals of the B-spline auxiliary constraints.
        Input:
            X          : Variables
        Output:
            values, res : Residuals values and residuals
        """

        cp = X[self.cp_idx].reshape(-1, 3)   # Control points
        n  = X[self.n_idx].reshape(-1, 3)    # Normals at sampled points
        E  = X[self.E_idx]                   # E values at sampled points
        G  = X[self.G_idx]                   # G values at sampled points
        Delta = X[self.Delta_idx]           # Delta values at sampled points
        F  = X[self.F_idx]                   # F values at sampled points
        L  = X[self.L_idx]                   # L values at sampled points
        M  = X[self.M_idx]                   # M values at sampled points
        N  = X[self.N_idx]                   # N values at sampled points
        K  = X[self.K_idx]                   # K values at sampled points

        # Compute su, sv
        self.bspline.controlpoints = cp.reshape(self.shapeControlPoints)
        su  = self.bspline.derivative(self.u_pts, self.v_pts, d=(1,0)).reshape(-1,3)
        sv  = self.bspline.derivative(self.u_pts, self.v_pts, d=(0,1)).reshape(-1,3)
        suu = self.bspline.derivative(self.u_pts, self.v_pts, d=(2,0)).reshape(-1,3)
        suv = self.bspline.derivative(self.u_pts, self.v_pts, d=(1,1)).reshape(-1,3)
        svv = self.bspline.derivative(self.u_pts, self.v_pts, d=(0,2)).reshape(-1,3)
      
        # Compute values and residuals for each auxiliary constraint
        values_n, res_n = self.compute_values_res_n(n, su, sv)
        values_EFG, res_EFG = self.compute_values_res_efg(E, F, G, su, sv)
        values_Delta, res_Delta = self.compute_values_res_delta(Delta, E, F, G)
        values_LMN, res_LMN = self.compute_values_res_lmn(L, M, N, suu, suv, svv, n)
        values_K, res_K = self.compute_values_res_k(K, L, M, N)
        
        # Combine all values and residuals
        values =  np.concatenate( (values_n, 
                                   values_EFG, 
                                   values_Delta, 
                                   values_LMN,
                                   values_K
                                   ) )
        
        res    =  np.concatenate( (res_n, 
                                   res_EFG, 
                                   res_Delta, 
                                   res_LMN,
                                   res_K
                                   ) )
        
        print("Diff Deltas:", np.linalg.norm(Delta - self.prevDelta))
    

        # Update previous Delta
        self.prevDelta = Delta
        
        
        return values, res


    def func(self, X):
        # Compute the residuals of the B-spline auxiliary constraints.

        cp = X[self.cp_idx].reshape(-1, 3)   # Control points
        n  = X[self.n_idx].reshape(-1, 3)    # Normals at sampled points
        E  = X[self.E_idx]                   # E values at sampled points
        G  = X[self.G_idx]                   # G values at sampled points
        Delta = X[self.Delta_idx]           # Delta values at sampled points
        F  = X[self.F_idx]                   # F values at sampled points
        L  = X[self.L_idx]                   # L values at sampled points
        M  = X[self.M_idx]                   # M values at sampled points
        N  = X[self.N_idx]                   # N values at sampled points
        K  = X[self.K_idx]                   # K values at sampled points

        su = self.bspline.derivative(self.u_pts, self.v_pts, d=(1,0)).reshape(-1,3)
        sv = self.bspline.derivative(self.u_pts, self.v_pts, d=(0,1)).reshape(-1,3)
        suu = self.bspline.derivative(self.u_pts, self.v_pts, d=(2,0)).reshape(-1,3)
        suv = self.bspline.derivative(self.u_pts, self.v_pts, d=(1,1)).reshape(-1,3)
        svv = self.bspline.derivative(self.u_pts, self.v_pts, d=(0,2)).reshape(-1,3)


        # Energy =: (n - su x sv)
        res_n = (n - np.cross(su, sv)).flatten()
        res_E = (E - np.einsum('ij,ij->i', su, su)).flatten()
        res_F = (F - np.einsum('ij,ij->i', su, sv)).flatten()
        res_G = (G - np.einsum('ij,ij->i', sv, sv)).flatten()
        res_Delta = (Delta - (E*G - F**2)).flatten()
        res_L = (L - np.einsum('ij,ij->i', suu, n)/np.sqrt(Delta)).flatten()
        res_M = (M - np.einsum('ij,ij->i', suv, n)/np.sqrt(Delta)).flatten()
        res_N = (N - np.einsum('ij,ij->i', svv, n)/np.sqrt(Delta)).flatten()
        res_K = (K - (L*N - M**2)/Delta).flatten()

        res = np.concatenate( (res_n, res_E, res_F, res_G, res_Delta, res_L, res_M, res_N, res_K) )

        
        return res