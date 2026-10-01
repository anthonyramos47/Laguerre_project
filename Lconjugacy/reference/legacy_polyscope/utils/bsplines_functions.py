import numpy as np
import matplotlib.pyplot as plt
import splipy as sp
from splipy import surface_factory
import polyscope as ps
import json
import os
import igl
import splipy as sp
from scipy.interpolate import bisplev, bisplrep
from geomdl.fitting import approximate_surface
from pathlib import Path
from scipy.spatial import KDTree
from scipy.optimize import minimize
from hanan.geometry.utils import *
#from hanan.geometry.geo_processing import normalize_vertices
from hanan.geometry.mesh import Mesh



# def normalsBspline(bsp, u_pts, v_pts):
#     """
#     Function to compute the normals of the surface
#     Input:
#         bsp: B-spline surface
#         u_pts: u points
#         v_pts: v points
#     """

#     # Compute the first and second derivatives
#     du  = bisplev(u_pts, v_pts, bsp, dx=1, dy=0).flatten()
#     dv  = bisplev(u_pts, v_pts, bsp, dx=0, dy=1).flatten()

#     print(du[:5])
    
#     du  = np.hstack((np.ones_like(u_pts.flatten())[:,None], v_pts.flatten()[:, None], du[:, None]))
#     dv  = np.hstack((u_pts.flatten()[:, None], np.ones_like(v_pts.flatten())[:, None], dv[:, None]))
    
#     # Compute the normal
#     n = np.cross(du, dv)
#     print(np.where(np.linalg.norm(n, axis=1) == 0))

def init_lconjugate_directions(bspline, u_sample_pts, v_sample_pts, RadiusValues, angle):
    """
    Initialize L-conjugate directions on the B-spline surface. We fix initial angles per sampled point. 
    
    Given initial angle, first L-conjugate direction is given as a = [cos(theta), sin(theta)] in the principal frame.
    The second L-conjugate direction is computed via the L-conjugacy condition for curve tangents:
        at . M . bt = 0, Mt = [[(k1**2 r - k1, 0], [0,  k2**2 r - k2]]
        t - directions are L-conjugate curve tangents 
        lc - L-conjugate directions (rulings of tangent developable surfaces)
    Input:
        bspline: splipy Surface object
        u_sample_pts: Sampled U parameter points
        v_sample_pts: Sampled V parameter points
        RadiusValues: Radius values at sampled points
        angle: Initial angle for L-conjugate direction [float or np.array]
    Returns:
        theta : Angles for L-conjugate direction 
        phi   : Angles for L-conjugate direction 
    """

    # Build mesh from sampled points
    V, F    = sample_bspline_surface(bspline, u_sample_pts, v_sample_pts)
    num_vertices = V.shape[0]
    mesh = Mesh()
    mesh.make_mesh(V, F)

    # Compute principal frame
    e1, e2, k1, k2, _ = principal_frame(bspline, u_sample_pts, v_sample_pts)

    theta = angle * np.ones(num_vertices)

    t1, t2, phi = compute_lconjugate_curve_tangents(e1, e2, k1, k2, RadiusValues, theta)

    ruling1, ruling2    = exchange_lconjugate_rulings_curve_tangents(e1, e2, k1, k2, t1, t2)
    
    return t1, t2, ruling1, ruling2, theta, phi

def char_circle_direction(e1, e2, k1, k2, r, tangent):
    """
    Compute the direction of the characteristic circle at a point given the ruling direction and principal frame.
    """

    I = np.array([[1, 0], [0, 1]])
    II = np.array([[k1, 0], [0, k2]])
    J = np.array([[0, -1], [1, 0]])
    M = (I - r * II)

    a = np.array([np.dot(tangent, e1), np.dot(tangent, e2)])
    # Get tanget to characteristic circle in parameter space
    c = unit(J @ M @ a)
    # Get the orthogonal direction to characteristic circle in parameter space
    cC = unit(J @ I @ c)
    charCircDir = unit(cC[0] * e1 + cC[1] * e2)

    return charCircDir
                                

def exchange_lconjugate_rulings_curve_tangents_at_point(e1, e2, k1, k2, dir1, dir2):
    """
    Given two L-conjugate directions dir1 and dir2 in principal frame,
    convert to its corresponding curve tangents or rulings at a point.
    """

    a = np.array([np.dot(dir1, e1), np.dot(dir1, e2)])
    b = np.array([np.dot(dir2, e1), np.dot(dir2, e2)])

    #JC =  [[0,-1][0,1]] . [[k1, 0], [0, k2]]
    JC = np.array([[ 0, -k2 ],
                     [ k1, 0   ]
                 ])
    
    # Compute conjugate direction in parameter space
    aC = JC @ a
    dir1_c = unit(aC[0] * e1 + aC[1] * e2)
    bC = JC @ b
    dir2_c = unit(bC[0] * e1 + bC[1] * e2)

    return dir1_c, dir2_c
    

def exchange_lconjugate_rulings_curve_tangents(e1, e2, k1, k2, dir1, dir2):
    """
    Given two L-conjugate directions dir1 and dir2 in principal frame, 
    its corresponding curve tangents or rulings.
    They are related by standard conjugacy conditions:
     a . C . aC = 0, C = [[k1, 0], [0, k2]]
    """

    num_samples = len(k1)
    dir1_c = np.zeros_like(dir1)
    dir2_c = np.zeros_like(dir2)


    # Get coefficients in principal frame
    a = np.column_stack([np.einsum('ij,ij->i', dir1, e1), np.einsum('ij,ij->i', dir1, e2)]) 
    b = np.column_stack([np.einsum('ij,ij->i', dir2, e1), np.einsum('ij,ij->i', dir2, e2)])

    for i in range(num_samples):     
        #JC =  [[0,-1][0,1]] . [[k1, 0], [0, k2]]
        JC = np.array([[ 0, -k2[i] ],
                      [ k1[i], 0   ]
                    ])
        
        C = np.array([[k1[i], 0],
                      [0, k2[i]]
                    ])
        # Compute conjugate direction in parameter space
        aC = JC @ a[i]

        dir1_c[i] = unit(aC[0] * e1[i] + aC[1] * e2[i])

        bC = JC @ b[i]
        dir2_c[i] = unit(bC[0] * e1[i] + bC[1] * e2[i])

        # Check conjugacy condition
        if np.abs(np.dot(a[i], C @ aC)) > 1e-5:
            print(f"Warning: Directions at index A {i} are not Conjugate. Check value: {np.dot(a[i], C @ aC)}")
        
        if np.abs(np.dot(b[i], C @ bC)) > 1e-5:
            print(f"Warning: Directions at index  B {i} are not Conjugate. Check value: {np.dot(b[i], C @ bC)}")
        
    
    return dir1_c, dir2_c

def check_lconjugate_rulings(e1,e2,k1,k2,r,lc1,lc2):
    """
    Check if lc1 and lc2 are L-conjugate rulings by checking the condition:
    a . C . aC = 0, C = [[(k2^-1 - r), 0], [0, k1^-1 - r]]
    """

    num_samples = len(k1)
    check_values = np.zeros(num_samples)

    # Get coefficients in principal frame
    a = np.column_stack([np.einsum('ij,ij->i', lc1, e1), np.einsum('ij,ij->i', lc1, e2)]) 
    b = np.column_stack([np.einsum('ij,ij->i', lc2, e1), np.einsum('ij,ij->i', lc2, e2)])

    for i in range(num_samples):     
        #C = [[(k2^-1 - r), 0], [0, k1^-1 - r]]
        C = np.array([[ 1/k2[i] - r[i], 0 ],
                      [ 0, 1/k1[i] - r[i]   ]
                    ])
        check_values[i] = np.dot(a[i], C @ b[i])

    # ids = np.where(np.abs(check_values) > 1e-5)[0]
    ids = np.where(np.abs(check_values) > 1e-5)[0]
    return check_values, ids

def compute_lconjugate_curve_tangents_at_point(e1, e2, k1, k2, r, theta):
    """
    Given angles theta of the first L-conjugate direction, compute the angles phi of the second L-conjugate direction
    """

    # JM = [[0,-1],[0,1]] . [[k1**2 r - k1, 0], [0, k2**2 r - k2]]
    JM = np.array([
                    [ 0,  - k2**2 * r + k2 ],
                    [k1**2 * r - k1, 0 ]
                ])

    a = np.array([np.cos(theta), np.sin(theta)])
    # Compute L-conjugate direction in parameter space
    b = unit(JM @ a)

    # Compute angle
    phi = np.arctan2(b[1], b[0])

    # Compute first L-conjugate direction in 3D
    t1 = np.cos(theta) * e1 + np.sin(theta) * e2
    # Compute L-conjugate direction in 3D
    t2 = np.cos(phi)   * e1 + np.sin(phi)   * e2

    return t1, t2, phi


def compute_lconjugate_curve_tangents(e1, e2, k1, k2, r, theta):
    """
    Given angles theta of the first L-conjugate direction, compute the angles phi of the second L-conjugate direction
    using the L-conjugacy condition for curve tangents:
        at . M . bt = 0, Mt = [[(k1**2 r - k1, 0], [0,  k2**2 r - k2]]
        t - directions are L-conjugate curve tangents 
        lc - L-conjugate directions (rulings of tangent developable surfaces)
    Input:
        k1: First principal curvature at sampled points
        k2: Second principal curvature at sampled points
        r:  Radius values at sampled points
        theta: Angles of first L-conjugate direction
    Returns:
        t1: First L-conjugate direction in 3D
        t2: Second L-conjugate direction in 3D
        phi: Angles of second L-conjugate direction
    """

    num_samples = len(k1)
    phi = np.zeros_like(k1)

    if isinstance(theta, float):
        # Single angle for all samples
        a = np.tile( [np.cos(theta), np.sin(theta)], (num_samples, 1) )
        # Compute first L-conjugate direction in 3D
        t1 = np.cos(theta) * e1 + np.sin(theta) * e2
    else:
        # Vector of angles for all samples
        a = np.column_stack([np.cos(theta), 
                      np.sin(theta)])
        # Compute first L-conjugate direction in 3D
        t1 = np.cos(theta)[:,None] * e1 + np.sin(theta)[:,None] * e2
        
    for i in range(len(theta)):     
        # JM = [[0,-1],[0,1]] . [[k1**2 r - k1, 0], [0, k2**2 r - k2]]
        JM = np.array([
                        [ 0,  - k2[i]**2 * r[i] + k2[i] ],
                        [k1[i]**2 * r[i] - k1[i], 0 ]
                    ])
        # Compute L-conjugate direction in parameter space
        b = unit(JM @ a[i])

        # Compute angle
        phi[i] = np.arctan2(b[1], b[0])
    
    # Compute L-conjugate direction in 3D
    t2 = np.cos(phi)[:,None]   * e1 + np.sin(phi)[:,None]   * e2

    return t1, t2, phi

def init_lconjugate_boundary(bspline, u_sample_pts, v_sample_pts, RadiusValues, angle):
    """
    Initialize the angles at the boundary of the B-spline surface for the direction field optimization.
    Input:
        bspline: splipy Surface object
        u_sample_pts: Sampled U parameter points
        v_sample_pts: Sampled V parameter points
        RadiusValues: Radius values at sampled points
        angle: Initial angle for L-conjugate direction [float or np.array]
    Returns:
        theta : Angles for L-conjugate direction 
        phi   : Angles for L-conjugate direction 
    """

    # Build mesh from sampled points
    V, F    = sample_bspline_surface(bspline, u_sample_pts, v_sample_pts)
    num_vertices = V.shape[0]
    mesh = Mesh()
    mesh.make_mesh(V, F)

    # Extract boundary curves and corners
    boundaryCurves = mesh.boundary_split_corners(corner_tol = 0.8)
    corners = mesh.corners(corner_tol = 0.8)


    # Compute principal frame
    e1, e2, k1, k2, n = principal_frame(bspline, u_sample_pts, v_sample_pts)
    
    if isinstance(angle, float):
        # Single angle for all samples
        theta = angle * np.ones(num_vertices)
    else:
        theta = angle 
    
    # Compute L-conjugate directions
    _, _, phi = compute_lconjugate_curve_tangents(e1, e2, k1, k2, RadiusValues, theta)


    indicesBoundaryTheta = []
    valsBoundaryTheta = []
    indicesBoundaryPhi = []
    valsBoundaryPhi = []

    
    # Fix theta at the boundary
    for c, bc in enumerate(boundaryCurves):

        # Skip corners
        for i in range(0, len(bc)-1):
            vi = bc[i]
            vj = bc[i+1]

            # Edge vector
            edge_vec = unit(V[vj] - V[vi])

            # Project direction to tangent at vi
            n_i = n[vi]
            edge_vec = unit(edge_vec - np.dot(edge_vec, n_i) * n_i)

            y = np.dot(edge_vec, e2[vi])
            x = np.dot(edge_vec, e1[vi])
            angle = np.arctan2(y, x)

            if c%2 == 0:
                theta[vi] = angle
                indicesBoundaryTheta.append(vi)
                valsBoundaryTheta.append(angle)

                _, _, phi_i = compute_lconjugate_curve_tangents_at_point(e1[vi], e2[vi], k1[vi], k2[vi], RadiusValues[vi], theta[vi])
                phi[vi] = phi_i
            else:
                phi[vi] = angle
                _, _, theta_i = compute_lconjugate_curve_tangents_at_point(e1[vi], e2[vi], k1[vi], k2[vi], RadiusValues[vi], phi[vi])
                theta[vi] = theta_i
                indicesBoundaryPhi.append(vi)
                valsBoundaryPhi.append(angle)

    t1 = np.cos(theta)[:,None] * e1 + np.sin(theta)[:,None] * e2
    t2 = np.cos(phi)[:,None]   * e1 + np.sin(phi)[:,None]   * e2

    ruling1, ruling2    = exchange_lconjugate_rulings_curve_tangents(e1, e2, k1, k2, t1, t2)

    return t1, t2, ruling1, ruling2, theta, phi, indicesBoundaryTheta, valsBoundaryTheta, indicesBoundaryPhi, valsBoundaryPhi

def sphere_center_at_point(bspline, u, v, radiusPercentage):
    """
    Compute the sphere center at a given point on the B-spline surface, given the radius percentage.
    """
    _, _, k1, k2, n = principal_frame_at_point(bspline, u, v)

    r = 1/np.maximum(np.abs(k1), np.abs(k2)) * radiusPercentage

    sphere_center = bspline(u,v) -  r * n

    return sphere_center


def sphere_center_spline_fitting(bspline, uSampledPoints, vSampledPoints):
    """
    Fit radius to Bspline surface with outlier removal.
    outlier_std: remove values beyond this many standard deviations (default: 3.0)
    """
    
    u_range = (uSampledPoints[0], uSampledPoints[-1])
    v_range = (vSampledPoints[0], vSampledPoints[-1])

    sphereCenters = lambda u, v: sphere_center_at_point(bspline, u, v, radiusPercentage=1.0)

    sphereCentersSpline = fit_bspline_surface(sphereCenters, u_range, v_range, n_u=100, n_v=100, order=3)

    

    return sphereCentersSpline

def radius_spline_fitting(uSampledPoints, vSampledPoints, radiusValues, 
                        u_degree=2, v_degree=2, outlier_std=3.0):
    """
    Fit radius to Bspline surface with outlier removal.
    outlier_std: remove values beyond this many standard deviations (default: 3.0)
    """
    # Get grid points
    u_grid, v_grid = np.meshgrid(uSampledPoints, vSampledPoints, indexing='ij')
    u_grid = u_grid.flatten()
    v_grid = v_grid.flatten()
    radiusValues = radiusValues.flatten()
    
    # Remove outliers using median absolute deviation (robust to outliers)
    median = np.median(radiusValues)
    mad = np.median(np.abs(radiusValues - median))
    z_scores = 0.6745 * np.abs(radiusValues - median) / mad
    valid = z_scores < outlier_std
    
    # Fit filtered data
    radiusFunction = bisplrep(u_grid[valid], v_grid[valid], radiusValues[valid], 
                              kx=u_degree, ky=v_degree)
    
    return radiusFunction


def fit_bspline_surface(func, u_range, v_range, n_u=30, n_v=100, order=5):
    """Fit a B-spline surface to a parametric function.

    Parameters
    ----------
    func     : callable(u, v) -> (x, y, z)
    u_range  : (u_min, u_max)
    v_range  : (v_min, v_max)
    n_u, n_v : number of interpolation points per direction
    order    : spline order (degree + 1), default 4 (cubic)

    Returns
    -------
    splipy.Surface
    """
    # Build open uniform knot vectors over the parameter range
    def _make_knots(order, n_funcs, start, end):
        n_internal = n_funcs - order
        knots = ([start] * order
                 + list(np.linspace(start, end, n_internal + 2)[1:-1])
                 + [end] * order)
        return knots

    basis_u = sp.BSplineBasis(order=order, knots=_make_knots(order, n_u, *u_range))
    basis_v = sp.BSplineBasis(order=order, knots=_make_knots(order, n_v, *v_range))

    # Greville abscissae — optimal interpolation sites for B-splines
    u_pts = basis_u.greville()
    v_pts = basis_v.greville()

    # Evaluate the parametric function on the tensor-product grid
    pts = np.empty((n_u, n_v, 3))
    for i, u in enumerate(u_pts):
        for j, v in enumerate(v_pts):
            pts[i, j] = func(u, v)

    # Interpolate
    surf = surface_factory.interpolate(pts, [basis_u, basis_v], [u_pts, v_pts])
    return surf

def fit_bspline_grid_mesh(vertices, order=5):
    """Fit a B-spline surface to a grid of vertices.

    Parameters
    ----------
    vertices : array_like, shape (n_u, n_v, 3)
        Grid of 3D points
    u_params : array_like, shape (n_u,), optional
        Parameter values in u direction. If None, uses [0, 1, ..., n_u-1]
    v_params : array_like, shape (n_v,), optional
        Parameter values in v direction. If None, uses [0, 1, ..., n_v-1]
    order    : int, default 5
        Spline order (degree + 1)

    Returns
    -------
    splipy.Surface
    """
    vertices = np.asarray(vertices)
    n_u, n_v = vertices.shape[:2]
    
    
    u_params = np.arange(n_u, dtype=float)
    v_params = np.arange(n_v, dtype=float)
    
    u_range = (u_params[0], u_params[-1])
    v_range = (v_params[0], v_params[-1])
    
    # Build knot vectors
    def _make_knots(order, n_funcs, start, end):
        n_internal = n_funcs - order
        knots = ([start] * order
                 + list(np.linspace(start, end, n_internal + 2)[1:-1])
                 + [end] * order)
        return knots

    basis_u = sp.BSplineBasis(order=order, knots=_make_knots(order, n_u, *u_range))
    basis_v = sp.BSplineBasis(order=order, knots=_make_knots(order, n_v, *v_range))

    # Interpolate
    surf = surface_factory.interpolate(vertices, [basis_u, basis_v], [u_params, v_params])
    return surf

def sample_radius(radiusFunction, u_pts, v_pts):
    """
    Sample the radius Bspline function at given u,v points
    Input:
        radiusFunction: Bspline function of the radius
    Output:
        r_uv: sampled radius values
    """
    # Evaluate the radius function at the given points, get only the radius component (z)
    radii = radiusFunction(u_pts, v_pts).reshape(-1,3)[:,2] 
    return radii

def compute_rij_transport_vectorized(ei_1, ej_1, ni, nj):
    """
    Parallel transport ei_1 from plane ni to plane nj,
    then compute angle to ej_1.

    Parameters:
    -----------
    ei_1 : np.ndarray, shape (N, 3)
        Reference vectors in tangent plane at i
    ej_1 : np.ndarray, shape (N, 3)
        Reference vectors in tangent plane at j
    ni : np.ndarray, shape (N, 3)
        Unit normals at i
    nj : np.ndarray, shape (N, 3)
        Unit normals at j

    Returns:
    --------
    rij : np.ndarray, shape (N,)
        Rotation angles
    """
    # Rotation axis: ni x nj
    axis = np.cross(ni, nj)
    sin_angle = np.linalg.norm(axis, axis=1, keepdims=True)
    cos_angle = np.sum(ni * nj, axis=1, keepdims=True)

    # Normalize axis (handle parallel planes)
    safe_sin = np.where(sin_angle > 1e-10, sin_angle, 1.0)
    axis_normalized = axis / safe_sin

    # Rodrigues rotation: transport ei_1 to plane nj
    # v_rot = v*cos + (axis x v)*sin + axis*(axis·v)*(1-cos)
    axis_cross_ei = np.cross(axis_normalized, ei_1)
    axis_dot_ei = np.sum(axis_normalized * ei_1, axis=1, keepdims=True)

    ei_transported = (ei_1 * cos_angle +
                      axis_cross_ei * sin_angle +
                      axis_normalized * axis_dot_ei * (1 - cos_angle))

    # Handle parallel planes (sin_angle ≈ 0)
    parallel_same = (sin_angle < 1e-10) & (cos_angle > 0)
    parallel_opposite = (sin_angle < 1e-10) & (cos_angle <= 0)

    ei_transported = np.where(parallel_same, ei_1, ei_transported)
    ei_transported = np.where(parallel_opposite, -ei_1, ei_transported)

    # Compute signed angle in plane nj
    cos_r = np.sum(ei_transported * ej_1, axis=1)
    cross_ej = np.cross(ei_transported, ej_1)
    sin_r = np.sum(cross_ej * nj, axis=1)

    rij = np.arctan2(sin_r, cos_r)

    return rij


def compute_crossfield_singularities(mesh, theta, t1, normals, symmetry=4):
    """
    Compute cross-field singularity indices at each vertex.

    Singularities are points where the cross-field cannot be defined smoothly.
    The index at each vertex is computed by summing angle defects around its 1-ring,
    accounting for parallel transport between tangent planes.

    For a 4-symmetric cross-field (standard cross-field):
    - Regular vertices have index 0
    - Positive singularities have index +1/4 (sources)
    - Negative singularities have index -1/4 (saddles)

    By Gauss-Bonnet theorem: sum of all indices = Euler characteristic χ(M)

    Parameters:
    -----------
    mesh : Mesh
        The mesh object with vertices, faces, and topology
    theta : np.ndarray, shape (N,)
        Direction field angles at each vertex (angle of first direction w.r.t. t1)
    t1 : np.ndarray, shape (N, 3)
        Reference tangent vectors at each vertex (e.g., principal direction e1)
    normals : np.ndarray, shape (N, 3)
        Unit normals at each vertex
    symmetry : int, optional
        Symmetry order of the field (default=4 for cross-fields)

    Returns:
    --------
    indices : np.ndarray, shape (N,)
        Singularity index at each vertex
    singularity_vertices : np.ndarray
        Indices of vertices that are singularities (|index| > threshold)
    singularity_indices : np.ndarray
        Index values at singularity vertices
    """
    num_vertices = mesh.V
    indices = np.zeros(num_vertices)

    # Get ordered vertex adjacency (1-ring neighbors in order)
    adjacency = mesh.vertex_adjacency_list()

    # Get edge information for parallel transport lookup
    ei, ej = mesh.edge_vertices()

    # Build a dictionary for quick edge lookup: (vi, vj) -> edge_index
    edge_dict = {}
    for idx, (i, j) in enumerate(zip(ei, ej)):
        edge_dict[(i, j)] = idx
        edge_dict[(j, i)] = idx

    # Precompute all parallel transport angles
    alpha_all = compute_rij_transport_vectorized(
        t1[ei].reshape(-1, 3),
        t1[ej].reshape(-1, 3),
        normals[ei].reshape(-1, 3),
        normals[ej].reshape(-1, 3)
    )

    # Create transport lookup: edge_dict key -> transport angle (from i to j)
    transport_ij = {}
    for idx, (i, j) in enumerate(zip(ei, ej)):
        transport_ij[(i, j)] = alpha_all[idx]
        transport_ij[(j, i)] = -alpha_all[idx]  # Reverse direction

    # Compute index at each vertex
    for vi in range(num_vertices):
        neighbors = adjacency[vi]
        if len(neighbors) < 2:
            continue

        angle_sum = 0.0
        n_neighbors = len(neighbors)

        for k in range(n_neighbors):
            vj = neighbors[k]
            vk = neighbors[(k + 1) % n_neighbors]

            # Get parallel transport from vj to vk
            if (vj, vk) in transport_ij:
                alpha_jk = transport_ij[(vj, vk)]
            else:
                # Compute on the fly if not in precomputed (shouldn't happen for manifold mesh)
                alpha_jk = compute_rij_transport_vectorized(
                    t1[vj].reshape(1, 3),
                    t1[vk].reshape(1, 3),
                    normals[vj].reshape(1, 3),
                    normals[vk].reshape(1, 3)
                )[0]

            # Angle difference with parallel transport correction
            # For n-symmetry field, multiply by n
            diff = symmetry * (theta[vk] - theta[vj] - alpha_jk)

            # Wrap to [-pi, pi]
            diff = np.arctan2(np.sin(diff), np.cos(diff))
            angle_sum += diff

        # Index = total rotation / (2π), should be multiple of 1/symmetry
        indices[vi] = angle_sum / (2 * np.pi)

    # Find singularities (vertices with non-zero index)
    # Use threshold to account for numerical errors
    threshold = 0.1 / symmetry
    singular_mask = np.abs(indices) > threshold
    singularity_vertices = np.where(singular_mask)[0]
    singularity_indices = indices[singularity_vertices]

    # Round indices to nearest multiple of 1/symmetry for cleaner output
    singularity_indices = np.round(singularity_indices * symmetry) / symmetry

    return indices, singularity_vertices, singularity_indices


def visualize_singularities(name, singularity_vertices, singularity_indices,
                            vertex_positions, radius=0.015):
    """
    Visualize cross-field singularities in Polyscope.

    Parameters:
    -----------
    name : str
        Name for the Polyscope point cloud
    singularity_vertices : np.ndarray
        Indices of singularity vertices
    singularity_indices : np.ndarray
        Index values at singularity vertices (+1/4 or -1/4 typically)
    vertex_positions : np.ndarray, shape (N, 3)
        Vertex positions
    radius : float
        Radius of singularity markers

    Returns:
    --------
    ps_cloud : polyscope point cloud
        The registered point cloud object
    """
    import polyscope as ps

    if len(singularity_vertices) == 0:
        print(f"No singularities found in {name}")
        return None

    # Get positions of singularities
    sing_positions = vertex_positions[singularity_vertices]

    # Create colors based on index sign (red for positive, blue for negative)
    colors = np.zeros((len(singularity_vertices), 3))
    for i, idx in enumerate(singularity_indices):
        if idx > 0:
            colors[i] = [1.0, 0.2, 0.2]  # Red for positive (source-like)
        else:
            colors[i] = [0.2, 0.2, 1.0]  # Blue for negative (saddle-like)

    # Register point cloud
    ps_cloud = ps.register_point_cloud(name, sing_positions, radius=radius)
    ps_cloud.add_color_quantity("index_sign", colors, enabled=True)
    ps_cloud.add_scalar_quantity("index_value", singularity_indices, enabled=False)

    # Print summary
    n_positive = np.sum(singularity_indices > 0)
    n_negative = np.sum(singularity_indices < 0)
    total_index = np.sum(singularity_indices)

    print(f"Cross-field singularities ({name}):")
    print(f"  Total singularities: {len(singularity_vertices)}")
    print(f"  Positive (index +1/4): {n_positive}")
    print(f"  Negative (index -1/4): {n_negative}")
    print(f"  Sum of indices: {total_index:.3f} (should equal Euler characteristic)")

    return ps_cloud


def control_points_grid(bspline):
    """
    Visualize the control points of a B-spline surface using Polyscope.
    Return the control points as a flattened array and the edges connecting them.
    Input:
        bspline: splipy Surface object
    Returns:
        control_points_flat: Flattened array of control points
        edges: Array of edges connecting control points
    """


    number_uPts, number_vPts, _ = bspline.controlpoints.shape

    # Create edges for visualization of control mesh (Grid like)
    edges = []
    for i in range(number_uPts):
        for j in range(number_vPts - 1):
            idx1 = i * number_vPts + j
            idx2 = i * number_vPts + (j + 1)
            edges.append([idx1, idx2])
    for j in range(number_vPts):
        for i in range(number_uPts - 1):
            idx1 = i * number_vPts + j
            idx2 = (i + 1) * number_vPts + j
            edges.append([idx1, idx2])

    edges = np.array(edges)
    control_points_flat = bspline.controlpoints.reshape(-1, 3)

    # Faces for visualization of control mesh (Grid like)
    faces = []
    for i in range(number_uPts - 1):
        for j in range(number_vPts - 1):
            idx1 = i * number_vPts + j
            idx2 = i * number_vPts + (j + 1)
            idx3 = (i + 1) * number_vPts + (j + 1)
            idx4 = (i + 1) * number_vPts + j
            faces.append([idx1, idx2, idx3, idx4])

    return control_points_flat, edges, faces

def search_nearest_uv(bspline, points):
    """
    Search the nearest (u,v) parameters on the B-spline surface for given 3D points.
    Input:
        bspline: splipy Surface object
        points: np.array of 3D points (N x 3)
    Returns:
        uv_params: np.array of (u,v) parameters (N x 2)
    """

    # Sample the B-spline surface densely to create a KDTree
    V_sampled, F_sampled, uv_F = bspline_to_mesh(bspline, 100, 100)

    TF_sampled = np.array(triangulate_quads(F_sampled))

    points = np.array(points).reshape(-1,3)

    _, faces_id, _ = igl.point_mesh_squared_distance(np.array(points), np.array(V_sampled), TF_sampled)

    # check if points is single or multiple

    if points.shape[0] == 1:
        
        if faces_id % 2 != 0:
            faces_id = (faces_id + 1) // 2
            uv_params = uv_F[faces_id][0]
        elif faces_id % 2 == 0:
            faces_id = faces_id // 2
            uv_params = uv_F[faces_id][0]
    
    else:

        uv_params = []

        for f_id in faces_id:
            if f_id % 2 != 0:

                uv_params.append(uv_F[(f_id + 1) // 2][0])
            elif f_id % 2 == 0:
                uv_params.append(uv_F[f_id // 2][0])    

    return np.array(uv_params)

def principal_frame_at_point(bspline, u, v):
    """
    Compute the principal frame at a given (u,v) parameter on the B-spline surface."""

    # First derivatives
    du  = bspline.derivative(np.array([u]), np.array([v]), d=(1, 0)).reshape(3)
    dv  = bspline.derivative(np.array([u]), np.array([v]), d=(0, 1)).reshape(3)
    # Second derivatives
    duv = bspline.derivative(np.array([u]), np.array([v]), d=(1, 1)).reshape(3)
    duu = bspline.derivative(np.array([u]), np.array([v]), d=(2, 0)).reshape(3)
    dvv = bspline.derivative(np.array([u]), np.array([v]), d=(0, 2)).reshape(3)

    # Compute the normal
    n = bspline.normal(np.array([u]), np.array([v])).reshape(3)

    # First fundamental form
    E = np.dot(du, du)
    F = np.dot(du, dv)
    G = np.dot(dv, dv)

    # Second fundamental form
    L = np.dot(duu, n)
    M = np.dot(duv, n)
    N = np.dot(dvv, n)


    I = np.array([[E, F],
                      [F, G]])
        
    II = np.array([[L, M],
                       [M, N]])


    # Shape operator
    S = np.linalg.inv(I).dot(II)

    # Eigen decomposition for principal curvatures and directions
    evals, evecs = np.linalg.eig(S)

    # Eigen decomposition
    eigvals, eigvecs = np.linalg.eig(S)
    # Sort eigenvalues and eigenvectors
    idx = np.argsort(eigvals)
    k1, k2 = eigvals[idx]
    v1, v2 = eigvecs[:, idx].T

    e1 = unit(v1[0]*du + v1[1]*dv)
    e2 = unit(v2[0]*du + v2[1]*dv)

    # Check orientation
    if np.dot(np.cross(e1, e2), n) < 0:
        e2 = -e2



    return e1, e2, k1, k2, n  

def compute_lconjugate_at_point(t1, t2, k1, k2, r, theta):
    """
    Compute the L-conjugate directions at a given point on the surface.
    Input:
        t1: First tangent direction (principal direction)
        t2: Second tangent direction (principal direction)
        k1: First principal curvature
        k2: Second principal curvature
        r: Radius of the central sphere
        theta: Angle of the first L-conjugate direction with respect to t1
    Returns:
        ruling1: First L-conjugate direction
        ruling2: Second L-conjugate direction
    """ 

    

    a = np.array([np.cos(theta), 
                  np.sin(theta)])
    
    J = np.array([[0, -1],
                  [1,  0]])
      
    M = np.array([[ (1.0/k2) - r, 0],
                    [0, (1.0/k1) - r ]
                ])
    
    b = unit( J @ M @ a)

    phi = np.arctan2(b[1], b[0])

    ruling1 = unit(np.cos(theta) * t1 + np.sin(theta) * t2)
    ruling2 = unit(np.cos(phi)   * t1 + np.sin(phi)   * t2)

    return ruling1, ruling2, theta, phi

def compute_lconjugate_curve_at_point(t1, t2, k1, k2, r, theta, phi):
    """
    Compute the L-conjugate directions at a given point on the surface.
    Input:
        t1: First tangent direction (principal direction)
        t2: Second tangent direction (principal direction)
        k1: First principal curvature
        k2: Second principal curvature
        r: Radius of the central sphere
        theta: Angle of the first L-conjugate direction with respect to t1
    Returns:
        cD1: First L-conjugate direction for curve integration
        cD2: Second L-conjugate direction for curve integration
    """

    a = np.array([np.cos(theta),
                    np.sin(theta)])
    
    b = np.array([np.cos(phi),
                    np.sin(phi)])
    
    J = np.array([[0, -1],
                    [1,  0]])
    
    M = np.array([[k1, 0],
                        [0, k2]])
        
    aC = unit( J @ M @ a )
    bC = unit( J @ M @ b )

    # Verify L-conjugacy condition
    MC = np.array([[r * k1**2 - k1, 0],
                    [0, r * k2**2 - k2]])
    
    #print("L-conjugacy check (should be close to 0):", aC @ MC @ bC)
            
    cD1 = unit(aC[0] * t1 + aC[1] * t2)
    cD2 = unit(bC[0] * t1 + bC[1] * t2)

    return cD1, cD2
    

def integration_step(bspline, point, dir, step_size):
    """
    Perform a single integration step on the B-spline surface along the given direction.
    Input:
        bspline: splipy Surface object
        point: Current point on the surface (3D vector)
        dir: Direction vector in 3D (tangent to the surface)
    Returns:
        u, v: New (u,v) parameters after the integration step
    """

    point = np.array(point).reshape(1,3)
    
    nextPoint = point + step_size * dir

    uv_parameter = project_points_to_bspline(bspline, nextPoint)
    return uv_parameter[0,0], uv_parameter[0,1]


def integrate_lconjugate_dir(bspline, radiusFunc, tangent1_angle, u_start, v_start, step_size=0.01, num_steps=20):
    """
    Integrate the L-conjugate direction field on the B-spline surface starting from (u_start, v_start).
    Input:
        bspline: splipy Surface object
        u_start: Starting u parameter
        v_start: Starting v parameter
        tangent1_angle: Starting angle of the first L-conjugate direction with respect to the first principal direction
        step_size: Step size for integration
        num_steps: Number of integration steps
    Returns:
        curve_points: List of 3D points along the integrated curve
    """

    curve_points = []
    u_curr = u_start
    v_curr = v_start
    
    runOpt = True
    step_count = 0

    uv_values = []
    centers = []

    while runOpt and step_count <= num_steps:
        # Get current point on surface
        p_curr = bspline(u_curr, v_curr)
        r_curr = radiusFunc(u_curr, v_curr)[2]  # Get radius value at current (u,v)
        curve_points.append(p_curr)
        uv_values.append([u_curr, v_curr])

        # Get principal frame at current (u,v)
        e1, e2, k1, k2, n = principal_frame_at_point(bspline, u_curr, v_curr)

        centers.append(p_curr + r_curr * n)

        t1, t2, _ = compute_lconjugate_curve_tangents_at_point(e1, e2, k1, k2, r_curr, tangent1_angle)

        # integration_step
        u_curr, v_curr = integration_step(bspline, p_curr, t1, step_size)
        step_count += 1

        if u_curr < 0 or u_curr > 1 or v_curr < 0 or v_curr > 1:
            runOpt = False

    return np.array(curve_points) , np.array(centers), np.array(uv_values)

def project_points_to_bspline(bspline, points):
    """
    Point tha a 3D point to a B-spline surface, returning the (u,v) parameters.
    """

    uv_params = []

    points = np.array(points).reshape(-1,3)

    initial_uv = np.atleast_2d(search_nearest_uv(bspline, points))

    for i, p in enumerate(points):
        # Initial guess for (u,v)
        
        u0, v0 = initial_uv[i]
        def objective(uv):
            sp = bspline(uv[0], uv[1])
            return np.linalg.norm(sp - p)**2

        res = minimize(objective, x0=[u0, v0], bounds=[(0, 1), (0, 1)], method='L-BFGS-B')
        uv_params.append(res.x)
        #print(res)

    return np.array(uv_params)


def save_bspline_to_json(bspline, filepath):
    """
    Save a splipy B-spline surface to JSON format compatible with Rhino import.
    
    Input:
        bspline: splipy Surface object
        filepath: Path where to save the JSON file
    """
    
    # Get the basis functions
    basis_u = bspline.bases[0]
    basis_v = bspline.bases[1]
    
    # Get degree (order - 1)
    degree_u = basis_u.order - 1
    degree_v = basis_v.order - 1
    
    # Get knot vectors (remove the extra first and last knots that were added for splipy)
    knots_u = list(basis_u.knots)
    knots_v = list(basis_v.knots)
    
    # Remove duplicate first and last knots if they were added
    # In splipy, knots typically have multiplicity = order at the ends
    # For Rhino, we need the internal knot vector representation
    # Remove first and last knot
    if len(knots_u) > 2:
        knots_u = knots_u[1:-1]
    if len(knots_v) > 2:
        knots_v = knots_v[1:-1]
    
    # Get control points
    control_points = bspline.controlpoints
    
    # Get the shape
    count_u, count_v = control_points.shape[0], control_points.shape[1]
    
    # Flatten control points and add weights (assuming weight = 1.0 for rational surfaces)
    control_points_list = []
    for u in range(count_u):
        for v in range(count_v):
            pt = control_points[u, v]
            # Add point coordinates and weight
            if len(pt) == 3:
                control_points_list.append([float(pt[0]), float(pt[1]), float(pt[2]), 1.0])
            elif len(pt) == 4:  # Already has weight
                control_points_list.append([float(pt[0]), float(pt[1]), float(pt[2]), float(pt[3])])
    
    # Create the data dictionary
    data = {
        "degreeU": int(degree_u),
        "degreeV": int(degree_v),
        "countU": int(count_u),
        "countV": int(count_v),
        "knotsU": [float(k) for k in knots_u],
        "knotsV": [float(k) for k in knots_v],
        "controlPoints": control_points_list
    }
    
    # Save to JSON file
    with open(filepath, 'w') as file:
        json.dump(data, file, indent=4)
    
    print(f"B-spline surface saved to {filepath}")
    print(f"Degree U: {degree_u}, Degree V: {degree_v}")
    print(f"Control points: {count_u} x {count_v} = {len(control_points_list)}")
    print(f"Knots U: {len(knots_u)}, Knots V: {len(knots_v)}")

def bspline_curvatures(bsp, u_pts, v_pts):
    """
    Function to compute the curvatures of the surface
    Input:
        bsp: B-spline surface
        u_pts: u points
        v_pts: v points
    Output:
        K: Gaussian curvature
        H: Mean curvature
        n: Normals
    """    
        
    # Compute normals
    #n = bsp.normal(u_pts, v_pts)

    # Compute derivatives
    # First derivatives
    du  = bsp.derivative(u_pts, v_pts,  d=(1, 0))
    dv  = bsp.derivative(u_pts, v_pts,  d=(0, 1))

    # Compute the first and second fundamental forms
    E = np.sum(du*du, axis=2)
    F = np.sum(du*dv, axis=2)
    G = np.sum(dv*dv, axis=2)

    n = np.cross(du, dv)
    # Determinant 
    det = E*G - F**2
    
    # Second derivatives
    duv = bsp.derivative(u_pts, v_pts,  d=(1, 1))
    duu = bsp.derivative(u_pts, v_pts,  d=(2, 0))
    dvv = bsp.derivative(u_pts, v_pts,  d=(0, 2))

    L = np.sum(duu*n, axis=2)/np.sqrt(det)
    M = np.sum(duv*n, axis=2)/np.sqrt(det)
    N = np.sum(dvv*n, axis=2)/np.sqrt(det)

    K = (L*N - M**2)/(E*G - F**2)
    H = (E*N + G*L - 2*F*M)/(2*(E*G - F**2))

    return K, H, n
    


def curvatures_graph(bsp, u_pts, v_pts):
    """
    Function to compute the curvatures of the surface
    """

    # Compute the first and second derivatives
    fu  = bisplev(u_pts[:,0], v_pts[0,:], bsp, dx=1, dy=0).flatten()
    fv  = bisplev(u_pts[:,0], v_pts[0,:], bsp, dx=0, dy=1).flatten()
    fuu = bisplev(u_pts[:,0], v_pts[0,:], bsp, dx=2, dy=0).flatten()
    fvv = bisplev(u_pts[:,0], v_pts[0,:], bsp, dx=0, dy=2).flatten()
    fuv = bisplev(u_pts[:,0], v_pts[0,:], bsp, dx=1, dy=1).flatten()

  
    # flatten each
    # Compute with flatten arrays
    # Reshape end result
    
    # Compute the normal
    norms_n = np.sqrt(fu**2 + fv**2 + 1)
 
    
    # Gauss curvature
    K = (fuu*fvv - fuv**2)/(norms_n**4)

    # Mean curvature
    H = (fuu - 2*fu*fuv*fv + fuu*fv*2 + fvv + fu**2*fvv)/(2*norms_n**3)

    K = K.reshape(u_pts.shape[0], v_pts.shape[0])
    H = H.reshape(u_pts.shape[0], v_pts.shape[0])

    return K, H  

def principal_frame(bsp, u_pts, v_pts):
    """
    Function to compute the principal frame per sampled points on the B-spline surface.
    Input:
        bsp: B-spline surface
        u_pts: u points
        v_pts: v points
    """

    # Compute the first and second derivatives
    du  = bsp.derivative(u_pts, v_pts,  d=(1, 0)).reshape(-1, 3)
    dv  = bsp.derivative(u_pts, v_pts,  d=(0, 1)).reshape(-1, 3)

    duu = bsp.derivative(u_pts, v_pts,  d=(2, 0)).reshape(-1, 3)
    dvv = bsp.derivative(u_pts, v_pts,  d=(0, 2)).reshape(-1, 3)
    duv = bsp.derivative(u_pts, v_pts,  d=(1, 1)).reshape(-1, 3)

    n = bsp.normal(u_pts, v_pts).reshape(-1, 3)

    # Number of points
    num_pts = u_pts.shape[0]*v_pts.shape[0]
    e1 = np.zeros((num_pts, 3))
    e2 = np.zeros((num_pts, 3))

    kappa1 = np.zeros(num_pts)
    kappa2 = np.zeros(num_pts)

    for i in range(num_pts):
        # First fundamental form
        E = np.dot(du[i], du[i])
        F = np.dot(du[i], dv[i])
        G = np.dot(dv[i], dv[i])

        # Second fundamental form
        L = np.dot(duu[i], n[i])
        M = np.dot(duv[i], n[i])
        N = np.dot(dvv[i], n[i])

        I = np.array([[E, F],
                      [F, G]])
        
        II = np.array([[L, M],
                       [M, N]])
        

        # Shape operator matrix
        S = np.linalg.inv(I).dot(II)

        # Eigen decomposition
        eigvals, eigvecs = np.linalg.eig(S)
        # Sort eigenvalues and eigenvectors
        idx = np.argsort(eigvals)
        k1, k2 = eigvals[idx]
        v1, v2 = eigvecs[:, idx].T

        e1[i] = unit(v1[0]*du[i] + v1[1]*dv[i])
        e2[i] = unit(v2[0]*du[i] + v2[1]*dv[i])

        # Check orientation
        if np.dot(np.cross(e1[i], e2[i]), n[i]) < 0:
            e2[i] = -e2[i]

        kappa1[i] = k1
        kappa2[i] = k2

    return e1, e2, kappa1, kappa2, n.reshape(-1, 3)


def compute_lconjugate_directions(e1, e2, k1, k2, r, theta):
    """
    Compute the L-conjugate directions for a given direction and angle.
    General formula    (k2^-1 -r)a1b1 + (k1^-1-r)a2b2 = 0;
    solve for b1, b2 assuming that a = cos(angle)e1 + sin(angle)e2; a1 = cos(angle), a2 = sin(angle)
    Args:
        e1: First direction vector.
        e2: Second direction vector.
        r: Ratio of the principal curvatures.
        theta: Angle in radians of the first L-conjugate direction with respect to e1.
    Returns:
        ruling1, ruling2: L-conjugate direction vectors.
    """

    # Number of directions
    n = e1.shape[0]

    ruling1 = np.zeros_like(e1)
    ruling2 = np.zeros_like(e2)

    if np.isscalar(theta):
        a = np.array([np.cos(theta), np.sin(theta)]*n).reshape(n,2)
    else:
        a = np.zeros((n,2))
        a[:,0] = np.cos(theta)
        a[:,1] = np.sin(theta)

    if np.isscalar(r):
        r = r * np.ones(n)
    else:
        r = r

    J = np.array([[0, -1],
                  [1,  0]])
    
    # Angle of the conjugate direction
    phi = np.zeros(n)

    for i in range(n):
        
        M = np.array([[ (1.0/k2[i]) - r[i], 0],
                      [0, (1.0/k1[i]) - r[i] ]
                    ])        
        b = unit(J @ M @ a[i])

    
        phi[i] = np.arctan2(b[1], b[0]) 

        ruling1[i] = unit(a[i,0] * e1[i] + a[i,1] * e2[i])
        ruling2[i] = unit(b[0]   * e1[i] + b[1]   * e2[i])
    return ruling1, ruling2, theta, phi

def export_crossfield(path, name, samplePts, ruling1, ruling2):
    """
    Export the cross field directions as lines in obj and dat as directions
    Input:
        ruling1: First direction vector.
        ruling2: Second direction vector.
    """

    n = ruling1.shape[0]
    column_major_data = ruling1.flatten(order='F')

    D1_path = os.path.join(path, name+"_D1.dat")
    D2_path = os.path.join(path, name+"_D2.dat")
    crossfield_path = os.path.join(path, name+"_crossfield_directions.dat")
    crossfieldD1_obj_path = os.path.join(path, name+"_crossFieldD1.obj")
    crossfieldD2_obj_path = os.path.join(path, name+"_crossFieldD2.obj")
    
    with open(D1_path, "w") as f:
        # Write the header: rows columns
        f.write(f"3 {n}\n")
        
        # Write the flattened data
        for value in column_major_data:
            f.write(f"{value}\n")
    
    column_major_data = ruling2.flatten(order='F')
    with open(D2_path, "w") as f:
        # Write the header: rows columns
        f.write(f"3 {n}\n")
        
        # Write the flattened data
        for value in column_major_data:
            f.write(f"{value}\n")

    with open(crossfield_path, "w") as f:
        for i in range(n):
            f.write(f"{ruling1[i,0]}, {ruling1[i,1]}, {ruling1[i,2]}, {ruling2[i,0]}, {ruling2[i,1]}, {ruling2[i,2]}\n")

    edges = [[i, i+1] for i in range(0, n*2, 2)]

    with open(crossfieldD1_obj_path, "w") as f:
        for i in range(n):
            p = samplePts[i]
            d1 = ruling1[i]*0.5

            f.write(f"v {p[0]} {p[1]} {p[2]}\n")
            f.write(f"v {p[0]+d1[0]} {p[1]+d1[1]} {p[2]+d1[2]}\n")
    
        for edge in edges:
            f.write(f"l {edge[0]+1} {edge[1]+1}\n")

    with open(crossfieldD2_obj_path, "w") as f:
        for i in range(n):
            p  = samplePts[i]
            d2 = ruling2[i]*0.5

            f.write(f"v {p[0]} {p[1]} {p[2]}\n")
            f.write(f"v {p[0]+d2[0]} {p[1]+d2[1]} {p[2]+d2[2]}\n")
    
        for edge in edges:
            f.write(f"l {edge[0]+1} {edge[1]+1}\n")


def read_bspline_json(dir):
    """
    Function to read the bspline surface from a json file
    Input: 
        dir: Direction to the json file
    """

    # Open and read the JSON file
    with open(dir, 'r') as file:
        data = json.load(file)
    
    
    # The order of the surface in each direction is the degree + 1
    order_u = data["degreeU"] + 1
    order_v = data["degreeV"] + 1
    
    # Print the order of the surface in each direction
    # 
    print("Reading B-Spline from Json:")
    print("U dir order: ", order_u )
    print("V dir order: ", order_v )

    # Get the knots
    knots_u = data["knotsU"]
    knots_v = data["knotsV"]

    # Get max knots values
    max_knots_u = knots_u[-1]
    max_knots_v = knots_v[-1]

    # Get min knots values
    min_knots_u = knots_u[0]
    min_knots_v = knots_v[0]

    # Fix knots, add one copy of the first and last knot
    knots_u = [knots_u[0]] + knots_u + [knots_u[-1]]
    knots_v = [knots_v[0]] + knots_v + [knots_v[-1]]

    # Normalize knots to the interval [0, 1]
    knots_u = (np.array(knots_u) - min_knots_u) / (max_knots_u-min_knots_u)
    knots_v = (np.array(knots_v) - min_knots_v) / (max_knots_v-min_knots_v)
    
    # Get the control points
    control_points = np.array(data["controlPoints"]).reshape(-1,4)
    control_points = np.array(control_points[:,:3])

    # Scale control points
    flat_ctrl_pts  = control_points.reshape(-1,3)
    norm_ctrl_pts  = normalize_vertices(flat_ctrl_pts, 2)
    control_points = norm_ctrl_pts

    # Create the B-splines basis
    basis_u = sp.BSplineBasis(order_u, knots_u)
    basis_v = sp.BSplineBasis(order_v, knots_v)

    # Reshape control points to (count_u, count_v, 3) in C-order
    count_u = basis_u.num_functions()
    count_v = basis_v.num_functions()
    control_points = control_points.reshape(count_u, count_v, 3)

    # Create the B-spline surface
    bspline = sp.Surface(basis_u, basis_v, control_points, raw=True)

    # Return bspline
    return bspline


def sample_grid(u_sample, v_sample, delta_u_Min=0.01, delta_u_Max=0.01, delta_v_Min=0.01, delta_v_Max=0.01):
    """
    Function to sample the grid points to be used for the Bspline surface
    """

    # Compute the grid points
    u_pts = np.linspace(0 + delta_u_Min, 1 - delta_u_Max, u_sample)
    v_pts = np.linspace(0 + delta_v_Min, 1 - delta_v_Max, v_sample)

    return u_pts, v_pts

def central_spheres(bspline, u_vals, v_vals):
    """
    Function to compute the central spheres of the surface per each point
    Input:
        bspline: Bspline surface
        u_vals: u values np.array(sample X 3)
        v_vals: v values np.array(sample X 3)
    """

    # Compute mean curvature
    _, H, n = curvatures_par(bspline, u_vals, v_vals)

    #mean_H = np.mean(H)

    # Search for H close to zero
    #idx = np.where(H < 0.0001)

    #print("idx len", len(idx[0]))

    #H0 = H.copy()

    #H[idx[0]] = mean_H

    # Compute the radius
    r_H = 1/H

    # Evaluate surface
    s_uv = bspline(u_vals, v_vals)

    # Compute the center
    c = s_uv + r_H[:,:,None]*n

    return c, r_H, H, n

    
def offset_spheres(bsp, u_vals, v_vals, offset):
    """
    Function to compute the offset spheres of the surface per each point
    Input:
        bsp: Bspline surface S(u,v): R^2 -> R^3
        u_vals: u values np.array(sample X 3)
        v_vals: v values np.array(sample X 3)
        offset: offset value
    """

    n = bsp.normal(u_vals, v_vals)

    c = bsp(u_vals, v_vals) + (offset/2)*n

    r = offset/2*np.ones((u_vals.shape[0], v_vals.shape[0]))

    return c, r

def sphere_congruence_derivatives(bsp, r_uv, u_vals, v_vals):
    """
    Function to compute the derivatives of the central spheres with respect to u and v
    Input:
        bsp: Bspline surface S(u,v): R^2 -> R^3
        ruv: (t: knots, c: control points; k: degrees) Bspline surface of the radius r(u,v): R^2 -> R 
        u_vals: u values np.array(sample X 3)
        v_vals: v values np.array(sample X 3)
    """

    # Compute the derivatives of the surface s(u,v)
    su = bsp.derivative(u_vals, v_vals, d=(1, 0))
    sv = bsp.derivative(u_vals, v_vals, d=(0, 1))

    n = bsp.normal(u_vals, v_vals)

    # Compute the derivatives of the Gauss map
    nu, nv = normal_derivatives_uv(bsp, u_vals, v_vals)
    
    # Compute the radius function r(u,v) at grid points
    r = bisplev(u_vals, v_vals, r_uv)

    # Compute the derivatives of the radius
    ru = bisplev(u_vals, v_vals, r_uv, dx=1, dy=0)
    rv = bisplev(u_vals, v_vals, r_uv, dx=0, dy=1)

    # Compute the derivatives of the center
    cu = su + ru[:,:,None]*n + r[:,:,None]*nu
    cv = sv + rv[:,:,None]*n + r[:,:,None]*nv

    return cu, cv


def normal_derivatives_uv(bsp, u_vals, v_vals):
    """
    Function tha compute the derivatives of the normal vector in the u and v directions
    Input:
        bsp: Bspline surface S(u,v): R^2 -> R^3
        u_vals: u values np.array(sample X 3)
        v_vals: v values np.array(sample X 3)
    """

    # Compute the derivatives of the surface s(u,v)
    su = bsp.derivative(u_vals, v_vals, d=(1, 0))
    sv = bsp.derivative(u_vals, v_vals, d=(0, 1))

    suu = bsp.derivative(u_vals, v_vals, d=(2, 0))
    svv = bsp.derivative(u_vals, v_vals, d=(0, 2))
    suv = bsp.derivative(u_vals, v_vals, d=(1, 1))

    # Compute the normal
    n = np.cross(su, sv)
    n_norm = np.linalg.norm(n, axis=2)[:, :, None]

    # Compute the derivatives of the normal
    # nu = f1/n_norm + f2/n_norm - n*(f1 + f2)@n)/n_norm**3
    suu_sv = np.cross(suu, sv, axis=2)
    su_suv = np.cross(su, suv, axis=2)
    suv_sv = np.cross(suv, sv, axis=2)
    su_svv = np.cross(su, svv, axis=2)


    nu = suu_sv/n_norm + su_suv/n_norm - n*np.sum((suu_sv + su_suv)*n, axis=2)[:,:,None]/n_norm**3
    nv = suv_sv/n_norm + su_svv/n_norm - n*np.sum((suv_sv + su_svv)*n, axis=2)[:,:,None]/n_norm**3

    return nu, nv


def r_uv_fitting(u_vals, v_vals, r_H, u_degree=5, v_degree=5):
    """
    Function to fit the function r(u,v) of radius of the central spheres, to a Bspline surface.
    Input:
        u_vals: u values np.array(sample X 3)
        v_vals: v values np.array(sample X 3)
        r_H: radius of the central spheres
    Output:
        r_uv: Bspline surface of the radius
    """

    # Get grid points
    u_grid, v_grid = np.meshgrid(u_vals, v_vals, indexing='ij')

    u_grid = u_grid.flatten()
    v_grid = v_grid.flatten()
    r_H = r_H.flatten()

    # Fit the radius to a Bspline surface
    r_uv = bisplrep(u_grid, v_grid, r_H, kx= u_degree, ky= v_degree)

    return r_uv


# ====================== Plotting Functions =================

def plot_scalar_value(ax, Surface, scalar, name):
    """
    Function to plot the scalar value on a surface. Meant to be used to plot the curvature values.
    Input:
        Surface       : Is a matrix u_size * v_size * 3, with n the number of points in the u direction and m the number of points in the v direction
        scalar        :  Is a matrix u_size * v_size, with the scalar value to be plotted per each point in the surface
        name          :    The name of the plot str
    """

    scalar_flatten = scalar.flatten()

    # Create a colormap
    cmap = plt.cm.RdBu # You can choose any existing colormap as a starting point
    norm = plt.Normalize(vmin=min(scalar_flatten), vmax=max(scalar_flatten))
    
    ax.plot_surface(Surface[:, :, 0], Surface[:, :, 1], Surface[:, :, 2], cmap =plt.cm.plasma,  facecolors=cmap(norm(scalar)), 
    vmin=0, vmax=1, alpha=0.8)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    # # Add colorbar for reference
    mappable = plt.cm.ScalarMappable(cmap=cmap)
    mappable.set_array(scalar_flatten)
    colorbar = plt.colorbar(mappable, ax=ax, orientation='vertical', shrink=0.6)
    colorbar.set_label(name+' Values')
    colorbar.set_ticks(np.linspace(np.min(scalar_flatten), np.max(scalar_flatten), 5))


def add_control_points(ax, control_points):
    """
    Function to add the control points to the plot
    Input:
        ax             : The axis of the plot
        control_points : The control points of the surface
    """
    ax.scatter(control_points[:, 0], control_points[:, 1], control_points[:, 2], c='r', marker='o', label='Control Points')
    # Set the aspect ratio of the plot to be equal
    min_x = np.min(control_points, axis=0)
    max_x = np.max(control_points, axis=0)
    scale = np.max(max_x - min_x)
    ax.set_box_aspect((max_x-min_x)/scale)


def plot_surface(ax, Surface, name):
    """
    Function to plot the B-spline surface
    Input:
        Surface       : Is a matrix u_size * v_size * 3, with n the number of points in the u direction and m the number of points in the v direction
        control_points: Is a matrix num_ctrl pots * 3, with the control points of the surface
        name          :    The name of the plot str
    """

    print("Surface shape: ", Surface.shape)

    # Plot the B-spline surface
    ax.plot_surface(Surface[:, :, 0], Surface[:, :, 1], Surface[:, :, 2], alpha=0.8,
    label=name)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    #ax.legend()

    #ax.set_title('B-spline surface '+name)
    

def draw_sphere(xCenter, yCenter, zCenter, r):
    #draw sphere
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:20j]
    x=np.cos(u)*np.sin(v)
    y=np.sin(u)*np.sin(v)
    z=np.cos(v)
    # shift and scale sphere
    x = r*x + xCenter
    y = r*y + yCenter
    z = r*z + zCenter

    return x, y, z


def approx_surface_from_data(file, box_size=2):
    """
    Function to approximate the surface from the data. 
    Input:
        file: File with the data
    """
    print("Reading data from file: ", file)

    # Read the data
    data = np.loadtxt(file)

    print("Data shape: ", data.shape)

    # Number of u, v pts 
    u_pts = int(data[0,0])
    v_pts = int(data[0,1])

    # Degree of u,v 
    u_degree = int(data[1,0])
    v_degree = int(data[1,1])

    # Points fitting
    #pts = normalize_vertices(data[2:], 2)
    pts = data[2:]

    # Approximate the surfaces
    geom_bsp = approximate_surface(pts, u_pts, v_pts, u_degree, v_degree)

    # Get controp points
    ctrl_pts = geom_bsp.ctrlpts

    # Knots
    u_knots = geom_bsp.knotvector_u
    v_knots = geom_bsp.knotvector_v

    # Order 
    u_order = geom_bsp.order_u
    v_order = geom_bsp.order_v


    # Create Bspline basis 
    basis_u = sp.BSplineBasis(u_order, u_knots)
    basis_v = sp.BSplineBasis(v_order, v_knots)

    # Create the B-spline surface
    bsp = sp.Surface(basis_u, basis_v, ctrl_pts)

    return bsp


# ====================== Optimization GUI Functions =================


def get_spline_data(choice_data, surface_dir, bspline_surf_name):
    """ Function to get the B-spline surface data
    Input:
        choice_data: Choice of the data type
        surface_dir: Directory of the surface
        bspline_surf_name: B-spline surface name
    Output:
        bspline: B-spline surface
    """

    if choice_data == 0:
        # Define the path to the B-spline surface
        bspline_surf_path = os.path.join(surface_dir, bspline_surf_name + ".json")
        print("bspline_surf_path:", bspline_surf_path)

        # Load the B-spline surface
        control_points, knots_u, knots_v, order_u, order_v = read_bspline_json(bspline_surf_path)

        # Scale control points
        ctrl_pts_shape = control_points.shape
        flat_ctrl_pts  = control_points.reshape(-1,3)
        norm_ctrl_pts  = normalize_vertices(flat_ctrl_pts, 2)
        control_points = norm_ctrl_pts.reshape(ctrl_pts_shape)

        # Create the B-splines basis
        basis_u = sp.BSplineBasis(order_u, knots_u) 
        basis_v = sp.BSplineBasis(order_v, knots_v) 

        # Create the B-spline surface
        bspline = sp.Surface(basis_u, basis_v, control_points)
    else:

        data_bspline =  bspline_surf_name+".dat"
        print("data_bspline:", data_bspline)
        data_bspline = os.path.join(surface_dir, data_bspline)
        bspline = approx_surface_from_data(data_bspline)

    return bspline

def init_sphere_congruence(mid_init, bspline, u_pts, v_pts, sample):
    """
    Function to get a sphere congruence.
    Input:
        mid_init: type of initialization
        bspline: B-spline surface
        u_pts: u points
        v_pts: v points
        sample: sample size
    Output:
        c: center of the spheres
        r_H: radius of the spheres
        H: mean curvature
        n: normals
    """

    if mid_init == 0:
        # Compute central spheres radius and normals
        _, r, _, n = central_spheres(bspline, u_pts, v_pts) 
    else: 
        n = bspline.normal(u_pts, v_pts)
        r = 5*np.ones((sample[0], sample[1])) 
    
    return r, n

def sample_bspline_surface(bspline, u_pts, v_pts):
    """
    Function to sample the B-spline surface at given u and v points.
    Input:
        bspline: B-spline surface
        u_pts: u points
        v_pts: v points
    Output:
        V, F: Vertices and faces of the mesh
    """
    # Mesh Visualization ========================== 
    S_flat = bspline(u_pts, v_pts).reshape(-1,3)

    sample = (len(u_pts), len(v_pts))

    # Get Grid as quad mesh V and F
    V = S_flat
    # Faces F_i = [i, i+1, sample[1]*i + i, sample[1]*i + i]
    #F = np.array([  (sample[1]*j) +np.array([ i, i+1, sample[1] + i + 1, sample[1] + i])   for j in range(sample[1] - 1)for i in range(sample[0] - 1)] )
    F = np.array([  np.array([ sample[1]*i+j, sample[1]*i+j+1, sample[1]*(i+1)+(j+1), sample[1]*(i+1)+j])   for j in range(sample[1] - 1)for i in range(sample[0] - 1)] )


    return V, F

def bspline_to_mesh(bspline, num_u_pts, num_v_pts):
    """
    Function to convert the B-spline surface to a mesh.
    Input:
        bspline: B-spline surface
        num_u_pts: number of u points
        num_v_pts: number of v points
    Output:
        V: Vertices of the mesh
        F: Faces of the mesh
        uvF: Dictionary of u,v values per face
    """


    # Sample u and v points
    u_pts = np.linspace(0, 1, num_u_pts)
    v_pts = np.linspace(0, 1, num_v_pts)

    # Mesh Visualization ========================== 
    S_flat = bspline(u_pts, v_pts).reshape(-1,3)

    sample = (len(u_pts), len(v_pts))

    # Get Grid as quad mesh V and F
    V = S_flat
    # Faces F_i = [i, i+1, sample[1]*i + i, sample[1]*i + i]
    #F = np.array([  (sample[1]*j) +np.array([ i, i+1, sample[1] + i + 1, sample[1] + i])   for j in range(sample[1] - 1)for i in range(sample[0] - 1)] )
    F = np.array([  np.array([ sample[1]*i+j, sample[1]*i+j+1, sample[1]*(i+1)+(j+1), sample[1]*(i+1)+j])   for j in range(sample[1] - 1)for i in range(sample[0] - 1)] )

      # Get dictionary of u,v values per face 
    uvF = {}
    for f_i in range(F.shape[0]):
        u_idx = F[f_i]//sample[1]
        v_idx = F[f_i]%sample[1]

        uvF[f_i] = ( (u_pts[u_idx[0]], v_pts[v_idx[0]]),
                     (u_pts[u_idx[1]], v_pts[v_idx[1]]),
                     (u_pts[u_idx[2]], v_pts[v_idx[2]]),
                     (u_pts[u_idx[3]], v_pts[v_idx[3]]) )

    return V, F, uvF


def closest_grid_points(p, grid_v):
    """
    Find the closest grid point to a given point p
    """
    kd_tree = KDTree(grid_v)

    _, i = kd_tree.query(p)

    return grid_v[i], i

def foot_points(p, V, F, u_pts, v_pts, Bsp, u_range=(0,1), v_range=(0,1)):
    """
    Find the foot point of a point p in the B-spline surface
    """
    #n = len(u_pts)
    m = len(v_pts)

    idx = np.zeros(len(p), dtype=int)


    _, f_i, vf = igl.point_mesh_squared_distance(p, V, F)

    prop_vf = np.zeros((len(p), 3))
    for i in range(len(p)):
        _, vertex_face_idx = closest_grid_points(vf[i], V[F[f_i[i]]])
        
        idx[i] = F[f_i[i],vertex_face_idx]
       
        prop_vf[i] = V[F[f_i[i],vertex_face_idx]]

    u_idx = idx//m
    v_idx = idx%m

    # print("u_idx, v_idx", u_idx[:5], v_idx[:5])


    # print("Shape F", F.shape)
    # # Test conversion
    # print("\n \n Test")
    # print("V[ u*m + v]", V[u_idx[0]*m + v_idx[0]])
    # print("upt, vpt", u_pts[73], v_pts[60])
    # print("bsp", Bsp(u_pts[73], v_pts[60]))

    x0 = np.c_[u_pts[u_idx], v_pts[v_idx]]


    # print("x0 shape", x0.shape)

    # print("bsp X0 1", Bsp(u_pts[u_idx][0], v_pts[u_idx][0]))
    # print("bsp X0", Bsp(x0[0,0], x0[1,0]))

    # Find the closest point in the B-spline surface
    # Define function to minimize 
    def min_dist(x, p):
        xu = x[0]
        xv = x[1]
        ev_b = Bsp(xu, xv)
        return np.linalg.norm(ev_b - p)
    
    for i in range(len(x0)):
        res = minimize(min_dist, x0[i], args=(p[i]), bounds=[u_range, v_range])
        x0[i] = res.x

    print("res: ", res)

    return x0, prop_vf
