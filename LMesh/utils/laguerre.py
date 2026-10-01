"""
Laguerre-geometry routines for L-meshes: discrete Laguerre channel surfaces,
spherical panels (spheres cut by their neighbours' contact circles), and
plane / isotropic-hyperplane fits to sphere families.

Moved out of hanan.geometry.geo_processing — research code built on hanan.
"""

import numpy as np
import triangle as tr

# pymeshlab is only needed for remeshing and has no wheel on some platforms —
# import it lazily so importing this module never fails because of it.
try:
    import pymeshlab as ml
except ImportError:
    ml = None

from hanan.geometry.primitives import (
    circle_3d,
    circular_arc,
    project_to_sphere,
    sphere_inversion,
)
from hanan.geometry.conical import cone_strip_mesh, oriented_cone
from hanan.geometry.construction import merge_meshes
from hanan.geometry.algebraic import unit


# Smallest 2D bounding-box area a spherical panel may have before it is treated as
# degenerate (a point rather than a surface). Below this, the per-triangle area
# constraint handed to Triangle stops being meaningful and refinement blows up.
AREA_TOL = 1e-12

# A panel can clear AREA_TOL and still be unmeshable: the guard above measures the
# *bounding box*, which a diagonal sliver inflates while its true area stays ~0.
# Triangle is asked for a quality mesh (q28) with a max-area bound, and those two
# constraints are unsatisfiable on a needle -- it inserts Steiner points forever
# without ever raising, so the RuntimeError fallback never fires. These two catch
# the shape, not just the size: below either, skip refinement and fan the corners.
POLY_AREA_TOL = 1e-12   # true (shoelace) polygon area
ASPECT_TOL    = 1e4     # bbox long side / short side


def compute_discrete_laguerre_channel_surface(centers, radii):

    cones_V_list = []
    cones_F_list = []

    spheres = []


    for i in range(len(centers)-1):
        ci = centers[i]
        ri = radii[i]
        cj = centers[i+1]
        rj = radii[i+1]
        kappai, kappaj, axis, tip, angle = oriented_cone(ci, ri, cj, rj)

        V, F = cone_strip_mesh(ci, ri, cj, rj, n=100)

        cones_V_list.append(V)
        cones_F_list.append(F)

    sph_V_list = []
    sph_F_list = []

    for i in range(1,len(centers)-1):
        
        ci = centers[i-1]
        ri = radii[i-1]
        cj = centers[i]
        rj = radii[i]
        ck = centers[i+1]
        rk = radii[i+1]

        kappai, kappaij, axis, tip, angle = oriented_cone(ci, ri, cj, rj)

        circle1 = circle_3d(kappaij[0], axis, kappaij[1], pts=100)

        kappak, kappajk, axis, tip, angle = oriented_cone(ck, rk, cj, rj)
        circle2 = circle_3d(kappajk[0], -axis, kappajk[1], pts=100)

        #sphere_panel_V,sphere_panel_F = loft_curves(circle1, circle2)

        v_len = len(circle1)
        u_len = 10
        sphere_panel_V = []
        for i in range(len(circle1)):
            dir = unit(circle2[i] - circle1[i])
            distance = np.linalg.norm(circle2[i] - circle1[i])
            sphere_panel_V.append( circle1[i][None, :] + np.linspace(0, distance, num=u_len)[:, None] * dir[None, :] )
        sphere_panel_V = np.concatenate(sphere_panel_V, axis=0)

        # Project vertices onto sphere
        sphere_panel_V = cj + rj * unit(sphere_panel_V - cj)

        sphere_panel_F = []
        # Create faces like grids 
        for i in range(v_len):
            for j in range(u_len-1):
                v0 = i*u_len + j
                v1 = ((i+1)%v_len)*u_len + j
                v2 = ((i+1)%v_len)*u_len + (j+1)
                v3 = i*u_len + (j+1)
                sphere_panel_F.append( [v0, v1, v2, v3] )
        

        # v_len = len(circle1)
        # u_len = 10
        # vertices = []
        # for i in range(len(circle1)):
        #     xlinspace = np.linspace(circle1[i,0], circle2[i,0], num=u_len)
        #     ylinspace = np.linspace(circle1[i,1], circle2[i,1], num=u_len)
        #     zlinspace = np.linspace(circle1[i,2], circle2[i,2], num=u_len)
        #     vertices.append( np.vstack((xlinspace, ylinspace, zlinspace)).T )

        # vertices = np.concatenate(vertices, axis=0)

        # # Project vertices onto sphere
        # #vertices = cj + rj * unit(vertices - cj)

        # sphereFaces = []
        # # Create faces like grids 
        # for i in range(v_len):
        #     for j in range(u_len-1):
        #         v0 = i*u_len + j
        #         v1 = ((i+1)%v_len)*u_len + j
        #         v2 = ((i+1)%v_len)*u_len + (j+1)
        #         v3 = i*u_len + (j+1)
        #         sphereFaces.append( [v0, v1, v2, v3] )
        

        spheres.append( (cj, rj) )
        sph_V_list.append(sphere_panel_V)
        sph_F_list.append(sphere_panel_F)

    
    cones_V, cones_F = merge_meshes(cones_V_list, cones_F_list)
    sph_V, sph_F = merge_meshes(sph_V_list, sph_F_list)

    return (cones_V, cones_F), (sph_V, sph_F), spheres


def save_discrete_laguerre_channel_surface(name, points, centers):
    """
    Export a discrete Laguerre channel surface (cone strips + sphere patches) as OBJ files.

    Args:
        name: Output filename prefix (files are written as name_cones.obj, name_spheres_panels.obj, etc.).
        points: Contact curve points (N, 3).
        centers: Sphere centers (N, 3); radii are derived as distance from points.
    """
    from hanan.geometry.io import write_obj

    radius = np.linalg.norm(points - centers, axis=1)

    (cones_V, cones_F), (sph_V, sph_F), spheres = compute_discrete_laguerre_channel_surface(centers, radius)

    write_obj(name+"_cones.obj", cones_V, cones_F)

    write_obj(name+"_spheres_panels.obj", sph_V, sph_F)

    contactCurve_path = name+"_contactCurve.obj"

    with open(contactCurve_path, 'w') as f:
        for p in points:
            f.write(f"v {p[0]} {p[1]} {p[2]}\n")

        for i in range(len(points)-1):
            f.write(f"l {i+1} {i+2}\n")

    centerCurve_path = name+"_centers.obj"
    with open(centerCurve_path, 'w') as f:
        for p in centers:
            f.write(f"v {p[0]} {p[1]} {p[2]}\n")

        for i in range(len(centers)-1):
            f.write(f"l {i+1} {i+2}\n")

    normals_path = name+"_normals.obj"

    with open(normals_path, 'w') as f:
        for i in range(len(points)):
            p = points[i]
            c = centers[i]
            f.write(f"v {p[0]} {p[1]} {p[2]}\n")
            f.write(f"v {c[0]} {c[1]} {c[2]}\n")
            f.write(f"l {2*i+1} {2*i+2}\n")

    sphere_data = name+"_spheres_planes.dat"

    with open(sphere_data, 'w') as f:
        for s in spheres:
            c, r = s
            f.write(f"s {c[0]} {c[1]} {c[2]} {r}\n")


def spherical_panel_improved(boundary, center, radius, target_edge_length=None, iterations=4):
    """
    Create a spherical panel with improved inversion approach.

    Args:
        boundary: Array of points defining the boundary curve (Nx3)
        center: Center of the sphere (3,)
        radius: Radius of the sphere
        target_edge_length: Target edge length as percentage (default: 2.5)
        iterations: Number of remeshing iterations (default: 4)

    Returns:
        vertices: Vertices of the mesh on the sphere
        faces: Face connectivity
    """
    

    if target_edge_length is None:
        target_edge_length = 2.5

    # Ensure boundary points are on the sphere
    boundary = project_to_sphere(boundary, center, radius)

    # Compute barycenter and project to sphere
    barycenter = np.mean(boundary, axis=0)
    barycenter = project_to_sphere(barycenter, center, radius)

    # Choose inversion center at antipodal point)
    inv_center = barycenter - 2 * (barycenter - center)
    inv_rad    = 0.8 * radius

    n_vertices = len(boundary)
    faces = [[i, (i + 1) % n_vertices, n_vertices] for i in range(n_vertices)]

    # Get vertices as the inverted points with respect to the sphere with center at inv_b
    vertices = sphere_inversion(np.vstack([boundary, barycenter]), inv_center, inv_rad)

    # Remesh the mesh
    ms = ml.MeshSet()
    vertices = np.array(vertices, dtype=np.float64)
    face_matrix = np.array(faces, dtype=np.int32)
    m = ml.Mesh(vertices, face_matrix)
    ms.add_mesh(m, "temp_mesh")

    ms.apply_filter(
            'meshing_isotropic_explicit_remeshing',
            iterations=4,
            targetlen=ml.PercentageValue(2.9),            # Adjust based on your mesh scale
        )

    # for _ in range(2):
    #     ms.apply_filter(
    #         'meshing_surface_subdivision_catmull_clark'
    #         )

    new_mesh = ms.current_mesh()
    new_vertices = new_mesh.vertex_matrix()
    new_faces = new_mesh.face_matrix()

    # Return the inversion of the new vertices
    vertices = sphere_inversion(new_vertices, inv_center, inv_rad)

    vertices = project_to_sphere(vertices, center, radius)

    return vertices, new_faces


def project_to_2d(points_3d):
    centroid = points_3d.mean(axis=0)
    centered = points_3d - centroid
    _, _, Vt = np.linalg.svd(centered)
    u, v     = Vt[0], Vt[1]
    pts_2d   = np.column_stack([centered @ u, centered @ v])
    return pts_2d, centroid, u, v


def reconstruct_3d(pts_2d, centroid, u, v):
    return centroid + pts_2d[:, 0:1] * u + pts_2d[:, 1:2] * v


def spherical_panel_no_remesh(boundary, center, radius,
                               min_angle=28, target_triangles=200):
    center     = np.asarray(center, dtype=float)
    boundary   = project_to_sphere(boundary, center, radius)
    barycenter = project_to_sphere(
        boundary.mean(axis=0, keepdims=True), center, radius)[0]
    inv_center = barycenter - 2.0 * (barycenter - center)
    inv_radius = 0.8 * radius
    plane_boundary         = sphere_inversion(boundary, inv_center, inv_radius)
    pts_2d, centroid, u, v = project_to_2d(plane_boundary)

    # Remove near-duplicate points that cause invalid segment geometry
    tol = 1e-10
    keep = np.ones(len(pts_2d), dtype=bool)
    for i in range(1, len(pts_2d)):
        if np.linalg.norm(pts_2d[i] - pts_2d[i - 1]) < tol:
            keep[i] = False
    pts_2d = pts_2d[keep]

    n = len(pts_2d)
    if n < 3:
        # Degenerate: return a flat fan from the centroid
        c3d = reconstruct_3d(np.zeros((1, 2)), centroid, u, v)
        c3d = sphere_inversion(c3d, inv_center, inv_radius)
        c3d = project_to_sphere(c3d, center, radius)
        return c3d, None

    bbox     = pts_2d.max(axis=0) - pts_2d.min(axis=0)
    area     = bbox[0] * bbox[1]
    # Guard against a degenerate (near-zero) bounding box. The threshold must stay
    # above the point where the area constraint below stops being representable:
    # Triangle receives max_area = area/target_triangles as text, and an area that
    # underflows the formatting is read as "no bound", sending it into effectively
    # unbounded refinement (it grinds rather than returning). Treat anything that
    # small as a degenerate panel — it is a point, not a surface.
    if area < AREA_TOL:
        c3d = reconstruct_3d(pts_2d[:1], centroid, u, v)
        c3d = sphere_inversion(c3d, inv_center, inv_radius)
        c3d = project_to_sphere(c3d, center, radius)
        return c3d, None

    # Degenerate-shape check. A sliver passes the bounding-box test above but
    # cannot be filled with well-shaped triangles, so quality refinement never
    # terminates. Fan the boundary corners instead: no Steiner points, no
    # refinement, always terminates, and for a panel this thin the fan and a
    # "proper" mesh are visually indistinguishable anyway.
    x, y = pts_2d[:, 0], pts_2d[:, 1]
    poly_area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    short, long_ = min(bbox), max(bbox)
    aspect = (long_ / short) if short > 0 else np.inf

    if poly_area < POLY_AREA_TOL or aspect > ASPECT_TOL:
        fan = np.array([[0, i, i + 1] for i in range(1, n - 1)], dtype=np.int32)
        return project_to_sphere(
            sphere_inversion(
                reconstruct_3d(pts_2d, centroid, u, v),
                inv_center, inv_radius),
            center, radius), fan

    max_area = area / target_triangles
    segments = np.array([[i, (i + 1) % n] for i in range(n)])
    try:
        tri_out = tr.triangulate(
            {'vertices': pts_2d, 'segments': segments},
            # %g keeps small constraints in exponent form instead of truncating
            # them to 0.0000000000, which Triangle would read as unbounded.
            f'q{min_angle}a{max_area:g}Dp'
        )
    except RuntimeError:
        # Fall back: plain constrained Delaunay without quality refinement
        try:
            tri_out = tr.triangulate(
                {'vertices': pts_2d, 'segments': segments},
                'pD'
            )
        except RuntimeError:
            return project_to_sphere(
                sphere_inversion(
                    reconstruct_3d(pts_2d, centroid, u, v),
                    inv_center, inv_radius),
                center, radius), None

    verts_2d     = tri_out['vertices']
    faces        = tri_out['triangles'].astype(np.int32)
    verts_3d     = reconstruct_3d(verts_2d, centroid, u, v)
    sphere_verts = sphere_inversion(verts_3d, inv_center, inv_radius)
    sphere_verts = project_to_sphere(sphere_verts, center, radius)
    return sphere_verts, faces


def spherical_polygon(center, radius, points_on_sphere):
    """
    Spherical polygon boundary via circular arcs through points projected onto the sphere.

    Args:
        center: Sphere center (3,).
        radius: Sphere radius.
        points_on_sphere: Polygon vertices to project (N, 3).

    Returns:
        Projected boundary points (N, 3).
    """

    center = np.array(center)
    points_on_sphere = np.array(points_on_sphere)

    # Project points to sphere
    projected_points = project_to_sphere(points_on_sphere, center, radius)

    spherePolygon = []

    for i in range(len(projected_points)):
        p0 = projected_points[i]
        p1 = projected_points[(i + 1) % len(projected_points)]
        normal = unit(np.cross(p0 - center, p1 - center))
        arc = circular_arc(center, normal, radius, p0, p1, num_points=40)
        spherePolygon.append(arc)

    spherePolygon = np.vstack(spherePolygon)

    return projected_points


def plane_tangent_spheres_approximation(centers, radii, n0=None, h0=None):
    """
    Best-fit oriented plane tangent to a set of spheres using spherical-coordinate parametrisation.

    Args:
        centers: Sphere centers (N, 3).
        radii: Sphere radii (N,).
        n0: Initial guess for normal vector (3,) — optional.
        h0: Initial guess for plane offset — optional.

    Returns:
        n: Optimised unit normal vector (3,).
        h: Optimised plane offset.
        cost: Final objective value.
    """

    from scipy.optimize import minimize

    centers = np.array(centers)
    radii = np.array(radii)

    def objective(params):
        theta, phi, h = params
        nx = np.sin(theta) * np.cos(phi)
        ny = np.sin(theta) * np.sin(phi)
        nz = np.cos(theta)
        n = np.array([nx, ny, nz])
        residuals = centers @ n + h - radii
        return np.sum(residuals**2)

    # Determine initial guess
    if n0 is not None:
        n_init = np.array(n0) / np.linalg.norm(n0)

        if h0 is not None:
            h_init = h0
        else:
            h_init = np.mean(radii - centers @ n_init)
    else:
        # Fallback to linear least squares initialization
        A = np.column_stack([centers, np.ones(len(centers))])
        x0_linear = np.linalg.lstsq(A, radii, rcond=None)[0]
        n_init = x0_linear[:3] / np.linalg.norm(x0_linear[:3])
        h_init = x0_linear[3] / np.linalg.norm(x0_linear[:3])

    # Convert to spherical coordinates
    theta_init = np.arccos(np.clip(n_init[2], -1, 1))
    phi_init   = np.arctan2(n_init[1], n_init[0])

    result = minimize(objective, [theta_init, phi_init, h_init], 
                      method='L-BFGS-B', 
                      options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-15})

    # Extract optimized parameters
    theta, phi, h = result.x
    n = np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta)
    ])

    return n, h, result.fun


def fit_isotropic_hyperplane(V):
    """
    Fit a hyperplane x3 = n·(x0, x1, x2) + h to 4-D vertices via least squares.

    Args:
        V: Vertices of shape (N, 4) or list of 4-tuples; each row is [x0, x1, x2, x3].

    Returns:
        n: Unit normal vector (3,).
        h: Offset/intercept scalar.
    """
    V = np.asarray(V, dtype=np.float64)
    X, y = V[:, :3], V[:, 3]
    
    # Solve least-squares: [X | 1] @ [n; h] = y
    coeffs = np.linalg.lstsq(np.column_stack([X, np.ones(len(X))]), y, rcond=None)[0]
    n = coeffs[:3]
    h = coeffs[3]
    
    # Normalize n to unit length
    norm_n = np.linalg.norm(n)
    if norm_n > 1e-14:
        n, h = n / norm_n, h / norm_n
    else:
        n, h = np.array([0., 0., 0.]), np.mean(y)
    
    return n, h
