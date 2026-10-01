import numpy as np
import os
import polyscope as ps
import igl
from scipy.optimize import minimize
from scipy.spatial import KDTree

def torsal_directions(lc, lu, lv, du, dv):
    """
    Function to compute the torsal directions.
    This is going to compute the torsal directions at a point vc on the surface with 
    line lc. We use the determinant of 
    [vt, lt, lc] = 0; vt = ut du + vt dv; lt = ut lu + vt lv
    We will find ut:vt
    Input:
        lc : Line at computation place (np.array)
        lu : Line derivative u (np.array)
        lv : Line derivative v (np.array)
        du : Surface derivative u (np.array)
        dv : Surface derivative v (np.array)
    """
   
    # t = ut/vt
    # We solve the quadratic equation
    # ut^2[vu, lu, lc] + ut vt ([vu, lv, lc] + [dv, lu, lc]) + vt^2 [lv, dv, lc] = 0
    # <=> ut^2 g0 + ut vt g1 + vt^2 g2 = 0
    lc = unit(lc)
    
    g0 = vec_dot(du, np.cross(lu, lc))
    g1 = vec_dot(du, np.cross(lv, lc)) + vec_dot(dv, np.cross(lu, lc))
    g2 = vec_dot(dv, np.cross(lv, lc))

    # Discriminant
    disc = g1**2 - 4*g0*g2

    # Init torsal directions
    ut1 = np.zeros(len(lc))
    vt1  = np.zeros(len(lc))
    vt2 = np.zeros(len(lc))
    ut2 = np.zeros(len(lc))

    # ut/vt =(-g1 +/- sqrt(g1^2 - 4*g0*g2))/2*g0
    # ut = (-g1 +/- sqrt(g1^2 - 4*g0*g2))
    # vt = 2*g0

    # For disc > 0
    idx = np.where(disc > 1e-8)[0]
    ut1[idx] = (-g1[idx] + np.sqrt(disc[idx]))
    ut2[idx] = (-g1[idx] - np.sqrt(disc[idx]))
    vt1[idx]  = vt2[idx] =  2*g0[idx]
    
    # For disc < 0
    idx = np.where(disc <= 1e-8)[0]
    print(f"Disc < 0 : {len(idx)}")
    if idx.size > 0:
        opt_t1, opt_t2 = approximate_torsal(lc[idx], lu[idx], lv[idx], du[idx], dv[idx])

        ut1[idx]  = opt_t1[:,0]
        vt1[idx]  = opt_t1[:,1]

        ut2[idx]  = opt_t2[:,0]
        vt2[idx]  = opt_t2[:,1]


    t1 = ut1[:,None]*du + vt1[:,None]*dv
    t2 = ut2[:,None]*du + vt2[:,None]*dv

    

    ut1 /= np.linalg.norm(t1, axis=1)
    vt1 /= np.linalg.norm(t1, axis=1)

    ut2 /= np.linalg.norm(t2, axis=1)
    vt2 /= np.linalg.norm(t2, axis=1)

    t1 /= np.linalg.norm(t1, axis=1)[:, None]
    t2 /= np.linalg.norm(t2, axis=1)[:, None]
   
    return t1, t2, ut1, vt1, ut2, vt2, idx
    
    
def lc_info_at_grid_points(l):
    """ Function to compute the line congruence information for torsal computations.
    Input:
        l: Line congruence (u,v,3)
    Return:
        lc: Line congruence at the baricenter of the faces
        lu: Line congruence u direction
        lv: Line congruence v direction
    """
    # Get line congruence at grid points
    l0 = l[:-1,:-1]
    l1 = l[:-1,1:]
    l2 = l[1:,1:]
    l3 = l[1:,:-1]

    lc = (l0 + l1 + l2 + l3)/4

    # lu 
    lu = l2 - l0 
    lv = l1 - l3

    return lc, lu, lv



def torsal_dir_vec(tv, tf, e_i):
    
    # Get vertices
    vi, vj, vk = tv[tf[:,0]], tv[tf[:,1]], tv[tf[:,2]]  

    # Get directions
    ei, ej, ek = e_i[tf[:,0]], e_i[tf[:,1]], e_i[tf[:,2]]

    # Compute edges
    eij = ej - ei
    eik = ek - ei

    # Compute vectors
    vij = vj - vi
    vik = vk - vi

    # Compute barycenter of directions
    ec = np.sum( e_i[tf], axis = 1)/3

    # Compute barycenter of vertices
    barycenters = (vi + vj+ vk)/3

    # Cross products
    vijXec = np.cross(vij, ec)
    vikXec = np.cross(vik, ec)

    # Compute coefficients
    g0 = np.sum(eij*vijXec, axis=1)
    g1 = np.sum(eij*vikXec, axis=1) + np.sum(eik*vijXec, axis=1)
    g2 = np.sum(eik*vikXec, axis=1)

    # Compute discriminant
    disc = g1**2 - 4*g0*g2
    

    # Get indices with negative discriminant, positive discriminant and small discriminant (i.e almost zero)
    negative_disc = np.where((disc < 0) & (np.abs(disc) > 1e-7))
    zero_disc = np.where((np.abs(disc) <= 1e-7))
    pos_disc = np.where((disc > 0) & (np.abs(disc) > 1e-7))

    # check disjoint sets
    assert len(np.intersect1d(negative_disc, zero_disc)) == 0
    assert len(np.intersect1d(negative_disc, pos_disc)) == 0
    assert len(np.intersect1d(zero_disc, pos_disc)) == 0

    # Init torsal directions
    t1 = np.zeros((len(disc), 3))
    t2 = np.zeros((len(disc), 3))

    # For discriminant near zero < 1e-7
    t1[zero_disc] = (-g1[zero_disc] + np.sqrt(abs(disc[zero_disc])))[:, None] * vij[zero_disc] + (2 * g0[zero_disc])[:, None] * vik[zero_disc]
    t2[zero_disc] = (-g1[zero_disc] - np.sqrt(abs(disc[zero_disc])))[:, None] * vij[zero_disc] + (2 * g0[zero_disc])[:, None] * vik[zero_disc]

    # For negative discriminant with absolute value > 1e-5
    t1[negative_disc] = np.zeros((len(negative_disc), 3))
    t2[negative_disc] = np.zeros((len(negative_disc), 3))

    # For positive discriminant > 1e-5
    t1[pos_disc] = (-g1[pos_disc] + np.sqrt(disc[pos_disc]))[:, None] * vij[pos_disc] + (2 * g0[pos_disc])[:, None] * vik[pos_disc]
    t2[pos_disc] = (-g1[pos_disc] - np.sqrt(disc[pos_disc]))[:, None] * vij[pos_disc] + (2 * g0[pos_disc])[:, None] * vik[pos_disc]

    # Normalize
    t1[zero_disc] /= np.linalg.norm(t1[zero_disc], axis=1)[:, None]
    t2[zero_disc] /= np.linalg.norm(t2[zero_disc], axis=1)[:, None]

    t1[pos_disc] /= np.linalg.norm(t1[pos_disc], axis=1)[:, None]
    t2[pos_disc] /= np.linalg.norm(t2[pos_disc], axis=1)[:, None]

    # Init cosines vectors
    cos_tors = np.zeros(len(disc))

    # Compute cross products
    t1Xec = np.cross(t1, ec)
    t2Xec = np.cross(t2, ec)

    # Get indices of nonzero vectors
    nonzeroInd = np.where(np.linalg.norm(t1Xec, axis=1) * np.linalg.norm(t2Xec, axis=1) >= 1e-7)

    # Compute cosines for nonzero vectors
    cos_tors[nonzeroInd] = np.sum( abs(t1Xec[nonzeroInd] * t2Xec[nonzeroInd]), axis=1) / (np.linalg.norm(t1Xec[nonzeroInd], axis=1) * np.linalg.norm(t2Xec[nonzeroInd], axis=1))

    # Compute cosines for zero vectors
    cos_tors[np.where(np.linalg.norm(t1Xec, axis=1) * np.linalg.norm(t2Xec, axis=1) < 1e-7)] = -1

    # print(f"f : 214 \n  disc : {disc[214]} \n ei : {ei[214]} \n ej : {ej[214]} \n ek : {ek[214]} \n eij : {eij[214]} \n eik : {eik[214]} \n vij : {vij[214]} \n vik : {vik[214]} \n g0 : {g0[214]} \n g1 : {g1[214]} \n g2 : {g2[214]} \n t1 : {t1[214]} \n t2 : {t2[214]} \n vijXec : {vijXec[214]} \n vikXec : {vikXec[214]} \n ec : {ec[214]} \n barycenters : {barycenters[214]} \n cos_tors : {cos_tors[214]}")

    return  barycenters, t1, t2, cos_tors


def solve_torsal(vi, vj, vk, ei, ej, ek) :
    """ Function to solve the torsal directions analytically
    Input:
        vi, vj, vk: vertices
        vvi, vvj, vvk: second envelope vertices
    """

    # Get edges
    vij = vj - vi 
    vik = vk - vi

    eij = ej - ei 
    eik = ek - ei
    

    ec = (ei + ej + ek)/3

    vijxec = np.cross(vij, ec)
    vikxec = np.cross(vik, ec)

    # g0 
    g0 = np.sum(eij*vijxec, axis=1)

    # g1
    g1 = np.sum(eij*vikxec, axis=1) + np.sum(eik*vijxec, axis=1)

    # g2
    g2 = np.sum(eik*vikxec, axis=1)


    disc = g1**2 - 4*g0*g2 

    t1 = np.zeros_like(vij)
    t2 = np.zeros_like(vij)

    a1 = np.zeros(len(vij))
    a2 = np.zeros(len(vij))
    b1 = np.zeros(len(vij))

    # Valid
    valid = np.zeros(len(vij))

    # indices disc >0 
    idx = np.where(disc >= 0)[0]

    a1[idx] = (-g1[idx] + np.sqrt(g1[idx]**2 - 4*g0[idx]*g2[idx]))
    a2[idx] = (-g1[idx] - np.sqrt(g1[idx]**2 - 4*g0[idx]*g2[idx]))
    b1[idx] = 2*g0[idx]

    # sol
    t1[idx] = (-g1[idx] + np.sqrt(g1[idx]**2 - 4*g0[idx]*g2[idx]))[:, None]*vij[idx] + 2*g0[idx,None]*vik[idx]
    t2[idx] = (-g1[idx] - np.sqrt(g1[idx]**2 - 4*g0[idx]*g2[idx]))[:, None]*vij[idx] + 2*g0[idx,None]*vik[idx]

    # Normalize
    t1[idx] = unit(t1[idx])
    t2[idx] = unit(t2[idx])

    # Put 1 on valid disc
    valid[idx] = 1

    # For disc < 0 we approximate the solution
    app_idx = np.where(disc < 0)[0]
    for i in app_idx:
        a1[i] = approximate_torsal(100, g0[i], g1[i], g2[i])
        a2[i] = approximate_torsal(-100, g0[i], g1[i], g2[i])
        b1[i] = 1

        t1[i] = unit(a1[i]*vij[i] + b1[i]*vik[i])

        t2[i] = unit(a2[i]*vij[i] + b1[i]*vik[i])

    return t1, t2, a1, a2, b1, valid



def vv_second(vvi, vvj, vvk, f, numV):
    """ Compute second envelope 
    """

    vv = np.zeros((numV, 3))
    nv = np.zeros(numV)

    for i in range(len(f)):      
        vv[f[i,0]] += vvi[i]
        vv[f[i,1]] += vvj[i]
        vv[f[i,2]] += vvk[i]

        nv[f[i,0]] += 1
        nv[f[i,1]] += 1
        nv[f[i,2]] += 1
    
    vv /= nv[:, None]

    return vv




def compute_disc(tv, tf, e_i):
    """ Function to compute the discriminant of the torsal directions
    Input:
        tv: vertices
        tf: faces
        e_i: edge directions normalized
    """


    # # Compute the edge vectors per each face
    vi, vj, vk = tv[tf[:,0]], tv[tf[:,1]], tv[tf[:,2]]

    # # Compute the edge vectors per each face
    vij = vj - vi
    vik = vk - vi

    # Set up X 
    eij = e_i[tf[:,1]] - e_i[tf[:,0]]
    eik = e_i[tf[:,2]] - e_i[tf[:,0]]

    ec = np.sum( e_i[tf], axis = 1) / 3

    # A = [vij, eik, ec] + [eij, vik, ec], where [ , , ] denotes determinant
    # A = gamma11 +  gamma12
    eikXec = np.cross(eik, ec)
    vikXec = np.cross(vik, ec)

    det1 = np.sum(vij*eikXec, axis=1)
    det2 = np.sum(eij*vikXec, axis=1)

    # b = [eij, eik, ec]  c = [vij, vik, ec]

    gamma0 = np.sum(eij*eikXec, axis=1)
    gamma2 = np.sum(vij*vikXec, axis=1)

    A = det1 + det2 

    return A, A**2 - 4*gamma0*gamma2


def planarity_check(t1, tt1, ec):
    """ Function to check the planarity of the torsal directions
    Input:
        t1: Torsal direction
        tt1: Second envelope torsal direction
        ec: Lince congruence joining barycenters of the faces
    """

    t1 = unit(t1)
    tt1 = unit(tt1)
    ec = unit(ec)
    t1_tt1 = np.cross(t1, tt1)
    # Check planarity
    planar = abs(vec_dot(t1_tt1, ec))

    #planar = abs(vec_dot(nt1, t1)) + abs(vec_dot(nt1, tt1)) + abs(vec_dot(nt1, ec))

    # # Replace nan with 0
    planar[np.where(np.isnan(planar))] = 1

    return planar



def compute_torsal_angles(t1, t2, ec):
    """ Function to compute the torsal angles between two cross fields
    Input:
        t1: Torsal direction
        t2: Second torsal direction
        ec: Lince congruence joining barycenters of the faces
    """

    # Compute nt1 
    nt1 = np.cross(t1, ec)
    nt1 /= np.linalg.norm(nt1, axis=1)[:, None]

    # Compute nt2
    nt2 = np.cross(t2, ec)
    nt2 /= np.linalg.norm(nt2, axis=1)[:, None]

    # Compute torsal angles
    torsal_angles = np.arccos(np.sum(nt1*nt2, axis=1))

    return torsal_angles, nt1, nt2



def torsal_dir_show(baricenter, t1, t2, size=0.005, rad=0.0005,  color=(1,1,1), name=""):

    # Torsal directions t1
    t1_dir_i = baricenter.reshape(-1,3) + size*t1
    t1_dir_f = baricenter.reshape(-1,3) - size*t1

    # Torsal directions t2 
    t2_dir_i = baricenter.reshape(-1,3) + size*t2
    t2_dir_f = baricenter.reshape(-1,3) - size*t2
    
    t2_nodes = np.concatenate((t2_dir_i, t2_dir_f), axis=0)
    t1_nodes = np.concatenate((t1_dir_i, t1_dir_f), axis=0)

    t1_edges = np.array([[i, i + len(t1_dir_i)] for i in range(len(t1_dir_i))])

    t1_net = ps.register_curve_network(name+"t1", 
                                       t1_nodes, 
                                       t1_edges, 
                                       color=color)
    t2_net = ps.register_curve_network(name+"t2", t2_nodes, t1_edges, 
                                        color=color)
    t1_net.set_radius(rad, relative=False) 
    t2_net.set_radius(rad, relative=False)

def save_torsal(baricenter, t1, t2, size=0.005, rad=0.0005, path="", type=1):

    # Get the last name of name
    name = path.split('/')[-1]

    # Torsal directions t1
    t1_dir_i = baricenter.reshape(-1,3) 
    t1_dir_f = baricenter.reshape(-1,3) + size*t1


    # Torsal directions t2 
    t2_dir_i = baricenter.reshape(-1,3) 
    t2_dir_f = baricenter.reshape(-1,3) + size*t2

    
    t2_nodes = np.concatenate((t2_dir_i, t2_dir_f), axis=0)
    t1_nodes = np.concatenate((t1_dir_i, t1_dir_f), axis=0)

    t1_edges = np.array([[i, i + len(t1_dir_i)] for i in range(len(t1_dir_i))])

    # Check if path exist
    if not os.path.exists(path):
        os.makedirs(path)

    
    if type==1:
        # Create two files for each vector field
        file1 = os.path.join(path, name+'_TD1.obj')
        file2 = os.path.join(path, name+'_TD2.obj')
    elif type==2:
        file1 = os.path.join(path, name+'_QD1.obj')
        file2 = os.path.join(path, name+'_QD2.obj')
    else: 
        file1 = os.path.join(path, name+'_MD1.obj')
        file2 = os.path.join(path, name+'_MD2.obj')


    # Write the first vector field
    with open(file1, 'w') as f:
        for v in t1_nodes:
            f.write('v {} {} {}\n'.format(v[0], v[1], v[2]))
        for l in t1_edges:
            f.write('l {} {}\n'.format(l[0]+1, l[1]+1))

    # Write the second vector field
    with open(file2, 'w') as f:
        for v in t2_nodes:
            f.write('v {} {} {}\n'.format(v[0], v[1], v[2]))
        for l in t1_edges:
            f.write('l {} {}\n'.format(l[0]+1, l[1]+1))

def get_torsal_q_mesh(V, F, L):
    """ Function that compute the torsal directions given a polyhedral surface
    with a line congruence per vertex
    Input:
        V: Vertices
        F: Faces
        L: Line congruence
    Output:
        t1: Torsal direction 1
        t2: Torsal direction 2
    """

    l0, l1, l2, l3 = L[F[:, 0]], L[F[:, 1]], L[F[:, 2]], L[F[:, 3]]

    lu = l2 - l0
    lv = l1 - l3

    v0, v1, v2, v3 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]], V[F[:, 3]]

    du = v2 - v0
    dv = v1 - v3

    lc = (l0 + l1 + l2 + l3)/4
    vc = (v0 + v1 + v2 + v3)/4

    t1, t2, _, _, _, _, _  = torsal_directions(lc, lu, lv, du, dv)

    return t1, t2, vc

def get_torsal_mesh(V, F, L):
    """ Function that compute the torsal directions given a polyhedral surface
    with a line congruence per vertex
    Input:
        V: Vertices
        F: Faces
        L: Line congruence
    Output:
        t1: Torsal direction 1
        t2: Torsal direction 2
    """

    l0, l1, l2 = L[F[:, 0]], L[F[:, 1]], L[F[:, 2]]

    lu = l1 - l0
    lv = l2 - l0

    v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]

    du = v1 - v0
    dv = v2 - v0

    lc = (l0 + l1 + l2)/3
    vc = (v0 + v1 + v2)/3

    t1, t2, _, _, _, _, valid  = torsal_directions(lc, lu, lv, du, dv)

    return t1, t2, vc, valid




def triangulate_quads(quads, vertices):
    """
    Triangulate a list of quads into triangles.

    Args:
    - quads (list of list of int): List of quads, where each quad is a list of four vertex indices.

    Returns:
    - list of list of int: List of triangles, where each triangle is a list of three vertex indices.
    """
    triangles = []

    Num_V = len(vertices)
    print("Num_V", Num_V)
    new_vertices = np.array(vertices)
    for quad in quads:
        # Ensure the quad has exactly 4 vertices
        if len(quad) == 4:

            Num_V += 1

            # Compute barycenter
            vc = np.sum(vertices[quad], axis=0)/4

            # Add the barycenter to the vertices
            new_vertices = np.vstack((new_vertices, vc))
        
            # First triangle 
            triangles.append([quad[0], quad[1], Num_V-1])
            # Second triangle
            triangles.append([quad[1], quad[2], Num_V-1])
            # Third triangle
            triangles.append([quad[2], quad[3], Num_V-1])
            # Fourth triangle
            triangles.append([quad[3], quad[0], Num_V-1])

        else:
            print("Error: Quad does not have exactly 4 vertices.", quad)

    return triangles, new_vertices

def triangulate_quads_diag(quads):
    """
    Triangulate a list of quads into triangles.

    Args:
    - quads (list of list of int): List of quads, where each quad is a list of four vertex indices.

    Returns:
    - list of list of int: List of triangles, where each triangle is a list of three vertex indices.
    """
    triangles = []
    for quad in quads:
        # Ensure the quad has exactly 4 vertices
        if len(quad) == 4:
            # First triangle from first, second, and third vertices
            triangles.append([quad[0], quad[1], quad[2]])
            # Second triangle from first, third, and fourth vertices
            triangles.append([quad[0], quad[2], quad[3]])
        else:
            print("Error: Quad does not have exactly 4 vertices.", quad)
    return triangles

def np_pop(arr, idx):
    """ Function to pop an element from a numpy array
    """
    return np.delete(arr, idx, axis=0)