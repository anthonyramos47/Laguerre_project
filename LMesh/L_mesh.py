# Import the necessary libraries
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

# Use all available CPU cores for MKL/OpenBLAS (must be set before numpy import)
_n_cores = str(os.cpu_count())
os.environ.setdefault("MKL_NUM_THREADS", _n_cores)
os.environ.setdefault("OMP_NUM_THREADS", _n_cores)
os.environ.setdefault("NUMEXPR_NUM_THREADS", _n_cores)

import numpy as np
from scipy.linalg import null_space
from pickle import dump, load

# Import Mesh class and utility functions from the geometry module
from hanan.geometry.mesh import Mesh
from hanan import glyphs
from hanan.geometry.conical import oriented_cone
from hanan.geometry.utils import (
    unit,
    project_point_plane,
    distance_point_plane,
    circular_arc,
    mathematica_v,
    project_to_sphere,
    merge_meshes,
    normalize_vertices,
    write_obj,
    line_plane_inter,
    hex_to_norm_rgb,
    iso_sphere_to_sphere,
    i_csphere_sphere,
    i_point_to_or_plane,
    i_csphere_inverse_point,
    offset_curve,
    catmull_clark_subdivision,
    plane_normal,
    dist_point_line,
    foot_point_line,
    loft_curves,
    merge_curves,
    reflect_point,
)
from LMesh.utils.laguerre import (
    plane_tangent_spheres_approximation,
    spherical_panel_improved,
    spherical_panel_no_remesh,
    spherical_polygon,
    fit_isotropic_hyperplane,
)
from hanan.geometry.primitives import reflect_plane
                        

# Optimization imports
from hanan.optimization.optimizer import Optimizer
from .Optimization.Or_contactFree import Or_contactFree
from .Optimization.Tangential_Pos import TDPositive



rad_tol = 0.00001


def _hce(center, normal, radius):
    """Return a half-conical-edge record as a plain dict."""
    return {"center": center, "normal": normal, "radius": abs(radius), "c_arc": None}


def _progress(iterable, total=None, desc="", width=32, stream=None):
    """
    Minimal dependency-free progress bar (works in a terminal and in Jupyter).

    Yields the items of *iterable*, redrawing a single line in place.  Falls back
    to plain iteration when the stream is not a TTY and not a Jupyter shell.
    """
    if stream is None:
        stream = sys.stdout
    if total is None:
        try:
            total = len(iterable)
        except TypeError:
            total = None

    # Redraw in place only where it makes sense; otherwise just iterate.
    live = getattr(stream, "isatty", lambda: False)() or "ipykernel" in sys.modules
    if not live or not total:
        yield from iterable
        return

    t0 = time.perf_counter()

    def draw(i):
        frac = i / total
        filled = int(width * frac)
        bar = "#" * filled + "-" * (width - filled)
        elapsed = time.perf_counter() - t0
        eta = (elapsed / i * (total - i)) if i else 0.0
        stream.write(f"\r{desc} |{bar}| {i}/{total} "
                     f"({frac:5.1%}) {elapsed:5.1f}s eta {eta:5.1f}s")
        stream.flush()

    draw(0)
    for i, item in enumerate(iterable, 1):
        yield item
        # Cap redraws so huge loops do not spend their time on I/O.
        if i == total or i % max(1, total // 200) == 0:
            draw(i)
    stream.write("\n")
    stream.flush()

    

def _merge_meshes_or_empty(V_list, F_list):
    """merge_meshes, or an empty (V, F) when there is nothing to merge."""
    if not V_list:
        return np.zeros((0, 3)), []
    return merge_meshes(V_list, F_list)


def _merge_curves_or_empty(curves):
    """merge_curves over the curves with at least two points, as (P, E)."""
    curves = [np.asarray(c, dtype=float) for c in curves if c is not None and len(c) >= 2]
    if not curves:
        return np.zeros((0, 3)), np.zeros((0, 2), dtype=int)
    return merge_curves(curves)


def _polylines(P, E):
    """Split a curve network into polylines: runs of edges that continue each other."""
    chains = []
    for a, b in E:
        if chains and chains[-1][-1] == a:
            chains[-1].append(b)
        else:
            chains.append([a, b])
    return [P[c] for c in chains]


def _collapsed(arc, tol=rad_tol):
    """True when all points of an arc coincide (its sphere has radius ~0)."""
    return float(np.ptp(np.asarray(arc), axis=0).max()) < tol


class Lmesh:
    """
    Class representing the L-mesh structure.

    This class contains methods for initializing the mesh, computing spherical faces,
    conical edge structures, vertex planes, and for visualizing the entire L-mesh.

    """
    def __init__(self):
        self.base_mesh = None
        self.top       = None
        self.dual_top  = None
        self.HCE = {}            # Half Conical Edges: keyed by (i, j) sphere-vertex pairs.
        self.CE  = {}            # Conical Edges: cone data [tip, axis, angle].
        self.contact_points = {} # Contact points keyed by (sphere_vertex_idx, plane_face_idx).
        self.orientation = 1

        # Sphere data now lives directly in self.top.vertices as an (N_vertices, 4)
        # array: columns [cx, cy, cz, r].  Spheres live on the L-mesh VERTICES
        # (one per vertex).  Access via properties sv_centers → top.vertices[:, :3]
        # and sv_radii → top.vertices[:, 3].  (There is no separate self.SV.)

        # Plane data — (N_faces, 4) array: columns [nx, ny, nz, h] (Hesse form n·x + h = 0)
        # Oriented planes now live on the L-mesh FACES (one per face).
        # Access via properties pf_normals → [:, :3]  and pf_dist → [:, 3]
        self.PF = np.empty((0, 4))

        # Panel caches — rebuilt by compute_spherical_panels / compute_panel_meshes
        # Spheres live on VERTICES → these are per-vertex.
        self.sv_boundary_curves = []    # per-vertex list of arc arrays
        self.sv_boundary_curve  = []    # per-vertex concatenated boundary
        self.sv_panels          = []    # per-vertex (V, F) or None

        # Plane-polygon data — planes live on FACES → one entry per L-mesh face
        self.vp_polygons = []   # per-face list of point arrays

        # Second-envelope data (reflected L-mesh structure)
        self.se_vp_polygons        = []  # reflected vertex-plane polygons
        self.se_contact_points     = {}  # contact points for reflected planes
        self.se_HCE                = {}  # half conical edges for second envelope
        self.se_sv_boundary_curve  = []  # per-face concatenated boundary
        self.se_sv_boundary_curves = []  # per-face list of arc arrays
        self.se_sv_panels          = []  # per-face (V, F) or None

        self.VIso = None

        self.Optimizer = Optimizer()

        # Message log — warnings/info collected instead of printed to stdout.
        # Inspect with print(lmesh.log) or lmesh.print_log(); write out with
        # lmesh.export_log(path).  Set lmesh.verbose = True to also echo to
        # stdout as messages arrive.
        self._log = []
        self.verbose = False

    # ------------------------------------------------------------------ #
    # Message log                                                         #
    # ------------------------------------------------------------------ #

    @property
    def log(self):
        """Collected log messages as a single string."""
        return "\n".join(self._log)

    def _log_msg(self, msg):
        """Record *msg* in the log (and echo it if self.verbose)."""
        self._log.append(str(msg))
        if self.verbose:
            print(msg)

    def clear_log(self):
        """Discard all collected log messages."""
        self._log = []

    def print_log(self):
        """Print every collected log message."""
        print(self.log)

    def export_log(self, path):
        """
        Write the collected log messages to *path* (default extension .out).

        Returns the path actually written.
        """
        if not os.path.splitext(path)[1]:
            path = path + ".out"
        folder = os.path.dirname(os.path.abspath(path))
        if folder:
            os.makedirs(folder, exist_ok=True)
        with open(path, "w") as f:
            f.write(self.log)
            if self._log:
                f.write("\n")
        return path

    # ------------------------------------------------------------------ #
    # Serialization: spheres + planes + topology only                     #
    # ------------------------------------------------------------------ #

    def lmesh_state(self):
        """
        Return the minimal state defining this L-mesh: spheres, planes, topology.

        Everything else (contact points, vertex polygons, conical edges,
        spherical panels, the Optimizer) is derived and is recomputed on load by
        init_L_mesh -> compute_lmesh_structure.

        Sphere and plane data are stored in ORIENTATION-FREE form, i.e. as the raw
        arrays before `self.orientation` is applied, so that load_lmesh() can pass
        them straight back to init_L_mesh without double-applying the orientation.

        Returns:
            dict: picklable arrays and metadata (no live objects).
        """
        return {
            "format":        "lmesh-state",
            "version":       1,
            # --- topology (base mesh; self.top is rebuilt from it) ---
            "base_vertices": np.asarray(self.base_mesh.vertices, dtype=float),
            "base_faces":    [list(map(int, f)) for f in self.base_mesh.faces],
            # --- spheres, per L-mesh vertex: centers (V,3) + radii (V,) ---
            "centers":       np.asarray(self.sv_centers, dtype=float).copy(),
            "radii":         np.asarray(self.sv_radii,   dtype=float).copy(),
            # --- planes, per L-mesh face: Hesse form (F,4), orientation removed ---
            "planes":        np.asarray(self.orientation * self.PF, dtype=float).copy(),
            # --- metadata needed to reproduce the same construction ---
            "orientation":   int(self.orientation),
            "combinatorics": self.combinatorics,
        }

    def save_lmesh(self, path):
        """
        Pickle only the spheres, planes and topology of this L-mesh.

        Parameters:
            path (str): destination .pkl file.
        """
        with open(path, 'wb') as f:
            dump(self.lmesh_state(), f)
        return path

    def load_lmesh_state(self, state):
        """
        Rebuild this L-mesh in place from a state dict produced by lmesh_state().

        Reconstructs the base mesh and topology, restores the saved spheres and
        planes, and recomputes the full derived structure via init_L_mesh
        (mode="OrPlanes" -> compute_lmesh_structure).

        Parameters:
            state (dict): as returned by lmesh_state() / stored by save_lmesh().

        Returns:
            self
        """
        if state.get("format") != "lmesh-state":
            raise ValueError(
                "Not an L-mesh state pickle (expected format='lmesh-state'). "
                "Old whole-object pickles should be loaded with pickle.load directly."
            )

        base = Mesh()
        base.make_mesh(np.asarray(state["base_vertices"], dtype=float),
                       state["base_faces"])

        # init_L_mesh re-applies `orientation` to the planes, so feed it the
        # orientation-free arrays exactly as they were stored.
        self.init_L_mesh(
            base,
            orientation=state["orientation"],
            combinatorics=state["combinatorics"],
            mode="OrPlanes",
            planes=np.asarray(state["planes"],  dtype=float),
            centers=np.asarray(state["centers"], dtype=float),
            radii=np.asarray(state["radii"],   dtype=float),
        )
        return self

    @property
    def sv_centers(self):
        return self.top.vertices[:, :3]

    @property
    def sv_radii(self):
        return self.top.vertices[:, 3]

    @property
    def pf_normals(self):
        return self.PF[:, :3]

    @property
    def pf_dist(self):
        return self.PF[:, 3]

    def reset(self):
        """
        Re-initialize the L-mesh from its base mesh (polyhedra mode).
        """
        self.init_L_mesh(self.base_mesh, orientation=self.orientation, mode="polyhedra")


    # ------------------------------------------------------------------ #
    # Private helpers for init_L_mesh                                    #
    # ------------------------------------------------------------------ #

    def _build_topology(self, mesh):
        """
        Build self.top from *mesh* according to self.combinatorics.

        NEW CONVENTION (spheres on vertices, planes on faces):

        For "primal" combinatorics (the default) the L-mesh shares the base-mesh
        combinatorics directly: L-mesh VERTICES are base-mesh vertices (or-spheres
        live on vertices) and L-mesh FACES are base-mesh faces (oriented planes
        live on faces).  self.top.vertices holds real xyz coordinates.

        For "dual" combinatorics the roles are taken on the combinatorial dual:
        L-mesh vertices correspond to base-mesh faces and L-mesh faces correspond
        to inner base-mesh vertices.

        Side effect: stores self._inner_vertices so _init_mode_polyhedra can use it
        without recomputing.
        """
        V = mesh.vertices
        F = mesh.faces

        if self.combinatorics == "primal":
            # Spheres on vertices, planes on faces — build top directly on (V, F).
            self.top = Mesh()
            self.top.make_mesh(V, F)

            self._inner_vertices = mesh.inner_vertices()
            self.create_map_indices()

        elif self.combinatorics == "dual":
            dual_faces    = mesh.dual_top()
            dual_vertices = np.zeros((len(mesh.face_barycenters()), 3))
            self.top = Mesh()
            self.top.make_mesh(dual_vertices, dual_faces)

            self._inner_vertices = mesh.inner_vertices()
            self.create_map_indices()

        else:
            raise ValueError(
                f"Unknown combinatorics '{self.combinatorics}'. "
                f"Valid options are: 'primal', 'dual'."
            )

    def _init_mode_polyhedra(self, mesh, V):
        """
        Initialize plane and sphere data from polyhedral mesh geometry.

        Planes  — one per L-mesh FACE, built from face normals and face
                  barycenters (Hesse normal form), stored in self.PF.
        Spheres — one per L-mesh VERTEX, centered at the vertex position with
                  radius 0, stored in self.top.vertices as [cx, cy, cz, r].

        Assumes "primal" combinatorics (base-mesh vertices ↔ spheres,
        base-mesh faces ↔ planes).
        """
        self.vertex_normals = self.orientation * mesh.vertex_normals
        self.face_normals   = self.orientation * mesh.face_normals

        # Hesse distance: h = -<n, p> for a point p on the plane
        h = -np.einsum('ij,ij->i', mesh.face_barycenters(), self.face_normals)

        # One plane per face — store as 4-vectors (nx, ny, nz, h) in self.PF
        self.PF = np.stack(
            (self.face_normals[:, 0], self.face_normals[:, 1],
             self.face_normals[:, 2], h),
            axis=1
        )

        # One sphere per L-mesh vertex, initialised at the vertex position, radius 0.
        # Stored in-place in self.top.vertices as (N, 4): [cx, cy, cz, r].
        centers = np.array(self.top.vertices[:, :3], dtype=float)
        self.top.vertices = np.column_stack((centers, np.zeros(len(centers))))

    def _init_mode_sph_congruence(self, mesh, kwargs):
        """
        Initialize sphere data from provided centers/radii, and plane data either
        from an optional 'planes' kwarg or from mesh normals.

        Required kwargs: 'centers', 'radii'
        Optional kwargs: 'planes'
        """
        if 'radii' not in kwargs or 'centers' not in kwargs:
            raise ValueError(
                "Mode 'SphCongruence' requires both 'radii' and 'centers' keyword arguments."
            )

        radii      = kwargs['radii']
        centers    = kwargs['centers']
        n_vertices = len(self.top.vertices)

        if len(radii) != n_vertices or len(centers) != n_vertices:
            raise ValueError(
                f"Length of 'radii' and 'centers' must match the number of L-mesh vertices "
                f"({n_vertices}). Received: radii={len(radii)}, centers={len(centers)}."
            )

        # Spheres live on vertices — store in self.top.vertices as [cx, cy, cz, r]
        self.top.vertices = np.column_stack((np.array(centers, dtype=float), np.array(radii, dtype=float)))

        if 'planes' in kwargs:
            planes = kwargs['planes'].copy()
            norms  = np.linalg.norm(planes[:, :3], axis=1, keepdims=True)
            planes /= norms
            self.PF = self.orientation * planes

        elif self.combinatorics == "primal":
            # One plane per face — initial normal from base-mesh face normals, then
            # fit a plane tangent to the spheres sitting on that face's vertices.
            mesh_face_normals = self.orientation * mesh.face_normals
            distance_origin   = -np.einsum('ij,ij->i', mesh.face_barycenters(), mesh_face_normals)

            spheres_per_face = self.top.faces  # vertex indices per face = spheres per face

            self.face_normals = np.zeros_like(mesh_face_normals)
            self.PF           = np.zeros((len(self.top.faces), 4))

            for fi in range(len(self.top.faces)):
                normals = mesh_face_normals[fi]
                h       = distance_origin[fi]

                centers = self.sv_centers[spheres_per_face[fi]]
                radii   = self.sv_radii[spheres_per_face[fi]]

                # Approximate plane using base normal as initial guess
                na, ha, _ = plane_tangent_spheres_approximation(centers, radii, normals, h)

                self.face_normals[fi] = na
                self.PF[fi]           = np.concatenate((na, [ha]))

        elif self.combinatorics == "dual":
            # Dual: planes from base-mesh vertex normals (one plane per dual face).
            self.vertex_normals = self.orientation * mesh.vertex_normals
            h = -np.einsum('ij,ij->i', mesh.vertices, self.vertex_normals)
            self.PF = np.stack(
                (self.vertex_normals[:, 0], self.vertex_normals[:, 1],
                 self.vertex_normals[:, 2], h),
                axis=1
            )

    def _init_mode_or_planes(self, kwargs):
        """
        Initialize plane data from a provided 'planes' array (Hesse normal form),
        and optionally sphere data from 'centers'/'radii'.

        Required kwargs: 'planes'  (shape (n, 4), columns: nx, ny, nz, h)
        Optional kwargs: 'centers' (shape (m, 3)), 'radii' (shape (m,))
        """
        if 'planes' not in kwargs:
            raise ValueError(
                "Mode 'OrPlanes' requires a 'planes' keyword argument "
                "defining planes in Hesse normal form (n, h)."
            )

        planes = kwargs['planes'].copy()
        norms  = np.linalg.norm(planes[:, :3], axis=1, keepdims=True)
        planes /= norms

        n_vertices = len(self.top.vertices)

        if 'centers' in kwargs and 'radii' in kwargs:
            centers = kwargs['centers']
            radii   = kwargs['radii']
            if len(radii) != n_vertices or len(centers) != n_vertices:
                raise ValueError(
                    f"Length of 'radii' and 'centers' must match the number of "
                    f"L-mesh vertices ({n_vertices})."
                )
            # Spheres on vertices — store in self.top.vertices as [cx, cy, cz, r]
            self.top.vertices = np.column_stack((np.array(centers, dtype=float), np.array(radii, dtype=float)))
        else:
            self.top.vertices = np.zeros((n_vertices, 4))

        if len(planes) != len(self.top.faces):
            raise ValueError(
                f"Length of 'planes' ({len(planes)}) must match the number of "
                f"L-mesh faces ({len(self.top.faces)})."
            )

        # Planes on faces
        self.PF = self.orientation * planes

    # ------------------------------------------------------------------ #

    def init_L_mesh(self, mesh, orientation=1, combinatorics="primal", mode="polyhedra", **kwargs):
        """
        Initialize an L-mesh from a base mesh.

        Parameters:
            mesh: Input Mesh object carrying combinatorial and geometric information.
            orientation (int): Global orientation of the L-mesh (+1 or -1).
            combinatorics (str): Which combinatorial structure to use:
                - "primal" (default) — L-mesh shares base-mesh combinatorics: or-spheres
                                       live on base-mesh VERTICES, oriented planes on
                                       base-mesh FACES.
                - "dual"             — L-mesh built on the combinatorial dual: spheres on
                                       base-mesh faces, planes on inner base-mesh vertices.
            mode (str): How to initialize the plane and sphere data:
                - "polyhedra"     (default) — planes from face normals (one per face),
                                              spheres at vertices with radius 0.
                - "SphCongruence"           — spheres from provided centers/radii; planes
                                              from optional 'planes' kwarg or inferred from
                                              mesh normals.
                - "OrPlanes"                — planes from provided 'planes' kwarg (Hesse
                                              normal form); spheres from optional
                                              centers/radii or initialized to zero.
            **kwargs:
                planes  (array, shape (F,4)): Oriented planes in Hesse form (n, h), per face.
                centers (array, shape (V,3)): Sphere centers, per vertex.
                radii   (array, shape (V,)):  Sphere radii, per vertex.

        Raises:
            ValueError: If required kwargs for the selected mode are missing or have
                        incompatible sizes.
        """
        self.orientation  = orientation
        self.combinatorics = combinatorics
        self.base_mesh = mesh

        V = mesh.vertices

        # ------------------------------------------------------------------ #
        # Step 1 — Build topology (self.top).                                 #
        # self.top  : the L-mesh combinatorics (verts = spheres, faces = planes)
        # ------------------------------------------------------------------ #
        self._build_topology(mesh)

        # ------------------------------------------------------------------ #
        # Step 2 — Initialize sphere data (self.top.vertices, per vertex) and  #
        #          plane data (self.PF, per face) according to the chosen mode.#
        # ------------------------------------------------------------------ #
        if mode == "polyhedra":
            self._init_mode_polyhedra(mesh, V)

        elif mode == "SphCongruence":
            self._init_mode_sph_congruence(mesh, kwargs)

        elif mode == "OrPlanes":
            self._init_mode_or_planes(kwargs)

        else:
            raise ValueError(
                f"Unknown mode '{mode}'. "
                f"Valid modes are: 'polyhedra', 'SphCongruence', 'OrPlanes'."
            )

        # ------------------------------------------------------------------ #
        # Step 3 — Compute the full L-mesh structure from the initialized     #
        #          planes and spheres.                                        #
        # ------------------------------------------------------------------ #
        
        self.compute_lmesh_structure()
        #self.compute_dual_topology()
        #self.compute_cyclographic_representation()
        #self.compute_isotropic_mesh()

    def update_spheres_planes(self, new_centers, new_radii, new_normals, new_distances):
        """
        Update the spheres and planes of the L-mesh.

        Parameters:
            new_centers (array-like): New sphere centers, per VERTEX (V, 3).
            new_radii (array-like): New sphere radii, per VERTEX (V,).
            new_normals (array-like): New plane normals, per FACE (F, 3).
            new_distances (array-like): New plane offsets h in n·x + h = 0, per FACE (F,).
        """
        self.top.vertices = np.column_stack((np.array(new_centers, dtype=float),
                                             np.array(new_radii,   dtype=float)))

        new_normals = np.array(new_normals, dtype=float)
        lengths = np.linalg.norm(new_normals, axis=1, keepdims=True)
        new_normals /= lengths
        new_distances = np.array(new_distances, dtype=float) / lengths[:, 0]
        self.PF = np.hstack((new_normals, new_distances.reshape(-1, 1)))

        self.compute_lmesh_structure()

    def compute_dual_topology(self):
        """
        Compute the dual topology of the L-mesh.

        This involves creating a new Mesh object where the roles of vertices and faces are swapped.
        The dual vertices correspond to the spherical faces, and the dual faces correspond to the vertex planes.
        """
        # Create dual topology
        dual_faces = self.top.dual_top()

        self.dual_top = Mesh()
        self.dual_top.make_mesh(np.zeros((len(self.top.vertices), 4)), dual_faces)

    def compute_cyclographic_representation(self):
        """
        Compute the cyclographic representation of the L-mesh.

        This involves mapping the spherical faces and vertex planes to a dual topology
        and storing the resulting vertices and faces in self.dual_top.
        """        

        # Map spherical faces to dual vertices
        n = len(self.sv_centers)
        dual_vertices = np.zeros((n, 4))
        dual_vertices[:, :3] = self.sv_centers
        dual_vertices[:,  3] = self.sv_radii

        self.dual_top.vertices = dual_vertices


    def subdivide_lmesh(self):
        """
        Subdivide the L-mesh and update the spheres and oriented planes accordingly.

        Spheres live on the L-mesh VERTICES (self.top.vertices as [cx, cy, cz, r]),
        which is exactly the cyclographic representation of the sphere congruence,
        so self.top *is* the mesh to subdivide — Catmull-Clark is applied to it
        directly in 4D cyclographic coordinates.  Averaging in that space maps back
        to a valid sphere, so the subdivided vertices are the refined spheres.

        Oriented planes live on the FACES (self.PF), so after subdivision one plane
        is refit per new face, tangent to the spheres of that face's vertices, using
        the parent face's plane as the initial guess.
        """

        # Cyclographic coordinates of the spheres: [cx, cy, cz, r] per vertex.
        cyclo_vertices = np.asarray(self.top.vertices, dtype=float)
        cyclo_faces    = self.top.faces

        # Perform subdivision (using Catmull-Clark as an example)
        new_vertices, new_faces, parent_faces = catmull_clark_subdivision(cyclo_vertices, cyclo_faces)

        child_to_parent = {child: parent for parent, children in parent_faces.items() for child in children}

        # Planes are per-FACE, so the initial guess for a new face comes from the
        # parent face's plane in self.PF (not from top.vertices, which holds spheres).
        old_planes = np.asarray(self.PF, dtype=float)

        vertex_planes = np.zeros((len(new_faces), 4))

        # Each face runs its own L-BFGS-B fit, so this is the slow part of the
        # subdivision — show progress, and record the fit residuals so a bad
        # approximation is visible instead of silently producing a broken plane.
        fit_costs = np.zeros(len(new_faces))
        for i, face in _progress(list(enumerate(new_faces)),
                                 total=len(new_faces),
                                 desc="  subdivide: fitting planes"):

            centers = new_vertices[face, :3]
            radii   = new_vertices[face, 3]

            prev_plane = old_planes[child_to_parent[i]]

            # Approximate new planes using previos as initial guess for the optimization
            na, ha, cost = plane_tangent_spheres_approximation(centers, radii, prev_plane[:3], prev_plane[3])

            fit_costs[i]     = cost
            vertex_planes[i] = np.concatenate((na, [ha]))

        # Tangency-fit quality: cost is the sum of squared residuals of
        # (c.n + h - r) over the face's spheres, so it should be ~0 for a plane
        # that really is tangent to all of them.
        self._subdiv_plane_fit_costs = fit_costs
        n_bad = int(np.sum(fit_costs > 1e-10))
        self._log_msg(f"subdivide_lmesh: plane fits over {len(new_faces)} faces — "
                      f"max={fit_costs.max():.3e} mean={fit_costs.mean():.3e} "
                      f"({n_bad} above 1e-10)")
        if n_bad:
            worst = np.argsort(fit_costs)[-5:][::-1]
            for w in worst:
                self._log_msg(f"  worst face {w}: cost={fit_costs[w]:.3e} "
                              f"valence={len(new_faces[w])}")

        # Rebuild the L-mesh: spheres on the refined vertices, planes on the faces.
        self.top = Mesh()
        self.top.make_mesh(new_vertices, new_faces)
        self.PF  = vertex_planes

        self.compute_lmesh_structure()

    def _apply_optimizer_result(self):
        """
        Unpack, normalize, and apply the optimizer variables back to the L-mesh.

        Reads cv, rv (spheres, per vertex) and npf, hpf (planes, per face) from the
        Optimizer, re-normalizes the plane normals, and updates the sphere data in
        self.top.vertices (per vertex) and self.PF (faces).  Sphere data is written
        into self.top.vertices as (N, 4).

        NOTE: the variable names npf/hpf are kept for backward compatibility but now
        denote per-FACE plane data, not per-vertex.
        """
        cv, rv, npf, hpf = self.Optimizer.unpack("cv", "rv", "npf", "hpf")

        cv  = cv.reshape((-1, 3))
        npf = npf.reshape((-1, 3))

        # Re-normalize plane normals and scale distances accordingly
        nvp_lengths = np.linalg.norm(npf, axis=1)
        npf = npf / nvp_lengths[:, np.newaxis]
        hpf = hpf / nvp_lengths

        self.top.vertices = np.column_stack((cv, rv))

        self.PF = np.hstack((npf, hpf.reshape((-1, 1))))

    def lmesh_planes_lc_optimization_step(self, weightsDic):
        """
        Perform a single optimization step to refine the L-mesh structure.

        Parameters:
            weightsDic (dict): Dictionary of weights for different objective terms.
        """
        self.Optimizer.set_term_weights(weightsDic)
        self.Optimizer.get_gradients()
        self.Optimizer.optimize_step()

        self._apply_optimizer_result()

        self.compute_contact_points()
        self.compute_vertex_polygons()
        self.compute_conical_edge_structures()
        self.compute_spherical_panels()

    def improve_contact(self):
        self.setup_optimization()
        

        max_iterations = 20
        for _ in range(max_iterations):
            self.Optimizer.get_gradients()
            self.Optimizer.optimize_step()

        #self.Optimizer.print_energy_report()
        print("L-mesh Optimized")
        print(f"Oriented Contact Deviation: {self.compute_oriented_contact_dev():.3e}")
        self._apply_optimizer_result()
        self.compute_lmesh_structure()


    def compute_lmesh_structure(self):
        """
        Compute the full L-mesh structure:
            - Spherical faces
            - Vertex planes
            - Conical edge structures
            - Spherical panels
        It requires that the spherical faces and vertex planes are already initialized.
        """

        steps = [
            ("compute_contact_points",          self.compute_contact_points),
            ("compute_vertex_polygons",         self.compute_vertex_polygons),
            ("compute_conical_edge_structures", self.compute_conical_edge_structures),
            ("compute_spherical_panels",        self.compute_spherical_panels),
        ]

        total_start = time.perf_counter()
        for name, fn in steps:
            t0 = time.perf_counter()
            fn()
            self._log_msg(f"  {name:<35} {time.perf_counter() - t0:.4f}s")
        self._log_msg(f"  {'TOTAL':<35} {time.perf_counter() - total_start:.4f}s")


    def compute_contact_points(self):
        """
        Compute contact points between spherical vertices and face planes.

        Loop over spherical VERTICES (spheres); for each, the incident FACES carry
        the oriented planes.  The contact point of sphere vi touching plane fi is
        c_vi - r_vi * n_fi.  The dict key is (sphere_idx, plane_idx) = (vi, fi).
        """
        self.contact_points = {}

        # Faces incident to each vertex = planes touching each sphere
        vf_adj = self.top.vertex_face_adjacency_list()
        nPF    = self.PF[:, :3]  # plane normals, per face

        # Loop over spherical vertices
        for vi in range(len(self.top.vertices)):

            center = self.sv_centers[vi]
            radius = self.sv_radii[vi]

            for fi in vf_adj[vi]:
                self.contact_points[(vi, fi)] = center - radius * nPF[fi]

    def get_contact_points(self, keys=False):
        """
        Return the contact points of the L-mesh as an array.

        Each contact point is the point where sphere ``vi`` touches the oriented
        plane ``fi`` of an incident face: ``c_vi - r_vi * n_fi``.  A sphere has
        one contact point per incident face, so the total is the number of
        (vertex, face) incidences, not the number of vertices.

        ``compute_contact_points`` is called first if the cache is empty, so this
        is safe to call at any point after the L-mesh is initialised.

        Args:
            keys (bool): also return the ``(sphere_idx, plane_idx)`` key for each
                         row, in the same order as the returned points.

        Returns:
            np.ndarray (N, 3): the contact points.
            If ``keys`` is True, returns ``(points, keys)`` where ``keys`` is an
            (N, 2) int array of ``(sphere_idx, plane_idx)`` pairs.
        """
        if not self.contact_points:
            self.compute_contact_points()

        # Sort by (sphere, plane) so the ordering is deterministic across calls.
        items = sorted(self.contact_points.items())

        if not items:
            pts = np.empty((0, 3))
            return (pts, np.empty((0, 2), dtype=int)) if keys else pts

        pts = np.array([v for _, v in items], dtype=float)

        if keys:
            return pts, np.array([k for k, _ in items], dtype=int)
        return pts

    def compute_vertex_polygons(self):
        """
        Store the oriented planes of the L-mesh as polygons (one polygon per FACE).

        For each face (plane), collect the contact points of the spheres sitting on
        its vertices, projected onto the plane.  Kept under the historical attribute
        name ``vp_polygons`` (now indexed by face).
        """

        lF  = self.top.faces      # face = plane; its vertices are spheres
        nPF = self.PF[:, :3]
        hPF = self.PF[:, 3]

        self.vp_polygons = []

        for fi in range(len(lF)):
            sphere_verts = lF[fi]
            polygon = [
                project_point_plane(self.contact_points[(vi, fi)], nPF[fi], hPF[fi])
                for vi in sphere_verts
            ]
            self.vp_polygons.append(polygon)


    def compute_conical_edge_structures(self):
        """
        Compute the conical edge structures of the L-mesh.

        A conical edge sits between two adjacent SPHERES (now on vertices) along an
        L-mesh edge.  The two PLANES (now on faces) on either side of the edge supply
        the four contact points that bound the conical strip.
        """

        # Sphere pair = edge endpoints (vertices); plane pair = the two adjacent faces
        # svi, svj   = self.top.edge_vertices()   # spheres
        # pfi_, pfj_ = self.top.edge_faces()      # planes
        svi, svj, pfi_, pfj_ = self.top.edge_vertices_faces()

        self.HCE = {}

        # ── filter to interior edges only (both adjacent plane-faces exist) ──
        valid = np.where((pfi_ != -1) & (pfj_ != -1))[0]
        if len(valid) == 0:
            return

        ei  = svi[valid];  ej  = svj[valid]     # sphere-vertex indices
        pfi = pfi_[valid]; pfj = pfj_[valid]    # plane-face indices

        # ── vectorised contact-circle geometry ─────────────────────────────
        centers = self.sv_centers  # (N,3)
        radii   = self.sv_radii    # (N,)

        c1 = centers[ei];  r1 = radii[ei]   # (E,3), (E,)
        c2 = centers[ej];  r2 = radii[ej]

        diff = c2 - c1                                         # (E,3)
        dc   = np.linalg.norm(diff, axis=1)                    # (E,)
        cn   = diff / dc[:, None]                              # (E,3) unit vectors

        td2 = dc**2 - (r1 - r2)**2
        if np.any(td2 < -1e-3):
            bad = np.where(td2 < -1e-3)[0]
            raise ValueError(
                f"Negative tangential distance on edges {valid[bad].tolist()}"
            )

        fac     = (r1 - r2) / dc                               # (E,)
        td      = np.sqrt(np.abs(td2))                         # (E,)
        circi_c = c1 + (r1 * fac)[:, None] * cn               # (E,3)
        circj_c = c2 + (r2 * fac)[:, None] * cn               # (E,3)
        circi_r = np.abs(r1) / dc * td                         # (E,)
        circj_r = np.abs(r2) / dc * td                         # (E,)

        # ── batch arc computation — fully vectorised over all E edges ──────
        E = len(valid)
        num_pts = 20

        # Gather the four contact points for each edge: (E, 3) each
        pij1 = np.array([self.contact_points[(ei[k], pfj[k])] for k in range(E)])
        pij2 = np.array([self.contact_points[(ei[k], pfi[k])] for k in range(E)])
        pji1 = np.array([self.contact_points[(ej[k], pfi[k])] for k in range(E)])
        pji2 = np.array([self.contact_points[(ej[k], pfj[k])] for k in range(E)])

        def _batch_arcs(centers, normals, radii, p0s, p1s):
            """
            Compute circular arcs for all E edges at once.
            Returns (E, num_pts, 3) array; degenerate (radius≈0) rows are marked
            by setting the whole row to the center point.
            """
            degen = np.abs(radii) < 1e-12  # (E,)

            # Radial directions from circle center to start/end points — (E, 3)
            v1 = p0s - centers
            v2 = p1s - centers

            # Project out the normal component: v -= dot(v, n) * n
            v1 -= np.einsum('ij,ij->i', v1, normals)[:, None] * normals
            v2 -= np.einsum('ij,ij->i', v2, normals)[:, None] * normals

            # Normalise rows (safe: degenerate edges handled separately)
            n1 = np.linalg.norm(v1, axis=1, keepdims=True)
            n2 = np.linalg.norm(v2, axis=1, keepdims=True)
            n1 = np.where(n1 < 1e-12, 1.0, n1)
            n2 = np.where(n2 < 1e-12, 1.0, n2)
            v1 /= n1;  v2 /= n2

            # Sweep angle per edge — (E,)
            cos_a = np.clip(np.einsum('ij,ij->i', v1, v2), -1.0, 1.0)
            angle = np.arccos(cos_a)                    # (E,)

            # Second basis vector: Gram-Schmidt v2 ⊥ v1 and ⊥ normal — (E, 3)
            u2 = v2 - np.einsum('ij,ij->i', v2, v1)[:, None] * v1
            u2 -= np.einsum('ij,ij->i', u2, normals)[:, None] * normals
            nu2 = np.linalg.norm(u2, axis=1, keepdims=True)
            nu2 = np.where(nu2 < 1e-12, 1.0, nu2)
            u2 /= nu2

            # Sample parameter: (num_pts,) per edge → broadcast to (E, num_pts)
            t = np.linspace(0, 1, num_pts)              # (num_pts,)
            theta = angle[:, None] * t[None, :]         # (E, num_pts)

            # Arc points: center + r*(cos(θ)*v1 + sin(θ)*u2)  →  (E, num_pts, 3)
            arcs = (centers[:, None, :]
                    + radii[:, None, None] * (
                        np.cos(theta)[:, :, None] * v1[:, None, :]
                        + np.sin(theta)[:, :, None] * u2[:, None, :]
                    ))

            # For degenerate edges replace entire arc with the center point
            arcs[degen] = centers[degen, None, :]
            return arcs

        arcs_i = _batch_arcs(circi_c, cn, circi_r, pij1, pij2)  # (E, num_pts, 3)
        arcs_j = _batch_arcs(circj_c, cn, circj_r, pji1, pji2)  # (E, num_pts, 3)

        # ── store dicts with pre-computed arcs ─────────────────────────────
        for k in range(E):
            self.HCE[(ei[k], ej[k])] = _hce(circi_c[k], cn[k], circi_r[k])
            self.HCE[(ej[k], ei[k])] = _hce(circj_c[k], cn[k], circj_r[k])
            self.HCE[(ei[k], ej[k])]["c_arc"] = arcs_i[k]
            self.HCE[(ej[k], ei[k])]["c_arc"] = arcs_j[k]



    def compute_spherical_panels(self):
        """
        Compute the panel for each sphere (now living on a vertex).

        For each sphere, collect boundary circular arcs from neighbouring spheres
        (vertex neighbours), merge and order the points, and generate the panel.
        """
        n = len(self.sv_centers)
        self.sv_boundary_curve  = [None] * n
        self.sv_boundary_curves = [[]   ] * n
        self.sv_panels          = [None ] * n

        # Sphere neighbours = vertex neighbours (HCE is keyed by sphere-vertex pairs)
        adj_vv = self.top.vertex_adjacency_list()
        for i in range(n):
            radius = self.sv_radii[i]

            if abs(radius) < rad_tol:
                self.sv_panels[i] = (self.sv_centers[i], None)
            else:
                boundary        = []
                boundary_curves = []

                for j in adj_vv[i]:
                    if (i, j) in self.HCE:
                        arc = self.HCE[(i, j)]["c_arc"]
                        if arc is not None and len(arc) > 1:
                            boundary.append(arc[:-1])
                            boundary_curves.append(arc)
                        else:
                            ci, cj = self.sv_centers[i], self.sv_centers[j]
                            ri, rj = radius, self.sv_radii[j]
                            self._log_msg(
                                f"Warning: HCE ({i},{j}) has no valid arc: {arc}\n"
                                f"  sph {i}: c={np.array2string(ci, precision=4)} r={ri:.4f}\n"
                                f"  sph {j}: c={np.array2string(cj, precision=4)} r={rj:.4f}")
                    else:
                        self._log_msg(f"Warning: HCE key ({i},{j}) not found.")

                self.sv_boundary_curves[i] = boundary_curves
                if boundary:
                    self.sv_boundary_curve[i] = np.concatenate(boundary, axis=0)

                       

 
    def _contact_deviations_raw(self, sphere_indices):
        """
        Shared helper: compute signed contact deviation for each (sphere, plane) pair.

        Loops over the given spheres (vertices); for each, its incident faces carry
        the planes.  The contact point for sphere vi touching plane fi is
        c_vi - r_vi * n_fi, and the deviation is its signed distance to the plane.

        Returns:
            deviations_sphere (n_vertices,): accumulated per sphere (vertex).
            deviations_plane  (n_faces,):    accumulated per plane (face).
        """
        vf_adj = self.top.vertex_face_adjacency_list()
        nPF    = self.PF[:, :3]
        hPF    = self.PF[:, 3]

        deviations_sphere = np.zeros(len(self.top.vertices))
        deviations_plane  = np.zeros(len(self.top.faces))

        for vi in sphere_indices:
            center = self.sv_centers[vi]
            radius = self.sv_radii[vi]
            for fi in vf_adj[vi]:
                n_fi = nPF[fi]
                h_fi = hPF[fi]
                cp = center - radius * n_fi
                d  = distance_point_plane(cp, n_fi, h_fi)
                deviations_sphere[vi] += d
                deviations_plane[fi]  += d

        return deviations_sphere, deviations_plane

    def compute_oriented_contact_dev(self):
        """
        Compute the total absolute oriented contact deviation (inner spheres only).
        """
        dev_sphere, _ = self._contact_deviations_raw(self.top.inner_vertices())
        return float(np.sum((dev_sphere)**2))

    def compute_oriented_contact_dev_per_element(self):
        """
        Compute the absolute oriented contact deviation per sphere and per plane.

        Returns:
            deviations_sphere (n_vertices,): summed absolute deviation per sphere.
            deviations_plane  (n_faces,):    summed absolute deviation per plane.
        """
        dev_sphere, dev_plane = self._contact_deviations_raw(range(len(self.top.vertices)))
        return np.abs(dev_sphere), np.abs(dev_plane)

    def compute_oriented_contact_dev_per_face(self):
        """
        Compute the signed oriented contact deviation per sphere (vertex).

        Returns:
            deviations (n_vertices,): signed deviation summed over adjacent planes per sphere.
        """
        dev_sphere, _ = self._contact_deviations_raw(range(len(self.top.vertices)))
        return dev_sphere

    def setup_optimization(self):
        """
        Setup data structures for optimization of the L-mesh.
        """

        cv  = self.sv_centers.copy()   # sphere centers, per vertex
        rv  = self.sv_radii.copy()     # sphere radii,  per vertex
        npf = self.pf_normals          # plane normals, per face
        hpf = self.pf_dist             # plane offsets, per face

        # Declare variables and set initial values
        self.Optimizer.add_variable("cv",  cv.flatten())   # Center of spheres (per vertex)
        self.Optimizer.add_variable("rv",  rv)             # Radius of spheres (per vertex)
        self.Optimizer.add_variable("npf", npf.flatten())  # Normals of planes (per face)
        self.Optimizer.add_variable("hpf", hpf)            # Plane distances from origin (per face)

        # Get connectivity of L-mesh: faces (planes) incident to each vertex (sphere)
        vertexPlane_adjacentFaces = self.top.vertex_face_adjacency_list()
        sphericalFaces_edges = self.top.edge_faces()
        vertexPlane_vertexPlane_edges = self.top.edge_vertices()

        innerSF = self.top.inner_faces
        # Get initial values for normals 
        vertex_normals = self.orientation * self.base_mesh.vertex_normals[self.base_mesh.inner_vertices()] 

        #Oriented contact constraint
        Or_contact = Or_contactFree()
        Or_contact.name = "or_contact"
        self.Optimizer.add_objective_term(Or_contact, args=(self.top, ), w=1, ce=1)


        # pos_tangential_dist = TDPositive()
        # pos_tangential_dist.name = "pos_tangential_dist"
        # self.Optimizer.add_objective_term(pos_tangential_dist, args=(sphericalFaces_edges,), w=0.01, ce=1)

        # Spherical face adjacency list
        sphericalFace_adjacentFaces = self.top.face_face_adjacency_list()

        self.Optimizer.unitize_variable("npf", 3, 5)
        # Reset iteration bookkeeping — call after all variables and terms are set up
        self.Optimizer.initialize_optimizer(verbose=False, adaptive_mu=False)

    def sphere_centers_mesh(self):
        """
        Return vertices and faces of a mesh representing the centers of or-spheres.

        Spheres now live on vertices, so the centers mesh shares the L-mesh
        combinatorics directly: vertices = sphere centers, faces = L-mesh faces.
        """
        sphereCenters = self.sv_centers
        faces = self.top.faces
        return sphereCenters, faces
    
    def create_map_indices(self):
        """
        Create mapping between L-mesh indices and Base mesh indices stored in a dict.

        NEW CONVENTION (spheres on vertices, planes on faces):

        - "primal": the L-mesh shares the base-mesh combinatorics directly, so
          L-mesh vertex i ≡ base vertex i (sphere) and L-mesh face j ≡ base face j
          (plane). The index maps are identities.
        - "dual":   L-mesh vertices correspond to base-mesh faces and L-mesh faces
          to inner base-mesh vertices.
        """
        if self.combinatorics == "primal":
            # Spheres on vertices ≡ base vertices; planes on faces ≡ base faces.
            n_v = len(self.top.vertices)
            n_f = len(self.top.faces)
            self.LmeshVertex_to_BaseMeshVertex = {vi: vi for vi in range(n_v)}
            self.BaseMeshVertex_to_LmeshVertex = {vi: vi for vi in range(n_v)}
            self.LmeshFace_to_BaseMeshFace     = {fi: fi for fi in range(n_f)}
            self.BaseMeshFace_to_LmeshFace     = {fi: fi for fi in range(n_f)}
            return

        # ---- dual combinatorics -------------------------------------------
        # Get inner vertices of Base mesh
        in_v = self.base_mesh.inner_vertices()

        # Map L-mesh face indices to Base mesh inner vertex indices
        self.LmeshFace_to_BaseMeshInnerVertex = {li: in_v[li] for li in range(len(in_v))}

        # Map Base mesh inner vertex indices to L-mesh face indices
        self.BaseMeshInnerVertex_to_LmeshFace = {in_v[li]: li for li in range(len(in_v))}

        # Map L-mesh vertex plane indices to Base mesh face indices
        self.LmeshVertex_to_BaseMeshFace = {vi: self.base_mesh.faces[vi] for vi in range(len(self.top.vertices))}

        # Map Base mesh face indices to L-mesh vertex plane indices
        self.BaseMeshFace_to_LmeshVertex = {vi: vi for vi, fi in enumerate(self.base_mesh.faces)}

    def compute_contact_circles(self, sph1_id, sph2_id):
        """
        Compute the contact circles between two spheres.

        Calculates the circle centers and radii where two oriented spheres are tangent
        to an enveloping cone. Returns two contact circles (one per sphere) and their
        common normal direction.

        Parameters:
            sph1_id (int): Index of the first spherical face.
            sph2_id (int): Index of the second spherical face.

        Returns:
            tuple: (circ1_c, circ1_r, circ2_c, circ2_r, cn) where circ1_c and circ2_c
                   are circle centers, circ1_r and circ2_r are radii, and cn is the
                   common normal direction.
        """
        c1, r1 = self.sv_centers[sph1_id], self.sv_radii[sph1_id]
        c2, r2 = self.sv_centers[sph2_id], self.sv_radii[sph2_id]

        ar1 = abs(r1)
        ar2 = abs(r2)
        dc = np.linalg.norm(c2 - c1)  # Distance between sphere centers.
        cn = unit(c2 - c1)  # Unit vector from c1 to c2.
        fac = (r1 - r2) / dc 

        if dc**2 - (r1 - r2)**2 < -1e-3:
            raise ValueError(
                f"Negative Tangential Distance between spheres {sph1_id} and {sph2_id}: "
                f"{dc**2 - (r1 - r2)**2:.6f}\n"
                f"sph1 = {mathematica_v(c1)}; r1 = {r1};\n"
                f"sph2 = {mathematica_v(c2)}; r2 = {r2};"
            )
            

        # Compute centers and radii for the contact circles.
        circ1_c = c1 + r1 * fac * cn
        circ2_c = c2 + r2 * fac * cn
        td = np.sqrt(np.abs(dc**2 - (r1 - r2)**2))
        circ1_r = ar1 / dc * td
        circ2_r = ar2 / dc * td

        return circ1_c, circ1_r, circ2_c, circ2_r, cn


    def get_spheres(self):
        """
        Function to get the spheres of the L-mesh.
        
        Returns:
            list: List of Spherical_face objects representing the spheres.
        """
        return self.sv_centers.copy(), self.sv_radii.copy()
    
    def get_or_planes(self):
        """
        Get the oriented planes of the L-mesh.

        Returns:
            tuple: (normals, distances) where normals is an array of plane normals
                   and distances is an array of plane offsets h in n·x + h = 0.
        """

        # self.PF is the single source of truth for plane data (n, h), per face.
        normals   = self.PF[:, :3].copy()
        distances = self.PF[:, 3].copy()

        return normals, distances
    
    
    def compute_isotropic_mesh(self):
        """
        Compute an isotropic mesh. 
        The isotropic map of planes is defined as:
            iso(v, n) = alpha * [n0, n1, n2, ..., -(v*n)]
        The isotropic spheres are defined as:
            iso(c, r) = [(r+c_3)/2, -c1, -c2, (r-c3)/2, 1].[x_1^2 + x2^2, x1, x2, 1, -x3] == 0
        where alpha = 1 / (1 + n[-1]).
        """
        
        self.Iso_vertices = None
        self.Iso_spheres = None
        # Get the number of Oriented planes PV 
        normals, h = self.get_or_planes()    
        self._log_msg(f"Number of Oriented planes: {len(normals)}")
        alpha = 1 / (1 + normals[:, -1])
        self.Iso_vertices = alpha[:, None] * np.concatenate((normals[:, :-1], h.reshape(-1, 1)), axis=1)

        # Get the spheres of the L-mesh
        centers, radii = self.get_spheres()

        # Compute the isotropic spheres
        r_c3  = (radii + centers[:,2]) / 2
        r_c3m = (radii - centers[:,2]) / 2

        self.Iso_spheres = np.concatenate((r_c3[:, None], -centers[:, 0][:, None], -centers[:, 1][:, None], r_c3m[:, None]), axis=1)
    
   
    def i_csphere_inverse_mesh(self, i_center, i_radius)-> None:
        """
        Compute the inverse of an isotropic sphere.
        
        Parameters:
            i_center (array-like): Center of the isotropic cylindrical sphere.
            i_radius (float): Radius of the isotropic cylindrical sphere.    
        """

        # Get oriented planes of the L-mesh (per face)
        or_planes = self.PF

        for i_s in range(len(self.top.vertices)):
            new_center, new_radius = iso_sphere_to_sphere(
                i_csphere_sphere(i_center, i_radius, self.sv_centers[i_s], self.sv_radii[i_s])
            )
            self.top.vertices[i_s, :3] = new_center
            self.top.vertices[i_s,  3] = new_radius


        # Update planes (per face)
        for or_pi in range(len(or_planes)):

            # Get the normal and distance-from-origin of the oriented plane
            normal = or_planes[or_pi, :3]
            h = or_planes[or_pi, 3] # Distance from the plane to the origin (

            # Transform it to isotropic point
            i_pv = 1/ (1 + normal[-1]) * np.concatenate((normal[:-1], [h]), axis=0)

            # Invert the isotropic point and transform it to oriented plane
            new_or_plane = i_point_to_or_plane(i_csphere_inverse_point(i_center, i_radius, i_pv))

            self.PF[or_pi] = new_or_plane


    def lmesh_offset(self, offset):
        """
        Apply an offset to the L-mesh spheres.
        """

        new_centers = self.sv_centers
        new_radii   = self.sv_radii + offset

        init_normals   = self.PF[:, : 3]
        init_distances = self.PF[:, 3] + offset

        new_normals = init_normals.copy()
        new_distances = init_distances.copy()

        # Spheres sitting on each plane-face are that face's vertices
        spheres_per_face = self.top.faces

        for fi in range(len(self.top.faces)):

            normals = new_normals[fi]
            h     = new_distances[fi]

            if len(spheres_per_face[fi]) > 2:

                centers = new_centers[spheres_per_face[fi]]
                radii   = new_radii[spheres_per_face[fi]]
                # Approximate new planes using previous as initial guess for the optimization
                na, ha, _ = plane_tangent_spheres_approximation(centers, radii, normals, h)
                new_normals[fi]   = na
                new_distances[fi] = ha

        self.update_spheres_planes(new_centers, new_radii, new_normals, new_distances)

    
    def _annulus_geometry(self, cp_i, cp_j, circle_center, cone_axis, nc_i, nc_j, thickness):
        """Build one annulus by lofting two extruded circular arcs (outer and inner).

        The outer arc extrudes cp_i/cp_j by +thickness along nc_i/nc_j.
        The inner arc extrudes by -0.2*thickness (thin inner boundary).
        Both arcs are projected onto the cone axis to find their center and radius.

        Parameters:
            cp_i, cp_j   : contact points at vertices i and j for this sphere face
            circle_center: center of the circular arc
            cone_axis    : unit direction shared by both spheres
            nc_i, nc_j   : extrusion directions at vertices i and j
            thickness    : extrusion magnitude
        Returns:
            (V, F) arrays for the annulus mesh.
        """
        def _arc_from_offset(offset):
            
            ei  = cp_i + nc_i * offset
            ej  = cp_j + nc_j * offset
    
            radius = 0.5 * (np.linalg.norm(ei - circle_center) + np.linalg.norm(ej - circle_center))
            return circular_arc(circle_center, cone_axis, radius, ei, ej)

        outer_arc = _arc_from_offset(thickness)
        inner_arc = _arc_from_offset(-0.2 * thickness)
        return loft_curves(outer_arc, inner_arc)

    def planar_support_structure(self, thickness):
        """
        Create a planar support structure for the L-mesh.

        Requires the sphere-centers mesh to have planar faces for the structure to be
        planar. Every edge of a plane polygon is extruded along the normal of the plane
        through the centers of the spheres on that plane; every conical edge gets two
        annuli (one per sphere) bounded by the extruded contact points.

        Spheres live on vertices and planes on faces: an edge joins spheres (si, sj)
        and separates planes (pk, pl).

        Parameters:
            thickness (float): Thickness of the support structure.
        Returns:
            tuple: (vertices, faces) of the support structure mesh.
        """
        # Normal of each plane's sphere-centers plane, oriented like the plane itself
        centers_normals = np.zeros((len(self.top.faces), 3))
        list_support_planes_vertices = []
        list_support_planes_faces = []
        for f, spheres in enumerate(self.top.faces):
            if len(spheres) < 3:
                continue
            normal = unit(plane_normal(self.sv_centers[list(spheres)]))
            normal *= np.sign(np.dot(normal, self.PF[f, :3]))
            centers_normals[f] = normal

            # Extrude every edge of the plane polygon along that normal
            polygon = self.vp_polygons[f]
            for i in range(len(polygon)):
                vi = polygon[i]
                vj = polygon[(i + 1) % len(polygon)]
                p0 = vi + normal * thickness
                p1 = vj + normal * thickness
                p2 = vj - normal * 0.2 * thickness
                p3 = vi - normal * 0.2 * thickness
                list_support_planes_vertices.append([p0, p1, p2, p3])
                list_support_planes_faces.append([[0, 1, 2, 3]])

        # Annuli around the conical edges between inner planes
        inner_planes = set(int(f) for f in self.top.inner_faces)
        si, sj, pk, pl = self.top.edge_vertices_faces()   # spheres, planes
        for e in range(len(si)):
            if pk[e] not in inner_planes or pl[e] not in inner_planes:
                continue
            cone_axis = unit(self.sv_centers[sj[e]] - self.sv_centers[si[e]])
            nc_k, nc_l = centers_normals[pk[e]], centers_normals[pl[e]]

            # contact points are keyed (sphere, plane)
            cp_ik = self.contact_points[(si[e], pk[e])]
            cp_il = self.contact_points[(si[e], pl[e])]
            cp_jk = self.contact_points[(sj[e], pk[e])]
            cp_jl = self.contact_points[(sj[e], pl[e])]

            center_ij = self.HCE[(si[e], sj[e])]["center"]
            center_ji = self.HCE[(sj[e], si[e])]["center"]

            annulus_i = self._annulus_geometry(cp_ik, cp_il, center_ij, cone_axis, nc_k, nc_l, thickness)
            annulus_j = self._annulus_geometry(cp_jk, cp_jl, center_ji, cone_axis, nc_k, nc_l, thickness)
            for V, F in (annulus_i, annulus_j):
                list_support_planes_vertices.append(V)
                list_support_planes_faces.append(F)

        return merge_meshes(list_support_planes_vertices, list_support_planes_faces)

    def support_structure_second_envelope(self):
        """
        Compute the support structure that connects the primary L-mesh envelope
        to the second envelope.

        Requires both ``compute_lmesh_structure`` and ``compute_second_envelope``
        to have been called first (``extract_support_structure("second_envelope")``
        does the latter).

        Two families of panels are built:

        * **Plane-polygon quads** — for each inner plane, every edge of the primary
          polygon is joined to the matching edge of the reflected polygon, producing
          a ruled quad strip around the perimeter.

        * **Arc annuli** — for each edge between two inner planes, the primary
          circular arc ``HCE[(si, sj)]`` of sphere si is lofted against the
          corresponding second-envelope arc ``se_HCE[(si, sj)]``, and likewise for
          sphere sj.

        Returns:
            tuple: (V, F) merged mesh arrays, or None if the second envelope has
            not been computed.
        """
        if not self.se_vp_polygons:
            self._log_msg("Second envelope not computed. Call compute_second_envelope() first.")
            return None

        inner_planes = set(int(f) for f in self.top.inner_faces)   # planes live on faces
        verts_list = []
        faces_list = []

        # ── plane-polygon quads ────────────────────────────────────────────
        for f in sorted(inner_planes):
            poly_primary = self.vp_polygons[f]
            poly_second  = self.se_vp_polygons[f]

            if (poly_primary is None or poly_second is None
                    or len(poly_primary) < 2 or len(poly_second) < 2
                    or len(poly_primary) != len(poly_second)):
                continue

            n = len(poly_primary)
            for i in range(n):
                p0 = poly_primary[i]
                p1 = poly_primary[(i + 1) % n]
                p2 = poly_second[(i + 1) % n]
                p3 = poly_second[i]
                verts_list.append([p0, p1, p2, p3])
                faces_list.append([[0, 1, 2, 3]])

        # ── arc annuli ─────────────────────────────────────────────────────
        si, sj, pk, pl = self.top.edge_vertices_faces()   # spheres, planes
        for e in range(len(si)):
            if pk[e] not in inner_planes or pl[e] not in inner_planes:
                continue
            for key in ((int(si[e]), int(sj[e])), (int(sj[e]), int(si[e]))):
                hce, se_hce = self.HCE.get(key), self.se_HCE.get(key)
                if hce is None or se_hce is None:
                    continue
                arc, se_arc = hce["c_arc"], se_hce["c_arc"]
                if arc is not None and se_arc is not None:
                    V, F = loft_curves(arc, se_arc)
                    verts_list.append(V)
                    faces_list.append(F)

        if not verts_list:
            return None

        support_V, support_F = merge_meshes(verts_list, faces_list)
        return support_V, support_F

    def compute_second_envelope(self):
        """
        Compute the second envelope of the L-mesh.

        For each vertex plane, reflect it across the plane spanned by the centres
        of its adjacent spheres (the local sphere-centre plane).  The reflected
        planes together with the *same* spheres define a second L-mesh structure
        whose vertex polygons, contact points, conical edge arcs, and spherical
        panel boundaries are stored in the ``se_*`` attributes.

        Requires that ``compute_lmesh_structure`` has already been called.
        """

        lF  = self.top.faces          # face = plane; its vertices are spheres
        nPF = self.PF[:, :3]
        hPF = self.PF[:, 3]

        # ── 1. Reflect every plane (per face) ──────────────────────────────
        # se_normals[f] = reflected (n, h); falls back to original for under-determined faces
        n_all = nPF.copy()
        h_all = hPF.copy()

        se_normals = n_all.copy()
        se_distances = h_all.copy()

        for fi in range(len(lF)):
            adj = lF[fi]   # spheres (vertices) sitting on this plane-face

            # If there are fewer than 3 spheres, we cannot reliably compute a reflection
            # plane, so we approximate using the original plane with flipped normal
            if len(adj) < 3:
                normal_init = n_all[fi]
                h_init = h_all[fi]
                centers = self.sv_centers[adj]
                radii  = self.sv_radii[adj]

                na, ha, _ = plane_tangent_spheres_approximation(centers, radii, -normal_init, h_init)
                se_normals[fi]   = na
                se_distances[fi] = ha
            else:
                centers = self.sv_centers[adj]
                mirror_n = unit(plane_normal(centers))
                mirror_h = -mirror_n @ np.mean(centers, axis=0)
                nr, hr = reflect_plane(n_all[fi], h_all[fi], mirror_n, mirror_h)
                se_normals[fi]   = nr
                se_distances[fi] = hr

        # ── 2. Contact points with reflected planes ─────────────────────────
        # Loop spheres (vertices); incident faces carry the reflected planes.
        vf_adj = self.top.vertex_face_adjacency_list()
        self.se_contact_points = {}
        for vi in range(len(self.top.vertices)):
            center = self.sv_centers[vi]
            radius = self.sv_radii[vi]
            for fi in vf_adj[vi]:
                self.se_contact_points[(vi, fi)] = center - radius * se_normals[fi]

        # ── 3. Plane polygons (per face) of the second envelope ─────────────
        self.se_vp_polygons = []
        for fi in range(len(lF)):
            sphere_verts = lF[fi]
            polygon = [
                project_point_plane(
                    self.se_contact_points[(vi, fi)],
                    se_normals[fi],
                    se_distances[fi]
                )
                for vi in sphere_verts
            ]
            self.se_vp_polygons.append(polygon)

        # ── 4. Conical edge arcs for the second envelope ────────────────────
        # Sphere pair = edge vertices; plane pair = the two adjacent faces.
        svi, svj   = self.top.edge_vertices()
        pfi_, pfj_ = self.top.edge_faces()

        self.se_HCE = {}

        valid = np.where((pfi_ != -1) & (pfj_ != -1))[0]
        if len(valid) == 0:
            return

        ei  = svi[valid];  ej  = svj[valid]    # sphere-vertex indices
        vpi = pfi_[valid]; vpj = pfj_[valid]   # plane-face indices

        centers = self.sv_centers
        radii   = self.sv_radii

        c1 = centers[ei];  r1 = radii[ei]
        c2 = centers[ej];  r2 = radii[ej]

        diff = c2 - c1
        dc   = np.linalg.norm(diff, axis=1)
        cn   = diff / dc[:, None]

        td2  = dc**2 - (r1 - r2)**2
        fac  = (r1 - r2) / dc
        td   = np.sqrt(np.abs(td2))

        circi_c = c1 + (r1 * fac)[:, None] * cn
        circj_c = c2 + (r2 * fac)[:, None] * cn
        circi_r = np.abs(r1) / dc * td
        circj_r = np.abs(r2) / dc * td

        E = len(valid)
        num_pts = 20

        pij1 = np.array([self.se_contact_points[(ei[k], vpi[k])] for k in range(E)])
        pij2 = np.array([self.se_contact_points[(ei[k], vpj[k])] for k in range(E)])
        pji1 = np.array([self.se_contact_points[(ej[k], vpj[k])] for k in range(E)])
        pji2 = np.array([self.se_contact_points[(ej[k], vpi[k])] for k in range(E)])

        def _batch_arcs(centers_b, normals_b, radii_b, p0s, p1s):
            degen = np.abs(radii_b) < 1e-12
            v1 = p0s - centers_b
            v2 = p1s - centers_b
            v1 -= np.einsum('ij,ij->i', v1, normals_b)[:, None] * normals_b
            v2 -= np.einsum('ij,ij->i', v2, normals_b)[:, None] * normals_b
            n1 = np.linalg.norm(v1, axis=1, keepdims=True)
            n2 = np.linalg.norm(v2, axis=1, keepdims=True)
            n1 = np.where(n1 < 1e-12, 1.0, n1)
            n2 = np.where(n2 < 1e-12, 1.0, n2)
            v1 /= n1;  v2 /= n2
            cos_a = np.clip(np.einsum('ij,ij->i', v1, v2), -1.0, 1.0)
            angle = np.arccos(cos_a)
            u2 = v2 - np.einsum('ij,ij->i', v2, v1)[:, None] * v1
            u2 -= np.einsum('ij,ij->i', u2, normals_b)[:, None] * normals_b
            nu2 = np.linalg.norm(u2, axis=1, keepdims=True)
            nu2 = np.where(nu2 < 1e-12, 1.0, nu2)
            u2 /= nu2
            t = np.linspace(0, 1, num_pts)
            theta = angle[:, None] * t[None, :]
            arcs = (centers_b[:, None, :]
                    + radii_b[:, None, None] * (
                        np.cos(theta)[:, :, None] * v1[:, None, :]
                        + np.sin(theta)[:, :, None] * u2[:, None, :]
                    ))
            arcs[degen] = centers_b[degen, None, :]
            return arcs   # (E, P, 3)

        arcs_ij = _batch_arcs(circi_c, cn, circi_r, pij1, pij2)
        arcs_ji = _batch_arcs(circj_c, cn, circj_r, pji1, pji2)

        for k in range(E):
            key_ij = (int(ei[k]), int(ej[k]))
            key_ji = (int(ej[k]), int(ei[k]))
            self.se_HCE[key_ij] = {"c_arc": arcs_ij[k]}
            self.se_HCE[key_ji] = {"c_arc": arcs_ji[k]}

        # ── 5. Spherical panel boundaries ───────────────────────────────────
        n = len(self.sv_centers)
        self.se_sv_boundary_curve  = [None] * n
        self.se_sv_boundary_curves = [[]]   * n
        self.se_sv_panels          = [None] * n

        adj_ff = self.top.vertex_adjacency_list()
        for i in range(n):
            radius = self.sv_radii[i]
            if abs(radius) < rad_tol:
                self.se_sv_panels[i] = (self.sv_centers[i], None)
                continue
            boundary        = []
            boundary_curves = []
            for j in adj_ff[i]:
                if (i, j) in self.se_HCE:
                    arc = self.se_HCE[(i, j)]["c_arc"]
                    if arc is not None and len(arc) > 1:
                        boundary.append(arc[:-1])
                        boundary_curves.append(arc)
            self.se_sv_boundary_curves[i] = boundary_curves
            if boundary:
                self.se_sv_boundary_curve[i] = np.concatenate(boundary, axis=0)

    def trivial_support_structure(self, thickness):
        """
        Trivial support structure: every edge of a plane polygon extruded along the
        plane normal, and every conical-edge arc offset towards its sphere center.

        Parameters:
            thickness (float): extrusion thickness (0.3 * thickness on the inner side).
        Returns:
            tuple: (vertices, faces) of the support structure mesh.
        """
        vertices = []
        faces = []
        faceCount = 0

        # Planes live on faces: polygon f lies on plane PF[f]
        for f, polygonVertices in enumerate(self.vp_polygons):
            normal = self.PF[f, :3]
            for i in range(len(polygonVertices)):
                vi = polygonVertices[i]
                vj = polygonVertices[(i + 1) % len(polygonVertices)]

                # Extend the edge in the direction of the normal to create a quad
                p0 = vi + normal * thickness
                p1 = vj + normal * thickness
                p2 = vj - normal * 0.3 * thickness
                p3 = vi - normal * 0.3 * thickness

                vertices.extend([p0, p1, p2, p3])
                faces.append([faceCount, faceCount + 1, faceCount + 2, faceCount + 3])
                faceCount += 4

        # Conical edges: sphere pair = edge vertices (HCE exists for inner edges only)
        sphEdge_i, sphEdge_j = self.top.edge_vertices()
        for sphi, sphj in zip(sphEdge_i, sphEdge_j):
            conicalEdge_ij = self.HCE.get((sphi, sphj))
            conicalEdge_ji = self.HCE.get((sphj, sphi))
            if conicalEdge_ij is None or conicalEdge_ji is None:
                continue

            c_ij = self.sv_centers[sphi]
            c_ji = self.sv_centers[sphj]

            circArc_ij = conicalEdge_ij["c_arc"]
            circArc_ji = conicalEdge_ji["c_arc"]

            # Offset each arc towards its sphere center
            direction_ij = unit(c_ij - circArc_ij)
            direction_ji = unit(c_ji - circArc_ji)

            annuliVertices_ij, annuliFaces_ij = offset_curve(circArc_ij, direction_ij, thickness, thickness * 0.3)
            annuliVertices_ji, annuliFaces_ji = offset_curve(circArc_ji, direction_ji, thickness, thickness * 0.3)

            vertices.extend(annuliVertices_ij)
            faces.extend([[faceCount + i for i in face] for face in annuliFaces_ij])
            faceCount += len(annuliVertices_ij)

            vertices.extend(annuliVertices_ji)
            faces.extend([[faceCount + i for i in face] for face in annuliFaces_ji])
            faceCount += len(annuliVertices_ji)

        return np.array(vertices), faces

    def sphere_center_mesh(self):
        """
        Create a mesh representing the centers of the spheres.

        Returns:
            tuple: (vertices, faces) where vertices is an array of vertex positions
                   and faces is an array of face indices.
        """
        innerVertexPlanes = self.top.inner_vertices()
        vertexPlaneSphereAdjacency = self.top.vertex_face_adjacency_list()

        centers = self.sv_centers
        faces = []

        for i in innerVertexPlanes:
            faces.append(vertexPlaneSphereAdjacency[i])
        return centers, faces

    # ------------------------------------------------------------------ #
    # Export — panel builders                                             #
    #                                                                     #
    # Each helper returns plain (vertices, faces) lists so the same code   #
    # serves both the main envelope and the second envelope; the caller    #
    # passes the relevant panel/arc/polygon containers.                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _write_wireframe(path, header, curves):
        """
        Write a set of polylines as an OBJ wireframe (v + l elements).

        Parameters:
            path (str): output file.
            header (str): comment written at the top of the file.
            curves (list): list of point arrays, each an open polyline.
        """
        curves = [c for c in curves if c is not None and len(c) >= 2]
        if not curves:
            return
        with open(path, 'w') as f:
            f.write(f"# {header}\n")
            for curve in curves:
                for v in curve:
                    f.write(f"v {v[0]} {v[1]} {v[2]}\n")
            idx = 1
            for curve in curves:
                f.write("l " + " ".join(str(idx + q) for q in range(len(curve))) + "\n")
                idx += len(curve)

    @staticmethod
    def _write_points(path, header, points):
        """Write a points-only OBJ (bare `v` lines, no faces)."""
        if len(points) == 0:
            return
        with open(path, 'w') as f:
            f.write(f"# {header}\n")
            for p in points:
                f.write(f"v {p[0]} {p[1]} {p[2]}\n")

    @staticmethod
    def _write_segments(path, header, segments):
        """Write segments as an OBJ wireframe: vertex pairs plus `l` elements."""
        if len(segments) == 0:
            return
        with open(path, 'w') as f:
            f.write(f"# {header}\n")
            for p0, p1 in segments:
                f.write(f"v {p0[0]} {p0[1]} {p0[2]}\n")
                f.write(f"v {p1[0]} {p1[1]} {p1[2]}\n")
            for q in range(len(segments)):
                f.write(f"l {2*q + 1} {2*q + 2}\n")

    # ------------------------------------------------------------------ #
    # Export — public entry points                                        #
    # ------------------------------------------------------------------ #

    def export_data(self, folder, name='lmesh'):
        """
        Export the L-mesh: panel geometry, sphere/plane data and a state pickle.

        This is a thin orchestrator over the individual exporters:
            export_extended_lmesh  -> panel OBJs (+ degenerate points/segments)
            export_lmesh_data      -> or_spheres.dat / or_planes.dat
            export_iso_mesh        -> isotropic mesh OBJ
            export_centers         -> sphere centers OBJ
            export_base_mesh       -> base mesh OBJ
            save_lmesh             -> spheres + planes + topology pickle

        Support structures are NOT exported here — call export_support_structure()
        or export_axes_nodes() explicitly.

        Parameters:
            folder (str): Path to the folder where the data will be saved.
            name (str): Base name for the exported files (without extension).
        """
        if not os.path.exists(folder):
            os.makedirs(folder)

        self.export_extended_lmesh(folder, name)
        self.export_lmesh_data(folder, name)
        self.export_iso_mesh(folder, name)
        self.export_centers(folder, name)
        self.export_base_mesh(folder, name)
        # The pickle's natural suffix is "lmesh"; avoid the "lmesh_lmesh.pkl"
        # doubling when the base name is already "lmesh" (or empty).
        pkl_name = f"{name}_lmesh.pkl" if name and name != "lmesh" else "lmesh.pkl"
        self.save_lmesh(os.path.join(folder, pkl_name))

        print("L-mesh exported to folder:", folder)

    def export_lmesh_data(self, folder, name='lmesh'):
        """
        Export the or-spheres and or-planes as .dat files.

        Format (adjacency lines follow the geometry, one per element):
            or_spheres.dat:  s cx cy cz r      then  adj i j k ...
            or_planes.dat:   p nx ny nz h      then  adj i j k ...

        Spheres live on vertices, planes on faces, so sphere adjacency is vertex
        adjacency and plane adjacency is face adjacency.
        """
        if not os.path.exists(folder):
            os.makedirs(folder)

        pre = f"{name}_" if name else ""
        n_sph = len(self.top.vertices)
        with open(os.path.join(folder, pre + "or_spheres.dat"), 'w') as f:
            f.write(f"# Or-sphere data\n# Total number of spheres: {n_sph}\n")
            for i in range(n_sph):
                c, r = self.sv_centers[i], self.sv_radii[i]
                f.write(f"s {c[0]} {c[1]} {c[2]} {r}\n")
            sph_adj = self.top.vertex_adjacency_list()
            for i in range(n_sph):
                f.write("adj " + " ".join(str(j) for j in sph_adj[i]) + "\n")

        n_pl = len(self.PF)
        with open(os.path.join(folder, pre + "or_planes.dat"), 'w') as f:
            f.write(f"# Or-plane data\n# Total number of planes: {n_pl}\n")
            for i in range(n_pl):
                n_, h_ = self.PF[i][:3], self.PF[i][3]
                f.write(f"p {n_[0]} {n_[1]} {n_[2]} {h_}\n")
            pl_adj = self.top.face_face_adjacency_list()
            for i in range(n_pl):
                f.write("adj " + " ".join(str(j) for j in pl_adj[i]) + "\n")

    def export_iso_mesh(self, folder, name='lmesh'):
        """Export the isotropic mesh (normalized) as an OBJ."""
        if not os.path.exists(folder):
            os.makedirs(folder)
        self.compute_isotropic_mesh()
        pre = f"{name}_" if name else ""
        write_obj(os.path.join(folder, pre + "iso_mesh.obj"),
                  normalize_vertices(self.Iso_vertices), self.top.faces)

    def export_base_mesh(self, folder, name='lmesh'):
        """Export the base mesh the L-mesh was built from, as an OBJ.

        The base mesh is already carried in the state pickle; writing it here
        too keeps the exported folder self-contained, so the input geometry can
        be inspected alongside the panels without unpickling.
        """
        if self.base_mesh is None:
            return
        if not os.path.exists(folder):
            os.makedirs(folder)
        pre = f"{name}_" if name else ""
        write_obj(os.path.join(folder, pre + "base_mesh.obj"),
                  self.base_mesh.vertices, self.base_mesh.faces)

    def export_centers(self, folder, name='lmesh'):
        """Export the mesh of or-sphere centers as an OBJ."""
        if not os.path.exists(folder):
            os.makedirs(folder)
        vertices, faces = self.sphere_centers_mesh()
        pre = f"{name}_" if name else ""
        write_obj(os.path.join(folder, pre + "sphere_centers.obj"), vertices, faces)

    def export_extended_lmesh(self, folder, name='lmesh'):
        """
        Export the extended L-mesh panel geometry as OBJ files.

        Writes extract_mesh(wireframe=True) — and, when the second envelope has been
        computed, extract_second_envelope(wireframe=True) with a "_se" infix:

            _sph_panels.obj          spherical panels
            _cone_panels.obj         conical edge strips
            _flat_panels.obj         flat (plane) panels
            _sph_panels_wf.obj       spherical panel boundary curves
            _flat_panels_wf.obj      flat panel boundary curves

        Degenerate elements are written separately so every OBJ stays valid:

            _degenerated_spheres.obj  centers of |r| < rad_tol spheres (points)
            _degenerated_cones.obj    collapsed conical edges (line elements)

        Parameters:
            folder (str): Path to the folder where the data will be saved.
            name (str): Base name for the exported files (without extension).
        """
        if not os.path.exists(folder):
            os.makedirs(folder)

        # `prefix` carries its trailing separator (or is empty), so an empty base
        # name yields bare filenames without a leading "_".
        self._write_envelope(folder, f"{name}_" if name else "",
                             self.extract_mesh(wireframe=True))

        # ── second envelope (only if computed) ─────────────────────────────
        if self.se_vp_polygons:
            self._write_envelope(folder, f"{name}_se_" if name else "se_",
                                 self.extract_second_envelope(wireframe=True))
            print("Second envelope exported to folder:", folder)

    def _write_envelope(self, folder, prefix, mesh):
        """Write one extract_mesh() dict as the OBJ files listed in export_extended_lmesh."""
        def path(f):
            return os.path.join(folder, prefix + f)

        write_obj(path("sph_panels.obj"), *mesh["spherical_panels"])
        self._write_points(path("degenerated_spheres.obj"),
                           "Degenerate or-spheres (|r| < rad_tol) — centers only",
                           mesh.get("degenerated_spheres", []))

        write_obj(path("cone_panels.obj"), *mesh["conical_panels"])
        if "degenerated_cones" in mesh:
            V, E = mesh["degenerated_cones"]
            self._write_segments(path("degenerated_cones.obj"),
                                 "Degenerate conical edges — line segments",
                                 [(V[a], V[b]) for a, b in E])

        write_obj(path("flat_panels.obj"), *mesh["flat_panels"])

        if "spherical_wireframe" in mesh:
            self._write_wireframe(path("sph_panels_wf.obj"),
                                  "Spherical panel boundary curves",
                                  _polylines(*mesh["spherical_wireframe"]))
        if "flat_wireframe" in mesh:
            self._write_wireframe(path("flat_panels_wf.obj"),
                                  "Flat panel boundary curves",
                                  _polylines(*mesh["flat_wireframe"]))

    def export_support_structure(self, folder, name='lmesh', flag='trivial', thickness=0.05):
        """
        Export a support structure mesh to an OBJ file.

        Parameters:
            folder (str): Path to the output folder (created if absent).
            name (str): Base filename (without extension).
            flag (str): Which support structure to export — one of
                ``'trivial'``, ``'planar'``, or ``'second_envelope'``.
            thickness (float): Thickness passed to trivial/planar structures
                (ignored for ``'second_envelope'``).
        """
        valid_flags = ('trivial', 'planar', 'second_envelope')
        if flag not in valid_flags:
            raise ValueError(f"flag must be one of {valid_flags}, got '{flag}'")

        if not os.path.exists(folder):
            os.makedirs(folder)

        result = self.extract_support_structure(flag, thickness)
        if result is None:
            self._log_msg(f"Support structure ({flag}) could not be computed.")
            return
        V, F = result

        flag_suffix = {'trivial': 'trivial_support_structure',
                       'planar': 'planar_support_structure',
                       'second_envelope': 'second_env_support_structure'}
        out_path = os.path.join(folder, name + f"_{flag_suffix[flag]}.obj")
        write_obj(out_path, V, F)
        print(f"Support structure ({flag}) exported to: {out_path}")

    def axes_nodes(self, length=0.05, flag="trivial"):
        """
        Compute axis-node line segments for visualization/export.

        Each returned segment is a pair (p_start, p_end) representing a short
        axis rod anchored at a contact point (trivial) or polygon vertex (planar).

        Parameters:
            length (float): Full length of each axis rod.
            flag (str): ``'trivial'`` or ``'planar'``.

        Returns:
            nodes (np.ndarray, (N,3)): start points of each segment.
            tips  (np.ndarray, (N,3)): end   points of each segment.
        """
        nodes, tips = [], []

        if flag == "trivial":
            inn_v  = self.top.inner_vertices()        # spheres live on vertices
            vf_adj = self.top.vertex_face_adjacency_list()
            for si in inn_v:
                center = self.sv_centers[si]
                for vpi in vf_adj[si]:                # incident faces = planes
                    cp  = self.contact_points[(si, vpi)]
                    d   = center - cp
                    nd  = np.linalg.norm(d)
                    if nd < 1e-12:
                        continue
                    d = d / nd
                    # rod: starts 0.2*length away from center, ends at length toward center
                    nodes.append(cp + 0.2 * length * d)
                    tips.append(cp + length * d)

        elif flag == "planar":
            # Planes live on faces now; spheres on a plane-face are its vertices.
            inn_f = set(self.top.inner_faces)
            lF    = self.top.faces
            nPF   = self.pf_normals
            for fi, polygon in enumerate(self.vp_polygons):
                if fi not in inn_f:
                    continue
                adj_sfs = lF[fi]
                if len(adj_sfs) < 3:
                    continue
                centers = self.sv_centers[adj_sfs]
                normal  = unit(plane_normal(centers))
                # orient normal consistently with the face-plane normal
                normal *= np.sign(np.dot(normal, nPF[fi]))
                for p in polygon:
                    nodes.append(p - 0.2 * length * normal)
                    tips.append(p + length * normal)

        else:
            raise ValueError(f"flag must be 'trivial' or 'planar', got '{flag}'")

        return np.array(nodes, dtype=float), np.array(tips, dtype=float)

    def normals_segments(self, length=0.05, flip=False):
        """Return (verts, edges) for all normal segments — no rendering calls."""
        inn_f = set(self.top.inner_faces)
        inn_v = set(self.top.inner_vertices())
        sign  = -1.0 if flip else 1.0

        all_nodes, all_tips = [], []

        def _emit(p, d):
            nd = np.linalg.norm(d)
            if nd < 1e-12:
                return
            all_nodes.append(p)
            all_tips.append(p + sign * length * d / nd)

        # Contact-point normals: loop spheres (vertices), incident faces are planes
        vf_adj = self.top.vertex_face_adjacency_list()
        for vi in inn_v:
            center = self.sv_centers[vi]
            for fi in vf_adj[vi]:
                cp = self.contact_points[(vi, fi)]
                _emit(cp, center - cp)
        # Arc-midpoint normals: boundary curves are per sphere (vertex)
        for vi in inn_v:
            center = self.sv_centers[vi]
            for arc in self.sv_boundary_curves[vi]:
                arc = np.asarray(arc)
                if len(arc) < 2:
                    continue
                p = arc[len(arc) // 2]
                _emit(p, center - p)

        # Polygon-edge normals: vp_polygons are per face (plane); its vertices are spheres
        nPF = self.pf_normals
        lF  = self.top.faces
        for fi, polygon in enumerate(self.vp_polygons):
            if fi not in inn_f or len(polygon) < 2:
                continue
            plane_n = nPF[fi]
            n_verts = len(polygon)
            sphere_verts = lF[fi]
            for k in range(n_verts):
                mid = 0.5 * (np.asarray(polygon[k]) +
                             np.asarray(polygon[(k + 1) % n_verts]))
                adj_center = self.sv_centers[sphere_verts[k]]
                d = plane_n * np.sign(np.dot(plane_n, adj_center - mid))
                _emit(mid, d)

        if not all_nodes:
            return None, None
        n     = len(all_nodes)
        verts = np.empty((2 * n, 3))
        verts[0::2] = np.array(all_nodes)
        verts[1::2] = np.array(all_tips)
        edges = np.column_stack([np.arange(0, 2 * n, 2), np.arange(1, 2 * n, 2)])
        return verts, edges

    def export_normals(self, folder, name="lmesh", length=0.05, flip=False):
        """
        Export normal segments to an OBJ file as line elements.

        Covers contact-point normals, arc midpoint normals, and polygon-edge
        midpoint normals — the same three sets as ``normals_segments``.

        Parameters:
            folder (str): Output folder (created if absent).
            name (str): Base filename (without extension).
            length (float): Length of each normal rod.
            flip (bool): Reverse all normal directions when True.
        """
        inn_f = set(self.top.inner_faces)
        inn_v = set(self.top.inner_vertices())
        sign  = -1.0 if flip else 1.0

        nodes, tips = [], []

        def _emit(p, d):
            nd = np.linalg.norm(d)
            if nd < 1e-12:
                return
            nodes.append(np.asarray(p, dtype=float))
            tips.append(np.asarray(p, dtype=float) + sign * length * d / nd)

        lF     = self.top.faces
        nPF    = self.pf_normals
        vf_adj = self.top.vertex_face_adjacency_list()

        # Contact-point normals: loop spheres (vertices), incident faces are planes
        for vi in inn_v:
            center = self.sv_centers[vi]
            for fi in vf_adj[vi]:
                cp = self.contact_points[(vi, fi)]
                _emit(cp, center - cp)

        # Arc-midpoint normals: boundary curves are per sphere (vertex)
        for vi in inn_v:
            center = self.sv_centers[vi]
            for arc in self.sv_boundary_curves[vi]:
                arc = np.asarray(arc)
                if len(arc) < 2:
                    continue
                p = arc[len(arc) // 2]
                _emit(p, center - p)

        # Polygon-edge normals: vp_polygons are per face (plane); its vertices are spheres
        for fi, polygon in enumerate(self.vp_polygons):
            if fi not in inn_f or len(polygon) < 2:
                continue
            plane_n = nPF[fi]
            n_verts = len(polygon)
            sphere_verts = lF[fi]
            for k in range(n_verts):
                mid = 0.5 * (np.asarray(polygon[k]) +
                             np.asarray(polygon[(k + 1) % n_verts]))
                adj_center = self.sv_centers[sphere_verts[k]]
                d = plane_n * np.sign(np.dot(plane_n, adj_center - mid))
                _emit(mid, d)

        if not nodes:
            self._log_msg("export_normals: nothing to export")
            return

        if not os.path.exists(folder):
            os.makedirs(folder)

        out_path = os.path.join(folder, f"{name}_normals.obj")
        n = len(nodes)
        with open(out_path, "w") as f:
            for p in nodes:
                f.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
            for p in tips:
                f.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
            for i in range(n):
                f.write(f"l {i + 1} {n + i + 1}\n")
        print(f"Normals exported to: {out_path}")

    def export_axes_nodes(self, folder, name="lmesh", length=0.05, flag="trivial"):
        """
        Export axis-node segments to an OBJ file as line elements.

        Parameters:
            folder (str): Output folder (created if absent).
            name (str): Base filename (without extension).
            length (float): Full length of each axis rod.
            flag (str): ``'trivial'`` or ``'planar'``.
        """
        nodes, tips = self.axes_nodes(length=length, flag=flag)
        if len(nodes) == 0:
            self._log_msg(f"axes_nodes ({flag}): nothing to export")
            return

        if not os.path.exists(folder):
            os.makedirs(folder)

        out_path = os.path.join(folder, f"{name}_axes_nodes_{flag}.obj")
        with open(out_path, "w") as f:
            for p in nodes:
                f.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
            for p in tips:
                f.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
            n = len(nodes)
            for i in range(n):
                f.write(f"l {i + 1} {n + i + 1}\n")
        print(f"Axes nodes ({flag}) exported to: {out_path}")

    def compute_panel_meshes(self):
        """Triangulate the spherical panel (V, F) of every sphere into sv_panels.

        Called lazily by extract_mesh() (and so by the serializers and the export).
        """
        self.sv_panels = self._panels_from_boundaries(self.sv_boundary_curve)

    # ------------------------------------------------------------------ #
    # Mesh extraction — plain arrays for display (lmesh_serializers) and  #
    # export. Nothing here draws.                                         #
    # ------------------------------------------------------------------ #

    def extract_mesh(self, wireframe=False):
        """
        Panel meshes of the L-mesh (first envelope).

        Parameters:
            wireframe (bool): also return the panel boundary curves.

        Returns:
            dict with
                "spherical_panels"     (V, F)  panels of the inner spheres
                "conical_panels"       (V, F)  conical strips of the inner edges (quads)
                "flat_panels"          (V, F)  polygons of the inner planes
            only when such elements exist:
                "degenerated_spheres"  V       centers of spheres with |r| < rad_tol (points)
                "degenerated_cones"    (V, E)  conical edges collapsed to a segment
            with ``wireframe=True``:
                "spherical_wireframe"  (P, E)  boundary arcs of the spherical panels
                "flat_wireframe"       (P, E)  boundary polygons of the flat panels

            V, P are (n, 3) arrays, F a list of index lists (flat panels are
            polygons), E an (m, 2) array. The spherical panels are triangulated on
            first use (compute_panel_meshes).
        """
        if not any(p is not None and p[1] is not None for p in self.sv_panels):
            self.compute_panel_meshes()
        return self._extract_envelope(self.sv_panels, self.HCE, self.vp_polygons,
                                      self.sv_boundary_curve, self.sv_boundary_curves,
                                      wireframe)

    def extract_second_envelope(self, wireframe=False):
        """
        Panel meshes of the second envelope; same dict as extract_mesh().

        Calls compute_second_envelope() (each plane reflected across the plane of
        its sphere centers, same spheres) and triangulates its spherical panels.
        """
        self.compute_second_envelope()
        self.se_sv_panels = self._panels_from_boundaries(self.se_sv_boundary_curve)
        return self._extract_envelope(self.se_sv_panels, self.se_HCE, self.se_vp_polygons,
                                      self.se_sv_boundary_curve, self.se_sv_boundary_curves,
                                      wireframe)

    def _panels_from_boundaries(self, boundary_curve):
        """Spherical panel (V, F) per sphere from its boundary curve.

        Entries are (center, None) for spheres with |r| < rad_tol and None where
        there is no closed boundary.
        """
        n = len(self.sv_centers)

        def _gen_panel(i):
            r = self.sv_radii[i]
            if abs(r) < rad_tol:
                return i, (self.sv_centers[i], None)
            bc = boundary_curve[i] if i < len(boundary_curve) else None
            if bc is None or len(bc) < 3:
                return i, None
            return i, spherical_panel_no_remesh(bc, self.sv_centers[i], abs(r))

        panels = [None] * n
        with ThreadPoolExecutor() as pool:
            for i, panel in pool.map(_gen_panel, range(n)):
                panels[i] = panel
        return panels

    def _extract_envelope(self, panels, hce, polygons, boundary_curve, boundary_curves,
                          wireframe):
        """Build the extract_mesh() dict for one envelope (see extract_mesh)."""
        # Spheres live on vertices, planes on faces.
        inn_sph = sorted(int(i) for i in self.top.inner_vertices())
        inn_pl  = sorted(int(f) for f in self.top.inner_faces)
        out = {}

        # Spherical panels (+ degenerate spheres as points)
        Vs, Fs, degenerate = [], [], []
        for i in inn_sph:
            panel = panels[i] if i < len(panels) else None
            if panel is None:
                continue
            V, F = panel
            if F is None:
                degenerate.append(self.sv_centers[i])
            else:
                Vs.append(V)
                Fs.append(F)
        out["spherical_panels"] = _merge_meshes_or_empty(Vs, Fs)

        # Conical strips of the inner edges (+ collapsed ones as segments)
        edge_s0, edge_s1 = self.top.edge_vertices()   # sphere pair per edge
        edge_p0, edge_p1 = self.top.edge_faces()      # plane on each side (-1 = boundary)
        Vs, Fs, segments = [], [], []
        for k in self.top.inner_edges():
            if edge_p0[k] == -1 or edge_p1[k] == -1:
                continue
            strip = self._cone_strip(edge_s0[k], edge_s1[k], hce)
            if strip is None:
                continue
            V, F = strip
            if F is None:
                segments.append(V)
            else:
                Vs.append(V)
                Fs.append(F)
        out["conical_panels"] = _merge_meshes_or_empty(Vs, Fs)

        # Flat panels
        Vs, Fs = [], []
        for f in inn_pl:
            polygon = polygons[f] if f < len(polygons) else None
            if polygon is not None and len(polygon) >= 3:
                Vs.append(np.asarray(polygon, dtype=float))
                Fs.append([list(range(len(polygon)))])
        out["flat_panels"] = _merge_meshes_or_empty(Vs, Fs)

        if degenerate:
            out["degenerated_spheres"] = np.array(degenerate, dtype=float)
        if segments:
            out["degenerated_cones"] = (np.vstack(segments),
                                        np.arange(2 * len(segments)).reshape(-1, 2))

        if wireframe:
            arcs = [arc for i in inn_sph
                    if i < len(boundary_curve) and boundary_curve[i] is not None
                    for arc in boundary_curves[i]]
            out["spherical_wireframe"] = _merge_curves_or_empty(arcs)
            loops = [np.vstack([polygons[f], polygons[f][:1]]) for f in inn_pl
                     if f < len(polygons) and polygons[f] is not None and len(polygons[f]) >= 3]
            out["flat_wireframe"] = _merge_curves_or_empty(loops)
        return out

    def _cone_strip(self, i, j, hce):
        """
        Conical strip between spheres i and j: their two contact arcs lofted into quads.

        Parameters:
            i, j (int): sphere (vertex) indices of an edge.
            hce (dict): half conical edges, self.HCE or self.se_HCE.

        Returns:
            (V, F) quad strip; (segment (2, 3), None) when both arcs collapsed to a
            point (spheres of radius ~0); None when there is nothing to draw.

        A missing arc (c_arc None) is rebuilt from the other one by intersecting the
        cone's rulings with the plane of the missing contact circle. Current code
        computes both arcs for every inner edge, so this only matters for L-meshes
        built by older code; it needs the circle's "center"/"normal" (first envelope).
        """
        e_ij, e_ji = hce.get((i, j)), hce.get((j, i))
        curve_1 = None if e_ij is None else e_ij.get("c_arc")
        curve_2 = None if e_ji is None else e_ji.get("c_arc")
        if curve_2 is not None:
            curve_2 = curve_2[::-1]
        if curve_1 is None and curve_2 is None:
            return None
        if curve_2 is None:
            curve_2 = self._rebuild_arc(i, j, curve_1, e_ji)
        elif curve_1 is None:
            curve_1 = self._rebuild_arc(i, j, curve_2, e_ij)
        if curve_1 is None or curve_2 is None:
            return None
        curve_1 = np.asarray(curve_1, dtype=float)
        curve_2 = np.asarray(curve_2, dtype=float)

        if (len(curve_1) <= 2 or len(curve_2) <= 2
                or (_collapsed(curve_1) and _collapsed(curve_2))):
            return np.array([curve_1[-1], curve_2[-1]]), None

        n1 = len(curve_1)
        vertices = np.concatenate((curve_1, curve_2), axis=0)
        if not np.all(np.isfinite(vertices)):
            self._log_msg(f"Warning: NaN or Inf in cone strip {i},{j}. Skipping.")
            return None
        faces = [[k, k + 1, k + n1 + 1, k + n1] for k in range(n1 - 1)]
        return vertices, faces

    def _rebuild_arc(self, i, j, other, entry):
        """Missing contact arc of the cone (i, j), from the other arc and the plane
        of the missing circle (``entry`` = its HCE record); None if impossible."""
        if entry is None or "normal" not in entry or "center" not in entry:
            return None
        _, _, cone_axis, cone_vertex, _ = oriented_cone(
            self.sv_centers[i], self.sv_radii[i], self.sv_centers[j], self.sv_radii[j])
        plane = [entry["normal"], -(entry["center"] @ entry["normal"])]
        if cone_vertex is None:
            return [line_plane_inter([p, cone_axis], plane) for p in other]
        return [line_plane_inter([p, unit(cone_vertex - p)], plane) for p in other]

    def extract_support_structure(self, kind="trivial", thickness=0.05):
        """
        Support structure mesh of the L-mesh.

        Parameters:
            kind (str): "trivial", "planar" or "second_envelope" (joins the L-mesh to
                its second envelope; computes the second envelope first).
            thickness (float): extrusion thickness ("trivial" and "planar").

        Returns:
            (V, F), or None if it could not be built.
        """
        if kind == "trivial":
            return self.trivial_support_structure(thickness)
        if kind == "planar":
            return self.planar_support_structure(thickness)
        if kind == "second_envelope":
            self.compute_second_envelope()
            return self.support_structure_second_envelope()
        raise ValueError(f"kind must be 'trivial', 'planar' or 'second_envelope', got {kind!r}")


def load_lmesh(path):
    """
    Load an L-mesh from a state pickle written by Lmesh.save_lmesh().

    Only spheres, planes and topology are stored; the full L-mesh structure
    (contact points, vertex polygons, conical edges, spherical panels) is
    recomputed on load.

    Parameters:
        path (str): path to the .pkl file.

    Returns:
        Lmesh: a fully initialized L-mesh.
    """
    with open(path, 'rb') as f:
        state = load(f)
    return Lmesh().load_lmesh_state(state)
