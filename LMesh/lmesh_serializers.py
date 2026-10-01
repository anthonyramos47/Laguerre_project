"""
L-mesh display: turn an Lmesh into kayviz serializer dicts, or push it to the viewer.

Lmesh only computes geometry — extract_mesh(), extract_second_envelope(),
extract_support_structure() return plain (V, F) / (P, E) arrays. This module names
and colours them:

    lmesh_visualize(lmesh)                        # push to the viewer
    kv.push_objects(lmesh_scene(lmesh))           # the same, by hand
    lmesh_scene(lmesh, wireframe=True)            # + panel boundary curves
    lmesh_second_envelope_scene(lmesh)            # second envelope
    lmesh_support_scene(lmesh, kind="trivial")    # support structure

Mesh data for single glyphs (spheres, circles, cones) comes from hanan.glyphs.
"""

import numpy as np
from kayviz.serializers import surface_mesh, curve_network, point_cloud, scalar_quantity


# ── LMesh palette helpers ─────────────────────────────────────────────────────

def get_palette_names() -> list:
    """Return the list of available palette names."""
    from LMesh.palette import PALETTE_NAMES
    return PALETTE_NAMES


def resolve_palette(palette_name: str) -> dict:
    """Return the {spheres, planes, cones} RGB-tuple dict for *palette_name*.
    Falls back to 'shikon' if the name is unknown."""
    from LMesh.palette import get_palette
    return get_palette(palette_name)


# ── LMesh scene snapshot helpers ─────────────────────────────────────────────

def lmesh_mesh_objects(mesh: dict, prefix: str = "", palette: str = "shikon") -> list:
    """
    Serializer dicts for an Lmesh.extract_mesh()-style dict.

    Draws every key that is present: the three panel meshes, degenerate spheres as
    points and degenerate cones as lines, and the wireframes if they were extracted.
    """
    pal = resolve_palette(palette)
    pre = prefix + "_" if prefix else ""
    objects = []

    for key, name, color in (("spherical_panels", "Spherical_Face_Panels", pal["spheres"]),
                             ("flat_panels",      "Vertex_Plane_Polygons", pal["planes"]),
                             ("conical_panels",   "Conical_Edge_Strips",   pal["cones"])):
        V, F = mesh.get(key, ((), ()))
        if len(V):
            objects.append(surface_mesh(pre + name, V, F, color=color, opacity=1.0))

    if "degenerated_spheres" in mesh:
        objects.append(point_cloud(pre + "Spherical_Faces_Centers", mesh["degenerated_spheres"],
                                   color=pal["spheres"], radius=0.0025))
    if "degenerated_cones" in mesh:
        objects.append(curve_network(pre + "Conical_Edge_Lines", *mesh["degenerated_cones"],
                                     color=pal["cones"]))

    for key, name, color in (("spherical_wireframe", "Spherical_Wireframe", pal["spheres"]),
                             ("flat_wireframe",      "Flat_Wireframe",      pal["planes"])):
        if key in mesh and len(mesh[key][0]):
            objects.append(curve_network(pre + name, *mesh[key], color=(0, 0, 0)))
    return objects


def lmesh_scene(lmesh, prefix="", palette: str = "shikon", wireframe: bool = False) -> list:
    """Serializer dicts for the L-mesh (first envelope): lmesh.extract_mesh()."""
    return lmesh_mesh_objects(lmesh.extract_mesh(wireframe=wireframe), prefix, palette)


def lmesh_second_envelope_scene(lmesh, prefix="se", palette: str = "shikon",
                                wireframe: bool = False) -> list:
    """Serializer dicts for the second envelope: lmesh.extract_second_envelope()."""
    return lmesh_mesh_objects(lmesh.extract_second_envelope(wireframe=wireframe), prefix, palette)


def lmesh_support_scene(lmesh, kind: str = "trivial", thickness: float = 0.05,
                        name: str = None, color=(0.6, 0.6, 0.6)) -> list:
    """Serializer dicts for a support structure: lmesh.extract_support_structure(kind)."""
    result = lmesh.extract_support_structure(kind, thickness)
    if result is None:
        return []
    V, F = result
    return [surface_mesh(name or f"Support_{kind}", V, F, color=color, opacity=1.0)]


def lmesh_normals_scene(lmesh, length=0.05, flip=False, name="Normals") -> list:
    """Serializer dicts for the L-mesh normals (lmesh.normals_segments)."""
    verts, edges = lmesh.normals_segments(length=length, flip=flip)
    return [] if verts is None else [curve_network(name, verts, edges)]


def lmesh_axes_nodes_scene(lmesh, length=0.05, flag="trivial", name=None) -> list:
    """Serializer dicts for the axis-node segments (lmesh.axes_nodes)."""
    nodes, tips = lmesh.axes_nodes(length=length, flag=flag)
    if len(nodes) == 0:
        return []
    verts = np.empty((2 * len(nodes), 3))
    verts[0::2], verts[1::2] = nodes, tips
    edges = np.arange(2 * len(nodes)).reshape(-1, 2)
    return [curve_network(name or f"Axes_Nodes_{flag}", verts, edges)]


def lmesh_visualize(lmesh, prefix="", palette: str = "shikon", wireframe: bool = False,
                    second_envelope: bool = False, support: str = None,
                    thickness: float = 0.05) -> None:
    """
    Push the L-mesh to the kayviz viewer: kv.push_objects(lmesh_scene(lmesh)).

    Parameters:
        wireframe:       also draw the panel boundary curves.
        second_envelope: also draw the second envelope.
        support:         also draw a support structure ("trivial", "planar" or
                         "second_envelope").
    """
    import kayviz as kv
    objects = lmesh_scene(lmesh, prefix, palette, wireframe)
    if second_envelope:
        objects += lmesh_second_envelope_scene(lmesh, (prefix + "_se") if prefix else "se",
                                               palette, wireframe)
    if support:
        objects += lmesh_support_scene(lmesh, support, thickness)
    kv.push_objects(objects)


def lmesh_sphere_centers_scene(lmesh, name="Sphere Centers Mesh"):
    """
    Serialize the sphere-centers mesh of an L-mesh with a per-face planarity
    scalar field.

    Returns a list of two dicts:
      [0]  surface_mesh   — the centers mesh
      [1]  scalar_quantity — planarity defined on faces (0 = planar, >0 = non-planar)
    """
    from hanan.geometry.measures import planarity_measure

    scV, scF = lmesh.sphere_centers_mesh()
    planarity = planarity_measure(scV, scF)          # (F,) one value per face

    mesh_dict     = surface_mesh(name, scV, scF, color=(0.8, 0.8, 0.2), opacity=0.6)
    scalar_dict   = scalar_quantity(name, "Planarity", planarity, defined_on="faces")

    return [mesh_dict, scalar_dict]
