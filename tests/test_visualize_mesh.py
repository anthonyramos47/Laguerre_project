"""
Visual integration test — push a real mesh to the browser viewer.

This test spins up the full FastAPI app (same as run_server.sh) via
TestClient, loads triMeshTest_small.obj, and pushes it through the
script API's HTTP stack.  No browser is opened; it validates that the
full pipeline from OBJ file → serializer → /api/script/push → buffer
produces a renderable scene_update.

Run (requires PYTHONPATH or conftest.py to be active):
    pytest tests/test_visualize_mesh.py -v

To actually see it in a browser:
    python tests/test_visualize_mesh.py
"""

import os
import sys
import numpy as np
import pytest

from hanan import glyphs

# ── Project paths ─────────────────────────────────────────────────────────────
_ROOT  = os.path.dirname(os.path.dirname(__file__))
_DATA  = os.path.join(_ROOT, "data")
_SMALL = os.path.join(_DATA, "triMeshTest_small.obj")
_MESH  = os.path.join(_DATA, "triMeshTest.obj")


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_mesh(path):
    from hanan.geometry.utils import read_obj, normalize_vertices
    V, F = read_obj(path)
    V    = normalize_vertices(V)
    return V, F


def make_test_client():
    """Build a fresh FastAPI app + TestClient with a clean ScriptApp."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from kayviz.webscope.script_app import ScriptApp
    from kayviz.webscope import registry as reg

    script_app = ScriptApp()
    reg._apps.clear()
    reg._apps["script"] = script_app

    app = FastAPI()
    from kayviz.webscope.registry import mount_all
    mount_all(app)
    return TestClient(app, raise_server_exceptions=True), script_app


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestVisualizeMesh:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.client, self.script_app = make_test_client()

    def test_mesh_file_exists(self):
        assert os.path.exists(_SMALL), f"Mesh not found: {_SMALL}"

    def test_push_surface_mesh(self):
        from kayviz import serializers as ser
        V, F = load_mesh(_SMALL)
        obj  = ser.surface_mesh("Base Mesh", V, F, color=(0.6, 0.6, 0.8))

        r = self.client.post("/api/script/push", json={"objects": [obj]})
        assert r.status_code == 200
        assert r.json()["pushed"] == 1
        assert "Base Mesh" in self.script_app._buffer

    def test_pushed_mesh_has_correct_shape(self):
        from kayviz import serializers as ser
        V, F = load_mesh(_SMALL)
        obj  = ser.surface_mesh("Mesh", V, F)

        self.client.post("/api/script/push", json={"objects": [obj]})
        buf = self.script_app._buffer["Mesh"]

        assert buf["type"]      == "surface_mesh"
        assert len(buf["vertices"]) == V.shape[0] * 3  # flat [x,y,z,…]
        assert buf["face_size"] == 3                    # triangulated
        assert len(buf["faces"]) % 3 == 0

    def test_push_with_scalar_quantity(self):
        from kayviz import serializers as ser
        from hanan.geometry.mesh import Mesh
        V, F = load_mesh(_SMALL)

        mesh = Mesh()
        mesh.make_mesh(V, F)
        mean_curvature = np.linalg.norm(mesh.vertices, axis=1)  # stand-in scalar field

        mesh_obj   = ser.surface_mesh("Mesh", V, F, color=(0.7, 0.7, 0.7))
        scalar_obj = ser.scalar_quantity("Mesh", "curvature", mean_curvature)

        r = self.client.post("/api/script/push",
                             json={"objects": [mesh_obj, scalar_obj]})
        assert r.status_code == 200
        assert r.json()["pushed"] == 2
        assert "Mesh" in self.script_app._buffer

    def test_push_with_edges_flag(self):
        from kayviz import serializers as ser
        V, F = load_mesh(_SMALL)

        # show_edges=True returns [mesh, edge_child]; the edge overlay is a
        # child of the mesh, keyed by target rather than a scene object name.
        objs = ser.surface_mesh("Mesh", V, F, show_edges=True, edge_color=(1.0, 0.0, 0.0))
        assert isinstance(objs, list) and len(objs) == 2
        mesh_obj, edge_obj = objs
        assert mesh_obj["type"] == "surface_mesh"
        assert edge_obj["type"] == "edge_child"
        assert edge_obj["target"] == "Mesh"
        assert edge_obj["color"] == pytest.approx([1.0, 0.0, 0.0])

        r = self.client.post("/api/script/push", json={"objects": objs})
        assert r.status_code == 200
        assert r.json()["pushed"] == 2
        assert "Mesh" in self.script_app._buffer
        assert "Mesh_edges" not in self.script_app._buffer

    def test_push_with_point_cloud(self):
        from kayviz import serializers as ser
        V, F = load_mesh(_SMALL)

        cloud_obj = ser.point_cloud("Vertices", V, color=(1.0, 0.4, 0.0), radius=0.02)
        r = self.client.post("/api/script/push", json={"objects": [cloud_obj]})
        assert r.status_code == 200
        assert "Vertices" in self.script_app._buffer

    def test_push_with_vector_quantity(self):
        """Vector quantities are stored in the buffer as vector_quantity dicts."""
        from kayviz import serializers as ser
        from hanan.geometry.mesh import Mesh
        V, F = load_mesh(_SMALL)

        mesh = Mesh()
        mesh.make_mesh(V, F)
        normals = mesh.vertex_normals  # (N, 3) per-vertex normals

        mesh_obj   = ser.surface_mesh("Mesh", V, F)
        vector_obj = ser.vector_quantity("Mesh", "normals", normals, defined_on="vertices")

        assert vector_obj["type"]       == "vector_quantity"
        assert vector_obj["target"]     == "Mesh"
        assert vector_obj["name"]       == "normals"
        assert len(vector_obj["vectors"]) == V.shape[0] * 3  # flat

        r = self.client.post("/api/script/push",
                             json={"objects": [mesh_obj, vector_obj]})
        assert r.status_code == 200
        assert r.json()["pushed"] == 2

    def test_push_vectors_as_segments(self):
        """Visualize a vector field as a curve network of line segments."""
        from kayviz import serializers as ser
        from hanan.geometry.mesh import Mesh
        V, F = load_mesh(_SMALL)

        mesh = Mesh()
        mesh.make_mesh(V, F)
        normals = mesh.vertex_normals
        length  = 0.05

        # Each vector: base point V[i] → tip V[i] + length * N[i]
        tips      = V + length * normals
        seg_verts = np.empty((len(V) * 2, 3), dtype=np.float32)
        seg_verts[0::2] = V
        seg_verts[1::2] = tips
        seg_edges = np.array([[i*2, i*2+1] for i in range(len(V))], dtype=np.int32)

        vector_seg_obj = ser.curve_network("Normals", seg_verts, seg_edges,
                                           color=(0.2, 0.8, 0.4))
        r = self.client.post("/api/script/push", json={"objects": [vector_seg_obj]})
        assert r.status_code == 200
        assert "Normals" in self.script_app._buffer

    def test_push_vector_field(self):
        """vector_field serializer produces the right dict and is pushed correctly."""
        from kayviz import serializers as ser
        from hanan.geometry.mesh import Mesh
        V, F = load_mesh(_SMALL)

        mesh = Mesh()
        mesh.make_mesh(V, F)
        normals = mesh.vertex_normals

        obj = ser.vector_field("Normals", V, normals, length=0.05, color=(0.2, 0.8, 0.4))
        assert obj["type"]   == "vector_field"
        assert obj["name"]   == "Normals"
        assert obj["length"] == pytest.approx(0.05)
        assert len(obj["origins"]) == V.shape[0] * 3
        assert len(obj["vectors"]) == V.shape[0] * 3
        assert obj["color"] == pytest.approx([0.2, 0.8, 0.4])

        r = self.client.post("/api/script/push", json={"objects": [obj]})
        assert r.status_code == 200
        assert r.json()["pushed"] == 1
        assert "Normals" in self.script_app._buffer
        assert self.script_app._buffer["Normals"]["type"] == "vector_field"

    def test_load_replays_buffer(self):
        """After pushing, a fresh page load returns the full scene."""
        from kayviz import serializers as ser
        V, F = load_mesh(_SMALL)

        self.client.post("/api/script/push",
                         json={"objects": [ser.surface_mesh("Mesh", V, F)]})
        r = self.client.post("/api/script/load")
        assert r.status_code == 200
        objs = r.json()["objects"]
        assert len(objs) == 1
        assert objs[0]["name"] == "Mesh"

    def test_replace_mesh_in_place(self):
        """Pushing the same name twice updates, not duplicates."""
        from kayviz import serializers as ser
        V, F = load_mesh(_SMALL)

        obj1 = ser.surface_mesh("Mesh", V, F, color=(0.8, 0.8, 0.8))
        obj2 = ser.surface_mesh("Mesh", V, F, color=(1.0, 0.0, 0.0))
        self.client.post("/api/script/push", json={"objects": [obj1]})
        self.client.post("/api/script/push", json={"objects": [obj2]})

        assert len(self.script_app._buffer) == 1
        assert self.script_app._buffer["Mesh"]["color"] == pytest.approx([1.0, 0.0, 0.0])

    def test_clear_removes_mesh(self):
        from kayviz import serializers as ser
        V, F = load_mesh(_SMALL)

        self.client.post("/api/script/push",
                         json={"objects": [ser.surface_mesh("Mesh", V, F)]})
        self.client.post("/api/script/push", json={"action": "clear"})
        assert self.script_app._buffer == {}


# ── Standalone: open in real browser ─────────────────────────────────────────

if __name__ == "__main__":
    """
    Run directly to see the mesh in the browser:
        python tests/test_visualize_mesh.py
    """
    sys.path.insert(0, _ROOT)

    import kayviz as kv
    from hanan.geometry.mesh import Mesh

    V, F  = load_mesh(_SMALL)
    mesh  = Mesh()
    mesh.make_mesh(V, F)
    normals = mesh.vertex_normals

    kv.init()

    kv.register_surface_mesh("Base Mesh", V, F, color=(0.6, 0.6, 0.8), show_edges=True)
    kv.register_point_cloud("Vertices",  V, color=(1.0, 0.4, 0.0), radius=0.015)
    kv.add_scalar_quantity("Base Mesh", "height", V[:, 2])
    kv.register_vector_field("Normals", V, normals, length=0.05, color=(0.2, 0.8, 0.4))
    kv.register_surface_mesh("Sphere", *glyphs.sphere((0.5, 0.5, 0.5), 0.1), color=(0.8, 0.2, 0.2))
    kv.register_curve_network("Ring", *glyphs.circle(V[0], normals[0], 0.05), color=(0.9, 0.5, 0.1))

    print(f"Mesh: {V.shape[0]} vertices, {len(F)} faces")
    kv.show()
