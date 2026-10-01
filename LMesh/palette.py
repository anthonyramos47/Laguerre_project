"""
Shared colour palettes for L-mesh visualisation.

Imported by:
  - LMesh/L_mesh.py          (relative:  from .palette import ...)
  - server/serializers.py    (absolute:  from LMesh.palette import ...)
"""

from hanan.geometry.utils import hex_to_norm_rgb

# Raw hex palettes — {name: {spheres, planes, cones}}
_HEX_PALETTES = {
    "elegant_bold": {
        "spheres": "#0B1D51",
        "planes":  "#D9623E",
        "cones":   "#EAD9C4",
    },
    "nature_inspired": {
        "spheres": "#2A4D34",
        "planes":  "#C14F23",
        "cones":   "#E7C88D",
    },
    "high_contrast_modern": {
        "spheres": "#2F3E46",
        "planes":  "#B8336A",
        "cones":   "#F4EDE8",
    },
    "moody_artistic": {
        "spheres": "#1B1B1B",
        "planes":  "#045D69",
        "cones":   "#B76E44",
    },
    "koan_style": {
        "spheres": "#4B0632",
        "planes":  "#02011A",
        "cones":   "#C2B8A3",
    },
    "midnight_luxe": {
        "spheres": "#1A1A2E",
        "planes":  "#C5A572",
        "cones":   "#EAE7DC",
    },
    "scandinavian_minimal": {
        "spheres": "#2D3142",
        "planes":  "#BFC0C0",
        "cones":   "#FFFFFF",
    },
    "sunset_warmth": {
        "spheres": "#3D2C29",
        "planes":  "#E89E76",
        "cones":   "#F9E4D4",
    },
    "botanical_garden": {
        "spheres": "#1B4332",
        "planes":  "#95D5B2",
        "cones":   "#D8F3DC",
    },
    "desert_dusk": {
        "spheres": "#774936",
        "planes":  "#D4A373",
        "cones":   "#FAEDCD",
    },
    "ocean_depths": {
        "spheres": "#003049",
        "planes":  "#669BBC",
        "cones":   "#FDF0D5",
    },
    "cherry_blossom": {
        "spheres": "#29042F",
        "planes":  "#D4A5A5",
        "cones":   "#F8E8EE",
    },
    "industrial_chic": {
        "spheres": "#264653",
        "planes":  "#E76F51",
        "cones":   "#F4F1DE",
    },
    "mizuki": {
        "spheres": "#a282e7",
        "planes":  "#292748",
        "cones":   "#a1b0b3",
    },
    "shikon": {
        "spheres": "#0C226C",
        "planes":  "#610438",
        "cones":   "#b8d4e8",
    },
}

DEFAULT_PALETTE = "shikon"

# Pre-converted to normalised RGB tuples — {name: {spheres, planes, cones}}
PALETTES: dict = {
    name: {k: hex_to_norm_rgb(v) for k, v in pal.items()}
    for name, pal in _HEX_PALETTES.items()
}

PALETTE_NAMES: list = list(PALETTES.keys())


def get_palette(name: str) -> dict:
    """Return the normalised-RGB palette dict for *name*, falling back to DEFAULT_PALETTE."""
    return PALETTES.get(name, PALETTES[DEFAULT_PALETTE])
