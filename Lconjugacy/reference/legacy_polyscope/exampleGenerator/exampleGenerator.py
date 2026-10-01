import numpy as np
import splipy as sp
from splipy import surface_factory
import polyscope as ps
import polyscope.imgui as psim
from hanan.geometry.utils import make_param_grid_vec
from LconjugacyGUI.utils.bsplines_functions import read_bspline_json, \
                                sample_bspline_surface, \
                                bspline_curvatures, \
                                save_bspline_to_json

from LconjugacyGUI.exampleGenerator.functions import (helical_superellipsoid_tube,
                       
                       convex_seashell,
                       ammonite,
                       twisted_ellipsoid,
                       )

def nautilus_shell(u, v, growth_rate=0.2, n_turns=4):
    """
    Nautilus shell approximation with K > 0
    u: [0, 2π] - cross-section angle
    v: [0, 2π*n_turns] - spiral angle
    """

    # Exponential spiral (logarithmic)
    a = 1.0  # Initial radius
    b = growth_rate

    R = a * np.exp(b * v)  # Exponentially growing radius
    #R = 2 + a * np.sin(b * v)  # Cosine-based radius for more natural look

    # Cross-section (elliptical for more natural look)
    r_cross = 0.85 * R  # Cross-section radius grows with spiral

    # # Position along spiral
    x_center = R * np.cos(v)  # Add some waviness
    y_center = R * np.sin(v)  # Add some waviness
    z_center = 0.5 * v  # Slight rise
    # x_center = R * np.cos(v)  # Add some waviness
    # y_center = R * np.sin(v)  # Add some waviness
    # z_center = 

    # Add cross-section
    # Normal to spiral (simplified)
    x = x_center + r_cross * np.cos(u) * np.cos(v)
    y = y_center + r_cross * np.cos(u) * np.sin(v)
    z = z_center + r_cross * np.sin(u)

    return x, y, z

def convex_shell(u, v, a=2.0, b=1.5, c=1.0, pulse_amp=0.15, pulse_freq=3):
    """
    Convex shell with K > 0 everywhere.
    
    Built as a perturbed ellipsoid — for small pulse_amp, Gaussian curvature
    remains strictly positive everywhere (ellipsoid has K > 0 everywhere,
    and small C² perturbations preserve this).
    
    u: [0, π]       - polar (latitude) angle
    v: [0, 2π]      - azimuthal (longitude) angle
    
    Parameters
    ----------
    a, b, c     : semi-axes of the base ellipsoid
    pulse_amp   : amplitude of radial perturbation (keep < 0.3 for K > 0)
    pulse_freq  : frequency of the bumps along v
    """
    # Base ellipsoid
    # K = (1/(a²b²c²)) / (x²/a⁴ + y²/b⁴ + z²/c⁴)²  > 0 always
    x_e = a * np.sin(u) * np.cos(v)
    y_e = b * np.sin(u) * np.sin(v)
    z_e = c * np.cos(u)

    # Smooth radial perturbation — stays convex for small amplitudes
    # Modulate only along v (longitude), constant along u strips preserve convexity better
    r_perturb = 1.0 + pulse_amp * np.cos(pulse_freq * v) * np.sin(u)  # sin(u) → zero at poles

    x = x_e * r_perturb
    y = y_e * r_perturb
    z = z_e * r_perturb

    return x, y, z



def spiral_tube(u, v, major_radius=3.0, minor_radius=0.5, pitch=0.1, n_turns=3):
    """
    Tube that spirals upward
    u: [0, 2π] - around the tube cross-section
    v: [0, 2π*n_turns] - along the spiral path
    """

    # Spiral centerline
    R = major_radius
    x_center = R * np.cos(v)
    y_center = R * np.sin(v)
    z_center = pitch * v / (2 * np.pi)

    # Tube cross-section (circle of radius minor_radius)
    # Normal and binormal to spiral
    # Simplified: assume gentle spiral
    r = minor_radius

    x = x_center + r * np.cos(u) * np.cos(v)
    y = y_center + r * np.cos(u) * np.sin(v)
    z = z_center + r * np.sin(u)

    return x, y, z


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


# ── Generate and fit ──────────────────────────────────────────────────────

u_range = (-np.pi/2, np.pi/2)
v_range = (0, 6*np.pi)

save_path = "/path/to/Laguerre_project/Code/gui/surfaces"
name = "example"

function = convex_seashell
#function = helical_superellipsoid_tube  # Change to nautilus_shell or spiral_tube for different shapes

# Sampled mesh for comparison
uv, faces, u_pts, v_pts = make_param_grid_vec(u_range, v_range, 30, 100)
vertices = np.array([function(u, v) for u, v in uv])

# Fit B-spline
bspline = fit_bspline_surface(function, u_range, v_range,
                               n_u=30, n_v=100, order=6)
print(f"B-spline fitted: {bspline.controlpoints.shape[0]}x"
      f"{bspline.controlpoints.shape[1]} control points, "
      f"order ({bspline.order(0)}, {bspline.order(1)})")

# Evaluate fitted surface on the same grid for visual comparison

#fitted_pts = bspline(u_eval, v_eval).reshape(-1, 3)


VBspline, FBspline = sample_bspline_surface(bspline, u_pts, v_pts)
K, _, _ = bspline_curvatures(bspline, u_pts, v_pts)



def callback():
    global name, save_path, bspline

    changed, name = psim.InputText("Name", name)
    
    if psim.Button("Save B-spline to JSON"):
        
        
        save_bspline_to_json(bspline, f"{save_path}/{name}.json")
    


# ── Visualise ─────────────────────────────────────────────────────────────
ps.init()
ps.register_surface_mesh("Original", vertices, faces)
surface = ps.register_surface_mesh("B-spline fit", VBspline, FBspline)
surface.add_scalar_quantity("Gaussian Curvature", K.flatten(),
                              defined_on='vertices',
                              cmap='coolwarm',
                              enabled=True)

# ulin = np.linspace(0, 1, 30)
# vlin = np.linspace(0, 1, 100)

ps.set_user_callback(callback)
# bpline2 = read_bspline_json(f"{save_path}/{name}.json")
# VBspline2, FBspline2 = sample_bspline_surface(bpline2, ulin, vlin)
# surface2 = ps.register_surface_mesh("B-spline from JSON", VBspline2, FBspline2)

ps.show()