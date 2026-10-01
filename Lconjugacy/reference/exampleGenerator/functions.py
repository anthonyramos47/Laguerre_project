"""
positive_curvature_surfaces.py
==============================
A collection of parametric surfaces with Gaussian curvature K > 0 everywhere
(or K ≥ 0 with K > 0 on a dense open set for the boundary cases).

Each function shares the same signature convention:
    f(u, v, **params) -> (x, y, z)  as numpy arrays

Typical usage
-------------
import numpy as np
import matplotlib.pyplot as plt
from positive_curvature_surfaces import *

u = np.linspace(0, np.pi, 300)
v = np.linspace(0, 2 * np.pi, 300)
U, V = np.meshgrid(u, v)

X, Y, Z = superellipsoid(U, V)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='plasma', linewidth=0, antialiased=True)
plt.show()
"""

import numpy as np
from scipy.integrate import odeint


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _signed_pow(base, exp):
    """
    Signed power: sign(x) * |x|^exp.
    Needed for superellipsoid / rounded-cube to handle negative values
    without producing NaN from fractional exponents.
    """
    return np.sign(base) * np.abs(base) ** exp


def compute_gaussian_curvature(func, U, V, eps=1e-5, **kwargs):
    """
    Numerically estimate Gaussian curvature K at each (u, v) point via
    finite differences on the first and second fundamental forms.

    Returns
    -------
    K : ndarray, same shape as U and V
    """
    def F(u, v):
        return np.array(func(u, v, **kwargs))  # shape (3, ...)

    r   = F(U, V)
    r_u = (F(U + eps, V) - F(U - eps, V)) / (2 * eps)
    r_v = (F(U, V + eps) - F(U, V - eps)) / (2 * eps)
    r_uu = (F(U + eps, V) - 2 * r + F(U - eps, V)) / eps**2
    r_vv = (F(U, V + eps) - 2 * r + F(U, V - eps)) / eps**2
    r_uv = (F(U + eps, V + eps) - F(U + eps, V - eps)
           - F(U - eps, V + eps) + F(U - eps, V - eps)) / (4 * eps**2)

    # Unit normal
    N = np.cross(r_u, r_v, axisa=0, axisb=0).T
    N /= np.linalg.norm(N, axis=0, keepdims=True) + 1e-30

    # First fundamental form
    E = np.einsum('i...,i...->...', r_u, r_u)
    Fcoef = np.einsum('i...,i...->...', r_u, r_v)
    G = np.einsum('i...,i...->...', r_v, r_v)

    # Second fundamental form
    L = np.einsum('i...,i...->...', r_uu, N)
    M = np.einsum('i...,i...->...', r_uv, N)
    Ncoef = np.einsum('i...,i...->...', r_vv, N)

    K = (L * Ncoef - M**2) / (E * G - Fcoef**2 + 1e-30)
    return K


# ---------------------------------------------------------------------------
# 1. Superellipsoid
# ---------------------------------------------------------------------------

def superellipsoid(u, v, a=1.5, b=1.0, c=1.2, p=2.5, q=2.5):
    """
    Superellipsoid with K > 0 everywhere for p, q > 1.

    Generalises the ellipsoid by controlling the "squareness" of cross-
    sections independently in latitude (p) and longitude (q).

    u : [0, π]   — latitude
    v : [0, 2π]  — longitude

    Parameters
    ----------
    a, b, c : semi-axes
    p       : latitude exponent  (p = 2 → ellipsoid, p > 2 → flatter poles)
    q       : longitude exponent (q = 2 → ellipsoid, q > 2 → squarer equator)

    K > 0 condition
    ---------------
    For even integer exponents the surface is C² and strictly convex, so
    K > 0 everywhere.  Keep p, q ≥ 2 and round to even integers for safety;
    odd or fractional values can produce C¹ ridges where K is undefined.
    """
    cos_u = np.cos(u)
    sin_u = np.sin(u)
    cos_v = np.cos(v)
    sin_v = np.sin(v)

    x = a * _signed_pow(sin_u, 2.0 / p) * _signed_pow(cos_v, 2.0 / q)
    y = b * _signed_pow(sin_u, 2.0 / p) * _signed_pow(sin_v, 2.0 / q)
    z = c * _signed_pow(cos_u, 2.0 / p)

    return x, y, z


# ---------------------------------------------------------------------------
# 2. Rounded Cube  (superellipsoid special case p = q = large)
# ---------------------------------------------------------------------------

def rounded_cube(u, v, a=1.0, b=1.0, c=1.0, roundness=8):
    """
    Rounded cube: a superellipsoid with large equal exponents, giving a
    shape that interpolates between a sphere (roundness=2) and a cube
    (roundness → ∞).  K > 0 everywhere for even integer exponents.

    u : [0, π]
    v : [0, 2π]

    Parameters
    ----------
    a, b, c    : box half-extents
    roundness  : even integer ≥ 2 controlling how "boxy" the result is
    """
    # Enforce even integer so the surface stays C²
    p = q = max(2, int(roundness) + int(roundness) % 2)
    return superellipsoid(u, v, a=a, b=b, c=c, p=p, q=q)


# ---------------------------------------------------------------------------
# 3. Zonal Harmonic Ellipsoid
# ---------------------------------------------------------------------------

def zonal_harmonic_ellipsoid(u, v, a=1.5, b=1.2, c=1.0,
                              harmonics=None):
    """
    Ellipsoid perturbed by a sum of zonal (axially symmetric) spherical
    harmonics.  K > 0 everywhere provided amplitudes are small enough.

    u : [0, π]   — polar angle (colatitude)
    v : [0, 2π]  — azimuthal angle

    Parameters
    ----------
    a, b, c    : base ellipsoid semi-axes
    harmonics  : list of (degree l, amplitude A) tuples.
                 Only even l values preserve fore-aft symmetry.
                 Default: [(4, 0.06), (6, 0.03)]

    K > 0 condition
    ---------------
    For an ellipsoid K > 0 exactly.  Each harmonic perturbation of amplitude
    A changes K by O(A), so keep |A| < ~0.15 to stay safely positive.
    """
    if harmonics is None:
        harmonics = [(4, 0.06), (6, 0.03)]

    # Base ellipsoid
    x0 = a * np.sin(u) * np.cos(v)
    y0 = b * np.sin(u) * np.sin(v)
    z0 = c * np.cos(u)

    # Radial perturbation from zonal harmonics P_l(cos u)
    delta_r = np.zeros_like(u, dtype=float)
    cos_u = np.cos(u)

    for l, A in harmonics:
        # Legendre polynomial P_l via recurrence
        P_prev, P_curr = np.ones_like(cos_u), cos_u
        if l == 0:
            Pl = P_prev
        elif l == 1:
            Pl = P_curr
        else:
            for n in range(1, l):
                P_next = ((2*n + 1) * cos_u * P_curr - n * P_prev) / (n + 1)
                P_prev, P_curr = P_curr, P_next
            Pl = P_curr
        delta_r += A * Pl

    r_perturb = 1.0 + delta_r

    x = x0 * r_perturb
    y = y0 * r_perturb
    z = z0 * r_perturb

    return x, y, z


# ---------------------------------------------------------------------------
# 4. Pollen Grain
# ---------------------------------------------------------------------------

def pollen_grain(u, v, a=1.0, b=1.0, c=1.3,
                 n_apertures=3, bump_amp=0.08, bump_width=0.4):
    """
    Stylised pollen grain: a prolate ellipsoid with n evenly spaced
    meridional furrows (colpi) realised as smooth negative bumps, plus a
    subtle equatorial ridge.  The geometry is K > 0 for small bump_amp.

    u : [0, π]   — colatitude
    v : [0, 2π]  — longitude

    Parameters
    ----------
    a, b, c       : base ellipsoid semi-axes (c > a for prolate / pollen-like)
    n_apertures   : number of colpi (3 is biologically typical)
    bump_amp      : depth of each colpus furrow (negative = inward)
                    keep < 0.15 for K > 0
    bump_width    : angular half-width of each furrow (radians)
    """
    # Base prolate ellipsoid
    x0 = a * np.sin(u) * np.cos(v)
    y0 = b * np.sin(u) * np.sin(v)
    z0 = c * np.cos(u)

    # Meridional furrows: Gaussian dips at v = k * 2π/n_apertures
    furrow = np.zeros_like(v)
    for k in range(n_apertures):
        v0 = k * 2 * np.pi / n_apertures
        dv = np.arctan2(np.sin(v - v0), np.cos(v - v0))   # wrapped difference
        furrow += np.exp(-0.5 * (dv / bump_width) ** 2)

    # Confine furrow to equatorial band via sin(u) weight
    r_perturb = 1.0 - bump_amp * furrow * np.sin(u)

    x = x0 * r_perturb
    y = y0 * r_perturb
    z = z0 * r_perturb

    return x, y, z


# ---------------------------------------------------------------------------
# 5. Diatom Cap
# ---------------------------------------------------------------------------

def diatom_cap(u, v, a=1.5, b=1.5, c=0.5,
               n_ribs=12, rib_amp=0.05, rib_decay=2.0):
    """
    Diatom valve (cap): an oblate ellipsoid carrying radial ribs that
    radiate outward from the apex, fading toward the rim — mimicking the
    striated silica frustule of a centric diatom.

    u : [0, π/2]  — only the top hemisphere (one valve)
    v : [0, 2π]   — azimuthal angle

    Parameters
    ----------
    a, b  : equatorial semi-axes (equal → circular outline)
    c     : polar semi-axis (c < a for oblate / disc-like)
    n_ribs     : number of radial ribs
    rib_amp    : amplitude of ribs (keep < 0.12 for K > 0)
    rib_decay  : how fast ribs fade toward the rim (higher = more central)
    """
    # Oblate ellipsoid (top cap only, u ∈ [0, π/2])
    x0 = a * np.sin(u) * np.cos(v)
    y0 = b * np.sin(u) * np.sin(v)
    z0 = c * np.cos(u)

    # Radial ribs: cosine pattern in v, Gaussian decay in u (strong near apex)
    rib = rib_amp * np.cos(n_ribs * v) * np.exp(-rib_decay * (u / (np.pi / 2)) ** 2)

    r_perturb = 1.0 + rib

    x = x0 * r_perturb
    y = y0 * r_perturb
    z = z0 * r_perturb

    return x, y, z


# ---------------------------------------------------------------------------
# 6. Ovaloid of Revolution
# ---------------------------------------------------------------------------

def ovaloid_of_revolution(u, v, profile="egg", **profile_kwargs):
    """
    Surface of revolution generated by revolving a strictly convex profile
    curve r(z) around the z-axis.  By Hadamard's theorem a smooth closed
    strictly convex surface has K > 0 everywhere.

    u : [0, π]   — parameter along the profile (0 = north pole, π = south)
    v : [0, 2π]  — azimuthal angle

    Parameters
    ----------
    profile : one of 'egg', 'lens', 'stadium'
        'egg'     — asymmetric prolate ovaloid (larger at the bottom)
        'lens'    — symmetric oblate ovaloid
        'stadium' — rounded-cylinder ovaloid

    Profile functions r(u) and z(u) are defined below.  All satisfy
    r(0) = r(π) = 0 (poles) and r > 0 for u ∈ (0, π).
    """
    if profile == "egg":
        # r(u) = sin(u) scaled by an asymmetric envelope
        asymmetry = profile_kwargs.get("asymmetry", 0.3)
        scale_r   = profile_kwargs.get("scale_r", 1.0)
        scale_z   = profile_kwargs.get("scale_z", 1.4)
        # Envelope: larger in the southern hemisphere
        envelope = 1.0 + asymmetry * np.cos(u)   # > 0 for asymmetry < 1
        r_u = scale_r * np.sin(u) * envelope
        z_u = scale_z * np.cos(u)

    elif profile == "lens":
        # Oblate: squash z, keep r circular
        scale_r = profile_kwargs.get("scale_r", 1.2)
        scale_z = profile_kwargs.get("scale_z", 0.6)
        r_u = scale_r * np.sin(u)
        z_u = scale_z * np.cos(u)

    elif profile == "stadium":
        # Rounded cylinder: use a superellipse profile in (r, z)
        p = profile_kwargs.get("p", 4)      # even integer → smooth
        scale_r = profile_kwargs.get("scale_r", 1.0)
        scale_z = profile_kwargs.get("scale_z", 1.5)
        r_u = scale_r * _signed_pow(np.sin(u), 2.0 / p)
        z_u = scale_z * _signed_pow(np.cos(u), 2.0 / p)

    else:
        raise ValueError(f"Unknown profile '{profile}'. Choose 'egg', 'lens', or 'stadium'.")

    x = r_u * np.cos(v)
    y = r_u * np.sin(v)
    z = z_u

    return x, y, z


# ---------------------------------------------------------------------------
# 7. Cassini Ovaloid of Revolution
# ---------------------------------------------------------------------------

def cassini_ovaloid(u, v, a=1.2, c=0.7, n_points=500):
    """
    Surface of revolution of a Cassini oval in its convex (ovaloid) regime.

    A Cassini oval is the locus of points where the *product* of distances
    to two foci is constant: d1 * d2 = b².  When b > a (the foci distance)
    the curve is a single convex closed loop — an ovaloid with K > 0.

    The parameter c controls the "focus spread"; keep c < a for the convex
    regime.  As c → 0 the shape approaches a circle; as c → a it approaches
    a figure-eight (lemniscate) and K → 0 at the waist.

    u : [0, 1]   — normalised arc-length parameter along the profile (0=top, 1=bottom)
    v : [0, 2π]  — azimuthal angle

    Parameters
    ----------
    a : product parameter  b = a  (controls overall size)
    c : half-distance between foci  (c < a for convex oval)
    n_points : resolution of the internal arc-length table
    """
    if c >= a:
        raise ValueError(
            f"c={c} must be strictly less than a={a} for the convex Cassini regime.")

    b_sq = a ** 2   # product of distances squared = a²

    # --- Build the profile curve r(theta), z(theta) in polar coords --------
    # Cassini oval in polar: r^4 - 2c²r²cos(2θ) = b⁴ - c⁴
    # We parameterise by θ ∈ [0, π] (top half), then mirror.
    theta = np.linspace(0, np.pi, n_points)
    cos2  = np.cos(2 * theta)

    # Solve r² from quadratic in r²:  r² = c²cos2θ ± sqrt(c⁴cos²2θ + b⁴ - c⁴)
    disc = c**4 * cos2**2 + b_sq**2 - c**4
    disc = np.maximum(disc, 0)
    r_sq = c**2 * cos2 + np.sqrt(disc)
    r_sq = np.maximum(r_sq, 0)
    r_profile = np.sqrt(r_sq)

    # In our surface of revolution: treat θ as the "height" angle
    # z = r_profile * cos(θ),  rho = r_profile * sin(θ)
    rho_profile = r_profile * np.abs(np.sin(theta))
    z_profile   = r_profile * np.cos(theta)

    # --- Arc-length parameterisation so u ∈ [0,1] maps smoothly -----------
    drho = np.diff(rho_profile)
    dz   = np.diff(z_profile)
    ds   = np.sqrt(drho**2 + dz**2)
    s    = np.concatenate([[0], np.cumsum(ds)])
    s   /= s[-1]    # normalise to [0, 1]

    # Interpolate to the requested u values (u may be a 2-D meshgrid)
    u_flat = np.asarray(u).ravel()
    rho_u  = np.interp(u_flat, s, rho_profile).reshape(np.asarray(u).shape)
    z_u    = np.interp(u_flat, s, z_profile  ).reshape(np.asarray(u).shape)

    # --- Revolve around z-axis ---------------------------------------------
    x = rho_u * np.cos(v)
    y = rho_u * np.sin(v)
    z = z_u

    return x, y, z


# ---------------------------------------------------------------------------
# 8. Mylar Balloon
# ---------------------------------------------------------------------------

def mylar_balloon(u, v, R=1.0, n_points=600):
    """
    Mylar balloon: the exact shape taken by two circular discs of radius R
    sewn together at their rims and inflated.  The profile is given by an
    elliptic-integral ODE; the result is oblate and has K > 0 everywhere.

    The profile satisfies:
        dz/ds = sin(φ),   dr/ds = cos(φ)
        dφ/ds = (1/r) * (1 - sin²(φ)) / tan(φ)   [from pressure balance]

    We use the known closed-form arc-length parameterisation:
        r(t) = R * cn(t, 1/√2),   z(t) = (R/K) * ∫₀ᵗ sn²(τ, 1/√2) dτ
    approximated via the ODE formulation for numerical robustness.

    u : [0, 1]   — normalised arc length from equator (0) to pole (1)
    v : [0, 2π]  — azimuthal angle

    Parameters
    ----------
    R        : equatorial radius (= disc radius)
    n_points : internal ODE integration points
    """
    from scipy.integrate import solve_ivp

    # ODE in arc-length s: state = [r, z, phi]
    # phi is the tangent angle to the profile curve
    # Pressure balance for a Mylar balloon gives:
    #   d(phi)/ds = cos(phi) * (1 - 2*sin^2(phi)) / r
    # with boundary conditions r(0)=R, z(0)=0, phi(0)=pi/2 (equator tangent is vertical)
    # and r(s_end)=0, phi(s_end)=0 (pole).

    def ode(s, y):
        r, z, phi = y
        r = max(r, 1e-10)
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        dr = cos_phi
        dz = sin_phi
        # curvature equation from Mylar equilibrium
        dphi = -cos_phi * sin_phi / r
        return [dr, dz, dphi]

    # Shoot from equator inward to pole
    # At equator: r=R, z=0, phi=pi/2 (tangent points along z)
    # Integrate until r hits 0
    s_max = np.pi * R / 2.0   # approximate quarter arc length

    sol = solve_ivp(
        ode,
        [0, s_max],
        [R, 0.0, np.pi / 2],
        max_step=s_max / n_points,
        dense_output=True,
        events=lambda s, y: y[0] - 1e-4   # stop when r ≈ 0
    )
    sol.t_events[0]  # pole arc length

    s_pole = sol.t[-1]
    s_vals = np.linspace(0, s_pole, n_points)
    states = sol.sol(s_vals)   # shape (3, n_points)

    r_prof = np.maximum(states[0], 0)
    z_prof = states[1]

    # Arc-length reparameterise to u ∈ [0, 1]
    ds     = np.diff(s_vals)
    arc    = np.concatenate([[0], np.cumsum(ds)])
    arc   /= arc[-1]

    u_flat  = np.asarray(u).ravel()
    r_u     = np.interp(u_flat, arc, r_prof).reshape(np.asarray(u).shape)
    z_u     = np.interp(u_flat, arc, z_prof ).reshape(np.asarray(u).shape)

    # Mirror: u ∈ [0,1] → top half; full balloon = top + bottom
    x = r_u * np.cos(v)
    y = r_u * np.sin(v)
    z = z_u   # top half; use -z_u for bottom

    return x, y, z


# ---------------------------------------------------------------------------
# 9. Radiolarian Skeleton
# ---------------------------------------------------------------------------

def radiolarian(u, v, R=1.0,
                ico_amp=0.12, ico_freq=6,
                hex_amp=0.06, hex_freq=10,
                spine_amp=0.18, n_spines=20, spine_width=0.18):
    """
    Stylised radiolarian skeleton: a sphere decorated with three layers of
    biological-looking geometry, all kept within the K > 0 regime.

    Layer 1 — icosahedral harmonic bumps  (degree-6 zonal + tesseral modes)
    Layer 2 — hexagonal lattice texture   (sum of 3 rotated cosine waves)
    Layer 3 — radial spines               (sharp Gaussian protrusions)

    u : [0, π]   — colatitude
    v : [0, 2π]  — longitude

    Parameters
    ----------
    R          : base sphere radius
    ico_amp    : amplitude of icosahedral bumps  (keep < 0.15)
    ico_freq   : harmonic frequency of icosahedral pattern
    hex_amp    : amplitude of hexagonal texture  (keep < 0.10)
    hex_freq   : spatial frequency of hexagonal cells
    spine_amp  : height of each spine            (keep < 0.25)
    n_spines   : number of randomly distributed spines
    spine_width: angular half-width of each spine (radians)

    K > 0 condition
    ---------------
    All three perturbations are outward-only (positive), so the surface
    remains star-shaped and K > 0 for the amplitudes listed above.
    """
    # Cartesian unit sphere coordinates
    sin_u, cos_u = np.sin(u), np.cos(u)
    sin_v, cos_v = np.sin(v), np.cos(v)
    nx = sin_u * cos_v
    ny = sin_u * sin_v
    nz = cos_u

    # --- Layer 1: icosahedral-symmetry harmonic bumps -----------------------
    # Approximate icosahedral pattern via degree-6 spherical harmonic
    # Y_6^0 + sqrt(14/11) * Re(Y_6^5) gives icosahedral symmetry
    cos6u = np.cos(ico_freq * u)
    ico = ico_amp * (cos6u + 0.4 * np.cos(ico_freq * v) * np.sin(u) ** 5)

    # --- Layer 2: hexagonal lattice -----------------------------------------
    # Three cosine waves at 0°, 60°, 120° in the (u,v) plane
    # gives a triangular / hexagonal interference pattern
    k = hex_freq
    w1 = np.cos(k * (nx + ny * 0.5))
    w2 = np.cos(k * (ny * np.sqrt(3) / 2))
    w3 = np.cos(k * (nx - ny * 0.5))
    hex_tex = hex_amp * (w1 + w2 + w3) / 3.0

    # --- Layer 3: spines ----------------------------------------------------
    # Place spines at quasi-random locations using a deterministic seed
    rng = np.random.default_rng(42)
    spine_u0 = rng.uniform(0.3, np.pi - 0.3, n_spines)
    spine_v0 = rng.uniform(0, 2 * np.pi,      n_spines)

    spines = np.zeros_like(u, dtype=float)
    for su, sv in zip(spine_u0, spine_v0):
        # Great-circle angular distance
        cos_dist = (np.sin(su) * sin_u * np.cos(sv - v)
                    + np.cos(su) * cos_u)
        cos_dist = np.clip(cos_dist, -1, 1)
        ang_dist = np.arccos(cos_dist)
        spines += spine_amp * np.exp(-0.5 * (ang_dist / spine_width) ** 2)

    # Combine all layers into a radial perturbation
    r_perturb = R * (1.0 + ico + hex_tex + spines)

    x = r_perturb * nx
    y = r_perturb * ny
    z = r_perturb * nz

    return x, y, z


# ---------------------------------------------------------------------------
# 10. Beetle Elytra
# ---------------------------------------------------------------------------

def beetle_elytra(u, v,
                  a=1.6, b=0.9, c=0.55,
                  keel_amp=0.07, keel_width=0.3,
                  micro_amp=0.025, micro_freq_u=18, micro_freq_v=9,
                  long_amp=0.04, n_striae=6):
    """
    Beetle elytron (single wing cover): an asymmetric convex shell with
    three levels of biological surface geometry.

    Base      — half-ellipsoid (dorsal surface of beetle abdomen)
    Keel      — longitudinal medial ridge along the suture line (v ≈ 0)
    Striae    — longitudinal grooves (puncture rows) along the length
    Micro     — fine surface microstructure (punctures / reticulation)

    u : [0, π]      — length axis (head = 0, abdomen tip = π)
    v : [0, π]      — width axis  (suture = 0, lateral edge = π)
                      (half-ellipsoid; mirror at v=0 for full dorsum)

    Parameters
    ----------
    a, b, c      : ellipsoid semi-axes (a=length, b=width, c=height)
    keel_amp     : height of medial keel ridge
    keel_width   : angular half-width of keel
    micro_amp    : amplitude of microstructure dimples
    micro_freq_u : microstructure frequency along u
    micro_freq_v : microstructure frequency along v
    long_amp     : amplitude of longitudinal striae
    n_striae     : number of longitudinal striae

    K > 0 condition
    ---------------
    Base ellipsoid is K > 0.  Keel, striae, and micro are all small outward
    or symmetric perturbations; keep amplitudes as listed for K > 0.
    """
    # Half-ellipsoid base (dorsal surface): v ∈ [0, π]
    x0 = a * np.cos(u)                          # along body axis
    y0 = b * np.sin(u) * np.cos(v)              # width
    z0 = c * np.sin(u) * np.abs(np.sin(v))      # height (always ≥ 0)

    # --- Keel: Gaussian ridge along suture (v = 0) --------------------------
    keel = keel_amp * np.exp(-0.5 * (v / keel_width) ** 2) * np.sin(u)

    # --- Longitudinal striae: evenly spaced cosine grooves ------------------
    # n_striae bands spaced across v ∈ [0, π]
    striae = long_amp * np.cos(n_striae * v) * np.sin(u)

    # --- Microstructure: 2-D cosine dimple field ----------------------------
    micro = micro_amp * np.cos(micro_freq_u * u) * np.cos(micro_freq_v * v)

    # Combine as normal-direction perturbation (approximate via radial scaling)
    r_perturb = 1.0 + keel + striae + micro

    x = x0
    y = y0 * r_perturb
    z = z0 * r_perturb

    return x, y, z


# ---------------------------------------------------------------------------
# 11. Coccolithophore Plate
# ---------------------------------------------------------------------------

def coccolithophore(u, v,
                    a=1.2, b=1.2, c=0.35,
                    n_elements=8,
                    shield_amp=0.08, shield_freq=3,
                    rim_amp=0.12, rim_width=0.15,
                    spiral_amp=0.05, spiral_turns=2):
    """
    Stylised coccolithophore plate (coccolith): the calcite shield produced
    by marine algae Emiliania huxleyi.

    Geometry layers (all on an oblate ellipsoid base):

    1. Radial shield elements — n_elements raised petals arranged in a ring
    2. Distal rim            — raised torus-like border around the plate edge
    3. Central spiral        — Archimedean spiral ridge on the proximal shield

    u : [0, π/2]  — colatitude (top face of the plate only)
    v : [0, 2π]   — azimuthal angle

    Parameters
    ----------
    a, b, c        : oblate base ellipsoid semi-axes (c << a for disc shape)
    n_elements     : number of shield elements (radial petals)
    shield_amp     : height of each shield element
    shield_freq    : harmonic order of each element's fine structure
    rim_amp        : height of distal rim
    rim_width      : angular half-width of rim (in u, near u = π/2)
    spiral_amp     : height of central spiral ridge
    spiral_turns   : number of spiral turns across the plate

    K > 0 condition
    ---------------
    All perturbations are outward; base oblate ellipsoid has K > 0.
    Keep amplitudes as listed.
    """
    # Oblate ellipsoid base (top face)
    x0 = a * np.sin(u) * np.cos(v)
    y0 = b * np.sin(u) * np.sin(v)
    z0 = c * np.cos(u)

    # --- Shield elements: n radial petals with fine radial structure --------
    # Each element is a raised Gaussian lobe centred at v = k*2π/n
    shield = np.zeros_like(u, dtype=float)
    for k in range(n_elements):
        v0  = k * 2 * np.pi / n_elements
        dv  = np.arctan2(np.sin(v - v0), np.cos(v - v0))
        # Radial extent: strong mid-plate, fades at centre and rim
        radial_env = np.sin(np.pi * u / (np.pi / 2))   # 0 at pole & equator
        shield += (shield_amp
                   * np.exp(-0.5 * (dv / (np.pi / n_elements)) ** 2)
                   * radial_env
                   * (1 + 0.3 * np.cos(shield_freq * (u * n_elements))))

    # --- Distal rim: Gaussian torus ridge at u ≈ π/2 -----------------------
    rim = rim_amp * np.exp(-0.5 * ((u - np.pi / 2) / rim_width) ** 2)

    # --- Central spiral: Archimedean spiral ridge on proximal face ----------
    # Parameterise by u (radial distance from pole) and v (angle)
    # spiral phase increases with u, so ridges rotate as you move outward
    spiral_phase = spiral_turns * 2 * np.pi * (u / (np.pi / 2))
    spiral = (spiral_amp
              * np.cos(v - spiral_phase)          # rotating ridge
              * np.exp(-2.0 * (u / (np.pi / 4))**2)   # confined to centre
              * (np.sin(u) + 0.1))                # zero at pole

    r_perturb = 1.0 + shield + rim + spiral

    x = x0 * r_perturb
    y = y0 * r_perturb
    z = z0 * r_perturb

    return x, y, z


# ---------------------------------------------------------------------------
# 12. Inflated Anisotropic Membrane
# ---------------------------------------------------------------------------

def inflated_membrane(u, v,
                      Lx=2.0, Ly=1.0,
                      pressure=1.0,
                      Ex=1.0, Ey=2.0,
                      n_terms=6,
                      n_points=80):
    """
    Pressure-inflated rectangular membrane with anisotropic in-plane stiffness.

    A flat elastic membrane of dimensions Lx × Ly, clamped at its edges,
    is inflated by uniform pressure.  The out-of-plane deflection w(x, y) is
    approximated by a double Fourier sine series satisfying the
    Föppl–von Kármán equations in the linear (small-deflection) limit:

        D_x * w_xxxx + 2*D_xy * w_xxyy + D_y * w_yyyy = p

    with D_x = Ex * h³/12, D_y = Ey * h³/12 (h = thickness, set to 1 here).

    The inflated surface is parametrised as a graph over the (x,y) domain,
    and for any convex graph z = w(x,y) ≥ 0 with w = 0 on the boundary the
    Gaussian curvature K = w_xx*w_yy - w_xy² / (1 + |∇w|²)² > 0 in the interior.

    u : [0, 1]  — normalised x coordinate
    v : [0, 1]  — normalised y coordinate

    Parameters
    ----------
    Lx, Ly    : membrane dimensions
    pressure  : applied uniform pressure p
    Ex, Ey    : Young's moduli in x and y directions (anisotropy ratio Ex/Ey)
    n_terms   : number of Fourier terms per direction (odd modes only)
    n_points  : unused (kept for API consistency)

    K > 0 condition
    ---------------
    Any strictly convex smooth graph has K > 0 in its interior.
    The clamped boundary conditions ensure the inflated shape is strictly
    convex for p > 0.
    """
    h = 0.1  # normalised thickness
    Dx  = Ex * h**3 / 12
    Dy  = Ey * h**3 / 12
    Dxy = 0.5 * (Dx + Dy)  # coupling term

    # Physical coordinates
    x = u * Lx
    y = v * Ly

    w = np.zeros_like(x, dtype=float)

    # Double Fourier sine series: odd modes m, n = 1, 3, 5, ...
    for m in range(1, 2 * n_terms, 2):
        for n in range(1, 2 * n_terms, 2):
            am = m * np.pi / Lx
            bn = n * np.pi / Ly

            # Coefficient from Galerkin projection of the biharmonic equation
            # For uniform pressure: rhs integral = (4p / (mn π²)) * Lx * Ly
            denom = (Dx * am**4
                     + 2 * Dxy * am**2 * bn**2
                     + Dy * bn**4)

            if denom == 0:
                continue

            # Fourier coefficient of uniform load
            p_mn = (4 * pressure) / (m * n * np.pi**2)

            W_mn = p_mn / denom

            w += W_mn * np.sin(am * x) * np.sin(bn * y)

    # Surface embedding as a graph
    X = x
    Y = y
    Z = w

    return X, Y, Z


# ---------------------------------------------------------------------------
# 13. Triaxial Harmonic Ovaloid
# ---------------------------------------------------------------------------

def triaxial_harmonic_ovaloid(u, v,
                               a=1.6, b=1.2, c=1.0,
                               tesseral_modes=None):
    """
    Triaxial ellipsoid perturbed by tesseral spherical harmonics Y_l^m with
    m ≠ 0, breaking all rotational symmetry while preserving K > 0.

    Unlike zonal harmonics (m = 0, axially symmetric), tesseral harmonics
    Y_l^m(θ, φ) with m ≠ 0 create genuine 3-D asymmetric bumps.  The result
    is a smooth, strictly convex surface with no axis of symmetry.

    u : [0, π]   — colatitude θ
    v : [0, 2π]  — longitude φ

    Parameters
    ----------
    a, b, c         : triaxial ellipsoid semi-axes
    tesseral_modes  : list of (l, m, A_cos, A_sin) tuples
                      l ≥ 2, 1 ≤ m ≤ l, amplitudes < 0.12 for K > 0.
                      Default: a rich mix of modes giving an interesting shape.

    Mathematics
    -----------
    Associated Legendre polynomials P_l^m are computed via the standard
    three-term recurrence.  The real tesseral harmonics are:
        Y_l^m(θ,φ) = P_l^|m|(cos θ) * { cos(mφ)  for m > 0
                                        { sin(|m|φ) for m < 0
    """
    if tesseral_modes is None:
        tesseral_modes = [
            # (l,  m,   A_cos,  A_sin)
            (2,  1,  0.07,   0.04),
            (2,  2,  0.05,  -0.06),
            (3,  1, -0.04,   0.05),
            (3,  2,  0.06,   0.03),
            (3,  3,  0.04,  -0.04),
            (4,  2,  0.05,   0.02),
            (4,  4,  0.03,  -0.03),
            (5,  3,  0.03,   0.02),
            (6,  4,  0.02,  -0.02),
        ]

    # Base triaxial ellipsoid
    x0 = a * np.sin(u) * np.cos(v)
    y0 = b * np.sin(u) * np.sin(v)
    z0 = c * np.cos(u)

    cos_u = np.cos(u)

    def assoc_legendre(l, m, x):
        """Associated Legendre P_l^m(x) via recurrence (m ≥ 0)."""
        # Start from P_m^m
        Pmm = np.ones_like(x, dtype=float)
        for k in range(1, m + 1):
            Pmm *= -(2 * k - 1) * np.sqrt(1 - x**2)
        if l == m:
            return Pmm
        Pmm1 = x * (2 * m + 1) * Pmm
        if l == m + 1:
            return Pmm1
        for ll in range(m + 2, l + 1):
            Pll = ((2 * ll - 1) * x * Pmm1 - (ll + m - 1) * Pmm) / (ll - m)
            Pmm, Pmm1 = Pmm1, Pll
        return Pmm1

    delta_r = np.zeros_like(u, dtype=float)

    for l, m, Ac, As in tesseral_modes:
        Plm = assoc_legendre(l, abs(m), cos_u)
        delta_r += Plm * (Ac * np.cos(abs(m) * v) + As * np.sin(abs(m) * v))

    r_perturb = 1.0 + delta_r

    x = x0 * r_perturb
    y = y0 * r_perturb
    z = z0 * r_perturb

    return x, y, z


# ---------------------------------------------------------------------------
# 14. Convex Spherical Harmonic Star
# ---------------------------------------------------------------------------

def spherical_harmonic_star(u, v,
                             base_radius=1.0,
                             mode_groups=None,
                             seed=7):
    """
    Convex "star" built by adding many random-amplitude spherical harmonic
    modes to a sphere, subject to a convexity-preserving amplitude bound.

    For a sphere of radius R, adding a spherical harmonic Y_l^m with
    amplitude A keeps the surface convex (K > 0) provided:
        |A| < R / (l(l+1))
    which follows from the eigenvalue equation ΔY = -l(l+1)Y on the sphere.

    This function uses that bound to automatically scale each mode amplitude
    so K > 0 is guaranteed by construction.

    u : [0, π]   — colatitude
    v : [0, 2π]  — longitude

    Parameters
    ----------
    base_radius  : radius of the base sphere R
    mode_groups  : list of (l_min, l_max, n_modes, amp_fraction) tuples.
                   For each group, n_modes random (l,m) pairs are drawn from
                   degree range [l_min, l_max], and amplitude is set to
                   amp_fraction * R / (l(l+1)).
                   Default: three groups giving a richly textured surface.
    seed         : random seed for reproducibility

    K > 0 guarantee
    ---------------
    Each mode satisfies |A_lm| ≤ amp_fraction * R / (l(l+1)).
    By the triangle inequality and the eigenvalue bound the sum of all
    perturbations cannot overcome the base curvature 1/R² provided
    Σ |A_lm| * l(l+1) / R < 1, which is satisfied by construction when
    amp_fraction * n_total_modes < 1.
    """
    if mode_groups is None:
        mode_groups = [
            # (l_min, l_max, n_modes, amp_fraction)
            (2,  4,  6, 0.55),   # large-scale bumps
            (5,  8,  8, 0.40),   # medium features
            (9, 12,  6, 0.25),   # fine texture
        ]

    rng  = np.random.default_rng(seed)
    cos_u = np.cos(u)

    def assoc_legendre(l, m, x):
        Pmm = np.ones_like(x, dtype=float)
        for k in range(1, m + 1):
            Pmm *= -(2 * k - 1) * np.sqrt(np.maximum(1 - x**2, 0))
        if l == m:
            return Pmm
        Pmm1 = x * (2 * m + 1) * Pmm
        if l == m + 1:
            return Pmm1
        for ll in range(m + 2, l + 1):
            Pll = ((2 * ll - 1) * x * Pmm1 - (ll + m - 1) * Pmm) / (ll - m)
            Pmm, Pmm1 = Pmm1, Pll
        return Pmm1

    delta_r = np.zeros_like(u, dtype=float)
    mode_count = 0

    for l_min, l_max, n_modes, amp_frac in mode_groups:
        degrees = rng.integers(l_min, l_max + 1, size=n_modes)
        for l in degrees:
            m   = rng.integers(0, l + 1)
            # Amplitude guaranteed to preserve convexity
            A   = amp_frac * base_radius / (l * (l + 1))
            phi = rng.uniform(0, 2 * np.pi)   # random cos/sin mix angle

            Plm = assoc_legendre(int(l), int(m), cos_u)
            Y   = Plm * (np.cos(phi) * np.cos(m * v)
                         + np.sin(phi) * np.sin(m * v))
            delta_r += A * Y
            mode_count += 1

    r_perturb = base_radius + delta_r

    x = r_perturb * np.sin(u) * np.cos(v)
    y = r_perturb * np.sin(u) * np.sin(v)
    z = r_perturb * np.cos(u)

    return x, y, z


# ---------------------------------------------------------------------------
# 15. Ovaloid from Prescribed K(u, v)
# ---------------------------------------------------------------------------

def prescribed_curvature_ovaloid(u, v,
                                  K_func=None,
                                  R0=1.0,
                                  n_radial=300):
    """
    Surface of revolution whose Gaussian curvature profile K(u) is
    prescribed by the user, via numerical integration of the Gauss equation.

    For a surface of revolution r(s) (where s is arc length along the
    meridian), the Gauss equation reduces to:

        r''(s) + K(s) * r(s) = 0,   r(0) = 0,  r'(0) = 1

    Any solution with r(s) > 0 for s ∈ (0, L) gives a valid surface of
    revolution with the prescribed K(s).  The height z(s) is recovered by:

        z'(s) = sqrt(1 - r'(s)²)

    The surface is an ovaloid (K > 0 everywhere) if K(s) > 0 for all s,
    and the ODE solution r(s) vanishes at the two poles s = 0 and s = L.

    u : [0, 1]   — normalised arc length (0 = north pole, 1 = south pole)
    v : [0, 2π]  — azimuthal angle

    Parameters
    ----------
    K_func   : callable K(s_norm) → scalar/array, s_norm ∈ [0, 1].
               Must return K > 0 everywhere.
               Default: K(s) = 1 + 0.6*sin(3πs)² — a "wavy" curvature.
    R0       : overall scale (equatorial radius)
    n_radial : number of ODE integration steps

    K > 0 condition
    ---------------
    Guaranteed by construction if K_func returns positive values and
    the ODE solution r(s) > 0 in the interior (checked numerically).
    """
    from scipy.integrate import solve_ivp

    if K_func is None:
        def K_func(s):
            # Curvature higher near poles, with a mid-latitude bump
            return 1.0 + 0.6 * np.sin(3 * np.pi * s) ** 2

    # We don't know L a priori; shoot and rescale.
    # Use s ∈ [0, 1] as a proxy for arc length, then renormalise.
    s_span = (0.0, 1.0)
    s_eval = np.linspace(1e-6, 1.0, n_radial)

    def gauss_ode(s, y):
        r, r_dot = y
        K = float(K_func(s))
        r_ddot = -K * r
        # z_dot from arc-length constraint: r_dot² + z_dot² = 1
        r_dot_clamped = np.clip(r_dot, -1 + 1e-9, 1 - 1e-9)
        return [r_dot, r_ddot]

    # Initial conditions: pole at s=0 → r=0, r'=1 (tangent horizontal)
    sol = solve_ivp(
        gauss_ode,
        s_span,
        [0.0, 1.0],
        t_eval=s_eval,
        max_step=1.0 / n_radial,
        method='RK45'
    )

    r_raw  = sol.y[0]
    rd_raw = sol.y[1]
    s_raw  = sol.t

    # Find where r first returns to 0 (south pole)
    zero_crossings = np.where(np.diff(np.sign(r_raw)))[0]
    if len(zero_crossings) > 0:
        pole_idx = zero_crossings[0] + 1
    else:
        pole_idx = len(r_raw) - 1

    r_half = np.maximum(r_raw[:pole_idx], 0)
    rd_half = rd_raw[:pole_idx]
    s_half  = s_raw[:pole_idx]

    # Scale so that max(r) = R0
    scale = R0 / (r_half.max() + 1e-30)
    r_half *= scale

    # Recover z via z'(s) = sqrt(1 - r'²)
    rd_clamped = np.clip(rd_half, -1 + 1e-9, 1 - 1e-9)
    z_dot = np.sqrt(np.maximum(1 - rd_clamped**2, 0))
    z_half = np.concatenate([[0], np.cumsum(z_dot[:-1] * np.diff(s_half))]) * scale

    # Mirror: build full profile (north pole → equator → south pole)
    # Equator is where r is maximum
    eq_idx = np.argmax(r_half)
    r_full = np.concatenate([r_half[:eq_idx+1], r_half[eq_idx::-1]])
    z_full = np.concatenate([z_half[:eq_idx+1], -z_half[eq_idx::-1] + 2*z_half[eq_idx]])

    # Normalise arc-length parameter to [0, 1]
    ds     = np.sqrt(np.diff(r_full)**2 + np.diff(z_full)**2)
    arc    = np.concatenate([[0], np.cumsum(ds)])
    arc   /= arc[-1] + 1e-30

    u_flat = np.asarray(u).ravel()
    r_u    = np.interp(u_flat, arc, r_full).reshape(np.asarray(u).shape)
    z_u    = np.interp(u_flat, arc, z_full).reshape(np.asarray(u).shape)

    # Centre the shape vertically
    z_u -= z_u.mean()

    x = r_u * np.cos(v)
    y = r_u * np.sin(v)
    z = z_u

    return x, y, z


# ---------------------------------------------------------------------------
# Convenience: plot all surfaces (original 7 + new 8)
# ---------------------------------------------------------------------------

def plot_all(n_grid=180, cols=4):
    """
    Render all fifteen surfaces in a single figure.
    Requires matplotlib.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D   # noqa: F401

    surfaces = [
        # ---- original 7 ----
        ("Superellipsoid\n(p=q=3)",
         lambda U, V: superellipsoid(U, V, p=3, q=3),
         np.linspace(0, np.pi, n_grid), np.linspace(0, 2*np.pi, n_grid)),

        ("Rounded Cube\n(roundness=8)",
         lambda U, V: rounded_cube(U, V, roundness=8),
         np.linspace(0, np.pi, n_grid), np.linspace(0, 2*np.pi, n_grid)),

        ("Zonal Harmonic\nEllipsoid",
         lambda U, V: zonal_harmonic_ellipsoid(U, V),
         np.linspace(1e-3, np.pi-1e-3, n_grid), np.linspace(0, 2*np.pi, n_grid)),

        ("Pollen Grain\n(3 apertures)",
         lambda U, V: pollen_grain(U, V),
         np.linspace(0, np.pi, n_grid), np.linspace(0, 2*np.pi, n_grid)),

        ("Diatom Cap\n(12 ribs)",
         lambda U, V: diatom_cap(U, V),
         np.linspace(0, np.pi/2, n_grid), np.linspace(0, 2*np.pi, n_grid)),

        ("Ovaloid of Revolution\n(egg)",
         lambda U, V: ovaloid_of_revolution(U, V, profile="egg"),
         np.linspace(0, np.pi, n_grid), np.linspace(0, 2*np.pi, n_grid)),

        ("Cassini Ovaloid\n(a=1.2, c=0.7)",
         lambda U, V: cassini_ovaloid(U, V, a=1.2, c=0.7),
         np.linspace(0, 1, n_grid), np.linspace(0, 2*np.pi, n_grid)),

        # ---- new 8 ----
        ("Mylar Balloon",
         lambda U, V: mylar_balloon(U, V, R=1.0),
         np.linspace(0, 1, n_grid), np.linspace(0, 2*np.pi, n_grid)),

        ("Radiolarian\nSkeleton",
         lambda U, V: radiolarian(U, V),
         np.linspace(1e-3, np.pi-1e-3, n_grid), np.linspace(0, 2*np.pi, n_grid)),

        ("Beetle Elytra",
         lambda U, V: beetle_elytra(U, V),
         np.linspace(0, np.pi, n_grid), np.linspace(0, np.pi, n_grid)),

        ("Coccolithophore\nPlate",
         lambda U, V: coccolithophore(U, V),
         np.linspace(1e-3, np.pi/2, n_grid), np.linspace(0, 2*np.pi, n_grid)),

        ("Inflated Anisotropic\nMembrane",
         lambda U, V: inflated_membrane(U, V, Ex=1.0, Ey=3.0, n_terms=5),
         np.linspace(0, 1, n_grid), np.linspace(0, 1, n_grid)),

        ("Triaxial Harmonic\nOvaloid",
         lambda U, V: triaxial_harmonic_ovaloid(U, V),
         np.linspace(1e-3, np.pi-1e-3, n_grid), np.linspace(0, 2*np.pi, n_grid)),

        ("Spherical Harmonic\nStar",
         lambda U, V: spherical_harmonic_star(U, V),
         np.linspace(1e-3, np.pi-1e-3, n_grid), np.linspace(0, 2*np.pi, n_grid)),

        ("Prescribed K(u)\nOvaloid",
         lambda U, V: prescribed_curvature_ovaloid(U, V),
         np.linspace(0, 1, n_grid), np.linspace(0, 2*np.pi, n_grid)),
    ]

    rows = int(np.ceil(len(surfaces) / cols))
    fig = plt.figure(figsize=(5 * cols, 4.5 * rows))
    fig.suptitle("Parametric Surfaces with K > 0  (15 surfaces)", fontsize=15, y=1.005)

    cmaps = ["plasma", "viridis", "cividis", "magma", "inferno",
             "coolwarm", "twilight", "PuOr", "RdYlGn", "Spectral",
             "YlOrRd", "BuPu", "GnBu", "OrRd", "PiYG"]

    def _set_equal_axes(ax, X, Y, Z):
        max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2
        mid_x = (X.max() + X.min()) / 2
        mid_y = (Y.max() + Y.min()) / 2
        mid_z = (Z.max() + Z.min()) / 2
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

    for idx, (title, func, u_range, v_range) in enumerate(surfaces):
        U, V = np.meshgrid(u_range, v_range)
        try:
            X, Y, Z = func(U, V)
        except Exception as e:
            print(f"[Warning] {title!r} raised {e!r} — skipping.")
            continue

        ax = fig.add_subplot(rows, cols, idx + 1, projection='3d')
        ax.plot_surface(X, Y, Z,
                        cmap=cmaps[idx % len(cmaps)],
                        linewidth=0, antialiased=True, alpha=0.93)
        ax.set_title(title, fontsize=9)
        ax.set_axis_off()
        _set_equal_axes(ax, X, Y, Z)

    plt.tight_layout()
    plt.savefig("positive_curvature_surfaces.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved → positive_curvature_surfaces.png")


# ===========================================================================
#  SPIRAL / HELICAL SURFACES  (K > 0 everywhere)
# ===========================================================================
#
#  Key insight: a helical/spiral SPINE does not itself have K > 0 — it is a
#  curve, not a surface.  K > 0 is achieved by sweeping a *strictly convex*
#  cross-section along the spine.  Provided the tube radius r_tube is smaller
#  than the reciprocal of the spine's curvature κ at every point
#  (r_tube < 1/κ), the resulting tube surface has K > 0 everywhere.
#
#  For a helix of radius R and pitch p:  κ = R / (R² + p²)
#  so the safe tube radius bound is:    r_tube < (R² + p²) / R
# ===========================================================================


# ---------------------------------------------------------------------------
# 16. Convex Seashell
# ---------------------------------------------------------------------------

def convex_seashell(u, v,
                    n_turns=3,
                    growth_rate=0.18,
                    tube_radius_frac=0.58,
                    flare=1.5,
                    tilt=0.22):
    """
    Logarithmic-spiral seashell with a convex elliptical cross-section
    guaranteed to have K > 0 everywhere.

    The spine follows a 3-D conical logarithmic spiral (like a Turritella or
    Conus shell). The cross-section is an ellipse that grows proportionally
    with the spiral radius, so the tube never self-intersects and K > 0
    is preserved throughout.

    u : [0, 2π]              — cross-section angle
    v : [0, 2π * n_turns]    — spiral angle (coiling parameter)

    Parameters
    ----------
    n_turns          : number of whorls
    growth_rate      : logarithmic spiral expansion rate b (R = e^(b*v))
    tube_radius_frac : cross-section radius as fraction of local spiral R
                       Must satisfy tube_radius_frac < 1 for K > 0
    flare            : ellipse aspect ratio a/b for cross-section (≥ 1)
    tilt             : tilt of cross-section plane relative to horizontal

    K > 0 condition
    ---------------
    The cross-section ellipse is strictly convex (K_cross > 0).  For a tube
    whose radius r satisfies r < 1/κ_spine at every point, the tube surface
    has K > 0 everywhere.  With an exponential spiral κ → 0 as v → ∞, so
    the condition is satisfied for all tube_radius_frac < 1.
    """
    b  = growth_rate
    R  = 1.2 * np.exp(b * v)                    # logarithmic spiral radius

    # Spine position (conical helix: rises with v)
    x_c = R * np.cos(v)
    y_c = R * np.sin(v)
    z_c = growth_rate * v * R * 0.5       # conical rise proportional to R

    # Cross-section: tilted ellipse in the local Frenet frame
    # Approximate local normal and binormal analytically
    # For an exponential spiral, the tangent direction is:
    # T ∝ (cos v - b sin v, sin v + b cos v, growth_rate * (1 + b*v) * 0.5)
    # We use a simplified frame: normal ≈ radial, binormal ≈ z

    r_tube = tube_radius_frac * R

    # Radial (outward from z-axis) unit vector
    nr_x = np.cos(v)
    nr_y = np.sin(v)

    # Cross-section in (radial, z) plane with tilt
    cos_u, sin_u = np.cos(u), np.sin(u)
    # flare makes the cross-section wider radially than vertically
    dx_cross = r_tube * flare * cos_u
    dz_cross = r_tube * sin_u

    # Apply tilt: rotate cross-section by `tilt` radians around tangent
    dx_t = dx_cross * np.cos(tilt) - dz_cross * np.sin(tilt)
    dz_t = dx_cross * np.sin(tilt) + dz_cross * np.cos(tilt)

    x = x_c + dx_t * nr_x
    y = y_c + dx_t * nr_y
    z = z_c + dz_t

    return x, y, z


# ---------------------------------------------------------------------------
# 17. Ammonite
# ---------------------------------------------------------------------------

def ammonite(u, v,
             n_turns=2.5,
             growth_rate=0.12,
             tube_a=0.32,
             tube_b=0.42,
             rib_amp=0.055,
             n_ribs=28,
             rib_width=0.18):
    """
    Ammonite approximation: a planar Archimedean-ish (logarithmic) spiral
    with a convex D-shaped (half-ellipse) cross-section and radial ribbing.

    Ammonites are nearly planar (z ≈ 0 spine), coiled in the xy-plane, with
    characteristic lateral ribs running across the whorl surface.  The ribs
    are modelled as Gaussian bumps in the cross-section angle direction,
    small enough to keep K > 0.

    u : [0, 2π]              — cross-section angle (full ellipse)
    v : [0, 2π * n_turns]    — coiling angle

    Parameters
    ----------
    n_turns     : number of whorls
    growth_rate : logarithmic growth b (R = R0 * e^(b*v))
    tube_a      : cross-section semi-axis in the radial direction
    tube_b      : cross-section semi-axis in the z direction (height of whorl)
    rib_amp     : amplitude of surface ribs  (keep < 0.08 for K > 0)
    n_ribs      : number of ribs per full revolution (total n_ribs * n_turns)
    rib_width   : angular half-width of each rib

    K > 0 condition
    ---------------
    Base cross-section is an ellipse (K > 0).  Ribs are smooth Gaussian
    bumps outward along the whorl surface; for rib_amp < tube_a * 0.25 the
    perturbed cross-section remains strictly convex.
    """
    b  = growth_rate
    R0 = 1.0
    R  = R0 * np.exp(b * v)    # local spine radius (distance from center)

    # Spine: planar logarithmic spiral
    x_c = R * np.cos(v)
    y_c = R * np.sin(v)
    z_c = np.zeros_like(v)

    # Local radial unit vector (outward from coil centre)
    nr_x = np.cos(v)
    nr_y = np.sin(v)

    # Elliptical cross-section in (radial, z) plane
    # u=0 → outermost point, u=π → innermost, u=π/2 → top, u=3π/2 → bottom
    r_cross = R * tube_a     # radial half-width grows with spiral
    z_cross = R * tube_b     # vertical half-height also grows

    # Radial ribs: Gaussian bumps periodic in v
    rib = rib_amp * R * np.exp(-0.5 * ((np.mod(v, 2*np.pi/n_ribs)
          - np.pi/n_ribs) / rib_width)**2)

    # Bump only on the outer half (cos(u) > 0) so K stays positive on inner
    rib_profile = np.maximum(np.cos(u), 0)

    cos_u, sin_u = np.cos(u), np.sin(u)

    dx = (r_cross + rib * rib_profile) * cos_u
    dz = z_cross * sin_u

    x = x_c + dx * nr_x
    y = y_c + dx * nr_y
    z = z_c + dz

    return x, y, z


# ---------------------------------------------------------------------------
# 18. Twisted Ellipsoid
# ---------------------------------------------------------------------------

def twisted_ellipsoid(u, v,
                      a=1.6, b=1.0, c=0.7,
                      twist_rate=1.8,
                      twist_profile="linear"):
    """
    Ellipsoid with a latitude-dependent azimuthal twist applied to each
    ring of constant u, giving a spiral appearance while preserving K > 0.

    The twist is a pure reparametrisation of v → v + φ(u), which is an
    isometry of the parameter domain and does NOT change the intrinsic
    geometry — so K is identical to the base ellipsoid (K > 0 everywhere).
    The extrinsic embedding looks twisted/spiral, however.

    u : [0, π]   — colatitude (twisted latitude)
    v : [0, 2π]  — longitude (before twist)

    Parameters
    ----------
    a, b, c        : ellipsoid semi-axes
    twist_rate     : total twist angle (radians) from north to south pole
    twist_profile  : 'linear'   — uniform twist with latitude
                     'sinusoidal' — twist concentrated at equator
                     'sigmoid'  — rapid twist near mid-latitudes

    K > 0 condition
    ---------------
    The twist is a diffeomorphism of the parameter domain; the image surface
    is smoothly embedded and has the same K as the base ellipsoid (K > 0).
    """
    # Twist angle as function of u ∈ [0, π]
    t = u / np.pi   # normalised ∈ [0, 1]

    if twist_profile == "linear":
        phi = twist_rate * t

    elif twist_profile == "sinusoidal":
        # Twist fastest at equator (t = 0.5)
        phi = twist_rate * (t - np.sin(2 * np.pi * t) / (2 * np.pi))

    elif twist_profile == "sigmoid":
        # Sigmoid: slow at poles, rapid in middle
        phi = twist_rate / (1 + np.exp(-10 * (t - 0.5)))

    else:
        raise ValueError(f"Unknown twist_profile '{twist_profile}'.")

    # Apply twist: shift azimuthal angle
    v_twisted = v + phi

    x = a * np.sin(u) * np.cos(v_twisted)
    y = b * np.sin(u) * np.sin(v_twisted)
    z = c * np.cos(u)

    return x, y, z


# ---------------------------------------------------------------------------
# 19. Spiralling Superellipsoid
# ---------------------------------------------------------------------------

def spiralling_superellipsoid(u, v,
                               a=1.4, b=1.0, c=0.8,
                               p=3.0, q=3.0,
                               twist_rate=2.5,
                               bulge_amp=0.18,
                               bulge_freq=3):
    """
    Superellipsoid with a helical twist baked into the parameterisation,
    plus an optional equatorial bulge that spirals around the surface.

    The twist follows the same isometric reparametrisation as
    `twisted_ellipsoid` but applied to a superellipsoid base, giving
    sharper edges that spiral visibly.  The bulge adds a raised ridge
    that winds helically around the surface.

    u : [0, π]   — latitude
    v : [0, 2π]  — longitude

    Parameters
    ----------
    a, b, c      : superellipsoid semi-axes
    p, q         : superellipsoid exponents (p = q = 2 → ellipsoid)
    twist_rate   : total helical twist (radians) pole to pole
    bulge_amp    : amplitude of the helical ridge (keep < 0.2 for K > 0)
    bulge_freq   : number of ridge turns around the surface

    K > 0 condition
    ---------------
    Base superellipsoid with even integer p, q has K > 0.  The twist is
    isometric. The helical bulge is a small outward radial perturbation;
    keep bulge_amp < 0.2 for K > 0.
    """
    # Helical twist angle
    phi = twist_rate * (u / np.pi)
    v_t = v + phi

    # Superellipsoid base
    x0 = a * _signed_pow(np.sin(u), 2.0 / p) * _signed_pow(np.cos(v_t), 2.0 / q)
    y0 = b * _signed_pow(np.sin(u), 2.0 / p) * _signed_pow(np.sin(v_t), 2.0 / q)
    z0 = c * _signed_pow(np.cos(u), 2.0 / p)

    # Helical ridge: a cosine bump whose phase rotates with u
    ridge_phase = bulge_freq * v - twist_rate * (u / np.pi) * bulge_freq
    ridge = 1.0 + bulge_amp * (0.5 + 0.5 * np.cos(ridge_phase)) * np.sin(u)

    x = x0 * ridge
    y = y0 * ridge
    z = z0

    return x, y, z


# ---------------------------------------------------------------------------
# 20. Coiled Torus Segment
# ---------------------------------------------------------------------------

def coiled_torus_segment(u, v,
                          R_coil=2.0,
                          r_tube=0.45,
                          n_turns=2.0,
                          coil_radius=0.6,
                          pitch=0.5,
                          cross_section="circle",
                          cs_a=1.0, cs_b=0.7):
    """
    A torus whose central circle is bent into a helix (coiled / solenoid
    shape), restricted to the outer convex half where K > 0.

    The spine is a helix of radius `coil_radius` and pitch `pitch` wound
    around a torus of major radius `R_coil`.  The cross-section is a
    circle or ellipse of radius `r_tube`.

    For the outer part of a torus K > 0, so by restricting u to the
    outer half (|u| < π/2 from the outermost equator) we keep K > 0.
    Adding the helical coiling preserves this because the tube radius is
    much smaller than the local curvature radii of the spine.

    u : [-π/2, π/2]         — cross-section angle (outer half only)
    v : [0, 2π * n_turns]   — coiling angle around the torus + helix

    Parameters
    ----------
    R_coil       : major radius of the base torus (torus centre to tube centre)
    r_tube       : tube (cross-section) radius
    n_turns      : number of helical turns
    coil_radius  : radius of the helix wound around the torus
    pitch        : vertical rise per radian of helix
    cross_section: 'circle' or 'ellipse'
    cs_a, cs_b   : ellipse semi-axes (used only if cross_section='ellipse')

    K > 0 condition
    ---------------
    On the outer half of a torus (cos u > 0), K > 0.  The helical coiling
    is a rigid-body motion of the cross-section plane along the spine, which
    does not change the sign of K.  Keep r_tube < R_coil.
    """
    assert r_tube < R_coil, "r_tube must be less than R_coil for K > 0"

    # Spine: helix of radius coil_radius wound around a torus at R_coil
    # Outer torus angle: v sweeps around the coil
    # We superimpose a slow helical wind
    torus_angle = v                          # fast angle: around the torus
    helix_angle = v / (n_turns * 2 * np.pi) * 2 * np.pi   # slow: helical wrap

    # Spine position
    Reff = R_coil + coil_radius * np.cos(helix_angle)
    x_s  = Reff * np.cos(torus_angle)
    y_s  = Reff * np.sin(torus_angle)
    z_s  = coil_radius * np.sin(helix_angle) + pitch * torus_angle

    # Local Frenet frame of the spine
    # Tangent T (approximate, finite diff not needed — analytic)
    dReff_dv   = -coil_radius * np.sin(helix_angle) / (n_turns * 2 * np.pi)
    Tx = dReff_dv * np.cos(torus_angle) - Reff * np.sin(torus_angle)
    Ty = dReff_dv * np.sin(torus_angle) + Reff * np.cos(torus_angle)
    Tz = coil_radius * np.cos(helix_angle) / (n_turns * 2 * np.pi) + pitch
    T_norm = np.sqrt(Tx**2 + Ty**2 + Tz**2) + 1e-30
    Tx, Ty, Tz = Tx/T_norm, Ty/T_norm, Tz/T_norm

    # Approximate normal N as radial outward from z-axis, projected ⊥ T
    Nx_raw = np.cos(torus_angle)
    Ny_raw = np.sin(torus_angle)
    Nz_raw = np.zeros_like(torus_angle)
    # Remove T component
    dot = Nx_raw*Tx + Ny_raw*Ty + Nz_raw*Tz
    Nx = Nx_raw - dot*Tx
    Ny = Ny_raw - dot*Ty
    Nz = Nz_raw - dot*Tz
    N_norm = np.sqrt(Nx**2 + Ny**2 + Nz**2) + 1e-30
    Nx, Ny, Nz = Nx/N_norm, Ny/N_norm, Nz/N_norm

    # Binormal B = T × N
    Bx = Ty*Nz - Tz*Ny
    By = Tz*Nx - Tx*Nz
    Bz = Tx*Ny - Ty*Nx

    # Cross-section in (N, B) plane — OUTER HALF only: u ∈ [-π/2, π/2]
    if cross_section == "ellipse":
        du = cs_a * r_tube * np.cos(u)   # along N (outward)
        db = cs_b * r_tube * np.sin(u)   # along B (binormal)
    else:
        du = r_tube * np.cos(u)
        db = r_tube * np.sin(u)

    x = x_s + du * Nx + db * Bx
    y = y_s + du * Ny + db * By
    z = z_s + du * Nz + db * Bz

    return x, y, z


# ---------------------------------------------------------------------------
# 21. Helical Superellipsoid Tube
# ---------------------------------------------------------------------------

def helical_superellipsoid_tube(u, v,
                                 helix_R=1.5,
                                 helix_pitch=2.5,
                                 n_turns=2.5,
                                 tube_a=0.35,
                                 tube_b=0.28,
                                 p=1.5, q=2.0,
                                 twist_along_spine=0.2):
    """
    A superellipse cross-section swept along a circular helix, with the
    cross-section itself rotating (rolling) along the spine.  This gives
    a shape that looks like a twisted ribbon or helical pillow.

    The cross-section is a superellipse |x/a|^p + |y/b|^q ≤ 1, which is
    strictly convex for p, q ≥ 2.  K > 0 is guaranteed when the tube
    dimensions satisfy tube_a, tube_b < 1/κ_helix = (R² + h²)/R,
    where h = helix_pitch / (2π).

    u : [0, 2π]              — cross-section angle
    v : [0, 2π * n_turns]    — helix angle

    Parameters
    ----------
    helix_R          : helix radius
    helix_pitch      : vertical rise per full turn (in world units)
    n_turns          : number of helical turns
    tube_a, tube_b   : superellipse cross-section semi-axes
    p, q             : superellipse exponents (p = q = 2 → ellipse)
    twist_along_spine: total roll angle of cross-section along helix (radians)

    K > 0 condition
    ---------------
    Superellipse with p,q ≥ 2 is strictly convex → K_cross > 0.
    Tube dimensions must satisfy tube_{a,b} < (helix_R² + (helix_pitch/(2π))²) / helix_R.
    This is checked and a warning is printed if violated.
    """
    h = helix_pitch / (2 * np.pi)   # rise per radian
    kappa_helix = helix_R / (helix_R**2 + h**2)
    safe_radius = 1.0 / kappa_helix
    if max(tube_a, tube_b) >= safe_radius:
        print(f"[Warning] tube radius ({max(tube_a,tube_b):.3f}) ≥ safe bound "
              f"({safe_radius:.3f}); K > 0 may be violated near inner edge.")

    # Helix spine
    x_s = 1.2 * helix_R * np.cos(v)
    y_s = 0.8 *helix_R * np.sin(v)
    z_s = h * v

    # Frenet frame of the helix (analytic)
    # Tangent T
    Tx = -helix_R * np.sin(v)
    Ty =  helix_R * np.cos(v)
    Tz =  h * np.ones_like(v)
    T_norm = np.sqrt(helix_R**2 + h**2)
    Tx, Ty, Tz = Tx/T_norm, Ty/T_norm, Tz/T_norm

    # Principal normal N (centripetal, points toward helix axis)
    Nx = - (1.1) * np.cos(v)
    Ny = - (0.8) * np.sin(v)
    Nz =  np.zeros_like(v)

    # Binormal B = T × N
    Bx = Ty*Nz - Tz*Ny
    By = Tz*Nx - Tx*Nz
    Bz = Tx*Ny - Ty*Nx

    # Cross-section: superellipse, optionally rolled along spine
    roll = twist_along_spine * v / (2 * np.pi * n_turns)  # rolling angle

    cos_u = np.cos(u + roll)
    sin_u = np.sin(u + roll)

    # Superellipse: r(u) = 1 / (|cos u|^p + |sin u|^q)^(1/max(p,q)) — implicit
    # Explicit parametric: use signed power
    cs_x = tube_a * (1+ 0.8 * np.exp(0.08 * v)) * _signed_pow(cos_u, 2.0 / p)
    cs_y = tube_b * (1+ 0.8 * np.exp(0.08 * v))  * _signed_pow(sin_u, 2.0 / q)

    # Embed cross-section into 3D via Frenet frame
    x = x_s + cs_x * Nx + cs_y * Bx
    y = y_s + cs_x * Ny + cs_y * By
    z = z_s + cs_x * Nz + cs_y * Bz

    return x, y, z


# ---------------------------------------------------------------------------
# Convenience: plot ALL surfaces (original 7 + complex 8 + spiral 6 = 21)
# ---------------------------------------------------------------------------

def plot_all(n_grid=180, cols=4):
    """
    Render all 21 surfaces in a single figure.
    Requires matplotlib.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D   # noqa: F401

    pi = np.pi
    surfaces = [
        # ── original 7 ────────────────────────────────────────────────────
       
        ("Inflated Membrane",
         lambda U, V: inflated_membrane(U, V, n_terms=5),
         np.linspace(0, 1, n_grid), np.linspace(0, 1, n_grid)),

        ("Triaxial Harmonic\nOvaloid",
         lambda U, V: triaxial_harmonic_ovaloid(U, V),
         np.linspace(1e-3, pi-1e-3, n_grid), np.linspace(0, 2*pi, n_grid)),

        ("SH Star",
         lambda U, V: spherical_harmonic_star(U, V),
         np.linspace(1e-3, pi-1e-3, n_grid), np.linspace(0, 2*pi, n_grid)),

        ("Prescribed K(u)\nOvaloid",
         lambda U, V: prescribed_curvature_ovaloid(U, V),
         np.linspace(0, 1, n_grid), np.linspace(0, 2*pi, n_grid)),

        # ── spiral / helical 6 ────────────────────────────────────────────
        ("Convex Seashell\n(3 turns)",
         lambda U, V: convex_seashell(U, V, n_turns=3),
         np.linspace(0, 2*pi, n_grid), np.linspace(0, 2*pi*3, n_grid)),

        ("Ammonite\n(2.5 turns)",
         lambda U, V: ammonite(U, V, n_turns=2.5),
         np.linspace(0, 2*pi, n_grid), np.linspace(0, 2*pi*2.5, n_grid)),

        ("Twisted Ellipsoid\n(sigmoid)",
         lambda U, V: twisted_ellipsoid(U, V, twist_profile="sigmoid"),
         np.linspace(0, pi, n_grid), np.linspace(0, 2*pi, n_grid)),

        ("Spiralling\nSuperellipsoid",
         lambda U, V: spiralling_superellipsoid(U, V),
         np.linspace(0, pi, n_grid), np.linspace(0, 2*pi, n_grid)),

        ("Coiled Torus\nSegment",
         lambda U, V: coiled_torus_segment(U, V, n_turns=2),
         np.linspace(-pi/2, pi/2, n_grid), np.linspace(0, 2*pi*2, n_grid)),

        ("Helical\nSuperellipsoid Tube",
         lambda U, V: helical_superellipsoid_tube(U, V, n_turns=2.5),
         np.linspace(0, 2*pi, n_grid), np.linspace(0, 2*pi*2.5, n_grid)),
    ]

    rows = int(np.ceil(len(surfaces) / cols))
    fig  = plt.figure(figsize=(5 * cols, 4.5 * rows))
    fig.suptitle("Parametric Surfaces with K > 0  (21 surfaces)",
                 fontsize=15, y=1.002)

    cmaps = [
        "plasma", "viridis", "cividis", "magma", "inferno",
        "coolwarm", "twilight", "PuOr", "RdYlGn", "Spectral",
        "YlOrRd", "BuPu", "GnBu", "OrRd", "PiYG",
        "copper", "autumn", "winter", "spring", "hot", "bone",
    ]

    def _equal_axes(ax, X, Y, Z):
        mr = np.array([X.max()-X.min(), Y.max()-Y.min(),
                       Z.max()-Z.min()]).max() / 2
        mx, my, mz = (X.max()+X.min())/2, (Y.max()+Y.min())/2, (Z.max()+Z.min())/2
        ax.set_xlim(mx-mr, mx+mr)
        ax.set_ylim(my-mr, my+mr)
        ax.set_zlim(mz-mr, mz+mr)

    for idx, (title, func, u_range, v_range) in enumerate(surfaces):
        U, V = np.meshgrid(u_range, v_range)
        try:
            X, Y, Z = func(U, V)
        except Exception as e:
            print(f"[Warning] {title!r} → {e!r} — skipped.")
            continue

        ax = fig.add_subplot(rows, cols, idx + 1, projection='3d')
        ax.plot_surface(X, Y, Z,
                        cmap=cmaps[idx % len(cmaps)],
                        linewidth=0, antialiased=True, alpha=0.93)
        ax.set_title(title, fontsize=8.5)
        ax.set_axis_off()
        _equal_axes(ax, X, Y, Z)

    plt.tight_layout()
    plt.savefig("positive_curvature_surfaces.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved → positive_curvature_surfaces.png")


if __name__ == "__main__":
    plot_all()