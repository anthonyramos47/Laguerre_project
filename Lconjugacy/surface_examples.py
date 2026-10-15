"""Catalogue of parametric surfaces with K > 0, fitted as B-spline surfaces.

Every example is registered under a name with its function, its (u, v) range and
its parameters, so a notebook can list, fit and show them by name:

    import Lconjugacy.surface_examples as ex
    ex.names()                                   # all example names
    p = ex.get_params("minkowski_patch")         # ranges + every function parameter
    p["twist"] = 1.2
    bsp = ex.fit("minkowski_patch", **p)         # splipy.Surface on [0, 1]^2
    ex.curvature_report(bsp)                     # min / max K, share of K <= 0

The B-spline interpolates the function at the Greville points of an open uniform knot
vector; its parameter domain is always [0, 1]^2 (the (u, v) range is mapped onto it),
which is what Lconjugacy.pipeline expects. ``save`` writes it to data/bsplines/.

Sources
-------
* ``reference/exampleGenerator/functions.py`` - the surfaces of the original example
  generator (shells, ovaloids, ...). Not all of them are K > 0 on their whole range;
  ``curvature_report`` says, and ``Example.k_positive`` records what was checked.
* ``exampleGenerator.py`` (legacy) - nautilus_shell, convex_shell, spiral_tube, copied below.
* New examples (section "Support-function surfaces" and lse_graph) - K > 0 by
  construction, chosen to be less symmetric: no rotational symmetry, umbilics at
  irregular places, strongly varying principal curvatures, parametrisations that are
  not curvature-line or orthogonal.
* Freeform lofts (convex_loft: vaults, canopies, towers, shells) - lofts of planar
  convex sections, K > 0 for every random seed (criterion in that section) - and
  spiral_shell (nautilus / conch family). ``names(architectural=True)`` lists the
  examples suited to architectural geometry; ``random_lofts`` makes batches.
"""
from __future__ import annotations

import functools
import inspect
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import splipy as sp
from splipy import surface_factory

from Lconjugacy.reference.exampleGenerator import functions as gen

SURFACE_DIR = Path(__file__).resolve().parent.parent / "data" / "bsplines"
PI = np.pi


# ── Support-function surfaces (K > 0 by construction) ─────────────────────────
#
# A strictly convex body is given by its support function H (positively homogeneous of
# degree 1 on R^3). Its boundary point with unit normal n is grad H(n), and the radii
# of curvature at that point are the eigenvalues of Hess H(n) restricted to n^perp.
# So the surface n -> grad H(n) has K > 0 wherever that restricted Hessian is positive
# definite, and the parameters (u, v) of n give a regular parametrisation of it.
#
# * Ellipsoid with matrix A (semi-axes a, rotation R, A = R diag(a^2) R^T):
#   H = sqrt(x^T A x), grad H = A x / H.  Hess H is positive definite on n^perp.
# * Minkowski sum of bodies: support functions add, so the Hessians add - a sum of
#   ellipsoids is again K > 0 everywhere, whatever the rotations and axes.
# * Bump on the sphere at direction a: H = c |x| exp(kappa (a.x/|x| - 1)).
#   Adds Hessian ~ c kappa^2 near a, so |c| kappa^2 must stay below the smallest
#   radius of curvature of the base body (checked numerically by curvature_report).


def _random_rotations(rng, n):
    """n random rotation matrices (QR of Gaussian matrices, sign-fixed)."""
    Q, Rm = np.linalg.qr(rng.normal(size=(n, 3, 3)))
    Q = Q * np.sign(np.diagonal(Rm, axis1=1, axis2=2))[:, None, :]
    Q[np.linalg.det(Q) < 0, :, 0] *= -1
    return Q


def _ellipsoid_matrices(n_ellipsoids, seed, axis_range):
    rng = np.random.default_rng(seed)
    R = _random_rotations(rng, n_ellipsoids)
    axes = rng.uniform(*axis_range, size=(n_ellipsoids, 3))
    return np.einsum("kij,kj,klj->kil", R, axes**2, R)          # R diag(a^2) R^T


def _ellipsoids_point(n, A):
    """grad of sum_k sqrt(x^T A_k x) at unit normals n (..., 3)."""
    An = np.einsum("kij,...j->...ki", A, n)                     # (..., k, 3)
    h = np.sqrt(np.einsum("...ki,...i->...k", An, n))
    return np.sum(An / h[..., None], axis=-2)


def _bumps_point(n, directions, amps, sharpness):
    """grad of sum_k c_k |x| exp(kappa (a_k.x/|x| - 1)) at unit normals n."""
    g = np.einsum("kj,...j->...k", directions, n)               # a_k . n
    w = amps * np.exp(sharpness * (g - 1.0))                    # (..., k)
    # grad = w [n + kappa (a_k - g n)]
    return (np.sum(w * (1.0 - sharpness * g), axis=-1)[..., None] * n
            + sharpness * np.einsum("...k,kj->...j", w, directions))


def _sphere_normals(u, v):
    """u: polar angle, v: azimuth."""
    return np.stack([np.sin(u) * np.cos(v), np.sin(u) * np.sin(v), np.cos(u)], axis=-1)


def _split(P):
    return P[..., 0], P[..., 1], P[..., 2]


def minkowski_ellipsoids(u, v, n_ellipsoids=3, seed=1, axis_range=(0.25, 1.3)):
    """Minkowski sum of randomly rotated ellipsoids: K > 0 everywhere, no symmetry.

    u : (0, pi) polar angle of the normal, v : [0, 2 pi] azimuth of the normal.
    The poles u = 0, pi are singular points of the parametrisation, not of the surface.

    n_ellipsoids : number of ellipsoids summed (more = rounder, less elongated)
    seed : random rotations and semi-axes
    axis_range : (min, max) of the random semi-axes; a wider range = more anisotropic
    """
    A = _ellipsoid_matrices(n_ellipsoids, seed, axis_range)
    return _split(_ellipsoids_point(_sphere_normals(u, v), A))


def _bumpy_point(n, axes, n_bumps, amp, sharpness, dent_fraction, seed):
    rng = np.random.default_rng(seed)
    d = rng.normal(size=(n_bumps, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    signs = np.where(rng.uniform(size=n_bumps) < dent_fraction, -1.0, 1.0)
    amps = amp * signs * rng.uniform(0.5, 1.0, size=n_bumps)
    A = np.diag(np.asarray(axes, float) ** 2)[None]
    return _ellipsoids_point(n, A) + _bumps_point(n, d, amps, sharpness)


def bumpy_ovaloid(u, v, axes=(1.3, 1.0, 0.8), n_bumps=14, amp=0.02,
                  sharpness=6.0, dent_fraction=0.35, seed=2):
    """Ellipsoid with random bumps and dents added to its support function.

    Convex (K > 0) while amp * sharpness^2 is well below the smallest radius of
    curvature of the ellipsoid (min(axes)^2 / max(axes)); the defaults keep
    min K > 0 with a margin. Raise amp or sharpness to get close to K = 0.
    Many umbilics, scattered irregularly. u : (0, pi), v : [0, 2 pi] as in
    minkowski_ellipsoids.

    axes : semi-axes of the base ellipsoid
    n_bumps : number of bumps and dents at random directions
    amp : height of the bumps (support-function units); too large gives K < 0
    sharpness : how localised a bump is (its angular width ~ 1 / sqrt(sharpness))
    dent_fraction : share of the bumps that are dents
    seed : random bump directions, signs and heights
    """
    return _split(_bumpy_point(_sphere_normals(u, v), axes, n_bumps, amp,
                               sharpness, dent_fraction, seed))


def minkowski_patch(u, v, body="minkowski", center_theta=1.1, center_phi=0.6,
                    half_width=0.9, shear=0.5, twist=0.8, bend=0.3,
                    n_ellipsoids=3, seed=1, axis_range=(0.25, 1.3)):
    """Open patch of a convex body, parametrised by a sheared, twisted grid of normals.

    (u, v) in [-1, 1]^2. The normal is the gnomonic image of a deformed square
    (shear, swirl by twist * r^2 / (1 + r^2), quadratic bend) around the direction
    (center_theta, center_phi), so the parameter lines are neither orthogonal nor
    curvature lines. No poles, so the B-spline is regular everywhere.

    body : "minkowski" (sum of ellipsoids) or "bumpy" (bumpy_ovaloid, its defaults)
    center_theta : polar angle of the patch's central normal
    center_phi : azimuth of the patch's central normal
    half_width : size of the patch (in tangent-plane units of the normal)
    shear : shear of the parameter grid (0 = orthogonal-ish lines)
    twist : swirl of the parameter lines around the centre (radians at the rim)
    bend : quadratic bending of the parameter lines
    n_ellipsoids, seed, axis_range : the body, as in minkowski_ellipsoids
    """
    s = half_width * (u + shear * v + bend * v**2)
    t = half_width * (v + 0.5 * bend * u**2)
    r2 = s**2 + t**2
    ang = twist * r2 / (1 + r2)                 # smooth swirl (twist * r has a kink at r = 0)
    s, t = s * np.cos(ang) - t * np.sin(ang), s * np.sin(ang) + t * np.cos(ang)
    c = _sphere_normals(np.asarray(center_theta), np.asarray(center_phi))
    e1 = np.array([np.cos(center_theta) * np.cos(center_phi),
                   np.cos(center_theta) * np.sin(center_phi), -np.sin(center_theta)])
    e2 = np.cross(c, e1)
    n = c + s[..., None] * e1 + t[..., None] * e2
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    if body == "minkowski":
        P = _ellipsoids_point(n, _ellipsoid_matrices(n_ellipsoids, seed, axis_range))
    elif body == "bumpy":
        P = _bumpy_point(n, (1.3, 1.0, 0.8), 14, 0.02, 6.0, 0.35, seed)
    else:
        raise ValueError(f"unknown body '{body}'")
    return _split(P)


def lse_graph(u, v, n_planes=7, seed=3, softness=0.25, quad=0.35, slope=1.2):
    """Graph z = f(x, y) of a smoothed max of planes plus a small paraboloid.

    f = softness * log(sum_i exp((a_i . p + b_i) / softness)) + quad/2 |p|^2 is strictly
    convex (log-sum-exp is convex, the quadratic makes it strict), so K > 0 everywhere.
    It looks like a rounded faceted cap: nearly flat facets (K ~ quad^2, small) joined
    by sharply curved ridges (one principal curvature ~ slope^2 / softness), so the
    principal curvatures vary over orders of magnitude. (u, v) = (x, y) in [-1, 1]^2.
    Smaller softness -> sharper ridges; smaller quad -> flatter facets.

    n_planes : number of facets (planes in the smoothed max)
    seed : random plane slopes and offsets
    softness : rounding of the ridges (smaller = sharper ridges, larger K range)
    quad : strength of the added paraboloid (curvature of the facets)
    slope : typical slope of the planes (steepness of the cap)
    """
    rng = np.random.default_rng(seed)
    a = slope * rng.normal(size=(n_planes, 2)) / np.sqrt(2)
    b = 0.3 * rng.normal(size=n_planes)
    L = (np.multiply.outer(u, a[:, 0]) + np.multiply.outer(v, a[:, 1]) + b) / softness
    m = L.max(axis=-1, keepdims=True)
    f = softness * (m[..., 0] + np.log(np.exp(L - m).sum(axis=-1)))
    z = f + 0.5 * quad * (u**2 + v**2)
    return u, v, z


# ── Freeform lofts (K > 0 by construction) ────────────────────────────────────
#
# A loft of planar sections in the planes z = const. Section z is the convex curve with
# support function h(theta, z) in its plane:
#     X(theta, z) = (h n + h_theta t, z),   n = (cos theta, sin theta), t = n'.
# Then X_theta = rho t with rho = h + h_theta_theta (radius of curvature of the section),
# the unit normal is (n - h_z e3) / w, w = sqrt(1 + h_z^2), and
#     L = -rho / w,   M = 0,   N = h_zz / w   =>   K = -rho h_zz / (w^2 (EG - F^2)).
# So K > 0 exactly when every section is strictly convex (rho > 0) and h is strictly
# concave in z (h_zz < 0): the sections may translate (bent spine), change shape and
# rotate freely as long as these two bounds hold.
#
# h = A(z) + sum_{k>=1} b_k(z) cos k theta + c_k(z) sin k theta, with b_k, c_k random
# low-frequency trigonometric functions of z. k = 1 is a translation (the spine),
# k >= 2 shape the sections. With S0 >= sum_{k>=2} (k^2 - 1)(|b_k| + |c_k|) and
# S2 >= sum_{k>=1} |b_k''| + |c_k''| (both bounded from the coefficients),
#     A(z) = C - D (z - 1/2)^2 / 2,   D > S2,   min A > S0
# gives rho > 0 and h_zz < 0 everywhere. A projective map that keeps the surface on one
# side of the plane it sends to infinity preserves K > 0 (affine maps in particular).


def _perspective(P, strength, seed, R):
    """Projective map x -> x / (1 + strength (d . x) / R), d a random unit vector.
    With |x| <= R on the surface the denominator stays >= 1 - strength > 0, so the plane
    sent to infinity misses the surface and K > 0 is preserved."""
    if not 0 <= strength < 1:
        raise ValueError("perspective must be in [0, 1)")
    d = np.random.default_rng(seed).normal(size=3)
    d /= np.linalg.norm(d)
    return P / (1.0 + (strength / R) * (P @ d))[..., None]


def _loft_coefficients(seed, n_modes, z_modes, spine_amp, shape_amp, bulge, margin, twist=0.0):
    """Random coefficients, normalised so that min A = 1 (the smallest section has size
    ~1): S0 = shape_amp < 1 and the spine excursion sum |b_1| + |c_1| = spine_amp."""
    if not 0 <= shape_amp < 1:
        raise ValueError("shape_amp must be in [0, 1)")
    if not 0 <= bulge < 8:
        raise ValueError("bulge must be in [0, 8)")
    rng = np.random.default_rng(seed)
    omega = PI * np.arange(z_modes + 1)                         # z frequencies
    # coef[k, 0|1, j]: amplitude of cos / sin(k theta) for the z-mode j (phase[k, 0|1, j])
    coef = rng.normal(size=(n_modes + 1, 2, z_modes + 1)) / (1.0 + np.arange(z_modes + 1)) ** 3
    coef[0] = 0.0                                               # k = 0 is A(z)
    phase = rng.uniform(0, 2 * PI, size=coef.shape)
    k = np.arange(n_modes + 1)[:, None, None]
    coef[1] *= spine_amp / np.sum(np.abs(coef[1]))
    if n_modes >= 2:
        coef[2:] /= k[2:] ** 2                                  # smoother sections
        coef[2:] *= shape_amp / np.sum((k[2:] ** 2 - 1) * np.abs(coef[2:]))
    S0 = shape_amp
    # |d^2/dz^2 of b_k(z) cos k(theta - twist z)| <= sum_j (omega_j + k |twist|)^2 |coef_kj|
    kt = np.where(k >= 2, k * abs(twist), 0.0)
    S2 = np.sum((omega + kt) ** 2 * np.abs(coef))
    # min A = C - D/8 = 1 > S0 (sections convex); D = (1 + margin) S2 + bulge * C
    # (h_zz < 0), solved together: bulge is the concavity relative to the section size.
    D = ((1 + margin) * S2 + bulge) / (1 - bulge / 8)
    C = 1.0 + D / 8
    return coef, phase, omega, C, D, twist


def _loft_point(theta, z, coef, phase, omega, C, D, twist=0.0):
    f = coef * np.cos(omega * z[..., None, None, None] + phase)          # (..., k, 2, j)
    b, c = f.sum(-1)[..., 0], f.sum(-1)[..., 1]                          # (..., k)
    k = np.arange(coef.shape[0])
    # sections k >= 2 rotate by twist * z about their own centre (k = 1 is the spine)
    arg = k * theta[..., None] - np.where(k >= 2, k * twist, 0.0) * z[..., None]
    ck, sk = np.cos(arg), np.sin(arg)
    A = C - 0.5 * D * (z - 0.5) ** 2
    h = A + np.sum(b * ck + c * sk, -1)
    h_t = np.sum(k * (c * ck - b * sk), -1)
    n = np.stack([np.cos(theta), np.sin(theta)], -1)
    t = np.stack([-np.sin(theta), np.cos(theta)], -1)
    xy = h[..., None] * n + h_t[..., None] * t
    return np.concatenate([xy, z[..., None]], -1)


def convex_loft(u, v, seed=0, n_modes=4, z_modes=2, spine_amp=0.6, shape_amp=0.5,
                bulge=1.0, margin=0.3, height=3.0, perspective=0.0, perspective_seed=None,
                orientation="vertical", twist=0.0):
    """Freeform loft of planar convex sections, K > 0 for every seed.

    u : section angle theta ([0, 2 pi] closed; a sub-range gives an open shell),
    v : station z in [0, 1].
    seed      : random spine, section shapes and their variation along z
    n_modes   : highest Fourier mode of the sections (shape complexity)
    z_modes   : highest frequency of the variation along z (how fast sections change)
    spine_amp : excursion of the spine (k = 1 translation), in units of the smallest
                section (whose size is ~1)
    shape_amp : in [0, 1): the section's radius of curvature stays within
                A (1 +- shape_amp); 0 = circles, near 1 = almost flat stretches
    bulge     : concavity of the radius profile A(z) relative to the section size, in
                [0, 8): 0 = as straight as the bounds allow (nearly developable, K small),
                2-4 = clearly domed
    margin    : safety factor of both bounds (K_min grows with it)
    height    : z scale (affine, keeps K > 0)
    orientation : "vertical" (stations along z: towers) or "horizontal" (stations along
                  x, sections in the (y, z) planes: vaults; use theta in (0, pi) for the
                  upper part)
    perspective : 0..0.9, strength of a random projective map (keeps K > 0); sections
                  then lie in a pencil of planes instead of parallel ones
    perspective_seed : seed of the projective map (None = seed)
    twist     : total rotation (radians) of the section shapes about their centre from
                z = 0 to z = 1; included in the concavity bound, so K > 0 still holds
    """
    cf = _loft_coefficients(seed, n_modes, z_modes, spine_amp, shape_amp, bulge, margin, twist)
    u, v = np.broadcast_arrays(np.asarray(u, float), np.asarray(v, float))
    P = _loft_point(u, v, *cf)
    P[..., 2] = height * (P[..., 2] - 0.5)
    if perspective:
        # |x| bound from the construction, so the map is the same for any (u, v) grid:
        # |xy| <= max h + |h_theta| <= C + 2 sum k |coef|
        k = np.arange(cf[0].shape[0])[:, None, None]
        R = cf[3] + 2 * np.sum(k * np.abs(cf[0])) + 0.5 * height
        P = _perspective(P, perspective, seed if perspective_seed is None else perspective_seed, R)
    if orientation == "horizontal":
        P = P[..., [2, 0, 1]]
    elif orientation != "vertical":
        raise ValueError(f"unknown orientation '{orientation}'")
    return _split(P)



def spiral_shell(u, v, growth=0.15, rise=0.5, section=0.6, aspect=1.4, tilt=0.25,
                 roll=0.1, shape_amp=0.3, n_modes=3, seed=0):
    """Generalised nautilus / conch: a convex section swept along a conical log spiral.

    Spine c(v) = R(v) (cos v, sin v, rise), R = exp(growth v); v : spiral angle.
    The section lies in the meridian plane (e_r, e_z) and is scaled with R (section =
    size relative to R). Its support function is an ellipse (aspect = width along e_r /
    height) plus random modes k = 2..n_modes (shape_amp < 1 = their share of the
    ellipse's smallest radius of curvature, so the section stays convex), rotated by
    tilt + roll * v.
    u : angle of the section's normal in the meridian plane (u = 0: pointing along
    e_r, outwards), so the K > 0 band - the outer part, as for every tube - stays at
    the same u for all v even when the section rotates. The registry range was found
    with k_positive_u_range; rerun it after changing parameters.

    growth : logarithmic growth rate of the spiral (R = exp(growth v))
    rise : height of the spine relative to R (0 = planar nautilus, > 0 = conch)
    section : size of the section relative to R
    aspect : width / height of the elliptic section
    tilt : rotation of the section in the meridian plane (radians)
    roll : extra rotation of the section per radian of v (twisting shell)
    shape_amp : in [0, 1): strength of the random section modes
    n_modes : highest Fourier mode of the section shape
    seed : random section modes
    """
    rng = np.random.default_rng(seed)
    k = np.arange(2, n_modes + 1)
    a = rng.normal(size=(len(k), 2)) / k[:, None] ** 2
    ea, eb = aspect, 1.0                                        # ellipse semi-axes
    rho_min = min(ea, eb) ** 2 / max(ea, eb)
    if len(k):
        a *= shape_amp * rho_min / np.sum((k[:, None] ** 2 - 1) * np.abs(a))
    u, v = np.broadcast_arrays(np.asarray(u, float), np.asarray(v, float))
    phi = u - (tilt + roll * v)                                 # normal angle, unrotated
    q = np.sqrt((ea * np.cos(phi)) ** 2 + (eb * np.sin(phi)) ** 2)
    p = q + np.cos(np.multiply.outer(phi, k)) @ a[:, 0] + np.sin(np.multiply.outer(phi, k)) @ a[:, 1]
    dp = ((eb**2 - ea**2) * np.sin(phi) * np.cos(phi) / q
          + (k * (np.cos(np.multiply.outer(phi, k)) * a[:, 1]
                  - np.sin(np.multiply.outer(phi, k)) * a[:, 0])).sum(-1))
    # point with normal angle u of the rotated section: p n(u) + p' t(u)
    sx = p * np.cos(u) - dp * np.sin(u)
    sy = p * np.sin(u) + dp * np.cos(u)
    R = np.exp(growth * v)
    er = np.stack([np.cos(v), np.sin(v), 0 * v], -1)
    ez = np.array([0.0, 0.0, 1.0])
    P = R[..., None] * (er + rise * ez + section * (sx[..., None] * er + sy[..., None] * ez))
    return _split(P)


def _concave_profile(rng, sag, waviness, n_waves):
    """f(s) = -sag/2 s^2 + sum_j a_j cos(w_j s + phi_j) on s in [-1, 1], with
    sum a_j w_j^2 = waviness * sag, so f'' <= -(1 - waviness) sag < 0."""
    w = rng.uniform(0.6 * PI, 2.5 * PI, size=n_waves)
    a = rng.uniform(0.3, 1.0, size=n_waves)
    a *= waviness * sag / max(np.sum(a * w**2), 1e-12)
    ph = rng.uniform(0, 2 * PI, size=n_waves)
    f = lambda t: -0.5 * sag * t**2 + np.sum(a * np.cos(np.multiply.outer(t, w) + ph), -1)
    return f, sag / 2 + np.sum(a)                               # f and a bound of |f|


def translational_shell(u, v, seed=0, sag_u=0.8, sag_v=0.5, angle=70.0, span_u=2.0,
                        span_v=1.5, waviness=0.6, n_waves=3, slope_u=0.0, slope_v=0.0,
                        perspective=0.0):
    """Translational surface X(u, v) = a(u) + b(v) of two concave arches: K > 0 always.

    a(u) = (span_u u d1, f(u)) and b(v) = (span_v v d2, g(v)) lie in two vertical planes
    (directions d1, d2 at `angle` degrees). Then X_uv = 0, so M = 0 and
    K = f'' g'' N_z^2 / (EG - F^2) > 0 because f'' < 0, g'' < 0 and N_z != 0.
    This is the geometry of translational glass gridshells: the (u, v) grid has planar
    quads. (u, v) in [-1, 1]^2.

    seed : random waves of the two arch profiles
    sag_u, sag_v : depth of the arches (curvature of the parabolic part)
    angle : angle between the two arch planes in degrees (90 = rectangular plan)
    span_u, span_v : half-lengths of the plan along d1, d2
    waviness : in [0, 1): strength of the random waves on the arches (relative to sag)
    n_waves : number of waves per arch
    slope_u, slope_v : linear tilt of the arches (affine, keeps K > 0)
    perspective : 0..0.9, random projective map (keeps K > 0, quads stay planar)
    """
    if not 0 <= waviness < 1:
        raise ValueError("waviness must be in [0, 1)")
    rng = np.random.default_rng(seed)
    f, fb = _concave_profile(rng, sag_u, waviness, n_waves)
    g, gb = _concave_profile(rng, sag_v, waviness, n_waves)
    u, v = np.broadcast_arrays(np.asarray(u, float), np.asarray(v, float))
    al = np.deg2rad(angle)
    d1, d2 = np.array([1.0, 0.0]), np.array([np.cos(al), np.sin(al)])
    xy = span_u * u[..., None] * d1 + span_v * v[..., None] * d2
    z = f(u) + g(v) + slope_u * u + slope_v * v
    P = np.concatenate([xy, z[..., None]], -1)
    if perspective:
        R = span_u + span_v + fb + gb + abs(slope_u) + abs(slope_v)
        P = _perspective(P, perspective, seed + 1, R)
    return _split(P)


def freeform_shell(u, v, growth=0.14, growth_var=0.05, rise=0.6, rise_var=0.3,
                   section=0.55, aspect=1.3, aspect_end=1.0, tilt=0.2, roll=0.15,
                   shape_amp=0.4, n_modes=3, morph_freq=0.25, seed=0,
                   ribs=0, rib_amp=0.0, rib_sharpness=4.0, flare=0.0, flare_at=12.0):
    """Seashell family beyond the plain log spiral (nautilus, conch, Turritella, Murex...).

    Spine c(v) = R(v) (cos v, sin v, Z(v)), v the coiling angle, with a variable growth
    log R = growth v + growth_var sin(v / 2) and a variable rise Z = rise + rise_var
    sin(v / 3) (whorls that steepen and flatten). The section lies in the meridian plane,
    scales with R s(v) and is parametrised by its normal angle u (u = 0 outwards), like
    spiral_shell. Along v it
      * morphs between two random convex shapes (and from aspect to aspect_end),
      * rotates by tilt + roll v,
      * gets ribs: s(v) = 1 + rib_amp bump(ribs v), bump in [0, 1] sharpened by
        rib_sharpness (varices of Murex / ribs of Cardium),
      * flares: s grows by flare (v - flare_at)^2 after v = flare_at (an aperture lip).
    K > 0 only on the outer band, found by k_positive_u_range (ribs and flare narrow it).

    growth : mean log growth per radian of coiling
    growth_var : amplitude of the growth variation
    rise : height of the spine relative to R (0 = planar, > 1 = high spire)
    rise_var : amplitude of the rise variation
    section : section size relative to R
    aspect, aspect_end : width / height of the section at the start / end of the morph
    tilt, roll : rotation of the section (radians) and its rate per radian of v
    shape_amp : in [0, 1): strength of the random section modes
    n_modes : highest Fourier mode of the section shapes
    morph_freq : speed of the morph between the two section shapes (per radian of v)
    seed : random section shapes
    ribs : number of ribs per radian of coiling (0 = none)
    rib_amp : relative height of the ribs
    rib_sharpness : larger = narrower ribs
    flare : strength of the aperture flare (0 = none)
    flare_at : coiling angle where the flare starts
    """
    rng = np.random.default_rng(seed)
    k = np.arange(2, n_modes + 1)
    shapes = rng.normal(size=(2, len(k), 2)) / k[None, :, None] ** 2
    u, v = np.broadcast_arrays(np.asarray(u, float), np.asarray(v, float))
    t = 0.5 * (1 - np.cos(morph_freq * v))                     # 0..1 morph parameter
    ea = aspect + (aspect_end - aspect) * t
    rho_min = np.minimum(ea, 1.0) ** 2 / np.maximum(ea, 1.0)
    if len(k):
        norm = np.sum((k[None, :, None] ** 2 - 1) * np.abs(shapes), axis=(1, 2))
        shapes = shapes / norm[:, None, None]
    a = (1 - t)[..., None, None] * shapes[0] + t[..., None, None] * shapes[1]
    a = shape_amp * rho_min[..., None, None] * a                # keeps rho > 0
    phi = u - (tilt + roll * v)
    q = np.sqrt((ea * np.cos(phi)) ** 2 + np.sin(phi) ** 2)
    ck, sk = np.cos(np.multiply.outer(phi, k)), np.sin(np.multiply.outer(phi, k))
    p = q + np.sum(ck * a[..., 0] + sk * a[..., 1], -1)
    dp = ((1 - ea**2) * np.sin(phi) * np.cos(phi) / q
          + np.sum(k * (ck * a[..., 1] - sk * a[..., 0]), -1))
    sx = p * np.cos(u) - dp * np.sin(u)
    sy = p * np.sin(u) + dp * np.cos(u)
    scale = np.ones_like(v)
    if ribs and rib_amp:
        scale = scale + rib_amp * (0.5 * (1 + np.cos(ribs * v))) ** rib_sharpness
    if flare:
        scale = scale + flare * np.maximum(v - flare_at, 0.0) ** 2
    R = np.exp(growth * v + growth_var * np.sin(v / 2))
    Z = rise + rise_var * np.sin(v / 3)
    er = np.stack([np.cos(v), np.sin(v), 0 * v], -1)
    ez = np.array([0.0, 0.0, 1.0])
    P = R[..., None] * (er + Z[..., None] * ez
                        + (section * scale)[..., None] * (sx[..., None] * er + sy[..., None] * ez))
    return _split(P)


# ── Surfaces of the legacy exampleGenerator.py script ─────────────────────────

def nautilus_shell(u, v, growth_rate=0.2):
    """Tube around a logarithmic spiral (paper's nautilus). u : cross-section angle.

    growth_rate : logarithmic growth of the spiral and of the circular section
    """
    R = np.exp(growth_rate * v)
    r_cross = 0.85 * R
    x = R * np.cos(v) + r_cross * np.cos(u) * np.cos(v)
    y = R * np.sin(v) + r_cross * np.cos(u) * np.sin(v)
    z = 0.5 * v + r_cross * np.sin(u)
    return x, y, z


def convex_shell(u, v, a=2.0, b=1.5, c=1.0, pulse_amp=0.15, pulse_freq=3):
    """Perturbed ellipsoid. u : [0, pi] polar, v : [0, 2 pi] azimuth.

    a, b, c : semi-axes of the ellipsoid
    pulse_amp : amplitude of the waves around the equator (> 0.1 gives K < 0)
    pulse_freq : number of waves around the equator
    """
    pulse = 1.0 + pulse_amp * np.sin(pulse_freq * v) * np.sin(u) ** 2
    return (a * pulse * np.sin(u) * np.cos(v), b * pulse * np.sin(u) * np.sin(v),
            c * np.cos(u))


def spiral_tube(u, v, major_radius=3.0, minor_radius=0.5, pitch=0.1):
    """Circular tube around a helix. u : cross-section angle, v : helix angle.

    major_radius : radius of the helix
    minor_radius : radius of the tube
    pitch : rise of the helix per radian of v
    """
    x = (major_radius + minor_radius * np.cos(u)) * np.cos(v)
    y = (major_radius + minor_radius * np.cos(u)) * np.sin(v)
    z = minor_radius * np.sin(u) + pitch * v
    return x, y, z


# ── Registry ──────────────────────────────────────────────────────────────────

FIT_DEFAULT = dict(n_u=30, n_v=60, order=4)
LOFT_FIT = dict(n_u=20, n_v=20, order=7)       # as the Moebius inputs: 20 x 20, degree 6
SHELL_FIT = dict(n_u=20, n_v=40, order=7)
LONG_SHELL_FIT = dict(n_u=24, n_v=80, order=7)  # many whorls or ribs


@dataclass
class Example:
    func: Callable
    u_range: tuple
    v_range: tuple
    params: dict = field(default_factory=dict)   # overrides of the function defaults
    note: str = ""
    k_positive: bool = True     # K > 0 on the range, checked on the exact surface and the fit
    architectural: bool = False # freeform suitable for architectural geometry
    fit: dict = field(default_factory=dict)      # n_u / n_v / order overrides of FIT_DEFAULT


_e = 0.08          # distance kept from the poles of polar parametrisations
EXAMPLES: dict[str, Example] = {
    # new, K > 0 by construction
    "minkowski_ellipsoids": Example(minkowski_ellipsoids, (_e, PI - _e), (0, 2 * PI),
                                    note="sum of 3 rotated ellipsoids, closed in v"),
    "minkowski_cap":        Example(minkowski_ellipsoids, (0.35, 1.9), (0.3, 2.6),
                                    note="open patch of minkowski_ellipsoids in polar normal coords"),
    "minkowski_patch":      Example(minkowski_patch, (-1, 1), (-1, 1),
                                    note="sheared + twisted normal grid on the Minkowski body", architectural=True),
    "bumpy_ovaloid":        Example(bumpy_ovaloid, (_e, PI - _e), (0, 2 * PI),
                                    note="ellipsoid + random bumps/dents in its support function"),
    "bumpy_patch":          Example(minkowski_patch, (-1, 1), (-1, 1),
                                    params=dict(body="bumpy", center_theta=1.3, center_phi=-0.4,
                                                half_width=1.0, twist=0.6),
                                    note="sheared + twisted patch of bumpy_ovaloid"),
    "lse_graph":            Example(lse_graph, (-1, 1), (-1, 1),
                                    note="rounded facets: curvature varies over orders of magnitude", architectural=True),
    # architectural freeforms: convex lofts (K > 0 for any seed) and spiral shells
    "loft_vault":    Example(convex_loft, (0.35, PI - 0.35), (0, 1),
                             dict(orientation="horizontal", height=5.0, seed=0),
                             note="vault: varying sections along a bent spine", architectural=True, fit=LOFT_FIT),
    "loft_canopy":   Example(convex_loft, (0.2, PI - 0.2), (0, 1),
                             dict(orientation="horizontal", height=5.0, seed=4, perspective=0.6,
                                  spine_amp=1.0),
                             note="vault under a perspective map: sections in a pencil of planes",
                             architectural=True, fit=LOFT_FIT),
    "loft_tower":    Example(convex_loft, (0, 2 * PI), (0, 1),
                             dict(height=5.0, seed=1, bulge=2.0, n_modes=3, shape_amp=0.4),
                             note="closed bulging tower with changing sections", architectural=True, fit=LOFT_FIT),
    "loft_shell":    Example(convex_loft, (0.5, 3.6), (0, 1),
                             dict(height=3.0, seed=2, spine_amp=1.0, z_modes=3),
                             note="open freeform shell (Moebius-style loft patch)", architectural=True, fit=LOFT_FIT),
    "spiral_shell":  Example(spiral_shell, (-1.41, 1.41), (PI, 5 * PI),
                             note="generalised nautilus, outer band", architectural=True, fit=SHELL_FIT),
    "conch_shell":   Example(spiral_shell, (-1.38, 1.38), (2 * PI, 8 * PI),
                             dict(growth=0.1, rise=1.6, section=0.55, aspect=1.2, roll=0.2,
                                  shape_amp=0.5, seed=5),
                             note="rising conical spiral, rolling section", architectural=True, fit=SHELL_FIT),
    # freeform shells (outer band K > 0, found with k_positive_u_range)
    "murex_shell":   Example(freeform_shell, (-0.87, 0.9), (PI, 5 * PI),
                             dict(ribs=2, rib_amp=0.15, rib_sharpness=8, rise=0.9, seed=1),
                             note="shell with ribs (varices)", architectural=True, fit=LONG_SHELL_FIT),
    "turritella_shell": Example(freeform_shell, (-1.35, 1.38), (PI, 9 * PI),
                                dict(rise=2.4, rise_var=0.2, section=0.32, growth=0.06,
                                     growth_var=0.02, aspect=1.0, roll=0.0),
                                note="high spire, four whorls", architectural=True,
                                fit=LONG_SHELL_FIT),
    "flared_shell":  Example(freeform_shell, (-1.38, 1.35), (PI, 5 * PI),
                             dict(flare=0.012, flare_at=11.0, rise=0.4, seed=2),
                             note="aperture lip flaring out", architectural=True, fit=SHELL_FIT),
    "morphing_whorl": Example(freeform_shell, (-1.26, 1.29), (0, 4 * PI),
                              dict(rise=0.0, rise_var=0.6, aspect=1.7, aspect_end=0.8,
                                   shape_amp=0.7, seed=4, morph_freq=0.4, roll=0.25),
                              note="section morphs and rolls, rise oscillates", architectural=True,
                              fit=SHELL_FIT),
    "ribbed_conch":  Example(freeform_shell, (-1.18, 1.18), (2 * PI, 8 * PI),
                             dict(growth=0.1, rise=1.6, section=0.5, ribs=1.5, rib_amp=0.12,
                                  rib_sharpness=6, seed=3, roll=0.2),
                             note="rising spiral with ribs", architectural=True, fit=LONG_SHELL_FIT),
    # translational shells (K > 0 for any parameters, planar quads)
    "translational_roof": Example(translational_shell, (-1, 1), (-1, 1),
                                  dict(seed=0, sag_u=1.4, sag_v=1.0),
                                  note="two wavy concave arches, rhombic plan", architectural=True,
                                  fit=LOFT_FIT),
    "translational_skew": Example(translational_shell, (-1, 1), (-1, 1),
                                  dict(seed=3, angle=55.0, sag_u=1.6, sag_v=0.6, waviness=0.8),
                                  note="skewed plan, strongly waved arches", architectural=True,
                                  fit=LOFT_FIT),
    "translational_canopy": Example(translational_shell, (-1, 1), (-1, 1),
                                    dict(seed=5, sag_u=1.4, sag_v=0.9, perspective=0.6, slope_u=0.4),
                                    note="tilted, under a perspective map", architectural=True,
                                    fit=LOFT_FIT),
    # twisted lofts (K > 0 for every seed)
    "twisted_tower": Example(convex_loft, (0, 2 * PI), (0, 1),
                             dict(seed=3, twist=2.5, shape_amp=0.7, n_modes=2, height=7.0,
                                  spine_amp=0.3),
                             note="elliptic sections turning 2.5 rad", architectural=True, fit=LOFT_FIT),
    "twisted_vault": Example(convex_loft, (0.3, PI - 0.3), (0, 1),
                             dict(seed=2, twist=1.5, shape_amp=0.6, n_modes=3, height=6.0,
                                  orientation="horizontal", spine_amp=1.0),
                             note="vault whose section turns along it", architectural=True,
                             fit=LOFT_FIT),
    # legacy exampleGenerator.py
    "nautilus_shell":       Example(nautilus_shell, (-1.2, 1.2), (0, 6 * PI),
                                    note="outer part of the tube (K = 0 at u = +-pi/2)", architectural=True),
    "convex_shell":         Example(convex_shell, (_e, PI - _e), (0, 2 * PI), dict(pulse_amp=0.08),
                                    note="pulse_amp 0.15 (script default) has K < 0"),
    "spiral_tube":          Example(spiral_tube, (-1.3, 1.3), (0, 6 * PI),
                                    note="outer part of the tube"),
    # reference/exampleGenerator/functions.py
    "superellipsoid":       Example(gen.superellipsoid, (_e, PI - _e), (0, 2 * PI), dict(p=3, q=3)),
    "rounded_cube": Example(gen.rounded_cube, (_e, PI - _e), (0, 2 * PI),
        note="K > 0 but ~1e-6 on the faces: the 40x80 fit has K < 0", k_positive=False),
    "zonal_harmonic_ellipsoid": Example(gen.zonal_harmonic_ellipsoid, (_e, PI - _e), (0, 2 * PI)),
    "pollen_grain": Example(gen.pollen_grain, (_e, PI - _e), (0, 2 * PI),
        note="K < 0 on part of the range", k_positive=False),
    "diatom_cap": Example(gen.diatom_cap, (_e, PI / 2), (0, 2 * PI),
        note="K < 0 on part of the range", k_positive=False),
    "ovaloid_of_revolution": Example(gen.ovaloid_of_revolution, (_e, PI - _e), (0, 2 * PI),
                                     dict(profile="egg")),
    "cassini_ovaloid":      Example(gen.cassini_ovaloid, (0.02, 0.98), (0, 2 * PI)),
    "mylar_balloon": Example(gen.mylar_balloon, (0.02, 0.98), (0, 2 * PI),
        note="K = 0 (the function gives a flat surface)", k_positive=False),
    "radiolarian": Example(gen.radiolarian, (_e, PI - _e), (0, 2 * PI),
        note="K < 0 on part of the range", k_positive=False),
    "beetle_elytra": Example(gen.beetle_elytra, (_e, PI - _e), (_e, PI - _e),
        note="K < 0 on part of the range", k_positive=False),
    "coccolithophore": Example(gen.coccolithophore, (_e, PI / 2), (0, 2 * PI),
        note="K < 0 on part of the range", k_positive=False),
    "inflated_membrane": Example(gen.inflated_membrane, (0.02, 0.98), (0.02, 0.98),
                                    dict(n_terms=5),
        note="K < 0 on part of the range", k_positive=False),
    "triaxial_harmonic_ovaloid": Example(gen.triaxial_harmonic_ovaloid, (_e, PI - _e), (0, 2 * PI),
        note="K < 0 on part of the range", k_positive=False),
    "spherical_harmonic_star": Example(gen.spherical_harmonic_star, (_e, PI - _e), (0, 2 * PI),
        note="K < 0 on part of the range", k_positive=False),
    "prescribed_curvature_ovaloid": Example(gen.prescribed_curvature_ovaloid, (0.02, 0.98),
                                            (0, 2 * PI),
        note="K < 0 on part of the range", k_positive=False),
    "convex_seashell":      Example(gen.convex_seashell, (-1.0, 1.0), (0, 6 * PI),
                                    note="outer part; (-pi/2, pi/2) of exampleGenerator.py has K < 0", architectural=True),
    "ammonite": Example(gen.ammonite, (0, 2 * PI), (0, 5 * PI), dict(n_turns=2.5),
        note="K < 0 on part of the range", k_positive=False),
    "twisted_ellipsoid":    Example(gen.twisted_ellipsoid, (_e, PI - _e), (0, 2 * PI),
                                    dict(twist_profile="sigmoid")),
    "spiralling_superellipsoid": Example(gen.spiralling_superellipsoid, (_e, PI - _e), (0, 2 * PI),
        note="K < 0 on part of the range", k_positive=False),
    "coiled_torus_segment": Example(gen.coiled_torus_segment, (-1.2, 1.2), (0, 4 * PI),
                                    dict(n_turns=2), note="outer part of the tube"),
    "helical_superellipsoid_tube": Example(gen.helical_superellipsoid_tube, (0, 2 * PI),
                                           (0, 5 * PI), dict(n_turns=2.5),
        note="K < 0 on part of the range", k_positive=False),
}

NEW = ["minkowski_ellipsoids", "minkowski_cap", "minkowski_patch", "bumpy_ovaloid",
       "bumpy_patch", "lse_graph", "loft_vault", "loft_canopy", "loft_tower", "loft_shell",
       "spiral_shell", "conch_shell", "murex_shell", "turritella_shell", "flared_shell",
       "morphing_whorl", "ribbed_conch", "translational_roof", "translational_skew",
       "translational_canopy", "twisted_tower", "twisted_vault"]


def names(new_only: bool = False, k_positive_only: bool = False,
          architectural: bool = False) -> list[str]:
    """Example names; k_positive_only drops the ones with K <= 0 somewhere,
    architectural keeps the freeforms suited to architectural geometry."""
    out = list(NEW) if new_only else list(EXAMPLES)
    return [n for n in out if (EXAMPLES[n].k_positive or not k_positive_only)
            and (EXAMPLES[n].architectural or not architectural)]


# random families: loft_00007 = random_loft_params(7), shell_<seed>, roof_<seed>
RANDOM_NAME = re.compile(r"(loft|shell|roof)_(\d+)$")
RANDOM_LOFT = RANDOM_NAME                       # old name


def _random_params(name: str) -> tuple[str, dict] | None:
    m = RANDOM_NAME.match(name)
    if m is None or name in EXAMPLES:
        return None
    family, seed = m.group(1), int(m.group(2))
    gen_ = {"loft": random_loft_params, "shell": random_shell_params,
            "roof": random_roof_params}[family]
    kind, p = gen_(seed)
    return kind, dict(p)


def is_example(name: str) -> bool:
    return name in EXAMPLES or _random_params(name) is not None


def _resolve(name: str) -> tuple[Example, dict]:
    """Registry entry and its parameters; loft_<seed> names are random lofts."""
    if name in EXAMPLES:
        ex = EXAMPLES[name]
        defaults = {k: p.default for k, p in inspect.signature(ex.func).parameters.items()
                    if p.default is not inspect.Parameter.empty}
        return ex, {"u_range": tuple(ex.u_range), "v_range": tuple(ex.v_range),
                    **defaults, **ex.params}
    r = _random_params(name)
    if r is None:
        raise KeyError(f"unknown example '{name}' (see names(), or loft_/shell_/roof_<seed>)")
    kind, p = r
    return EXAMPLES[kind], p


def get_params(name: str) -> dict:
    """u_range, v_range and every parameter of the function (registry values first)."""
    return _resolve(name)[1]


def fit_settings(name: str) -> dict:
    """n_u, n_v, order used by fit(name) unless given."""
    if name.startswith("shell_") and name not in EXAMPLES:
        return dict(LONG_SHELL_FIT)                 # random shells may carry ribs
    return {**FIT_DEFAULT, **_resolve(name)[0].fit}


def describe(name: str) -> str:
    ex, _ = _resolve(name)
    doc = inspect.getdoc(ex.func) or ""
    flag = "" if ex.k_positive else "  [K <= 0 somewhere]"
    kind = "" if name in EXAMPLES else f"random {_random_params(name)[0]}, "
    return f"{name}  ({kind}{ex.func.__name__})  {ex.note}{flag}\n\n{doc}"


def evaluate(name: str, U, V, **params):
    """Points of the exact surface at parameter arrays U, V (in the function's own range)."""
    ex, p = _resolve(name)
    p.update(params)
    del p["u_range"], p["v_range"]
    x, y, z = ex.func(U, V, **p)
    shape = np.broadcast(U, V).shape
    return np.stack([np.broadcast_to(x, shape), np.broadcast_to(y, shape),
                     np.broadcast_to(z, shape)], axis=-1).astype(float)


def exact_curvature(name: str, n_u: int = 150, n_v: int = 150, **params):
    """Gaussian curvature of the exact surface (central differences) on an n_u x n_v grid
    of the (u, v) range. Returns u values, v values, K (n_u, n_v)."""
    p = get_params(name)
    p.update(params)
    (u0, u1), (v0, v1) = p.pop("u_range"), p.pop("v_range")
    h = 1e-4 * max(u1 - u0, v1 - v0)
    us, vs = np.linspace(u0, u1, n_u), np.linspace(v0, v1, n_v)
    U, V = np.meshgrid(us, vs, indexing="ij")
    f = lambda a, b: evaluate(name, a, b, **p)
    X = f(U, V)
    Xu, Xv = (f(U + h, V) - f(U - h, V)) / (2 * h), (f(U, V + h) - f(U, V - h)) / (2 * h)
    Xuu = (f(U + h, V) - 2 * X + f(U - h, V)) / h**2
    Xvv = (f(U, V + h) - 2 * X + f(U, V - h)) / h**2
    Xuv = (f(U + h, V + h) - f(U + h, V - h) - f(U - h, V + h) + f(U - h, V - h)) / (4 * h * h)
    n = np.cross(Xu, Xv)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    E, F, G = (np.sum(a * b, -1) for a, b in ((Xu, Xu), (Xu, Xv), (Xv, Xv)))
    L, M, N = (np.sum(a * n, -1) for a in (Xuu, Xuv, Xvv))
    return us, vs, (L * N - M**2) / (E * G - F**2)


def k_positive_u_range(name: str, u_search=None, shrink: float = 0.05, n: int = 200,
                       **params):
    """Largest u-interval inside u_search (default: the registered u_range) on which the
    exact surface has K > 0 for every v of v_range, shrunk by `shrink` of its length at
    both ends. Use it for tubes/shells after changing their parameters."""
    p = get_params(name)
    p.update(params)
    if u_search is not None:
        p["u_range"] = tuple(u_search)
    us, _, K = exact_curvature(name, n, n, **p)
    ok = np.all(K > 0, axis=1)
    best, start = (0, -1), None
    for i, good in enumerate(np.append(ok, False)):
        if good and start is None:
            start = i
        elif not good and start is not None:
            if i - start > best[1] - best[0] + 1:
                best = (start, i - 1)
            start = None
    if best[1] < best[0]:
        raise ValueError(f"{name}: K <= 0 somewhere on every u column")
    lo, hi = us[best[0]], us[best[1]]
    d = shrink * (hi - lo)
    return (float(lo + d), float(hi - d))

# ── Fitting ───────────────────────────────────────────────────────────────────

def _basis(order, n_funcs):
    n_internal = n_funcs - order
    knots = [0.0] * order + list(np.linspace(0, 1, n_internal + 2)[1:-1]) + [1.0] * order
    return sp.BSplineBasis(order=order, knots=knots)


def fit(name: str, n_u: int | None = None, n_v: int | None = None, order: int | None = None,
        **params) -> sp.Surface:
    """Interpolating B-spline (n_u x n_v control points, given order) on [0, 1]^2.

    name: a registry name or loft_<seed>. n_u, n_v, order default to fit_settings(name).
    params: u_range, v_range and any function parameter (see get_params).
    """
    fs = fit_settings(name)
    n_u, n_v, order = n_u or fs["n_u"], n_v or fs["n_v"], order or fs["order"]
    p = get_params(name)
    p.update(params)
    (u0, u1), (v0, v1) = p.pop("u_range"), p.pop("v_range")
    bu, bv = _basis(order, n_u), _basis(order, n_v)
    s, t = np.asarray(bu.greville()), np.asarray(bv.greville())
    U, V = np.meshgrid(u0 + s * (u1 - u0), v0 + t * (v1 - v0), indexing="ij")
    pts = evaluate(name, U, V, **p)
    if not np.all(np.isfinite(pts)):
        raise ValueError(f"{name}: non-finite points in the (u, v) range")
    return surface_factory.interpolate(pts, [bu, bv], [s, t])


def curvatures(bsp, n_u: int = 80, n_v: int = 80, margin: float = 0.0):
    """Sample grid, Gaussian and mean curvature of a B-spline on [margin, 1 - margin]^2."""
    s = np.linspace(margin, 1 - margin, n_u)
    t = np.linspace(margin, 1 - margin, n_v)
    du = bsp.derivative(s, t, d=(1, 0))
    dv = bsp.derivative(s, t, d=(0, 1))
    duu = bsp.derivative(s, t, d=(2, 0))
    duv = bsp.derivative(s, t, d=(1, 1))
    dvv = bsp.derivative(s, t, d=(0, 2))
    E, F, G = (np.sum(a * b, axis=-1) for a, b in ((du, du), (du, dv), (dv, dv)))
    n = np.cross(du, dv)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    L, M, N = (np.sum(a * n, axis=-1) for a in (duu, duv, dvv))
    det = E * G - F**2
    K = (L * N - M**2) / det
    H = (E * N + G * L - 2 * F * M) / (2 * det)
    return s, t, K, H


def curvature_report(bsp, n: int = 120) -> dict:
    """min / max K and the share of samples with K <= 0 (sign does not depend on n)."""
    _, _, K, H = curvatures(bsp, n, n)
    K = K[np.isfinite(K)]
    return dict(K_min=float(K.min()), K_max=float(K.max()),
                K_ratio=float(K.max() / K.min()) if K.min() > 0 else np.inf,
                frac_K_nonpositive=float(np.mean(K <= 0)))


def mesh(bsp, n_u: int = 80, n_v: int = 80):
    """Quad grid of the B-spline: vertices (n_u*n_v, 3), faces, K per vertex."""
    s, t, K, _ = curvatures(bsp, n_u, n_v)
    Vtx = bsp(s, t).reshape(-1, 3)
    i, j = np.meshgrid(np.arange(n_u - 1), np.arange(n_v - 1), indexing="ij")
    a = (i * n_v + j).ravel()
    Fc = np.stack([a, a + 1, a + n_v + 1, a + n_v], axis=1)
    return Vtx, Fc, K.ravel()


def control_net(bsp):
    """Control points (m*n, 3) and the edges of the control net."""
    C = np.asarray(bsp.controlpoints)[..., :3]
    m, n = C.shape[:2]
    idx = np.arange(m * n).reshape(m, n)
    E = np.concatenate([np.stack([idx[:, :-1].ravel(), idx[:, 1:].ravel()], 1),
                        np.stack([idx[:-1].ravel(), idx[1:].ravel()], 1)])
    return C.reshape(-1, 3), E


def save(bsp, filename: str, surface_dir=SURFACE_DIR) -> Path:
    """Write the B-spline to <surface_dir>/<filename>.json (pipeline.load_surface format)."""
    from Lconjugacy.utils.bsplines_functions import save_bspline_to_json
    path = Path(surface_dir) / f"{filename}.json"
    save_bspline_to_json(bsp, str(path))
    return path


# ── Batches of random lofts ───────────────────────────────────────────────────

LOFT_KINDS = ("loft_vault", "loft_canopy", "loft_tower", "loft_shell")


def random_loft_params(seed: int) -> tuple[str, dict]:
    """A random architectural loft: (registry name, parameters for fit). K > 0 always."""
    rng = np.random.default_rng(seed)
    kind = LOFT_KINDS[rng.integers(len(LOFT_KINDS))]
    p = get_params(kind)
    p.update(seed=int(seed), n_modes=int(rng.integers(2, 6)), z_modes=int(rng.integers(1, 4)),
             spine_amp=float(rng.uniform(0.2, 1.5)), shape_amp=float(rng.uniform(0.1, 0.7)),
             bulge=float(rng.uniform(0.3, 3.0)))
    if kind == "loft_canopy":
        p["perspective"] = float(rng.uniform(0.3, 0.8))
    if kind == "loft_shell":
        a = float(rng.uniform(0, 2 * PI))
        p["u_range"] = (a, a + float(rng.uniform(1.5, 3.5)))
    return kind, p


def random_lofts(n: int, first_seed: int = 0, out_dir=None, prefix: str = "loft",
                 n_u: int = 20, n_v: int = 20, order: int = 7) -> list[dict]:
    """n random architectural lofts as B-splines (default 20 x 20, degree 6, as the
    Moebius inputs). With out_dir, writes <prefix>_<seed>.json for each and
    <prefix>_manifest.json with the kind and parameters to rebuild every one."""
    out = []
    for seed in range(first_seed, first_seed + n):
        kind, p = random_loft_params(seed)
        bsp = fit(kind, n_u=n_u, n_v=n_v, order=order, **p)       # same as fit(f"loft_{seed}")
        item = dict(name=f"{prefix}_{seed:05d}", kind=kind, params=p, bspline=bsp,
                    **curvature_report(bsp))
        if out_dir is not None:
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            item["path"] = str(save(bsp, item["name"], out_dir))
        out.append(item)
    if out_dir is not None:
        import json
        manifest = [{k: v for k, v in it.items() if k != "bspline"} for it in out]
        with open(Path(out_dir) / f"{prefix}_manifest.json", "w") as fh:
            json.dump(manifest, fh, indent=2, default=float)
    return out


SHELL_KINDS = ("murex_shell", "turritella_shell", "flared_shell", "morphing_whorl", "ribbed_conch")


@functools.lru_cache(maxsize=256)
def _random_shell_cached(seed: int):
    rng = np.random.default_rng(10_000 + seed)
    kind = SHELL_KINDS[rng.integers(len(SHELL_KINDS))]
    p = get_params(kind)
    turns = float(rng.uniform(1.5, 2.5))
    v0 = float(rng.uniform(0.5, 2.0) * PI)
    p.update(seed=int(seed), growth=float(rng.uniform(0.07, 0.2)),
             growth_var=float(rng.uniform(0, 0.06)), rise=float(rng.uniform(0, 2.2)),
             rise_var=float(rng.uniform(0, 0.5)), section=float(rng.uniform(0.35, 0.7)),
             aspect=float(rng.uniform(0.8, 1.8)), aspect_end=float(rng.uniform(0.8, 1.8)),
             tilt=float(rng.uniform(-0.4, 0.4)), roll=float(rng.uniform(-0.3, 0.3)),
             shape_amp=float(rng.uniform(0, 0.7)), n_modes=int(rng.integers(2, 5)),
             morph_freq=float(rng.uniform(0.1, 0.5)),
             ribs=float(rng.choice([0, 0, 1.0, 1.5, 2.0])), rib_amp=float(rng.uniform(0.05, 0.15)),
             rib_sharpness=float(rng.uniform(3, 8)),
             flare=float(rng.choice([0.0, 0.0, 0.01])), v_range=(v0, v0 + 2 * PI * turns))
    p["flare_at"] = p["v_range"][1] - 2.0
    # tubes are K > 0 on their outer band only: find it on the exact surface
    p["u_range"] = tuple(round(x, 3) for x in k_positive_u_range(kind, u_search=(-PI, PI), n=160, **p))
    return kind, p


def random_shell_params(seed: int) -> tuple[str, dict]:
    """A random freeform shell (kind, parameters); u_range is its K > 0 band."""
    kind, p = _random_shell_cached(int(seed))
    return kind, dict(p)


ROOF_KINDS = ("translational_roof", "translational_skew", "translational_canopy",
              "twisted_tower", "twisted_vault")


def random_roof_params(seed: int) -> tuple[str, dict]:
    """A random translational shell or twisted loft (kind, parameters). K > 0 always."""
    rng = np.random.default_rng(20_000 + seed)
    kind = ROOF_KINDS[rng.integers(len(ROOF_KINDS))]
    p = get_params(kind)
    p["seed"] = int(seed)
    if kind.startswith("translational"):
        p.update(sag_u=float(rng.uniform(0.6, 2.0)), sag_v=float(rng.uniform(0.4, 1.6)),
                 angle=float(rng.uniform(50, 90)), span_u=float(rng.uniform(1.5, 3.0)),
                 span_v=float(rng.uniform(1.0, 2.5)), waviness=float(rng.uniform(0.2, 0.9)),
                 n_waves=int(rng.integers(1, 5)), slope_u=float(rng.uniform(-0.5, 0.5)),
                 slope_v=float(rng.uniform(-0.5, 0.5)),
                 perspective=float(rng.choice([0.0, rng.uniform(0.3, 0.7)])))
    else:
        p.update(twist=float(rng.uniform(0.8, 3.0)), shape_amp=float(rng.uniform(0.3, 0.8)),
                 n_modes=int(rng.integers(2, 4)), spine_amp=float(rng.uniform(0.1, 1.2)),
                 bulge=float(rng.uniform(0.5, 2.5)), z_modes=int(rng.integers(1, 3)))
    return kind, p
