"""Differentiable bound-Kepler mechanics and observer-frame geometry.

Longitude zero points toward the observer. ``peak_time`` in legacy APIs is the
longitude-zero reference epoch, not a guarantee of the exact flux maximum for
an inclined eccentric orbit. Positive z places the lens in front of the source.
"""

import jax
import numpy as np
from astropy import constants, units
from jax import numpy as jnp
from jaxoplanet.core.kepler import kepler

jax.config.update("jax_enable_x64", True)
AU_M = constants.au.value
M_SUN_KG = constants.M_sun.value
R_SUN_M = constants.R_sun.value
G = constants.G.value
C = constants.c.value
DAY_S = units.day.to(units.s)
_NODES, _WEIGHTS = np.polynomial.legendre.leggauss(256)


def orbital_period_days(a_m, total_mass_kg):
    """Kepler's period law, valid for all bound eccentricities."""
    return 2 * jnp.pi * jnp.sqrt(a_m**3 / (G * total_mass_kg)) / DAY_S


def conjunction_crossing_time_factor(e, periapsis_phase):
    """Circular/local transverse-speed ratio at longitude zero (edge-on)."""
    return jnp.sqrt((1-e)*(1+e)) / (1+e*jnp.cos(periapsis_phase))


def orbital_separation_m(phase, a_m, e, periapsis_phase):
    return a_m * (1-e)*(1+e) / (1+e*jnp.cos(phase-periapsis_phase))


def projected_position_m(phase, a_m, inclination, e, periapsis_phase):
    """Relative lens position (sky x, sky y, foreground z), in meters."""
    radius = orbital_separation_m(phase, a_m, e, periapsis_phase)
    return (radius*jnp.sin(phase), radius*jnp.cos(inclination)*jnp.cos(phase),
            radius*jnp.sin(inclination)*jnp.cos(phase))


def einstein_radius_m(a_m, sin_i, macc_kg, e, periapsis_phase, phase=0.0):
    """Einstein radius from instantaneous foreground line-of-sight separation.

    Returns zero behind the source. The positive placeholder avoids a sqrt(0)
    derivative contaminating masked rear-half-orbit gradients.
    """
    radius = orbital_separation_m(phase, a_m, e, periapsis_phase)
    z = radius*sin_i*jnp.cos(phase)
    front = z > 0
    safe_z = jnp.where(front, z, 1.)
    return jnp.where(front, jnp.sqrt(4*G*macc_kg*safe_z/C**2), 0.)


def impact_parameter_source_radii(a_m, i, rcomp_m, e=0., periapsis_phase=0.):
    """Projected offset at the longitude-zero epoch, in source radii.

    The three-argument form explicitly defaults to a circular orbit. This is
    not a minimization over orbital longitude for a general inclined orbit.
    """
    radius = orbital_separation_m(0., a_m, e, periapsis_phase)
    return jnp.abs(radius*jnp.cos(i)/rcomp_m)


def _symmetric_kepler(mean, eccentricity):
    reduced = jnp.fmod(mean, 2*jnp.pi)
    sign = jnp.where(reduced < 0, -1., 1.)
    magnitude = jnp.abs(reduced)
    reflected = magnitude > jnp.pi
    magnitude = jnp.where(reflected, 2*jnp.pi-magnitude, magnitude)
    sign = jnp.where(reflected, -sign, sign)
    sine, cosine = kepler(magnitude, eccentricity)
    return sign*sine, cosine


def orbital_phase(time_offsets_days, porb_days, e, periapsis_phase):
    """Time offsets to longitude using jaxoplanet's analytic-JVP Kepler solver."""
    E0 = 2*jnp.arctan2(jnp.sqrt(1-e)*jnp.sin(-periapsis_phase/2),
                       jnp.sqrt(1+e)*jnp.cos(-periapsis_phase/2))
    square = E0*E0
    # Stable E-sin(E) through order 15; omitted term <3e-32 at |E|<0.1.
    polynomial = (1/6 + square*(-1/120 + square*(1/5040 + square*(-1/362880
                  + square*(1/39916800 + square*(-1/6227020800
                  + square/1307674368000))))))
    e_minus_sin = jnp.where(jnp.abs(E0)<.1, E0*square*polynomial, E0-jnp.sin(E0))
    mean0 = (1-e)*E0 + e*e_minus_sin
    sine, cosine = _symmetric_kepler(mean0+time_offsets_days*2*jnp.pi/porb_days, e)
    # No snap-to-zero branch: it would erase derivatives near the wrap.
    return jnp.mod(jnp.arctan2(sine, cosine)+periapsis_phase, 2*jnp.pi)


def source_offset_source_radii(phase, a_m, rcomp_m, e, periapsis_phase, b_source):
    """Full projected distance; b_source is the longitude-zero source offset.

    Geometry is returned for either half-orbit; foreground gating is applied
    through the instantaneous Einstein radius, not by clipping longitude.
    """
    radius0 = orbital_separation_m(0., a_m, e, periapsis_phase)
    cos_i = b_source*rcomp_m/radius0
    radius = orbital_separation_m(phase, a_m, e, periapsis_phase)
    return radius*jnp.hypot(jnp.sin(phase), cos_i*jnp.cos(phase))/rcomp_m


def source_offset_einstein_radii(phase, a_m, rcomp_m, r_e_m, e, periapsis_phase, b_source):
    """Full offset in Einstein radii; caller supplies phase-dependent r_e_m."""
    offset = source_offset_source_radii(phase, a_m, rcomp_m, e, periapsis_phase, b_source)
    front = r_e_m > 0
    return jnp.where(front, offset*rcomp_m/jnp.where(front, r_e_m, 1.), jnp.inf)


def relative_polar_velocity_m_per_s(phase, a_m, total_mass_kg, e, periapsis_phase):
    anomaly = phase-periapsis_phase
    scale = jnp.sqrt(G*total_mass_kg/(a_m*(1-e)*(1+e)))
    return scale*e*jnp.sin(anomaly), scale*(1+e*jnp.cos(anomaly))


def crossing_time_days(a_m, inclination, macc_kg, total_mass_kg, rstar_m, e,
                       periapsis_phase, *, include_einstein=True):
    """Local straight-track chord duration at longitude zero, not an FWHM.

    Includes eccentric radial/transverse velocity and the local closest sky
    approach. It freezes velocity and the boundary radius over the crossing;
    a finite event's exact contact times require solving the orbit directly.
    Non-intersecting local tracks return NaN. Face-on systems cannot lens.
    """
    radius0 = orbital_separation_m(0., a_m, e, periapsis_phase)
    vr, vt = relative_polar_velocity_m_per_s(0., a_m, total_mass_kg, e, periapsis_phase)
    speed2 = vt*vt + (vr*jnp.cos(inclination))**2
    impact2 = (radius0*jnp.cos(inclination))**2 * vt*vt/speed2
    boundary = rstar_m
    if include_einstein:
        boundary = boundary + einstein_radius_m(a_m, jnp.sin(inclination), macc_kg,
                                               e, periapsis_phase)
    chord2 = boundary*boundary-impact2
    duration = 2*jnp.sqrt(jnp.where(chord2 > 0, chord2, 1.))/jnp.sqrt(speed2)/DAY_S
    return jnp.where((chord2 > 0) & (jnp.sin(inclination)>0), duration, jnp.nan)


def mean_relative_speed_m_per_s(a_m, total_mass_kg, e):
    """Time-average relative speed = ellipse perimeter / orbital period."""
    theta = jnp.asarray((_NODES+1)*np.pi/4)
    weights = jnp.asarray(_WEIGHTS*np.pi/4)
    integral = jnp.sum(weights*jnp.sqrt(1-jnp.asarray(e)[...,None]**2*jnp.sin(theta)**2), axis=-1)
    return 4*a_m*integral/(orbital_period_days(a_m, total_mass_kg)*DAY_S)
