"""Shared orbital geometry and mechanics helpers."""

import jax
from astropy import constants, units
from jax import numpy as jnp

jax.config.update("jax_enable_x64", True)

AU_M = constants.au.value
M_SUN_KG = constants.M_sun.value
R_SUN_M = constants.R_sun.value
G = constants.G.value
C = constants.c.value
DAY_S = units.day.to(units.s)


def orbital_period_days(a_m, total_mass_kg):
    """Compute orbital period in days from semi-major axis and total mass in SI."""
    return (2 * jnp.pi * jnp.sqrt(a_m**3 / (G * total_mass_kg))) / DAY_S


def orbital_separation_m(phase, a_m, e, periapsis_phase):
    """Compute binary separation in meters at true orbital longitude ``phase``."""
    return a_m * (1 - e**2) / (1 + e * jnp.cos(phase - periapsis_phase))


def einstein_radius_m(a_m, sin_i, macc_kg, e, periapsis_phase, phase=0.0):
    """Compute Einstein radius in meters at ``phase``."""
    separation = orbital_separation_m(phase, a_m, e, periapsis_phase)
    return jnp.sqrt((4 * G * macc_kg * separation * sin_i) / (C**2))


def impact_parameter_source_radii(a_m, i, rcomp_m):
    """Compute lens impact parameter in source radii."""
    return jnp.abs(a_m * jnp.cos(i) / rcomp_m)


def orbital_phase(time_offsets_days, porb_days, e, periapsis_phase):
    """Convert time offsets in days to true orbital longitudes in radians."""
    true_anomaly_at_peak = -periapsis_phase
    eccentric_anomaly_at_peak = 2 * jnp.arctan2(
        jnp.sqrt(1 - e) * jnp.sin(true_anomaly_at_peak / 2),
        jnp.sqrt(1 + e) * jnp.cos(true_anomaly_at_peak / 2),
    )
    mean_anomaly_at_peak = eccentric_anomaly_at_peak - e * jnp.sin(
        eccentric_anomaly_at_peak
    )
    mean_anomaly = jnp.mod(
        mean_anomaly_at_peak + (time_offsets_days / porb_days) * 2 * jnp.pi,
        2 * jnp.pi,
    )

    eccentric_anomaly = mean_anomaly
    for _ in range(10):
        eccentric_anomaly = eccentric_anomaly - (
            eccentric_anomaly - e * jnp.sin(eccentric_anomaly) - mean_anomaly
        ) / (1 - e * jnp.cos(eccentric_anomaly))

    true_anomaly = 2 * jnp.arctan2(
        jnp.sqrt(1 + e) * jnp.sin(eccentric_anomaly / 2),
        jnp.sqrt(1 - e) * jnp.cos(eccentric_anomaly / 2),
    )
    phase = jnp.mod(true_anomaly + periapsis_phase, 2 * jnp.pi)
    return jnp.where(jnp.abs(phase - 2 * jnp.pi) < 1e-12, 0.0, phase)


def source_offset_source_radii(phase, a_m, rcomp_m, e, periapsis_phase, b_source):
    """Compute projected lens-source separation in source radii."""
    phase_limit = 0.5 * jnp.pi
    phase_upper_limit = 2 * jnp.pi - phase_limit
    projected_phase = jnp.where(
        (phase > phase_limit) & (phase < phase_upper_limit),
        phase_limit,
        phase,
    )
    projected_separation = jnp.abs(
        orbital_separation_m(projected_phase, a_m, e, periapsis_phase)
        * jnp.sin(projected_phase)
        / rcomp_m
    )
    return jnp.hypot(projected_separation, b_source)


def source_offset_einstein_radii(
    phase,
    a_m,
    rcomp_m,
    r_e_m,
    e,
    periapsis_phase,
    b_source,
):
    """Compute projected lens-source separation in Einstein radii."""
    l_source_radii = source_offset_source_radii(
        phase,
        a_m,
        rcomp_m,
        e,
        periapsis_phase,
        b_source,
    )
    return l_source_radii / (r_e_m / rcomp_m)
