"""Unitless JAX helpers for deterministic self-lensing fitting.

This module is deliberately internal: public fitting functions continue to
accept Astropy quantities and return :class:`SelfLensingSystem` objects.  It
provides the numerical core those functions can call after their API-boundary
unit conversion.
"""

from jax import numpy as jnp

from ..lightcurves import (
    light_curve_flux,
    physical_parameters,
    radial_velocity_m_per_s,
)
from ..orbits import einstein_radius_m, orbital_period_days


def parameter_row(fitted, mcomp, rcomp, teff):
    """Pack five sampled and three fixed public-unit values for the model."""
    a, inclination, macc, eccentricity, periapsis_phase = fitted
    return jnp.asarray(
        [a, inclination, macc, mcomp, rcomp, teff, eccentricity, periapsis_phase]
    )


def physical_validity(params):
    """Return self-lensing feasibility and the orbital period in days."""
    a_m, inclination, macc_kg, _mcomp_kg, total_mass_kg, rcomp_m, _teff, e, phase, sin_i = physical_parameters(params)
    r_e = einstein_radius_m(a_m, sin_i, macc_kg, e, phase)
    impact = a_m * jnp.abs(jnp.cos(inclination)) / r_e
    b_max = 1.0 + rcomp_m / r_e
    return (impact <= b_max) & jnp.isfinite(r_e) & (r_e > 0), orbital_period_days(a_m, total_mass_kg)


def deterministic_models(times_lc, times_rv, params, peak_time):
    """Evaluate the existing unitless light-curve and RV physics."""
    light_curve = None if times_lc is None else light_curve_flux(times_lc, params, peak_time)
    radial_velocity = None if times_rv is None else radial_velocity_m_per_s(times_rv, params, peak_time)
    return light_curve, radial_velocity
