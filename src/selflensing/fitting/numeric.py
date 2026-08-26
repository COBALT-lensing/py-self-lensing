"""Unitless deterministic models used by the batched BlackJAX likelihood."""

from jax import numpy as jnp

from ..gp.PhysicalRelations import effective_temperature_K
from ..lightcurves import light_curve_flux, physical_parameters, radial_velocity_m_per_s
from ..orbits import einstein_radius_m, orbital_period_days


def parameter_row(fitted, mcomp, rcomp):
    """Pack the sampled and fixed public-unit values for one model row."""
    a, inclination, macc, eccentricity, periapsis_phase = fitted
    return jnp.asarray(
        [
            a,
            inclination,
            macc,
            mcomp,
            rcomp,
            effective_temperature_K(mcomp),
            eccentricity,
            periapsis_phase,
        ]
    )


def physical_properties(params):
    """Return feasibility, orbital period, and effective duration in days."""
    (
        a_m,
        inclination,
        macc_kg,
        _mcomp_kg,
        total_mass_kg,
        rcomp_m,
        _teff,
        eccentricity,
        periapsis_phase,
        sin_i,
    ) = physical_parameters(params)
    r_e_m = einstein_radius_m(
        a_m, sin_i, macc_kg, eccentricity, periapsis_phase
    )
    impact = a_m * jnp.abs(jnp.cos(inclination)) / r_e_m
    impact_limit = 1.0 + rcomp_m / r_e_m
    period_days = orbital_period_days(a_m, total_mass_kg)
    tau_eff_days = (
        period_days
        * (r_e_m + rcomp_m)
        / (jnp.pi * a_m * sin_i)
        * jnp.sqrt(1.0 - (impact / impact_limit) ** 2)
    )
    feasible = (
        jnp.isfinite(r_e_m)
        & (r_e_m > 0)
        & jnp.isfinite(impact)
        & (impact <= impact_limit)
    )
    return feasible, period_days, tau_eff_days


def deterministic_models(lc_times, rv_times, params, peak_time):
    """Evaluate the shared unitless deterministic light-curve and RV models."""
    light_curve = (
        None if lc_times is None else light_curve_flux(lc_times, params, peak_time)
    )
    radial_velocity = (
        None
        if rv_times is None
        else radial_velocity_m_per_s(rv_times, params, peak_time)
    )
    return light_curve, radial_velocity
