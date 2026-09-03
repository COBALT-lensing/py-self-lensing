"""Shared deterministic light-curve physics."""

from collections.abc import Mapping, Sequence

import jax
from astropy import constants, units
from jax import numpy as jnp

from .lensing import integrated_witt_mao_magnification_jax
from .orbits import (
    AU_M,
    C,
    G,
    M_SUN_KG,
    R_SUN_M,
    einstein_radius_m,
    impact_parameter_source_radii,
    orbital_period_days,
    orbital_phase,
    orbital_separation_m,
    source_offset_einstein_radii,
)

jax.config.update("jax_enable_x64", True)

PARAMETER_ORDER = (
    "a",
    "i",
    "Macc",
    "Mcomp",
    "Rcomp",
    "Teff",
    "e",
    "periapsis_phase",
)

_H = constants.h.value
_K_B = constants.k_B.value
_OBS_FREQ_HZ = (551 * units.nm).to(units.Hz, equivalencies=units.spectral()).value
_EV_BETA = 0.15 * (1 + 0.32) * (15 + 0.51) / (3 - 0.51)


def extract_value(value):
    if hasattr(value, "jd"):
        return value.jd
    if hasattr(value, "value"):
        return value.value
    return value


def as_light_curve_parameter_array(params, parameter_order=PARAMETER_ORDER):
    """Convert mappings or row-like inputs to a ``(batch, n_parameters)`` array.

    Parameters are expected in the same public units used by
    :class:`selflensing.models.SelfLensingSystem`: ``a`` in AU, masses in solar
    masses, ``Rcomp`` in solar radii, ``Teff`` in K, and angles in radians.
    """
    if isinstance(params, Mapping):
        columns = [
            jnp.atleast_1d(jnp.asarray(extract_value(params[name])))
            for name in parameter_order
        ]
        return jnp.stack(jnp.broadcast_arrays(*columns), axis=-1)

    if (
        isinstance(params, Sequence)
        and len(params) > 0
        and isinstance(params[0], Mapping)
    ):
        columns = [
            jnp.asarray([extract_value(row[name]) for row in params])
            for name in parameter_order
        ]
        return jnp.stack(columns, axis=-1)

    array = jnp.asarray(params)
    if array.ndim == 1:
        array = array[jnp.newaxis, :]
    if array.shape[-1] != len(parameter_order):
        raise ValueError(
            f"Expected {len(parameter_order)} parameters in order {parameter_order}, "
            f"got shape {array.shape}."
        )
    return array


def physical_parameters(params):
    a, i, macc, mcomp, rcomp, teff, e, periapsis_phase = params
    a_m = a * AU_M
    macc_kg = macc * M_SUN_KG
    mcomp_kg = mcomp * M_SUN_KG
    total_mass_kg = macc_kg + mcomp_kg
    rcomp_m = rcomp * R_SUN_M
    sin_i = jnp.sin(i)
    return (
        a_m,
        i,
        macc_kg,
        mcomp_kg,
        total_mass_kg,
        rcomp_m,
        teff,
        e,
        periapsis_phase,
        sin_i,
    )


def light_curve_phase(times_days, params, peak_time_days=0.0):
    """Convert times to orbital phase for one parameter row."""
    (
        a_m,
        _,
        _,
        _,
        total_mass_kg,
        _,
        _,
        e,
        periapsis_phase,
        _,
    ) = physical_parameters(params)
    porb_days = orbital_period_days(a_m, total_mass_kg)
    return orbital_phase(
        times_days - peak_time_days,
        porb_days,
        e,
        periapsis_phase,
    )


def self_lensing_amplification(times_days, params, peak_time_days=0.0):
    """Compute JAX-native self-lensing amplification for one parameter row."""
    (
        a_m,
        i,
        macc_kg,
        _,
        _,
        rcomp_m,
        _,
        e,
        periapsis_phase,
        sin_i,
    ) = physical_parameters(params)
    phase = light_curve_phase(times_days, params, peak_time_days)
    r_e_m = einstein_radius_m(a_m, sin_i, macc_kg, e, periapsis_phase)
    b_source = impact_parameter_source_radii(a_m, i, rcomp_m)
    offsets = source_offset_einstein_radii(
        phase,
        a_m,
        rcomp_m,
        r_e_m,
        e,
        periapsis_phase,
        b_source,
    )
    amplification = integrated_witt_mao_magnification_jax(
        offsets,
        r_e_m,
        rcomp_m,
    )
    valid = (
        jnp.isfinite(r_e_m)
        & (r_e_m > 0)
        & jnp.isfinite(rcomp_m)
        & (rcomp_m > 0)
        & jnp.all(jnp.isfinite(offsets))
    )
    return jnp.where(valid, amplification, jnp.ones_like(times_days))


def ellipsoidal_variations_amplification(times_days, params, peak_time_days=0.0):
    """Compute ellipsoidal-variation fractional amplification for one row."""
    (
        a_m,
        _,
        macc_kg,
        mcomp_kg,
        _,
        rcomp_m,
        _,
        e,
        periapsis_phase,
        sin_i,
    ) = physical_parameters(params)
    phase = light_curve_phase(times_days, params, peak_time_days)
    r_sep_m = orbital_separation_m(phase, a_m, e, periapsis_phase)
    return (
        -_EV_BETA
        * (macc_kg / mcomp_kg)
        * (rcomp_m / r_sep_m) ** 3
        * sin_i**2
        * jnp.cos(2 * phase)
    )


def radial_velocity_m_per_s(times_days, params, peak_time_days=0.0):
    """Compute source radial velocity for one parameter row."""
    (
        a_m,
        _,
        macc_kg,
        _,
        total_mass_kg,
        _,
        _,
        e,
        periapsis_phase,
        sin_i,
    ) = physical_parameters(params)
    phase = light_curve_phase(times_days, params, peak_time_days)
    true_anomaly = phase - periapsis_phase
    mu = G * total_mass_kg
    angular_momentum = jnp.sqrt(mu * a_m * (1 - e**2))
    velocity_scale = (macc_kg / total_mass_kg) * (mu / angular_momentum)
    radial_velocity = velocity_scale * e * jnp.sin(true_anomaly)
    tangential_velocity = velocity_scale * (1 + e * jnp.cos(true_anomaly))
    return (
        tangential_velocity * jnp.sin(phase)
        - radial_velocity * jnp.cos(phase)
    ) * sin_i


def doppler_boosting_alpha(params):
    """Compute the average spectral index used for Doppler boosting."""
    teff = physical_parameters(params)[6]
    x = _H * _OBS_FREQ_HZ / (_K_B * teff)
    e_x = jnp.exp(x)
    return (e_x * (3 - x) - 3) / (e_x - 1)


def doppler_boosting_amplification(times_days, params, peak_time_days=0.0):
    """Compute Doppler-boosting fractional amplification for one row."""
    rv = radial_velocity_m_per_s(times_days, params, peak_time_days)
    return (3 - doppler_boosting_alpha(params)) * (rv / C)


def light_curve_amplification(
    times_days,
    params,
    peak_time_days=0.0,
    *,
    sl=True,
    ev=True,
    db=True,
):
    """Compute combined light-curve amplification for one parameter row."""
    sl_amplification = (
        self_lensing_amplification(times_days, params, peak_time_days)
        if sl
        else jnp.ones_like(times_days)
    )
    ev_amplification = (
        ellipsoidal_variations_amplification(times_days, params, peak_time_days)
        if ev
        else 0.0
    )
    db_amplification = (
        doppler_boosting_amplification(times_days, params, peak_time_days)
        if db
        else 0.0
    )
    return sl_amplification * (1 + ev_amplification + db_amplification)


def light_curve_flux(
    times_days,
    params,
    peak_time_days=0.0,
    baseline=1.0,
    *,
    sl=True,
    ev=True,
    db=True,
):
    """Compute deterministic light-curve flux for one parameter row."""
    return baseline * light_curve_amplification(
        times_days,
        params,
        peak_time_days,
        sl=sl,
        ev=ev,
        db=db,
    )
