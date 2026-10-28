"""Memory-bounded differentiable photometric likelihoods.

The stable likelihood builds object models and relies on reverse-mode AD
through a nested fit/walker ``vmap``. For long light curves that can retain a
prohibitive time-by-time reverse tape. This experimental implementation uses
the existing pure light-curve physics but supplies a custom VJP for the scalar
photometric likelihood. Its backward pass evaluates the five parameter
directions sequentially with JVPs, so no full time-domain Jacobian is formed.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from selflensing.lightcurves import light_curve_amplification
from selflensing.models import DEFAULT_LIMITS


class PhotometryRow(NamedTuple):
    """Unitless static inputs for one photometric system.

    ``known`` is ``(Mcomp, Rcomp, Teff)`` in the public self-lensing units.
    Fitted parameters are ``(a, i, Macc, e, periapsis_phase)``.
    """

    times_days: jax.Array
    flux: jax.Array
    flux_err: jax.Array
    mask: jax.Array
    known: jax.Array
    peak_time_days: jax.Array


def _full_parameters(params, known):
    a, inclination, lens_mass, eccentricity, periapsis_phase = params
    companion_mass, companion_radius, temperature = known
    return jnp.asarray(
        (
            a,
            inclination,
            lens_mass,
            companion_mass,
            companion_radius,
            temperature,
            eccentricity,
            periapsis_phase,
        )
    )


def photometric_model(params, row):
    """Return the normalized composite model flux for one system."""
    return light_curve_amplification(
        row.times_days,
        _full_parameters(params, row.known),
        row.peak_time_days,
    )


def _raw_photometric_logprob(params, row):
    model = photometric_model(params, row) - 1.0
    residual = row.flux - model
    weights = row.mask.astype(model.dtype) / row.flux_err**2
    a, inclination, lens_mass, eccentricity, periapsis_phase = params
    bounds_ok = (
        (DEFAULT_LIMITS["a"][0] <= a)
        & (a <= DEFAULT_LIMITS["a"][1])
        & (DEFAULT_LIMITS["i"][0] <= inclination)
        & (inclination <= DEFAULT_LIMITS["i"][1])
        & (DEFAULT_LIMITS["Macc"][0] <= lens_mass)
        & (lens_mass <= DEFAULT_LIMITS["Macc"][1])
        & (DEFAULT_LIMITS["e"][0] <= eccentricity)
        & (eccentricity < DEFAULT_LIMITS["e"][1])
        & (DEFAULT_LIMITS["periapsis_phase"][0] <= periapsis_phase)
        & (periapsis_phase <= DEFAULT_LIMITS["periapsis_phase"][1])
    )
    finite = bounds_ok & jnp.all(jnp.isfinite(model)) & jnp.all(jnp.isfinite(residual))
    value = -0.5 * jnp.sum(weights * residual**2)
    return jnp.where(finite, value, -jnp.inf)


@jax.custom_vjp
def photometric_logprob(params, row):
    """Scalar photometric log likelihood with a five-direction custom VJP."""
    return _raw_photometric_logprob(params, row)


def _photometric_logprob_fwd(params, row):
    return _raw_photometric_logprob(params, row), (params, row)


def _photometric_logprob_bwd(residual, cotangent):
    params, row = residual
    directions = jnp.eye(params.shape[0], dtype=params.dtype)

    def directional_derivative(direction):
        return jax.jvp(
            lambda value: _raw_photometric_logprob(value, row),
            (params,),
            (direction,),
        )[1]

    # lax.map lowers to a scan, keeping only one directional tangent live.
    gradient = jax.lax.map(directional_derivative, directions)
    return cotangent * gradient, None


photometric_logprob.defvjp(_photometric_logprob_fwd, _photometric_logprob_bwd)


def batched_photometric_logprob_and_grad(params, rows):
    """Evaluate independent ``(fit, walker)`` likelihoods and gradients.

    The inner custom VJP computes gradients locally for each five-parameter
    walker, before the outer fit/walker vectorisation is applied.
    """
    value_and_grad = jax.value_and_grad(photometric_logprob)

    def one_row(row_params, row):
        values, gradients = jax.vmap(value_and_grad, in_axes=(0, None))(row_params, row)
        return values, gradients

    return jax.vmap(one_row, in_axes=(0, 0))(params, rows)
