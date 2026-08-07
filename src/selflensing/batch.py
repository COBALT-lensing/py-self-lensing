"""JAX-native batched light-curve generation."""

import jax
from jax import numpy as jnp

from .lightcurves import (
    PARAMETER_ORDER,
    as_light_curve_parameter_array,
    extract_value,
    light_curve_flux,
)
from .orbits import DAY_S

jax.config.update("jax_enable_x64", True)


def batched_light_curve(
    times,
    params,
    *,
    peak_times=0.0,
    baseline=1.0,
    sl=True,
    ev=True,
    db=True,
    parameter_order=PARAMETER_ORDER,
):
    """Generate a batch of deterministic light curves.

    ``times`` may be a shared ``(n_times,)`` array or a per-row
    ``(batch, n_times)`` array. ``peak_times`` and ``baseline`` may be scalars or
    one value per batch row. The result has shape ``(batch, n_times)`` and stays
    on the active JAX device.
    """
    param_array = as_light_curve_parameter_array(params, parameter_order)
    times_array = jnp.asarray(extract_value(times))
    peak_times_array = jnp.asarray(extract_value(peak_times))
    baseline_array = jnp.asarray(extract_value(baseline))

    batch_size = param_array.shape[0]
    if times_array.ndim == 1:
        times_array = jnp.broadcast_to(times_array, (batch_size, times_array.shape[0]))
    elif times_array.shape[0] != batch_size:
        raise ValueError(
            f"times has leading dimension {times_array.shape[0]}, "
            f"but params has batch size {batch_size}."
        )

    peak_times_array = jnp.broadcast_to(
        jnp.atleast_1d(peak_times_array),
        (batch_size,),
    )
    baseline_array = jnp.broadcast_to(jnp.atleast_1d(baseline_array), (batch_size,))

    return jax.vmap(
        lambda row_params, row_times, row_peak, row_baseline: light_curve_flux(
            row_times,
            row_params,
            row_peak,
            row_baseline,
            sl=sl,
            ev=ev,
            db=db,
        )
    )(param_array, times_array, peak_times_array, baseline_array)


batched_light_curve_jit = jax.jit(
    batched_light_curve,
    static_argnames=("sl", "ev", "db", "parameter_order"),
)


def make_batched_light_curve_generator(
    times,
    *,
    peak_times=0.0,
    baseline=1.0,
    sl=True,
    ev=True,
    db=True,
    parameter_order=PARAMETER_ORDER,
    device=None,
):
    """Return a compiled callable for repeatedly generating deterministic batches."""
    times = jax.device_put(jnp.asarray(extract_value(times)), device)
    peak_times = jax.device_put(jnp.asarray(extract_value(peak_times)), device)
    baseline = jax.device_put(jnp.asarray(extract_value(baseline)), device)

    @jax.jit
    def _generate(param_array):
        return batched_light_curve(
            times,
            param_array,
            peak_times=peak_times,
            baseline=baseline,
            sl=sl,
            ev=ev,
            db=db,
            parameter_order=parameter_order,
        )

    def generate(params):
        params = jax.device_put(
            as_light_curve_parameter_array(params, parameter_order),
            device,
        )
        return _generate(params)

    return generate


def _count_kernel_hyperparameters(kernel_type):
    count = 0
    for component in kernel_type:
        if component == "G":
            count += 2
        elif component in ("S", "E"):
            count += 3
        elif component == "F":
            count += 2
        else:
            raise ValueError(
                f"Unknown kernel component '{component}' in kernel_type={kernel_type!r}."
            )
    return count


def _as_batched_rows(values, batch_size, name):
    values = jnp.asarray(extract_value(values))
    if values.ndim == 0:
        return jnp.broadcast_to(values[jnp.newaxis], (batch_size,))
    if values.shape[0] != batch_size:
        raise ValueError(
            f"{name} has leading dimension {values.shape[0]}, "
            f"but mean_flux has batch size {batch_size}."
        )
    return values


def batched_gp_light_curve(
    key,
    times,
    mean_flux,
    gp_params,
    *,
    kernel_type="GS",
    noise_measure=0.0,
    times_unit="day",
):
    """Sample GP-noisy light curves around batched deterministic means.

    ``gp_params`` must have shape ``(batch, nhyp)`` using the same hyperparameter
    packing as ``selflensing.gp.LensingGP.make_kernel_fn``:
    ``G=[log10(nu_Hz), log10(sigma)]``,
    ``S/E=[log10(nu_Hz), Q, log10(sigma)]``, and
    ``F=[log10(tau_s), log10(sigma)]``.

    ``times`` may be shared ``(n_times,)`` or per-row ``(batch, n_times)``. Times
    are converted to seconds by default because the GP kernels use Hz.
    """
    mean_flux = jnp.asarray(extract_value(mean_flux))
    if mean_flux.ndim != 2:
        raise ValueError(
            f"mean_flux must have shape (batch, n_times), got {mean_flux.shape}."
        )

    batch_size = mean_flux.shape[0]
    gp_params = jnp.asarray(extract_value(gp_params))
    if gp_params.ndim == 1:
        gp_params = jnp.broadcast_to(
            gp_params[jnp.newaxis, :],
            (batch_size, gp_params.shape[0]),
        )
    elif gp_params.shape[0] != batch_size:
        raise ValueError(
            f"gp_params has leading dimension {gp_params.shape[0]}, "
            f"but mean_flux has batch size {batch_size}."
        )

    nhyp = _count_kernel_hyperparameters(kernel_type)
    if gp_params.shape[-1] != nhyp:
        raise ValueError(
            f"kernel_type={kernel_type!r} expects {nhyp} GP hyperparameters, "
            f"got shape {gp_params.shape}."
        )

    times = jnp.asarray(extract_value(times))
    if times.ndim == 1:
        times = jnp.broadcast_to(times, mean_flux.shape)
    elif times.shape != mean_flux.shape:
        raise ValueError(
            f"times must have shape {mean_flux.shape} or ({mean_flux.shape[1]},), "
            f"got {times.shape}."
        )

    if times_unit == "day":
        times = times * DAY_S
    elif times_unit not in ("s", "second"):
        raise ValueError("times_unit must be 'day', 's', or 'second'.")

    noise_measure = _as_batched_rows(noise_measure, batch_size, "noise_measure")
    keys = jax.random.split(key, batch_size)
    from .gp.LensingGP import build_model_additive, make_kernel_fn

    kernel_fn = make_kernel_fn(kernel_type)

    def sample_one(row_key, row_times, row_mean, row_params, row_noise):
        theta = jnp.concatenate(
            [row_params, jnp.array([0.0], dtype=row_params.dtype)]
        )
        gp = build_model_additive(
            theta=theta,
            nhyp=nhyp,
            kernel_fn=kernel_fn,
            T=row_times,
            include_lens_mean=0,
            noise_measure=row_noise,
            mean_vals=row_mean,
        )
        return gp.sample(row_key)

    return jax.vmap(sample_one)(keys, times, mean_flux, gp_params, noise_measure)


batched_gp_light_curve_jit = jax.jit(
    batched_gp_light_curve,
    static_argnames=("kernel_type", "times_unit"),
)


def make_batched_gp_light_curve_generator(
    times,
    gp_params,
    *,
    kernel_type="GS",
    noise_measure=0.0,
    times_unit="day",
    device=None,
):
    """Return a compiled callable for GP sampling around supplied mean fluxes."""
    times = jax.device_put(jnp.asarray(extract_value(times)), device)
    gp_params = jax.device_put(jnp.asarray(extract_value(gp_params)), device)
    noise_measure = jax.device_put(jnp.asarray(extract_value(noise_measure)), device)

    @jax.jit
    def _generate(key, mean_flux):
        return batched_gp_light_curve(
            key,
            times,
            mean_flux,
            gp_params,
            kernel_type=kernel_type,
            noise_measure=noise_measure,
            times_unit=times_unit,
        )

    def generate(key, mean_flux):
        mean_flux = jax.device_put(
            jnp.asarray(extract_value(mean_flux)),
            device,
        )
        return _generate(key, mean_flux)

    return generate


def batched_light_curve_with_gp_noise(
    key,
    times,
    params,
    gp_params,
    *,
    kernel_type="GS",
    noise_measure=0.0,
    peak_times=0.0,
    baseline=1.0,
    sl=True,
    ev=True,
    db=True,
    parameter_order=PARAMETER_ORDER,
    times_unit="day",
):
    """Generate deterministic batched light curves and add GP-based noise."""
    mean_flux = batched_light_curve(
        times,
        params,
        peak_times=peak_times,
        baseline=baseline,
        sl=sl,
        ev=ev,
        db=db,
        parameter_order=parameter_order,
    )
    return batched_gp_light_curve(
        key,
        times,
        mean_flux,
        gp_params,
        kernel_type=kernel_type,
        noise_measure=noise_measure,
        times_unit=times_unit,
    )
