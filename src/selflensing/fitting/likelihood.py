"""Likelihood builders for self-lensing fitting."""

import jax
import jax.numpy as jax_numpy
import numpy
import unxt
from astropy import units
from quax import quaxify
from quaxed import numpy as jnp

from ..generators import RVGenerator, LightcurveGenerator, GPLightcurveGenerator
from ..gp.PriorsSamplers import log_prior_bounds
from ..models import SelfLensingSystem, DEFAULT_LIMITS
from ._utils import _device_array, _extract_value, gauss_lnprior
from .numeric import deterministic_models, parameter_row, physical_properties


@quaxify
def combined_chi2(
    params,
    Mcomp,
    Rcomp,
    tau_eff,
    d,
    peak_time,
    rv_times,
    observed_rv,
    observed_rv_err,
    lc_times,
    observed_lc,
    observed_lc_err,
    gp_model,
    detected_periods,
    lensing_method,
    center_lc_model=False,
    # weight_lc_peak=False,
    period_min=None,
    period_max=None,
    model_sls=None,
    rv_mask=None,
    lc_mask=None,
    peak_time_prior_sigma=None,
    rv_source_ids=None,
):
    """Compute log-likelihood for self-lensing binary parameters.

    Non-lensing orientations are allowed: self-lensing feasibility is not a
    prior. An explicit tau_eff prior still requires a finite lensing duration.

    The @quaxify decorator allows astropy Quantity and Time objects to work
    with JAX tracers. However, astropy operations execute at Python level
    (on CPU), not in compiled JAX code. Inputs should be unitless arrays when
    called from JIT-traced code for best performance (see make_combined_chi2_vec).
    """
    # if a < DEFAULT_LIMITS["a"][0] or a > DEFAULT_LIMITS["a"][1]:
    #     return -jnp.inf

    # if i < DEFAULT_LIMITS["i"][0] or i > DEFAULT_LIMITS["i"][1]:
    #     return -jnp.inf

    # if Macc < DEFAULT_LIMITS["Macc"][0] or Macc > DEFAULT_LIMITS["Macc"][1]:
    #     return -jnp.inf

    # if (
    #     periapsis_phase < DEFAULT_LIMITS["periapsis_phase"][0]
    #     or periapsis_phase > DEFAULT_LIMITS["periapsis_phase"][1]
    # ):
    #     return -jnp.inf

    # # e = 1 is parabolic, so less than but not equal
    # if e < DEFAULT_LIMITS["e"][0] or e >= DEFAULT_LIMITS["e"][1]:
    #     return -jnp.inf

    Mcomp = unxt.Quantity(Mcomp, SelfLensingSystem.param_units["Mcomp"])
    if Rcomp is not None:
        Rcomp = unxt.Quantity(Rcomp, SelfLensingSystem.param_units["Rcomp"])
    if d is not None:
        d = unxt.Quantity(d, SelfLensingSystem.param_units["d"])
    if observed_rv is not None:
        observed_rv = unxt.Quantity(observed_rv, units.m / units.s)
    if observed_rv_err is not None:
        observed_rv_err = unxt.Quantity(observed_rv_err, units.m / units.s)
    model_params = params[:5]
    peak_time_offset = params[5] if peak_time_prior_sigma is not None else 0.0
    effective_peak_time = peak_time + peak_time_offset
    lp_bounds = log_prior_bounds(
        model_params,
        (
            DEFAULT_LIMITS["a"],
            DEFAULT_LIMITS["i"],
            DEFAULT_LIMITS["Macc"],
            DEFAULT_LIMITS["e"],
            DEFAULT_LIMITS["periapsis_phase"],
        ),
    )

    # ---- 2) Define the "True" branch for jax.lax.cond ----
    # This function is *only* called if lp_bounds is finite
    @quaxify
    def compute_full_logprob(_):
        a, i, Macc, e, periapsis_phase = model_params

        a = unxt.Quantity(a, SelfLensingSystem.param_units["a"])
        Macc = unxt.Quantity(Macc, SelfLensingSystem.param_units["Macc"])
        e = unxt.Quantity(e, SelfLensingSystem.param_units["e"])
        periapsis_phase = unxt.Quantity(
            periapsis_phase, SelfLensingSystem.param_units["periapsis_phase"]
        )
        i = unxt.Quantity(i, SelfLensingSystem.param_units["i"])

        # if model_sls is None:
        new_model_sls = SelfLensingSystem(
            a=a,
            i=i,
            Macc=Macc,
            Mcomp=Mcomp,
            Rcomp=Rcomp,
            e=e,
            periapsis_phase=periapsis_phase,
            d=d,
        )
        # else:
        #     new_model_sls = copy.deepcopy(model_sls)
        #     new_model_sls.a = a
        #     new_model_sls.Macc = Macc
        #     new_model_sls.Mcomp = Mcomp
        #     # Must set i after setting the masses to correctly pre-calc b
        #     new_model_sls.i = i
        #     new_model_sls.Rcomp = Rcomp
        #     new_model_sls.e = e
        #     new_model_sls.periapsis_phase = periapsis_phase
        #     new_model_sls.d = d

        def check_sls_props():
            def condition_value(condition):
                return jnp.asarray(
                    condition.value if hasattr(condition, "value") else condition
                )

            # A binary need not transit/lens to explain photometry or RVs.
            period = condition_value(new_model_sls.porb.to(units.day).value)
            props_ok = jnp.isfinite(period) & (period > 0)
            if period_min is not None:
                props_ok = jnp.logical_and(
                    props_ok,
                    condition_value(new_model_sls.porb.to(units.day).value)
                    >= period_min,
                )
            if period_max is not None:
                props_ok = jnp.logical_and(
                    props_ok,
                    condition_value(new_model_sls.porb.to(units.day).value)
                    <= period_max,
                )
            if tau_eff is not None:
                props_ok = jnp.logical_and(
                    props_ok,
                    jnp.logical_or(
                        jnp.isnan(tau_eff),
                        jnp.isfinite(new_model_sls.tau_eff.value),
                    ),
                )
            return props_ok

        def zero(*args, **kwargs):
            return 0.0

        def tau_eff_prior(model_tau_eff, tau_eff):
            model_tau_eff = model_tau_eff.to(units.day).value
            # tau_eff = tau_eff.to(units.day).value
            return gauss_lnprior(model_tau_eff, tau_eff, tau_eff / 100)

        def calc_errs():
            finite_detected_periods = jnp.where(
                jnp.isfinite(detected_periods),
                detected_periods,
                jnp.inf,
            )
            period_err = jax.lax.cond(
                jnp.any(jnp.isfinite(detected_periods)),
                lambda a, b: gauss_lnprior(
                    a,
                    b[jnp.argmin(jnp.abs(b - a))] if b.size > 0 else a,
                    0.1,
                ),
                zero,
                new_model_sls.porb.value,
                finite_detected_periods,
            )

            if tau_eff is None:
                tau_eff_err = zero()
            else:
                tau_eff_err = jax.lax.cond(
                    jnp.logical_not(jnp.isnan(tau_eff)),
                    tau_eff_prior,
                    zero,
                    new_model_sls.tau_eff,
                    tau_eff,
                )

            def calc_rv_chi2():
                rv_gen = RVGenerator(
                    new_model_sls, times=rv_times, peak_time=effective_peak_time
                )
                return profiled_rv_chi2(
                    observed_rv.to(units.m / units.s).value,
                    observed_rv_err.to(units.m / units.s).value,
                    rv_gen.rv.to(units.m / units.s).value,
                    rv_mask,
                    rv_source_ids,
                )

            def calc_lc_chi2():
                def _lc_chi2():
                    lc_gen = LightcurveGenerator(
                        new_model_sls,
                        times=lc_times,
                        peak_time=effective_peak_time,
                        lensing_method=lensing_method,
                    )
                    model_lc = lc_gen.flux_array
                    if hasattr(model_lc, "value"):
                        model_lc = model_lc.value
                    if center_lc_model:
                        model_lc = model_lc - 1.0
                    return chi2_masked(
                        observed_lc,
                        observed_lc_err,
                        model_lc,
                        lc_mask,
                    )

                def gp_chi2():
                    lc_gen = GPLightcurveGenerator(
                        new_model_sls,
                        times=lc_times,
                        peak_time=effective_peak_time,
                        lensing_method=lensing_method,
                        gp=gp_model(new_model_sls, lc_times.jd, observed_lc),
                    )
                    # new_gp.sls = new_model_sls
                    # new_gp.mean_vals = lc_gen.flux_array
                    lp = lc_gen.gp.logprob_single()
                    return jax.lax.cond(
                        jnp.isfinite(lp),
                        lambda: lp,
                        _lc_chi2,
                    )

                return gp_chi2() if gp_model is not None else _lc_chi2()

                # if gp is not None:
                #     gp.sls = model_sls
                #     gp.mean_vals = lc_gen.flux_array.value
                #     lc_chi2 = gp.logprob_single()
                #     # print("lc_chi2=", lc_chi2)
                # else:
                #     lc_err = observed_lc_err.value
                #     if weight_lc_peak:
                #         # Reduce the weight of points further from the peak
                #         lc_err *= 1 + (peak_time - lc_times).value ** 2
                #     lc_chi2 = chi2(
                #         observed_lc.value,
                #         observed_lc_err.value,
                #         lc_gen.flux_array.value,
                #     )

            rv_chi2 = calc_rv_chi2() if observed_rv is not None else zero()
            lc_chi2 = calc_lc_chi2() if observed_lc is not None else zero()

            # jax.debug.print(
            #     "rv={rv}, lc={lc}, period={period}, tau_eff={tau_eff}",
            #     rv=rv_chi2,
            #     lc=lc_chi2,
            #     period=period_err,
            #     tau_eff=tau_eff_err,
            # )

            peak_time_err = (
                0.0
                if peak_time_prior_sigma is None
                else gauss_lnprior(peak_time_offset, 0.0, peak_time_prior_sigma)
            )
            return rv_chi2 + lc_chi2 + period_err + tau_eff_err + peak_time_err

        return jax.lax.cond(
            check_sls_props(),
            calc_errs,
            return_neg_inf,
        )

    # ---- 3) Define the "False" branch for jax.lax.cond ----
    def return_neg_inf(*args, **kwargs):
        return -jnp.inf

    # ---- 4) JAX-native conditional branch ----
    # This correctly branches based on the *traced* value of jnp.isfinite
    return jax.lax.cond(
        jnp.isfinite(lp_bounds),
        compute_full_logprob,
        return_neg_inf,
        operand=None,
    )


def chi2(obs, obs_err, model):
    return -jnp.sum((obs - model) ** 2 / obs_err**2)


def chi2_masked(obs, obs_err, model, mask=None):
    if mask is None:
        return chi2(obs, obs_err, model)
    mask = jnp.asarray(mask, dtype=bool)
    weight = mask.astype(float)
    return -jnp.sum(weight * (obs - model) ** 2 / obs_err**2)


def _prepare_rv_source_ids(source_ids, observed_rv):
    """Validate per-observation labels and encode dense IDs per fit before JIT."""
    if source_ids is None:
        return None
    labels = numpy.asarray(source_ids)
    if observed_rv is None or labels.shape != numpy.shape(observed_rv):
        raise ValueError("rv_source_ids must have the same shape as observed_rv")
    if labels.ndim not in (1, 2):
        raise ValueError("rv_source_ids must be one- or two-dimensional")
    if labels.ndim == 1:
        return numpy.unique(labels, return_inverse=True)[1].astype(numpy.int32)
    return numpy.asarray([
        numpy.unique(row, return_inverse=True)[1] for row in labels
    ], dtype=numpy.int32)


def profiled_rv_chi2(obs, obs_err, orbital_model, mask=None, source_ids=None):
    """Negative RV chi squared after profiling one constant offset per source.

    ``source_ids`` are dense nonnegative integers smaller than the observation
    count, as produced by the likelihood builders. None retains one shared
    offset. Masked points contribute neither to offsets nor to chi squared.
    A source with one unmasked observation contributes zero chi squared.
    """
    active = jnp.ones_like(obs, dtype=bool) if mask is None else jnp.asarray(mask, dtype=bool)
    error = jnp.where(active, obs_err, 1.0)
    weight = jnp.where(active, 1.0 / error**2, 0.0)
    residual = jnp.where(active, obs - orbital_model, 0.0)
    if source_ids is None:
        weight_sum = jnp.sum(weight)
        systemic_velocity = jnp.sum(weight * residual) / jnp.where(weight_sum > 0, weight_sum, 1.0)
    else:
        ids = jax_numpy.asarray(source_ids, dtype=jax_numpy.int32)
        weight_sum = jax.ops.segment_sum(weight, ids, num_segments=obs.shape[0])
        residual_sum = jax.ops.segment_sum(weight * residual, ids, num_segments=obs.shape[0])
        offsets = residual_sum / jnp.where(weight_sum > 0, weight_sum, 1.0)
        systemic_velocity = offsets[ids]
    centered = jnp.where(active, residual - systemic_velocity, 0.0)
    return -jnp.sum(weight * centered**2)


def numeric_combined_chi2(
    params,
    Mcomp,
    Rcomp,
    tau_eff,
    peak_time,
    rv_times,
    observed_rv,
    observed_rv_err,
    lc_times,
    observed_lc,
    observed_lc_err,
    detected_periods,
    center_lc_model=False,
    period_min=None,
    period_max=None,
    rv_mask=None,
    lc_mask=None,
    rv_source_ids=None,
):
    """Object-free deterministic equivalent of :func:`combined_chi2`.

    All values use the public units accepted by ``combined_chi2``.  It is kept
    deliberately separate from that public, quantity-aware function so only
    the batched BlackJAX sampler uses this fully numeric path.
    """
    bounds = log_prior_bounds(
        params,
        (
            DEFAULT_LIMITS["a"],
            DEFAULT_LIMITS["i"],
            DEFAULT_LIMITS["Macc"],
            DEFAULT_LIMITS["e"],
            DEFAULT_LIMITS["periapsis_phase"],
        ),
    )
    model_params = parameter_row(params, Mcomp, Rcomp)
    _sl_possible, period, model_tau_eff = physical_properties(model_params)
    # Match combined_chi2: lensing feasibility is diagnostic, not a hard prior.
    valid = jnp.isfinite(period) & (period > 0)
    if period_min is not None:
        valid = valid & (period >= period_min)
    if period_max is not None:
        valid = valid & (period <= period_max)
    if tau_eff is not None:
        valid = valid & (jnp.isnan(tau_eff) | jnp.isfinite(model_tau_eff))

    model_lc, model_rv = deterministic_models(lc_times, rv_times, model_params, peak_time)
    result = bounds
    if observed_rv is not None:
        result = result + profiled_rv_chi2(
            observed_rv, observed_rv_err, model_rv, rv_mask, rv_source_ids
        )
    if observed_lc is not None:
        if center_lc_model:
            model_lc = model_lc - 1.0
        result = result + chi2_masked(observed_lc, observed_lc_err, model_lc, lc_mask)

    if detected_periods.size > 0:
        finite_periods = jnp.where(
            jnp.isfinite(detected_periods), detected_periods, jnp.inf
        )
        result = result + jax.lax.cond(
            jnp.any(jnp.isfinite(detected_periods)),
            lambda: gauss_lnprior(
                period,
                finite_periods[jnp.argmin(jnp.abs(finite_periods - period))],
                0.1,
            ),
            lambda: 0.0,
        )
    if tau_eff is not None:
        result = result + jax.lax.cond(
            jnp.logical_not(jnp.isnan(tau_eff)),
            lambda: gauss_lnprior(model_tau_eff, tau_eff, tau_eff / 100),
            lambda: 0.0,
        )
    return jnp.where(jnp.isfinite(bounds) & valid, result, -jnp.inf)



def make_combined_chi2_vec(
    Mcomp,
    Rcomp,
    tau_eff,
    d,
    peak_time,
    rv_times,
    observed_rv,
    observed_rv_err,
    lc_times,
    observed_lc,
    observed_lc_err,
    gp,
    detected_periods,
    lensing_method,
    center_lc_model=False,
    weight_lc_peak=False,
    period_min=None,
    period_max=None,
    model_sls=None,
    device=None,
    rv_source_ids=None,
):
    # Strip astropy units at Python level (before JIT tracing) to enable
    # better compiler optimization and reduce tracer overhead.
    # This converts astropy Time objects to float (JD) and Quantity objects to values.
    peak_time = _extract_value(peak_time)
    rv_times = _extract_value(rv_times)
    lc_times = _extract_value(lc_times)
    Mcomp = _extract_value(Mcomp)
    Rcomp = _extract_value(Rcomp)
    tau_eff = _extract_value(tau_eff)
    d = _extract_value(d)
    observed_rv = _extract_value(observed_rv)
    rv_source_ids = _prepare_rv_source_ids(rv_source_ids, observed_rv)
    rv_source_ids = _device_array(rv_source_ids, device)
    observed_rv_err = _extract_value(observed_rv_err)
    observed_lc = _extract_value(observed_lc)
    observed_lc_err = _extract_value(observed_lc_err)
    detected_periods = _extract_value(detected_periods)

    Mcomp = _device_array(Mcomp, device)
    Rcomp = _device_array(Rcomp, device)
    tau_eff = _device_array(tau_eff, device)
    d = _device_array(d, device)
    observed_rv = _device_array(observed_rv, device)
    observed_rv_err = _device_array(observed_rv_err, device)
    observed_lc = _device_array(observed_lc, device)
    observed_lc_err = _device_array(observed_lc_err, device)
    detected_periods = _device_array(detected_periods, device)

    @quaxify
    def _combined_chi2_vec(params):
        params = jax.device_put(jax_numpy.asarray(params), device)
        return jnp.asarray(
            jax.vmap(
                lambda p: combined_chi2(
                    p,
                    Mcomp,
                    Rcomp,
                    tau_eff,
                    d,
                    peak_time,
                    rv_times,
                    observed_rv,
                    observed_rv_err,
                    lc_times,
                    observed_lc,
                    observed_lc_err,
                    gp,
                    detected_periods,
                    lensing_method,
                    center_lc_model,
                    # weight_lc_peak,
                    period_min,
                    period_max,
                    model_sls,
                    rv_source_ids=rv_source_ids,
                )
            )(params)
        )

    # === JAX 0.10+ Compatibility and GPU Support ===
    # JAX 0.10+ removed the device= parameter from jax.jit(). Instead, device
    # placement is controlled via jax.device_put() and context managers.
    #
    # For GPU execution, we wrap the JIT with explicit device placement:
    # 1. All input data is pre-placed on GPU via _device_array()
    # 2. The wrapper ensures inputs to the JIT function come from the target device
    # 3. Results remain on-device until explicitly transferred back to host
    #
    # This approach ensures compatibility with JAX 0.10+ while maintaining
    # GPU acceleration when a GPU device is specified.
    if device is not None and hasattr(device, "platform") and device.platform == "gpu":
        # Use jax.jit with default device context
        combined_chi2_vec_jitted = jax.jit(_combined_chi2_vec)

        # Wrap to ensure GPU placement of entire computation
        def combined_chi2_vec(x):
            # Put inputs on GPU device, compute, result stays on GPU
            x_gpu = jax.device_put(x, device)
            return combined_chi2_vec_jitted(x_gpu)

    else:
        combined_chi2_vec = jax.jit(_combined_chi2_vec)

    def logprob(params):
        # emcee feeds/consumes NumPy; keep the vectorized likelihood on the
        # selected JAX device, then transfer just the log-probabilities back.
        result = combined_chi2_vec(params)
        # Ensure result is back on host for emcee (which only works with NumPy)
        return numpy.asarray(result)

    return logprob


def make_combined_chi2_vec_jax(
    Mcomp,
    Rcomp,
    tau_eff,
    d,
    peak_time,
    rv_times,
    observed_rv,
    observed_rv_err,
    lc_times,
    observed_lc,
    observed_lc_err,
    gp,
    detected_periods,
    lensing_method,
    center_lc_model=False,
    weight_lc_peak=False,
    period_min=None,
    period_max=None,
    model_sls=None,
    device=None,
    peak_time_prior_sigma=None,
    rv_source_ids=None,
):
    """Build a JAX-native vectorized log-probability function.

    Unlike make_combined_chi2_vec(), this keeps the log-probabilities on the
    selected JAX device so device-native samplers do not synchronize with NumPy
    on every MCMC step.
    """
    peak_time = _extract_value(peak_time)
    rv_times = _extract_value(rv_times)
    lc_times = _extract_value(lc_times)
    Mcomp = _extract_value(Mcomp)
    Rcomp = _extract_value(Rcomp)
    tau_eff = _extract_value(tau_eff)
    d = _extract_value(d)
    observed_rv = _extract_value(observed_rv)
    rv_source_ids = _prepare_rv_source_ids(rv_source_ids, observed_rv)
    rv_source_ids = _device_array(rv_source_ids, device)
    observed_rv_err = _extract_value(observed_rv_err)
    observed_lc = _extract_value(observed_lc)
    observed_lc_err = _extract_value(observed_lc_err)
    detected_periods = _extract_value(detected_periods)
    period_min = _extract_value(period_min)
    period_max = _extract_value(period_max)

    Mcomp = _device_array(Mcomp, device)
    Rcomp = _device_array(Rcomp, device)
    tau_eff = _device_array(tau_eff, device)
    d = _device_array(d, device)
    observed_rv = _device_array(observed_rv, device)
    observed_rv_err = _device_array(observed_rv_err, device)
    observed_lc = _device_array(observed_lc, device)
    observed_lc_err = _device_array(observed_lc_err, device)
    detected_periods = _device_array(detected_periods, device)

    @quaxify
    def _combined_chi2_vec(params):
        params = jax.device_put(jax_numpy.asarray(params), device)
        return jnp.asarray(
            jax.vmap(
                lambda p: combined_chi2(
                    p,
                    Mcomp,
                    Rcomp,
                    tau_eff,
                    d,
                    peak_time,
                    rv_times,
                    observed_rv,
                    observed_rv_err,
                    lc_times,
                    observed_lc,
                    observed_lc_err,
                    gp,
                    detected_periods,
                    lensing_method,
                    center_lc_model,
                    period_min,
                    period_max,
                    model_sls,
                    rv_source_ids=rv_source_ids,
                    peak_time_prior_sigma=peak_time_prior_sigma,
                )
            )(params)
        )

    combined_chi2_vec = jax.jit(_combined_chi2_vec)

    def logprob(params):
        params = jax.device_put(params, device)
        return combined_chi2_vec(params)

    return logprob


def make_batched_combined_chi2_vec_jax(
    Mcomp,
    Rcomp,
    tau_eff,
    d,
    peak_time,
    rv_times,
    observed_rv,
    observed_rv_err,
    lc_times,
    observed_lc,
    observed_lc_err,
    gp,
    detected_periods,
    lensing_method,
    center_lc_model=False,
    weight_lc_peak=False,
    period_min=None,
    period_max=None,
    model_sls=None,
    rv_mask=None,
    lc_mask=None,
    device=None,
    rv_source_ids=None,
):
    """Build a JAX-native batched log-probability function.

    Parameters with observations must already have a leading fit dimension.
    Ragged observations should be padded to a common length and accompanied by
    boolean masks with the same leading dimensions.
    """
    peak_time = _extract_value(peak_time)
    rv_times = _extract_value(rv_times)
    lc_times = _extract_value(lc_times)
    Mcomp = _extract_value(Mcomp)
    Rcomp = _extract_value(Rcomp)
    tau_eff = _extract_value(tau_eff)
    d = _extract_value(d)
    observed_rv = _extract_value(observed_rv)
    rv_source_ids = _prepare_rv_source_ids(rv_source_ids, observed_rv)
    rv_source_ids = _device_array(rv_source_ids, device)
    observed_rv_err = _extract_value(observed_rv_err)
    observed_lc = _extract_value(observed_lc)
    observed_lc_err = _extract_value(observed_lc_err)
    detected_periods = _extract_value(detected_periods)
    rv_mask = _extract_value(rv_mask)
    lc_mask = _extract_value(lc_mask)
    has_tau_eff = tau_eff is not None

    peak_time = _device_array(peak_time, device)
    rv_times = _device_array(rv_times, device)
    lc_times = _device_array(lc_times, device)
    Mcomp = _device_array(Mcomp, device)
    Rcomp = _device_array(Rcomp, device)
    tau_eff = _device_array(tau_eff, device)
    d = _device_array(d, device)
    observed_rv = _device_array(observed_rv, device)
    observed_rv_err = _device_array(observed_rv_err, device)
    observed_lc = _device_array(observed_lc, device)
    observed_lc_err = _device_array(observed_lc_err, device)
    detected_periods = _device_array(detected_periods, device)
    rv_mask = _device_array(rv_mask, device)
    lc_mask = _device_array(lc_mask, device)

    if gp is not None:
        raise NotImplementedError("Batched BlackJAX fitting does not support gp_model")

    @quaxify
    def _row_chi2_vec(
        params,
        row_Mcomp,
        row_Rcomp,
        row_tau_eff,
        row_d,
        row_peak_time,
        row_rv_times,
        row_observed_rv,
        row_observed_rv_err,
        row_lc_times,
        row_observed_lc,
        row_observed_lc_err,
        row_detected_periods,
        row_rv_mask,
        row_lc_mask,
        row_rv_source_ids,
    ):
        return jnp.asarray(
            jax.vmap(
                lambda p: combined_chi2(
                    p,
                    row_Mcomp,
                    row_Rcomp,
                    row_tau_eff,
                    row_d,
                    row_peak_time,
                    row_rv_times,
                    row_observed_rv,
                    row_observed_rv_err,
                    row_lc_times,
                    row_observed_lc,
                    row_observed_lc_err,
                    None,
                    row_detected_periods,
                    lensing_method,
                    center_lc_model,
                    period_min,
                    period_max,
                    model_sls,
                    row_rv_mask,
                    row_lc_mask,
                    rv_source_ids=row_rv_source_ids,
                )
            )(params)
        )

    @quaxify
    def _batched_chi2_vec(params):
        params = jax.device_put(jax_numpy.asarray(params), device)
        return jnp.asarray(
            (
                jax.vmap(_row_chi2_vec)(
                    params,
                    Mcomp,
                    Rcomp,
                    tau_eff,
                    d,
                    peak_time,
                    rv_times,
                    observed_rv,
                    observed_rv_err,
                    lc_times,
                    observed_lc,
                    observed_lc_err,
                    detected_periods,
                    rv_mask,
                    lc_mask,
                    rv_source_ids,
                )
                if has_tau_eff
                else jax.vmap(
                    lambda row_params,
                    row_Mcomp,
                    row_Rcomp,
                    row_d,
                    row_peak_time,
                    row_rv_times,
                    row_observed_rv,
                    row_observed_rv_err,
                    row_lc_times,
                    row_observed_lc,
                    row_observed_lc_err,
                    row_detected_periods,
                    row_rv_mask,
                    row_lc_mask,
                    row_rv_source_ids: _row_chi2_vec(
                        row_params,
                        row_Mcomp,
                        row_Rcomp,
                        None,
                        row_d,
                        row_peak_time,
                        row_rv_times,
                        row_observed_rv,
                        row_observed_rv_err,
                        row_lc_times,
                        row_observed_lc,
                        row_observed_lc_err,
                        row_detected_periods,
                        row_rv_mask,
                        row_lc_mask,
                        row_rv_source_ids,
                    )
                )(
                    params,
                    Mcomp,
                    Rcomp,
                    d,
                    peak_time,
                    rv_times,
                    observed_rv,
                    observed_rv_err,
                    lc_times,
                    observed_lc,
                    observed_lc_err,
                    detected_periods,
                    rv_mask,
                    lc_mask,
                    rv_source_ids,
                )
            )
        )

    batched_chi2_vec = jax.jit(_batched_chi2_vec)

    def logprob(params):
        params = jax.device_put(params, device)
        return batched_chi2_vec(params)

    return logprob


def make_batched_numeric_combined_chi2_vec_jax(
    Mcomp,
    Rcomp,
    tau_eff,
    peak_time,
    rv_times,
    observed_rv,
    observed_rv_err,
    lc_times,
    observed_lc,
    observed_lc_err,
    detected_periods,
    center_lc_model=False,
    period_min=None,
    period_max=None,
    rv_mask=None,
    lc_mask=None,
    device=None,
    rv_source_ids=None,
):
    """Build the numeric likelihood used exclusively by batched BlackJAX fits."""
    rv_source_ids = _prepare_rv_source_ids(rv_source_ids, _extract_value(observed_rv))
    values = [
        Mcomp,
        Rcomp,
        tau_eff,
        peak_time,
        rv_times,
        observed_rv,
        observed_rv_err,
        lc_times,
        observed_lc,
        observed_lc_err,
        detected_periods,
        rv_mask,
        lc_mask,
        rv_source_ids,
    ]
    (
        Mcomp,
        Rcomp,
        tau_eff,
        peak_time,
        rv_times,
        observed_rv,
        observed_rv_err,
        lc_times,
        observed_lc,
        observed_lc_err,
        detected_periods,
        rv_mask,
        lc_mask,
        rv_source_ids,
    ) = [_device_array(_extract_value(value), device) for value in values]

    def row_logprob(
        row_params,
        row_Mcomp,
        row_Rcomp,
        row_tau_eff,
        row_peak_time,
        row_rv_times,
        row_observed_rv,
        row_observed_rv_err,
        row_lc_times,
        row_observed_lc,
        row_observed_lc_err,
        row_detected_periods,
        row_rv_mask,
        row_lc_mask,
        row_rv_source_ids,
    ):
        return jax.vmap(
            lambda walker: numeric_combined_chi2(
                walker,
                row_Mcomp,
                row_Rcomp,
                row_tau_eff,
                row_peak_time,
                row_rv_times,
                row_observed_rv,
                row_observed_rv_err,
                row_lc_times,
                row_observed_lc,
                row_observed_lc_err,
                row_detected_periods,
                center_lc_model,
                period_min,
                period_max,
                row_rv_mask,
                row_lc_mask,
                rv_source_ids=row_rv_source_ids,
            )
        )(row_params)

    def _batched_logprob(params):
        params = jax.device_put(jax_numpy.asarray(params), device)
        if tau_eff is None:
            return jax.vmap(
                lambda *row: row_logprob(row[0], row[1], row[2], None, *row[3:])
            )(
                params,
                Mcomp,
                Rcomp,
                peak_time,
                rv_times,
                observed_rv,
                observed_rv_err,
                lc_times,
                observed_lc,
                observed_lc_err,
                detected_periods,
                rv_mask,
                lc_mask,
                rv_source_ids,
            )
        return jax.vmap(row_logprob)(
            params,
            Mcomp,
            Rcomp,
            tau_eff,
            peak_time,
            rv_times,
            observed_rv,
            observed_rv_err,
            lc_times,
            observed_lc,
            observed_lc_err,
            detected_periods,
            rv_mask,
            lc_mask,
            rv_source_ids,
        )

    batched_logprob = jax.jit(_batched_logprob)

    def logprob(params):
        return batched_logprob(jax.device_put(params, device))

    return logprob
