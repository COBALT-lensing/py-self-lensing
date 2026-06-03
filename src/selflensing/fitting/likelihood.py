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
):
    """Compute log-likelihood for self-lensing binary parameters.

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
    lp_bounds = log_prior_bounds(
        params,
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
        a, i, Macc, e, periapsis_phase = params

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
            return (
                (period_min is None or new_model_sls.porb >= period_min)
                and (period_max is None or new_model_sls.porb <= period_max)
                and (
                    tau_eff is None
                    or numpy.isnan(tau_eff)
                    or jnp.isfinite(new_model_sls.tau_eff.value)
                )
            )

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
                    not numpy.isnan(tau_eff),
                    tau_eff_prior,
                    zero,
                    new_model_sls.tau_eff,
                    tau_eff,
                )

            def calc_rv_chi2():
                rv_gen = RVGenerator(new_model_sls, times=rv_times, peak_time=peak_time)
                return chi2_masked(
                    observed_rv.to(units.m / units.s).value,
                    observed_rv_err.to(units.m / units.s).value,
                    rv_gen.rv.to(units.m / units.s).value,
                    rv_mask,
                )

            def calc_lc_chi2():
                def _lc_chi2():
                    lc_gen = LightcurveGenerator(
                        new_model_sls,
                        times=lc_times,
                        peak_time=peak_time,
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
                        peak_time=peak_time,
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

            return rv_chi2 + lc_chi2 + period_err + tau_eff_err

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
    return -jnp.sum((obs - model) ** 2 / obs_err**2) / len(obs)


def chi2_masked(obs, obs_err, model, mask=None):
    if mask is None:
        return chi2(obs, obs_err, model)
    mask = jnp.asarray(mask, dtype=bool)
    weight = mask.astype(float)
    n = jnp.maximum(jnp.sum(weight), 1.0)
    return -jnp.sum(weight * (obs - model) ** 2 / obs_err**2) / n



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
                    period_min,
                    period_max,
                    model_sls,
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
                    row_lc_mask: _row_chi2_vec(
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
                )
            )
        )

    batched_chi2_vec = jax.jit(_batched_chi2_vec)

    def logprob(params):
        params = jax.device_put(params, device)
        return batched_chi2_vec(params)

    return logprob
