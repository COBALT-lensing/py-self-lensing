# import copy
import emcee
import jax
import jax.numpy as jax_numpy
import logging
import numpy
import unxt
import warnings

from astropy import units
from quax import quaxify
from quaxed import numpy as jnp

from .models import SelfLensingSystem, DEFAULT_LIMITS
from .generators import (
    RVGenerator,
    LightcurveGenerator,
    GPLightcurveGenerator,
    SelfLensingLightcurveGenerator,
)
from .gp.PriorsSamplers import log_prior_bounds

warnings.filterwarnings("ignore", module="erfa.core")


def _jax_devices(platform):
    try:
        return jax.devices(platform)
    except RuntimeError:
        return []


def _select_fit_device(device=None, prefer_gpu=True):
    if device is not None and hasattr(device, "platform"):
        return device

    if device is None or str(device).lower() == "auto":
        gpu_devices = _jax_devices("gpu") if prefer_gpu else []
        return gpu_devices[0] if gpu_devices else jax.devices()[0]

    device_name = str(device).lower()
    platform, _, device_id = device_name.partition(":")
    if platform == "cuda":
        platform = "gpu"

    devices = _jax_devices(platform)
    if not devices:
        raise RuntimeError(f"No JAX {platform!r} devices are available")

    if device_id:
        try:
            return devices[int(device_id)]
        except (ValueError, IndexError) as exc:
            raise RuntimeError(f"No JAX {platform!r} device with index {device_id}") from exc

    return devices[0]


def _device_array(value, device):
    if value is None:
        return None
    return jax.device_put(jax_numpy.asarray(value), device)


def gauss_lnprior(x, mu, sigma=None):
    if sigma is None:
        sigma = mu / 10
    # https://stackoverflow.com/questions/49810234/using-emcee-with-gaussian-priors
    gauss_norm = jnp.log(1.0 / (jnp.sqrt(2 * jnp.pi) * sigma))
    return gauss_norm - 0.5 * (x - mu) ** 2 / sigma**2


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
):
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
            period_err = jax.lax.cond(
                jnp.size(detected_periods) > 0,
                lambda a, b: gauss_lnprior(
                    a, b[jnp.argmin(jnp.abs(b - a))] if b.size > 0 else a, 0.1
                ),
                zero,
                new_model_sls.porb.value,
                detected_periods,
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
                return chi2(
                    observed_rv.to(units.m / units.s).value,
                    observed_rv_err.to(units.m / units.s).value,
                    rv_gen.rv.to(units.m / units.s).value,
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
                    return chi2(
                        observed_lc,
                        observed_lc_err,
                        model_lc,
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


from functools import partial


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

    combined_chi2_vec = jax.jit(_combined_chi2_vec, device=device)

    def logprob(params):
        # emcee feeds/consumes NumPy; keep the vectorized likelihood on the
        # selected JAX device, then transfer just the log-probabilities back.
        return numpy.asarray(combined_chi2_vec(params))

    return logprob


@units.quantity_input
def self_lensing_system_from_fit(
    Mcomp: units.kg,
    peak_time,
    rv_times=None,
    observed_rv=None,
    observed_rv_err=None,
    lc_times=None,
    observed_lc=None,
    observed_lc_err=None,
    gp_model=None,
    Rcomp: units.m = None,
    Teff: units.K = None,
    tau_eff: units.day = None,
    d: units.pc = SelfLensingSystem.DEFAULT_D,
    false_alarm_probability=0.1,
    detect_periods=False,
    periods=[],
    period_min: units.day = None,
    period_max: units.day = None,
    weight_lc_peak=False,
    nwalkers=100,
    nsteps=5000,
    burn=500,
    lensing_method=SelfLensingLightcurveGenerator.DEFAULT_LENSING_METHOD,
    center_lc_model=False,
    pool=None,
    progress=True,
    prefer_gpu=True,
    device=None,
    stop_on_convergence=False,
    initial_walker_state=None,
):
    ndim = 5
    initial_walker_coords = None
    if initial_walker_state is not None:
        initial_walker_coords = getattr(
            initial_walker_state,
            "coords",
            initial_walker_state,
        )
        initial_walker_coords = numpy.asarray(initial_walker_coords)
        if initial_walker_coords.ndim != 2 or initial_walker_coords.shape[1] != ndim:
            raise ValueError(
                "initial_walker_state must have coordinates with shape "
                f"(nwalkers, {ndim})"
            )
        nwalkers = initial_walker_coords.shape[0]

    logging.debug(
        "fit: nwalkers=%s, nsteps=%s, stop_on_convergence=%s, "
        "initial_walker_state=%s",
        nwalkers,
        nsteps,
        stop_on_convergence,
        initial_walker_state is not None,
    )
    if observed_rv is None and observed_lc is None:
        raise RuntimeError("You must provide at least one of observed_rv, observed_ts")

    fit_device = _select_fit_device(device=device, prefer_gpu=prefer_gpu)
    logging.info(
        "fit: using JAX %s device %s",
        fit_device.platform,
        fit_device,
    )

    default_sls = SelfLensingSystem(
        0 * SelfLensingSystem.param_units["Macc"],
        0 * SelfLensingSystem.param_units["Mcomp"],
        0 * SelfLensingSystem.param_units["Rcomp"],
        0 * SelfLensingSystem.param_units["a"],
    )

    p0 = [
        numpy.random.uniform(bounds[0], bounds[1], nwalkers)
        for bounds in (
            DEFAULT_LIMITS["i"],
            DEFAULT_LIMITS["Macc"],
            DEFAULT_LIMITS["e"],
            DEFAULT_LIMITS["periapsis_phase"],
        )
    ]
    p0.insert(
        0,
        jnp.logspace(
            jnp.log10(DEFAULT_LIMITS["a"][0]),
            jnp.log10(DEFAULT_LIMITS["a"][1]),
            nwalkers,
        ),
    )

    p0 = numpy.asarray(list(zip(*p0)), dtype=float)

    # gp = None
    # if gp_model is not None:
    #     gp = gp_model(default_sls, lc_times.jd, observed_lc, nwalkers=nwalkers)

    Mcomp = Mcomp.to(SelfLensingSystem.param_units["Mcomp"])
    if Rcomp is not None:
        Rcomp = Rcomp.to(SelfLensingSystem.param_units["Rcomp"])
    if observed_rv is not None:
        observed_rv = observed_rv.to(units.m / units.s)
    if observed_rv_err is not None:
        observed_rv_err = observed_rv_err.to(units.m / units.s)
    if observed_lc is not None and hasattr(observed_lc, "to"):
        observed_lc = observed_lc.to(units.dimensionless_unscaled)
    if observed_lc_err is not None and hasattr(observed_lc_err, "to"):
        observed_lc_err = observed_lc_err.to(units.dimensionless_unscaled)

    from .lombscargle import get_periods

    MAX_PERIOD_THRESH = 0.9

    detected_lc_periods = []
    detected_rv_periods = []

    if detect_periods:
        lc_baseline = 0
        if lc_times is not None:
            lc_baseline = lc_times[-1].jd - lc_times[0].jd
            detected_lc_periods = get_periods(
                observed_lc, lc_times.jd, false_alarm_probability
            )

        if len(detected_lc_periods) > 0:
            detected_lc_periods = detected_lc_periods[
                detected_lc_periods < MAX_PERIOD_THRESH * lc_baseline
            ]

        rv_baseline = 0
        if rv_times is not None:
            rv_baseline = rv_times[-1].jd - rv_times[0].jd
            detected_rv_periods = get_periods(
                observed_rv, rv_times.jd, false_alarm_probability
            )

            if len(detected_rv_periods) > 0:
                detected_rv_periods = detected_rv_periods[
                    detected_rv_periods < MAX_PERIOD_THRESH * rv_baseline
                ]

    detected_periods = jnp.concatenate(
        [
            jnp.array(periods) if len(periods) > 0 else jnp.array([]),
            (
                jnp.array(detected_lc_periods)
                if len(detected_lc_periods) > 0
                else jnp.array([])
            ),
            (
                jnp.array(detected_rv_periods)
                if len(detected_rv_periods) > 0
                else jnp.array([])
            ),
        ]
    )

    # close_pool = False
    # if pool is None:
    #     from multiprocessing import Pool

    #     pool = Pool()
    #     close_pool = True
    # if pool is False:
    #     pool = None

    args = (
        Mcomp.value,
        Rcomp.value if Rcomp is not None else None,
        tau_eff.value if tau_eff is not None else None,
        d.value if d is not None else None,
        peak_time,
        rv_times,
        observed_rv.value if observed_rv is not None else None,
        observed_rv_err.value if observed_rv_err is not None else None,
        lc_times,
        observed_lc.value if hasattr(observed_lc, "value") else observed_lc,
        observed_lc_err.value if hasattr(observed_lc_err, "value") else observed_lc_err,
        gp_model,
        detected_periods,
        lensing_method,
        center_lc_model,
        weight_lc_peak,
        period_min,
        period_max,
        default_sls,
        fit_device,
    )

    # try:
    sampler = emcee.EnsembleSampler(
        nwalkers,
        ndim,
        make_combined_chi2_vec(*args),
        vectorize=True,
        # combined_chi2,
        # args=args,
        # pool=pool,
        # moves=[
        #     (emcee.moves.DEMove(), 0.8),
        #     (emcee.moves.DESnookerMove(), 0.2),
        # ],
    )
    starting_walker_state = (
        initial_walker_state if initial_walker_state is not None else p0
    )
    final_walker_state = None

    if stop_on_convergence:
        autocorr_check_interval = 100
        old_tau = numpy.inf

        for final_walker_state in sampler.sample(
            starting_walker_state,
            iterations=nsteps,
            progress=progress,
        ):
            if sampler.iteration % autocorr_check_interval:
                continue

            try:
                tau = sampler.get_autocorr_time(tol=0)
            except emcee.autocorr.AutocorrError:
                continue

            converged = numpy.all(numpy.isfinite(tau))
            converged &= numpy.all(tau * 100 < sampler.iteration)
            converged &= numpy.all(numpy.abs(old_tau - tau) / tau < 0.01)

            if converged:
                logging.info("fit: converged after %s steps", sampler.iteration)
                break

            old_tau = tau
    else:
        final_walker_state = sampler.run_mcmc(
            starting_walker_state,
            nsteps,
            progress=progress,
        )
    # finally:
    #     if close_pool:
    #         pool.close()
    flat_samples = sampler.get_chain(flat=True, discard=burn)
    res = flat_samples[jnp.argmax(sampler.get_log_prob(flat=True, discard=burn))]

    out_a, out_i, out_Macc, out_e, out_periapsis_phase = res

    return (
        SelfLensingSystem(
            a=out_a * SelfLensingSystem.param_units["a"],
            i=out_i * SelfLensingSystem.param_units["i"],
            Macc=out_Macc * SelfLensingSystem.param_units["Macc"],
            Rcomp=Rcomp,
            Mcomp=Mcomp,
            Teff=Teff,
            e=out_e * SelfLensingSystem.param_units["e"],
            periapsis_phase=out_periapsis_phase
            * SelfLensingSystem.param_units["periapsis_phase"],
            d=d,
        ),
        sampler,
        final_walker_state,
    )
