"""emcee fitting backend."""

import logging

import emcee
import numpy
from astropy import units
from quaxed import numpy as jnp

from ..generators import SelfLensingLightcurveGenerator
from ..models import SelfLensingSystem, DEFAULT_LIMITS
from ._utils import _select_fit_device
from .likelihood import make_combined_chi2_vec


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
    lensing_method=None,
    center_lc_model=False,
    progress=True,
    progress_desc=None,
    prefer_gpu=True,
    device=None,
    stop_on_convergence=False,
    initial_walker_state=None,
    rv_source_ids=None,
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
    if lensing_method is None:
        lensing_method = (
            SelfLensingLightcurveGenerator.default_lensing_method_for_device(fit_device)
        )
    logging.info(
        "fit: using JAX %s device %s with lensing_method=%s",
        fit_device.platform,
        fit_device,
        lensing_method,
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

    from ..lombscargle import get_periods

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

    sampler = emcee.EnsembleSampler(
        nwalkers,
        ndim,
        make_combined_chi2_vec(*args, rv_source_ids=rv_source_ids),
        vectorize=True,
    )
    starting_walker_state = (
        initial_walker_state if initial_walker_state is not None else p0
    )
    final_walker_state = None
    progress_kwargs = (
        {"desc": progress_desc} if progress and progress_desc is not None else None
    )

    if stop_on_convergence:
        autocorr_check_interval = 100
        old_tau = numpy.inf

        for final_walker_state in sampler.sample(
            starting_walker_state,
            iterations=nsteps,
            progress=progress,
            progress_kwargs=progress_kwargs,
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
            progress_kwargs=progress_kwargs,
        )

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
