import emcee
import jax
import logging
import numpy
import unxt
import warnings

from astropy import units
from quax import quaxify

from .models import SelfLensingSystem
from .generators import (
    RVGenerator,
    LightcurveGenerator,
    SelfLensingLightcurveGenerator,
)
from .gp.PriorsSamplers import log_prior_bounds

warnings.filterwarnings("ignore", module="erfa.core")

FIT_LIMITS = {
    "a": (0.01, 200),
    "i": (1.4, 0.5 * numpy.pi),
    "Macc": (0.5, 50),
    "e": (0, 0.999999),
    "periapsis_phase": (0, 2 * numpy.pi),
}


def gauss_lnprior(x, mu, sigma=None):
    if sigma is None:
        sigma = mu / 10
    # https://stackoverflow.com/questions/49810234/using-emcee-with-gaussian-priors
    gauss_norm = numpy.log(1.0 / (numpy.sqrt(2 * numpy.pi) * sigma))
    return gauss_norm - 0.5 * (x - mu) ** 2 / sigma**2


@quaxify
def combined_chi2(
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
    gp,
    detected_periods,
    lensing_method,
    weight_lc_peak=False,
    period_min=None,
    period_max=None,
    model_sls=None,
):
    # if a < FIT_LIMITS["a"][0] or a > FIT_LIMITS["a"][1]:
    #     return -numpy.inf

    # if i < FIT_LIMITS["i"][0] or i > FIT_LIMITS["i"][1]:
    #     return -numpy.inf

    # if Macc < FIT_LIMITS["Macc"][0] or Macc > FIT_LIMITS["Macc"][1]:
    #     return -numpy.inf

    # if (
    #     periapsis_phase < FIT_LIMITS["periapsis_phase"][0]
    #     or periapsis_phase > FIT_LIMITS["periapsis_phase"][1]
    # ):
    #     return -numpy.inf

    # # e = 1 is parabolic, so less than but not equal
    # if e < FIT_LIMITS["e"][0] or e >= FIT_LIMITS["e"][1]:
    #     return -numpy.inf
    print(type(Mcomp))
    Mcomp = unxt.Quantity(Mcomp, SelfLensingSystem.param_units["Mcomp"])
    Rcomp = unxt.Quantity(Rcomp, SelfLensingSystem.param_units["Rcomp"])
    lp_bounds = log_prior_bounds(params, list(FIT_LIMITS.values()))

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
        print("Mcomp, compute_full_logprob", Mcomp)
        model_sls = SelfLensingSystem(
            a=a,
            i=i,
            Macc=Macc,
            Mcomp=Mcomp,
            Rcomp=Rcomp,
            e=e,
            periapsis_phase=periapsis_phase,
        )
        # else:
        #     model_sls.a = a
        #     model_sls.Macc = Macc
        #     model_sls.Mcomp = Mcomp
        #     # Must set i after setting the masses to correctly pre-calc b
        #     model_sls.i = i
        #     model_sls.Rcomp = Rcomp
        #     model_sls.e = e
        #     model_sls.periapsis_phase = periapsis_phase

        if period_min is not None and model_sls.porb < period_min:
            return -numpy.inf
        if period_max is not None and model_sls.porb > period_max:
            return -numpy.inf

        period_err = 0
        if len(detected_periods) > 0:
            porb = model_sls.porb.value
            closest_period = detected_periods[
                numpy.argmin(numpy.abs(detected_periods - porb))
            ]
            period_err = gauss_lnprior(porb, closest_period, sigma=0.1)

        tau_eff_err = 0
        if tau_eff is not None and not numpy.isnan(tau_eff):
            if numpy.isnan(model_sls.tau_eff):
                # If the impact parameter is too large, then there is no crossing
                # We only enforce this if tau_eff has been provided, so that we
                # can still fit ellipsoidal variations and Doppler boosting when
                # there is no lensing.
                return -numpy.inf
            tau_eff_err = gauss_lnprior(
                model_sls.tau_eff.to(units.day).value,
                tau_eff.to(units.day).value,
                tau_eff.to(units.day).value / 100,
            )

        rv_chi2 = 0
        if observed_rv is not None:
            rv_gen = RVGenerator(model_sls, times=rv_times, peak_time=peak_time)
            rv_chi2 = chi2(
                observed_rv.to(units.m / units.s).value,
                observed_rv_err.to(units.m / units.s).value,
                rv_gen.rv.to(units.m / units.s).value,
            )

        lc_chi2 = 0
        if observed_lc is not None:
            lc_gen = LightcurveGenerator(
                model_sls,
                times=lc_times,
                peak_time=peak_time,
                lensing_method=lensing_method,
            )

            if gp is not None:
                gp.sls = model_sls
                gp.mean_vals = lc_gen.flux_array.value
                lc_chi2 = gp.logprob_single()
                # print("lc_chi2=", lc_chi2)
            else:
                lc_err = observed_lc_err.value
                if weight_lc_peak:
                    # Reduce the weight of points further from the peak
                    lc_err *= 1 + (peak_time - lc_times).value ** 2
                lc_chi2 = chi2(
                    observed_lc.value, observed_lc_err.value, lc_gen.flux_array.value
                )

        return rv_chi2 + lc_chi2 + period_err + tau_eff_err

    # ---- 3) Define the "False" branch for jax.lax.cond ----
    def return_neg_inf(_):
        return -jax.numpy.inf

    # ---- 4) JAX-native conditional branch ----
    # This correctly branches based on the *traced* value of jnp.isfinite
    return jax.lax.cond(
        jax.numpy.isfinite(lp_bounds),
        compute_full_logprob,  # Function to run if True
        return_neg_inf,  # Function to run if False
        operand=None,
    )


def chi2(obs, obs_err, model):
    return -numpy.sum((obs - model) ** 2 / obs_err**2) / len(obs)


def make_combined_chi2_vec(
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
    gp,
    detected_periods,
    lensing_method,
    weight_lc_peak=False,
    period_min=None,
    period_max=None,
    model_sls=None,
):

    @jax.jit
    @quaxify
    def combined_chi2_vec(params):
        params = jax.numpy.asarray(params)
        return jax.numpy.asarray(
            jax.vmap(
                lambda p: combined_chi2(
                    p,
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
                    gp,
                    detected_periods,
                    lensing_method,
                    weight_lc_peak,
                    period_min,
                    period_max,
                    model_sls,
                )
            )(params)
        )

    return combined_chi2_vec


@units.quantity_input
def self_lensing_system_from_fit(
    Mcomp: units.kg,
    Rcomp: units.m,
    peak_time,
    rv_times=None,
    observed_rv=None,
    observed_rv_err=None,
    lc_times=None,
    observed_lc=None,
    observed_lc_err=None,
    gp_model=None,
    Teff: units.K = SelfLensingSystem.DEFAULT_TEFF,
    tau_eff: units.day = None,
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
    pool=None,
    progress=True,
):
    logging.debug(f"fit: nwalkers={nwalkers}, nsteps={nsteps}")
    if observed_rv is None and observed_lc is None:
        raise RuntimeError("You must provide at least one of observed_rv, observed_ts")

    default_sls = SelfLensingSystem(
        0 * SelfLensingSystem.param_units["Macc"],
        0 * SelfLensingSystem.param_units["Mcomp"],
        0 * SelfLensingSystem.param_units["Rcomp"],
        0 * SelfLensingSystem.param_units["a"],
    )

    p0 = [
        numpy.random.uniform(bounds[0], bounds[1], nwalkers)
        for bounds in (
            FIT_LIMITS["i"],
            FIT_LIMITS["Macc"],
            FIT_LIMITS["e"],
            FIT_LIMITS["periapsis_phase"],
        )
    ]
    p0.insert(
        0,
        numpy.logspace(
            numpy.log10(FIT_LIMITS["a"][0]), numpy.log10(FIT_LIMITS["a"][1]), nwalkers
        ),
    )

    p0 = list(zip(*p0))

    gp = None
    if gp_model is not None:
        gp = gp_model(default_sls, lc_times.jd, observed_lc, nwalkers=nwalkers)

    Mcomp = Mcomp.to(SelfLensingSystem.param_units["Mcomp"])
    Rcomp = Rcomp.to(SelfLensingSystem.param_units["Rcomp"])

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

    detected_periods = numpy.concatenate(
        [periods, detected_lc_periods, detected_rv_periods]
    )
    close_pool = False
    if pool is None:
        from multiprocessing import Pool

        pool = Pool()
        close_pool = True
    if pool is False:
        pool = None

    args = (
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
        gp,
        detected_periods,
        lensing_method,
        weight_lc_peak,
        period_min,
        period_max,
        default_sls,
    )

    try:
        sampler = emcee.EnsembleSampler(
            nwalkers,
            len(p0[0]),
            make_combined_chi2_vec(*args),
            vectorize=True,
            # combined_chi2,
            # args=args,
            pool=pool,
            moves=[
                (emcee.moves.DEMove(), 0.8),
                (emcee.moves.DESnookerMove(), 0.2),
            ],
        )
        sampler.run_mcmc(
            p0,
            nsteps,
            progress=progress,
        )
    finally:
        if close_pool:
            pool.close()
    flat_samples = sampler.get_chain(flat=True, discard=burn)
    res = flat_samples[numpy.argmax(sampler.get_log_prob(flat=True, discard=burn))]

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
        ),
        sampler,
    )
