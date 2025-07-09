import emcee
import numpy

from astropy import units

from selflensing.models import SelfLensingSystem
from selflensing.generators import (
    RVGenerator,
    LightcurveGenerator,
    SelfLensingLightcurveGenerator,
)

FIT_LIMITS = {
    "a": (0.01, 50),
    "b": (0, 10),
    "Macc": (0.1, 50),
    "Rcomp": (0.1, 1000),
    "Mcomp": (0.1, 200),
    "Teff": (100, 100000),
    "e": (0, 1),
    "periapsis_phase": (0, 2 * numpy.pi),
}


def gauss_lnprior(x, mu, sigma=1):
    # https://stackoverflow.com/questions/49810234/using-emcee-with-gaussian-priors
    gauss_norm = numpy.log(1.0 / (numpy.sqrt(2 * numpy.pi) * sigma))
    return gauss_norm - 0.5 * (x - mu) ** 2 / sigma**2


def combined_chi2(
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
    # minimum_baseline_period,
    # maximum_baseline_period,
    lensing_method,
    model_sls=None,
):
    a, b, Macc, e, periapsis_phase = params

    if a < FIT_LIMITS["a"][0] or a > FIT_LIMITS["a"][1]:
        return -numpy.inf

    if Macc < FIT_LIMITS["Macc"][0] or Macc > FIT_LIMITS["Macc"][1]:
        return -numpy.inf

    if (
        periapsis_phase < FIT_LIMITS["periapsis_phase"][0]
        or periapsis_phase > FIT_LIMITS["periapsis_phase"][1]
    ):
        return -numpy.inf

    # e = 1 is parabolic, so less than but not equal
    if e < FIT_LIMITS["e"][0] or e >= FIT_LIMITS["e"][1]:
        return -numpy.inf

    a = a * SelfLensingSystem.param_units["a"]
    Macc = Macc * SelfLensingSystem.param_units["Macc"]
    e = e * SelfLensingSystem.param_units["e"]
    periapsis_phase = periapsis_phase * SelfLensingSystem.param_units["periapsis_phase"]
    b = b * SelfLensingSystem.param_units["b"]

    if model_sls is None:
        model_sls = SelfLensingSystem(
            a=a,
            b=b,
            Macc=Macc,
            Mcomp=Mcomp,
            Rcomp=Rcomp,
            e=e,
            periapsis_phase=periapsis_phase,
        )
    else:
        model_sls.a = a
        model_sls.b = b
        model_sls.Macc = Macc
        model_sls.Mcomp = Mcomp
        model_sls.Rcomp = Rcomp
        model_sls.e = e
        model_sls.periapsis_phase = periapsis_phase

    porb = model_sls.porb.value

    period_err = 0
    if len(detected_periods) > 0:
        # if porb > minimum_baseline_period:
        #     return -numpy.inf
        closest_period = detected_periods[
            numpy.argmin(numpy.abs(detected_periods - porb))
        ]
        period_err = gauss_lnprior(porb, closest_period)

    # if porb > maximum_baseline_period:
    #     return -numpy.inf

    if observed_rv is None:
        rv_chi2 = 0
    else:
        rv_gen = RVGenerator(model_sls, times=rv_times, peak_time=peak_time)
        rv_chi2 = chi2(
            observed_rv.value,
            observed_rv_err.to(units.m / units.s).value,
            rv_gen.rv.value,
        )

    if observed_lc is None:
        lc_chi2 = 0
    else:
        lc_gen = LightcurveGenerator(
            model_sls,
            times=lc_times,
            peak_time=peak_time,
            lensing_method=lensing_method,
        )
        lc_chi2 = chi2(
            observed_lc.value, observed_lc_err.value, lc_gen.flux_array.value
        )

    return rv_chi2 + lc_chi2 + period_err


def chi2(obs, obs_err, model):
    return -numpy.sum((obs - model) ** 2 / obs_err**2) / len(obs)


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
    Teff: units.K = SelfLensingSystem.DEFAULT_TEFF,
    nwalkers=100,
    nsteps=5000,
    burn=500,
    lensing_method=SelfLensingLightcurveGenerator.DEFAULT_LENSING_METHOD,
    pool=None,
    progress=True,
):
    if observed_rv is None and observed_lc is None:
        raise RuntimeError("You must provide at least one of observed_rv, observed_ts")

    p0 = list(
        zip(
            *[
                numpy.random.uniform(bounds[0], bounds[1], nwalkers)
                for bounds in (
                    FIT_LIMITS["a"],
                    FIT_LIMITS["b"],
                    FIT_LIMITS["Macc"],
                    FIT_LIMITS["e"],
                    FIT_LIMITS["periapsis_phase"],
                )
            ]
        )
    )

    from selflensing.lombscargle import get_periods

    MAX_PERIOD_THRESH = 0.9

    lc_baseline = 0
    detected_lc_periods = []
    if lc_times is not None:
        lc_baseline = lc_times[-1].jd - lc_times[0].jd
        detected_lc_periods = get_periods(observed_lc, lc_times.jd)

    if len(detected_lc_periods) > 0:
        detected_lc_periods = detected_lc_periods[
            detected_lc_periods < MAX_PERIOD_THRESH * lc_baseline
        ]

    rv_baseline = 0
    detected_rv_periods = []
    if rv_times is not None:
        rv_baseline = rv_times[-1].jd - rv_times[0].jd
        detected_rv_periods = get_periods(observed_rv, rv_times.jd)

        if len(detected_rv_periods) > 0:
            detected_rv_periods = detected_rv_periods[
                detected_rv_periods < MAX_PERIOD_THRESH * rv_baseline
            ]

    # baseline = max(lc_baseline, rv_baseline)
    # minimum_baseline_period = MIN_BASELINE_THRESH * baseline
    # maximum_baseline_period = MAX_BASELINE_THRESH * baseline

    detected_periods = numpy.concatenate([detected_lc_periods, detected_rv_periods])

    close_pool = False
    if pool is None:
        from multiprocessing import Pool

        pool = Pool()
        close_pool = True
    if pool is False:
        pool = None

    try:
        sampler = emcee.EnsembleSampler(
            nwalkers,
            len(p0[0]),
            combined_chi2,
            args=(
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
                # minimum_baseline_period,
                # maximum_baseline_period,
                lensing_method,
                SelfLensingSystem(
                    0 * SelfLensingSystem.param_units["Macc"],
                    0 * SelfLensingSystem.param_units["Mcomp"],
                    0 * SelfLensingSystem.param_units["Rcomp"],
                    0 * SelfLensingSystem.param_units["a"],
                ),
            ),
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

    out_a, out_Macc, out_e, out_periapsis_phase = res

    return (
        SelfLensingSystem(
            a=out_a * SelfLensingSystem.param_units["a"],
            b=0 * SelfLensingSystem.param_units["b"],
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
