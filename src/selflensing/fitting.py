import emcee
import numpy

from astropy import units

from selflensing.models import SelfLensingSystem
from selflensing.generators import RVGenerator, LightcurveGenerator


FIT_INIT = {
    "a": 5.0,
    "b": 0.0,
    "Macc": 10,
    "Rcomp": 1,
    "Mcomp": 1,
    "Teff": 5000,
    "e": 0,
    "periapsis_phase": 0,
}
FIT_LIMITS = {
    "a": (0.01, 50),
    # "a": (1.0, 1.2),
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
    expected_period,
    minimum_period,
    maximum_period,
):
    a, Macc, e, periapsis_phase = params

    if a < FIT_LIMITS["a"][0] or a > FIT_LIMITS["a"][1]:
        return -numpy.inf

    if Macc < FIT_LIMITS["Macc"][0] or Macc > FIT_LIMITS["Macc"][1]:
        return -numpy.inf

    if (
        periapsis_phase < FIT_LIMITS["periapsis_phase"][0]
        or periapsis_phase > FIT_LIMITS["periapsis_phase"][1]
    ):
        return -numpy.inf

    # e = 1 is parabolic
    if e < FIT_LIMITS["e"][0] or e >= FIT_LIMITS["e"][1]:
        return -numpy.inf

    model_sls = SelfLensingSystem(
        a=a * SelfLensingSystem.param_units["a"],
        b=0 * SelfLensingSystem.param_units["b"],
        Macc=Macc * SelfLensingSystem.param_units["Macc"],
        Mcomp=Mcomp,
        Rcomp=Rcomp,
        e=e * SelfLensingSystem.param_units["e"],
        periapsis_phase=periapsis_phase
        * SelfLensingSystem.param_units["periapsis_phase"],
    )

    period_err = 0
    if expected_period is not None:
        period_err = gauss_lnprior(model_sls.porb.value, expected_period)

    if minimum_period is not None and model_sls.porb.value < minimum_period:
        return -numpy.inf
    if maximum_period is not None and model_sls.porb.value > maximum_period:
        return -numpy.inf

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
        lc_gen = LightcurveGenerator(model_sls, times=lc_times, peak_time=peak_time)
        lc_chi2 = chi2(observed_lc.value, observed_lc_err.value, lc_gen.flux_array)

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
):
    if observed_rv is None and observed_lc is None:
        raise RuntimeError("You must provide at least one of observed_rv, observed_ts")

    p0 = list(
        zip(
            *[
                numpy.random.uniform(bounds[0], bounds[1], nwalkers)
                for bounds in (
                    FIT_LIMITS["a"],
                    FIT_LIMITS["Macc"],
                    FIT_LIMITS["e"],
                    FIT_LIMITS["periapsis_phase"],
                )
            ]
        )
    )

    from selflensing.lombscargle import get_top_period

    baseline = lc_times[-1].jd - lc_times[0].jd

    expected_period = get_top_period(observed_lc, times=lc_times.jd)
    minimum_period = None
    maximum_period = None
    if expected_period > (0.9 * baseline):
        expected_period = None
        minimum_period = 0.9 * baseline
        maximum_period = 2 * baseline

    from multiprocessing import Pool

    with Pool() as pool:
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
                expected_period,
                minimum_period,
                maximum_period,
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
            progress=True,
        )
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
