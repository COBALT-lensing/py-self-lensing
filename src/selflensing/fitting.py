import emcee
import numpy

from astropy import units

from selflensing.models import SelfLensingSystem
from selflensing.generators import RVGenerator


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


def rv_chi2(
    params,
    Mcomp,
    Rcomp,
    peak_time,
    times,
    observed_rv,
    observed_rv_err,
    expected_period,
    minimum_period,
    maximum_period,
):
    a, Macc, e, periapsis_phase = params
    cls = SelfLensingSystem

    if a < cls.fit_limits["a"][0] or a > cls.fit_limits["a"][1]:
        return -numpy.inf

    # if b < cls.fit_limits["b"][0] or b > cls.fit_limits["b"][1]:
    #     return -numpy.inf

    if Macc < cls.fit_limits["Macc"][0] or Macc > cls.fit_limits["Macc"][1]:
        return -numpy.inf

    if (
        periapsis_phase < cls.fit_limits["periapsis_phase"][0]
        or periapsis_phase > cls.fit_limits["periapsis_phase"][1]
    ):
        return -numpy.inf

    # e = 1 is parabolic
    if e < cls.fit_limits["e"][0] or e >= cls.fit_limits["e"][1]:
        return -numpy.inf

    model_sls = cls(
        a=a * cls.param_units["a"],
        b=0 * cls.param_units["b"],
        Macc=Macc * cls.param_units["Macc"],
        Mcomp=Mcomp,
        Rcomp=Rcomp,
        e=e * cls.param_units["e"],
        periapsis_phase=periapsis_phase * cls.param_units["periapsis_phase"],
    )

    rv_gen = RVGenerator(model_sls, peak_time)
    model_rv = rv_gen.rv(times, noise=None)

    rv_chi2 = -numpy.sum(
        (observed_rv.value - model_rv.value) ** 2
        / observed_rv_err.to(units.m / units.s).value ** 2
    ) / len(observed_rv)

    period_err = 0
    if expected_period is not None:
        period_err = gauss_lnprior(model_sls.porb.value, expected_period)

    if minimum_period is not None and model_sls.porb.value < minimum_period:
        return -numpy.inf
    if maximum_period is not None and model_sls.porb.value > maximum_period:
        return -numpy.inf

    return rv_chi2 + period_err


@units.quantity_input
def self_lensing_system_from_rv_fit(
    cls,
    observed_rv,
    observed_rv_err,
    times,
    Mcomp: units.kg,
    Rcomp: units.m,
    peak_time,
    Teff: units.K = SelfLensingSystem.DEFAULT_TEFF,
):
    nwalkers = 100
    nsteps = 5000

    p0 = list(
        zip(
            *[
                numpy.random.uniform(bounds[0], bounds[1], nwalkers)
                for bounds in (
                    cls.fit_limits["a"],
                    # cls.fit_limits["b"],
                    cls.fit_limits["Macc"],
                    cls.fit_limits["e"],
                    cls.fit_limits["periapsis_phase"],
                )
            ]
        )
    )

    from lombscargle import get_top_period

    baseline = times[-1].jd - times[0].jd

    expected_period = get_top_period(observed_rv, times=times.jd)
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
            rv_chi2,
            args=(
                Mcomp,
                Rcomp,
                peak_time,
                times,
                observed_rv,
                observed_rv_err,
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
    flat_samples = sampler.get_chain(flat=True)
    res = flat_samples[numpy.argmax(sampler.get_log_prob(flat=True))]

    out_a, out_Macc, out_e, out_periapsis_phase = res

    return (
        cls(
            a=out_a * cls.param_units["a"],
            b=0 * cls.param_units["b"],
            Macc=out_Macc * cls.param_units["Macc"],
            Rcomp=Rcomp,
            Mcomp=Mcomp,
            Teff=Teff,
            e=out_e * cls.param_units["e"],
            periapsis_phase=out_periapsis_phase * cls.param_units["periapsis_phase"],
        ),
        sampler,
    )
