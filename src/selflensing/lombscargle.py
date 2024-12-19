import numpy

from astropy.timeseries import LombScargle


def get_top_period(rv, times):
    maximum_frequency = 0.5
    minimum_frequency = 1.0 / max(times[-1] - times[0])
    frequency, power = LombScargle(times, rv).autopower(
        maximum_frequency=maximum_frequency, minimum_frequency=minimum_frequency
    )
    period = 1 / frequency
    return period[numpy.argmax(power)]
