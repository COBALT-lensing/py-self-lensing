from astropy.timeseries import LombScargle


def get_periods(lc, times):
    if len(times) == 0:
        return []
    maximum_frequency = 0.5
    minimum_frequency = 1.0 / (times[-1] - times[0])
    ls = LombScargle(times, lc)
    frequency, power = ls.autopower(
        maximum_frequency=maximum_frequency, minimum_frequency=minimum_frequency
    )

    fal = ls.false_alarm_level(0.01)
    return 1 / frequency[power > fal]
