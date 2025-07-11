from astropy.timeseries import LombScargle


def get_periods(lc, times, min_fal_prob=0.1):
    # Autopower fails with one point, false_alarm_level fails with exactly three points
    # In reality the lower limit could be higher but I'm reluctant to set an arbitrary
    # minimum.
    if len(times) <= 3:
        return []
    maximum_frequency = 0.5
    minimum_frequency = 1.0 / (times[-1] - times[0])
    ls = LombScargle(times, lc)
    frequency, power = ls.autopower(
        maximum_frequency=maximum_frequency, minimum_frequency=minimum_frequency
    )

    fal = ls.false_alarm_level(min_fal_prob)
    return 1 / frequency[power > fal]
