import numpy
import rustlens

from astropy import constants, units

from multiprocessing import Pool

import pandas as pd


def gen_mag_matrix(rE, l, Rcomp):
    return numpy.array(rustlens.multi_witt_mao_magnification(l, rE, Rcomp))


def estimate_lensing(min_mag, max_mag, Rcomp):
    rE_out = []
    b_out = []

    all_rE = numpy.linspace(0.001, 1, 10000)
    l = numpy.linspace(0, 10, 10000)

    chunk_size = 1000
    for i in range(0, len(all_rE), chunk_size):
        rE = all_rE[i : i + chunk_size]
        mags = gen_mag_matrix(rE, l, [Rcomp])

        match_idx = numpy.where((mags >= min_mag) & (mags <= max_mag))

        rE_match = rE[match_idx[0]]
        l_match = l[match_idx[1]]
        b_match = l_match / (rE_match * Rcomp)

        rE_out.append(rE_match)
        b_out.append(b_match)

    return numpy.concat(rE_out), numpy.concat(b_out)


MACC_MIN = 0.5 * units.M_sun
MACC_MAX = 50 * units.M_sun


def calc_a(rE):
    x = (rE**2 * constants.c**2) / (4 * constants.G)
    a_min = x / MACC_MAX
    a_max = x / MACC_MIN
    return a_min.to(units.au), a_max.to(units.au)


def calc_porb(a, b, rE, Rcomp, tau_eff):
    """
    Using some trig identities to eliminate sin i from the below
    """
    return (
        (tau_eff / numpy.sqrt(1 - ((b * rE) / (1 * Rcomp.unit + Rcomp)) ** 2))
        * (numpy.pi / (rE + Rcomp))
        * (
            a
            * numpy.sin(  # sin i
                (0.5 * numpy.pi) * units.rad - numpy.arcsin((b * rE) / a)
            )
        )
    )


ROUND_SIGFIGS = 3


def estimate_porb(tau_eff, min_mag, max_mag, Rcomp):
    rE, b = estimate_lensing(min_mag, max_mag, Rcomp.to(units.R_sun).value)
    rE *= units.R_sun
    b_max = 1 + Rcomp / rE
    b_mask = b <= b_max

    rE = rE[b_mask]
    b = b[b_mask]

    a_min, a_max = calc_a(rE)

    porb_min = numpy.round(calc_porb(a_min, b, rE, Rcomp, tau_eff).to(units.day), 4)
    porb_max = numpy.round(calc_porb(a_max, b, rE, Rcomp, tau_eff).to(units.day), 4)

    # Remove solutions where the max is less than 2 crossing times
    porb_max_filter = numpy.where(porb_max > (2 * tau_eff))
    porb_min = porb_min[porb_max_filter]
    porb_max = porb_max[porb_max_filter]

    # Reset min to at least 2 crossing times
    porb_min[numpy.where(porb_min < (2 * tau_eff))] = 2 * tau_eff

    # Make porb_min and porb_max unique by porb_min, keeping the largest porb_max for each porb_min

    df = pd.DataFrame({"porb_min": porb_min.value, "porb_max": porb_max.value})
    grouped = df.groupby("porb_min", as_index=False)["porb_max"].max()
    porb_min = units.Quantity(grouped["porb_min"].values, porb_min.unit)
    porb_max = units.Quantity(grouped["porb_max"].values, porb_max.unit)

    # Combine overlapping ranges in porb_min and porb_max
    # ranges = sorted(zip(porb_min, porb_max), key=lambda x: x[0])
    combined = [[porb_min[0], porb_max[0]]]
    last_end = porb_max[0]
    for start, end in zip(porb_min[1:], porb_max[1:]):
        if start > last_end:
            combined.append([start, end])
            last_end = end
        elif end > last_end:
            combined[-1][1] = end
            last_end = end
    Porb = [units.Quantity((s, e), porb_min.unit) for s, e in combined]
    return Porb


def eliminate_times(porb_ranges, tau_eff, times, aliases=True):
    """
    Eliminates tau_eff days around each time from the given porb_ranges.
    - porb_ranges: A list of time ranges (list of time quantity tuples).
    - tau_eff: A time quantity.
    - times: A single time quantity list.
    """
    if aliases:
        new_times = []
        for t in times:
            new_times.append(t)
            new_times.extend(calc_aliases(t))
        times = new_times
    return eliminate_ranges(
        porb_ranges,
        extend_times(times, tau_eff / 2),
    )


def eliminate_ranges(porb_ranges, times):
    """
    Eliminates any overlap between porb_ranges and times.
    Returns a list of (start, end) tuples from porb_ranges with overlaps removed.
    """
    result = []
    for p_start, p_end in porb_ranges:
        intervals = [(p_start, p_end)]
        for t_start, t_end in times:
            new_intervals = []
            for i_start, i_end in intervals:
                # No overlap
                if t_end <= i_start or t_start >= i_end:
                    new_intervals.append((i_start, i_end))
                else:
                    # Overlap on the left
                    if t_start > i_start:
                        new_intervals.append((i_start, min(t_start, i_end)))
                    # Overlap on the right
                    if t_end < i_end:
                        new_intervals.append((max(t_end, i_start), i_end))
            intervals = [iv for iv in new_intervals if iv[0] < iv[1]]
        result.extend(intervals)
    return result


def extend_ranges(ranges, delta):
    return [(start - delta, end + delta) for (start, end) in ranges]


def extend_times(times, delta):
    return [(time - delta, time + delta) for time in times]


def calc_aliases(p):
    return p / numpy.arange(1, p.value)
