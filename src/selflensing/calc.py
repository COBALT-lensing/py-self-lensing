import numpy
import rustlens

from astropy import constants, units

import pandas as pd

LENS_MASS_MIN = 0.5 * units.M_sun
LENS_MASS_MAX = 50 * units.M_sun
LENS_MASS_DIST_SIZE = 100
# Temporary simple lens mass distribution for testing
LENS_MASS_DIST = numpy.array(
    [
        numpy.linspace(LENS_MASS_MIN.value, LENS_MASS_MAX.value, LENS_MASS_DIST_SIZE),
        numpy.ones(LENS_MASS_DIST_SIZE),
    ]
)
LENS_MASS_DIST[1, 19] = 7
LENS_MASS_DIST[1, 20] = 10
LENS_MASS_DIST[1, 21] = 5
LENS_MASS_DIST[1, 59] = 3
LENS_MASS_DIST[1, 60] = 8
LENS_MASS_DIST[1, 61] = 6
LENS_MASS_DIST[1, 89] = 3
LENS_MASS_DIST[1, 90] = 8
LENS_MASS_DIST[1, 61] = 6

LENS_MASS_DIST[1, :] /= numpy.sum(LENS_MASS_DIST[1, :])


def lens_mass_pdist(n, min_mass=LENS_MASS_MIN.value, max_mass=LENS_MASS_MAX.value):
    return numpy.interp(
        numpy.linspace(min_mass, max_mass, n),
        LENS_MASS_DIST[0, :],
        LENS_MASS_DIST[1, :],
        0,
        0,
    )


def gen_mag_matrix(rE, l, Rcomp):
    return numpy.array(rustlens.multi_witt_mao_magnification(l, rE, Rcomp))


def estimate_lensing(min_mag, max_mag, Rcomp):
    rE_out = []
    b_out = []

    all_rE = numpy.linspace(0.001, 1, 1000)
    l = numpy.linspace(0, 10, 1000)

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


def calc_a(rE):
    x = (rE**2 * constants.c**2) / (4 * constants.G)
    a_min = x / LENS_MASS_MAX
    a_max = x / LENS_MASS_MIN
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


def estimate_porb(tau_eff, min_mag, max_mag, Rcomp):
    rE, b = estimate_lensing(min_mag, max_mag, Rcomp.to(units.R_sun).value)
    rE *= units.R_sun
    b_max = 1 + Rcomp / rE
    b_mask = b <= b_max

    rE = rE[b_mask]
    b = b[b_mask]
    a_min, a_max = calc_a(rE)

    # TODO: Use lens mass distribution to calculate a probability distribution scaled between a_min and a_max.
    #       This will give a 2-D array, containing a probability distribution for each rE between its a_min and a_max.
    #       i.e. for a given rE, we have a probability distribution for the possible a values.
    #       Reversed so that it's matched to a rather than mass.
    PDIST_SAMPLES = 100
    pdist = numpy.flip(lens_mass_pdist(PDIST_SAMPLES))

    porb_min = numpy.round(
        calc_porb(a_min, b, rE, Rcomp, tau_eff).to(units.day).value, 4
    )
    porb_max = numpy.round(
        calc_porb(a_max, b, rE, Rcomp, tau_eff).to(units.day).value, 4
    )
    porbs = numpy.linspace(porb_min, porb_max, PDIST_SAMPLES)

    # porbs = numpy.logspace(numpy.log10(porb_min), numpy.log10(porb_max), PDIST_SAMPLES)
    print(porbs)
    # pdist = numpy.tile(
    #     pdist, porbs.T.shape
    # ).T  # TODO NEXT: Eliminate the need for this. If the output from numpy.where below is a tuple, can I just take the second one as indices in pdist or something?

    porb_min_bin = numpy.floor(numpy.log10(numpy.min(porb_min[~numpy.isnan(porb_min)])))
    porb_max_bin = numpy.ceil(numpy.log10(numpy.max(porb_max[~numpy.isnan(porb_max)])))
    porb_bins = numpy.logspace(porb_min_bin, porb_max_bin, 100)
    print("porb_bins", porb_bins.shape, porb_bins)
    bin_starts, bin_ends, bin_probs = [], [], []

    # bin_idx = numpy.digitize(porbs, porb_bins)
    # total_bins = len(porb_bins)
    # for i in range(1, total_bins):
    #     print(i, total_bins)
    #     probs = pdist[(bin_idx == i).nonzero()[0]]
    #     combined_prob = numpy.sum(probs) - numpy.prod(probs)
    #     bin_starts.append(i - 1)
    #     bin_ends.append(i)
    #     bin_probs.append(combined_prob)

    for start, end in zip(porb_bins[:-1], porb_bins[1:]):
        bin_filter = numpy.where((porbs < end) & (porbs >= start))[0]
        print(start, end, porb_max_bin)
        bin_prob = numpy.sum(pdist[bin_filter]) - numpy.prod(pdist[bin_filter])
        bin_starts.append(start)
        bin_ends.append(end)
        bin_probs.append(bin_prob)
    return bin_starts, bin_ends, bin_probs

    # pdist applies to each porbs[:,*]

    # TODO: Then, can we assume the a probability distributions just scale linearly for porb_min and porb_max?

    # Remove solutions where the max is less than 2 crossing times
    # porb_max_filter = numpy.where(porb_max > (2 * tau_eff))
    # porb_min = porb_min[porb_max_filter]
    # porb_max = porb_max[porb_max_filter]
    # pdist = pdist[porb_max_filter]

    # Disabled the following because it's too complicated to handle the probabilities correctly...
    # We can show a vline at 2*tau_eff on the final plot instead and just ignore anything below that
    # # Reset min to at least 2 crossing times
    # porb_min_filter = numpy.where(porb_min < (2 * tau_eff))
    # porb_min[porb_min_filter] = 2 * tau_eff
    # # TODO: How to handle probs here? Need to calc the prob for 2*tau_eff and set it to that.
    # #       i.e. need to take all the probabilities that correspond
    # double_tau_eff_as = calc_a_from_porb(2 * tau_eff)
    # double_tau_eff_rEs = calc_rE(double_tau_eff_as)
    # double_tau_eff_probs = calc_a_prob(double_tau_eff_rEs)
    # pdist[porb_min_filter] = combine_probs(double_tau_eff_probs)

    # Make porb_min and porb_max unique by porb_min, keeping the largest porb_max for each porb_min

    # df = pd.DataFrame(
    #     {"porb_min": porb_min.value, "porb_max": porb_max.value, "prob": a_prob}
    # )
    # grouped = df.groupby("porb_min", as_index=False)["porb_max"].max()
    # porb_min = units.Quantity(grouped["porb_min"].values, porb_min.unit)
    # porb_max = units.Quantity(grouped["porb_max"].values, porb_max.unit)
    # a_prob = grouped["prob"]

    # # Combine overlapping ranges in porb_min and porb_max
    # # ranges = sorted(zip(porb_min, porb_max), key=lambda x: x[0])
    # # TODO: Put a call to combine_probs in here somewhere
    # combined = [[porb_min[0], porb_max[0]]]
    # last_end = porb_max[0]
    # for start, end in zip(porb_min[1:], porb_max[1:]):
    #     if start > last_end:
    #         combined.append([start, end])
    #         last_end = end
    #     elif end > last_end:
    #         combined[-1][1] = end
    #         last_end = end
    # Porb = [units.Quantity((s, e), porb_min.unit) for s, e in combined]
    # return Porb


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
