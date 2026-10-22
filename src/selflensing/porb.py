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
LENS_MASS_DIST[1, 91] = 6
# for n in range(15, 25):
#     LENS_MASS_DIST[1, n] = 10

LENS_MASS_DIST[1, :] /= numpy.sum(LENS_MASS_DIST[1, :])


def lens_mass_pdist(mass_samples):
    if isinstance(mass_samples, units.Quantity):
        mass_samples = mass_samples.to(units.M_sun).value
    return numpy.interp(
        mass_samples,
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
        b_match = l_match * Rcomp / rE_match

        rE_out.append(rE_match)
        b_out.append(b_match)

    return numpy.concat(rE_out), numpy.concat(b_out)


def _astropy_quantity(value, unit):
    """Normalize astropy/unxt values before mixing with Astropy constants."""
    converted = value.to(unit)
    return numpy.asarray(converted.value)*unit


def _separation_factor(e, periapsis_phase):
    e = numpy.asarray(e)
    if numpy.any(~numpy.isfinite(e)) or numpy.any((e<0) | (e>=1)):
        raise ValueError("Eccentricity must lie in [0, 1)")
    return (1-e)*(1+e)/(1+e*numpy.cos(periapsis_phase))


def calc_a(rE, *, b=0., e=0., periapsis_phase=0.):
    """Mass-limited a range from conjunction RE and impact d/RE.

    The keyword defaults explicitly specify an edge-on circular orbit. For
    eccentric systems r0=a*(1-e^2)/(1+e*cos(periapsis_phase)).
    """
    rE = _astropy_quantity(rE, units.m)
    factor = _separation_factor(e, periapsis_phase)
    z_mass = rE**2*constants.c**2/(4*constants.G)
    transverse = b*rE
    a_min = numpy.sqrt((z_mass/LENS_MASS_MAX)**2+transverse**2)/factor
    a_max = numpy.sqrt((z_mass/LENS_MASS_MIN)**2+transverse**2)/factor
    return a_min.to(units.au), a_max.to(units.au)


def calc_porb(a, b, rE, Rcomp, tau_eff, *, e=0., periapsis_phase=0.):
    """Invert the local eccentric crossing estimate (not an FWHM).

    b is the longitude-zero offset in Einstein radii. This inversion must use
    the same epoch, boundary and straight-track convention as tau_eff.
    """
    a = _astropy_quantity(a, units.m)
    rE = _astropy_quantity(rE, units.m)
    Rcomp = _astropy_quantity(Rcomp, units.m)
    tau_eff = _astropy_quantity(tau_eff, units.day)
    factor = _separation_factor(e, periapsis_phase)
    radius0 = a*factor
    cos_i = (b*rE/radius0).to_value(units.one)
    transverse = 1+e*numpy.cos(periapsis_phase)
    radial_y = e*numpy.sin(periapsis_phase)*cos_i
    speed_factor = numpy.hypot(transverse, radial_y)/numpy.sqrt((1-e)*(1+e))
    closest = b*rE*transverse/numpy.hypot(transverse, radial_y)
    chord = 2*numpy.sqrt((rE+Rcomp)**2-closest**2)
    period = tau_eff*2*numpy.pi*a*speed_factor/chord
    return numpy.where(numpy.abs(cos_i)<1, period.to_value(units.day), numpy.nan)*units.day


def calc_mass(rE, a, *, b=0., e=0., periapsis_phase=0.):
    """Lens mass using the foreground line-of-sight separation at the epoch."""
    rE = _astropy_quantity(rE, units.m)
    a = _astropy_quantity(a, units.m)
    radius0 = a*_separation_factor(e, periapsis_phase)
    z = numpy.sqrt(radius0**2-(b*rE)**2)
    return rE**2*constants.c**2/(4*constants.G*z)


def estimate_porb(tau_eff, min_mag, max_mag, Rcomp, *, e=0., periapsis_phase=0.):
    """Conditional period estimate for specified e/peri; omitted values mean circular.

    Uses a uniform-disc magnification grid and a local crossing-time estimate,
    not measured FWHM. Unknown e/peri require an explicit external prior.
    """
    rE, b = estimate_lensing(min_mag, max_mag, Rcomp.to(units.R_sun).value)
    rE *= units.R_sun
    b_max = 1 + Rcomp / rE
    b_mask = b <= b_max

    rE = rE[b_mask]
    b = b[b_mask]
    a_min, a_max = calc_a(rE, b=b, e=e, periapsis_phase=periapsis_phase)

    PDIST_SAMPLES = 1000

    # porb_min = numpy.round(
    #     calc_porb(a_min, b, rE, Rcomp, tau_eff).to(units.day).value, 4
    # )
    # porb_max = numpy.round(
    #     calc_porb(a_max, b, rE, Rcomp, tau_eff).to(units.day).value, 4
    # )
    # porbs = numpy.linspace(porb_min, porb_max, PDIST_SAMPLES)

    total = len(a_min)
    pdist = numpy.array([])
    porbs = [] * units.day
    for i, (_a_min, _a_max, _rE, _b) in enumerate(zip(a_min, a_max, rE, b)):
        print(f"\r{i} / {total}", end="")
        # print(_a_min, _a_max, _rE, _b)
        a_samples = (
            numpy.logspace(
                numpy.log10(_a_min.value), numpy.log10(_a_max.value), PDIST_SAMPLES
            )
            * _a_min.unit
        )
        # Rework lens_mass_pdist to use a differential density of the mass distribution, and to sample that distribution
        # at given mass ranges (in other words just normalise the value, but actually it won't make a difference now since
        # # the bins are all the smae width).
        # Then use those samples to weight the calculated porbs.
        # Then change the bin_prob calculation below to just sum the weighted porbs.
        # Finally return a differential density distribution for the porbs.
        mass_samples = calc_mass(_rE, a_samples, b=_b, e=e, periapsis_phase=periapsis_phase)
        new_pdist = lens_mass_pdist(mass_samples)
        pdist = numpy.append(pdist, new_pdist)
        porbs = numpy.append(porbs, calc_porb(a_samples, _b, _rE, Rcomp, tau_eff,
                                                e=e, periapsis_phase=periapsis_phase))
    # print(len(pdist), len(numpy.unique(pdist)), pdist[:100])
    print(porbs)

    finite = numpy.isfinite(porbs)
    porbs, pdist = porbs[finite], pdist[finite]
    porb_min_bin = numpy.floor(numpy.log10(numpy.min(porbs).value)) * porbs.unit
    porb_max_bin = numpy.ceil(numpy.log10(numpy.max(porbs).value)) * porbs.unit
    porb_bins = numpy.logspace(porb_min_bin.value, porb_max_bin.value, 100) * porbs.unit
    bin_starts, bin_ends, bin_counts = [], [], []

    for start, end in zip(porb_bins[:-1], porb_bins[1:]):
        bin_filter = numpy.where((porbs < end) & (porbs >= start))[0]
        # bin_prob = numpy.sum(pdist[bin_filter]) - numpy.prod(pdist[bin_filter])
        bin_count = numpy.sum(pdist[bin_filter])
        bin_starts.append(start)
        bin_ends.append(end)
        bin_counts.append(bin_count)
    bin_counts = list(numpy.array(bin_counts) / sum(bin_counts))
    return bin_starts, bin_ends, bin_counts


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
