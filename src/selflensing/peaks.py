"""Peak detection utilities for noisy light curves.

The detector is a two-stage procedure:

1. Use ``scipy.signal.find_peaks`` to propose cheap local-maximum candidates.
2. Re-score each candidate with a local weighted least-squares model comparison:
   a slanted baseline versus the same baseline plus a positive Gaussian pulse.

This keeps the first pass permissive while making the final score insensitive to
a local linear continuum slope.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize, signal, stats


# Median absolute deviation must be multiplied by this factor to estimate the
# standard deviation of Gaussian noise. This gives a robust scatter estimate that
# is not pulled upward too strongly by real peaks.
MAD_NORMAL_SCALE = 1.4826

# If the caller does not provide template widths, scan a compact log-spaced-ish
# bank in units of the median cadence. Two cadences catches barely resolved
# features; 32 cadences catches broad peaks without making the default scan too
# expensive or too prone to fitting long baseline structure.
DEFAULT_WIDTH_CADENCES = np.array([2.0, 4.0, 8.0, 16.0, 32.0])

# ``find_peaks`` is only a candidate generator. A 2.5 robust-sigma prominence is
# deliberately loose: it should admit plausible low-S/N candidates and let the
# likelihood-ratio score reject them later.
DEFAULT_PROMINENCE_SIGMA = 2.5

# The local fitting window is +/- this many template widths. Five Gaussian
# sigmas contains effectively all of the pulse while leaving enough off-peak data
# to estimate the intercept and slope of a local baseline.
DEFAULT_WINDOW_WIDTHS = 5.0

# The local model has three parameters: intercept, slope, and positive amplitude.
# Five points is the bare minimum that leaves more data than parameters and gives
# the fit a little leverage on both sides of the candidate.
MIN_LOCAL_FIT_POINTS = 5

# Default detection cuts. ``delta_chi2=25`` is roughly a 5-sigma fixed-candidate
# improvement for one extra amplitude parameter, while ``amplitude_snr=5`` keeps
# the fitted pulse amplitude itself significant. Because the code scans candidate
# locations and widths, these are screening thresholds rather than final global
# false-alarm probabilities.
DEFAULT_MIN_DELTA_CHI2 = 25.0
DEFAULT_MIN_AMPLITUDE_SNR = 5.0


@dataclass(frozen=True)
class PeakDetection:
    """Summary of a local baseline-aware peak detection."""

    index: int
    time: float
    width: float
    amplitude: float
    amplitude_err: float
    amplitude_snr: float
    delta_chi2: float
    p_value: float
    baseline_intercept: float
    baseline_slope: float
    window_start: float
    window_end: float
    n_points: int


def _as_float_array(values, name: str) -> np.ndarray:
    if hasattr(values, "jd"):
        values = values.jd
    elif hasattr(values, "value"):
        values = values.value

    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def _robust_sigma(values: np.ndarray) -> float:
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    if mad > 0:
        return MAD_NORMAL_SCALE * mad
    std = np.std(values)
    return float(std) if std > 0 else 1.0


def _default_widths(time: np.ndarray) -> np.ndarray:
    cadence = np.median(np.diff(np.sort(time)))
    if cadence <= 0:
        raise ValueError("time values must not all be identical")
    return cadence * DEFAULT_WIDTH_CADENCES


def _gaussian_template(centered_time: np.ndarray, width: float) -> np.ndarray:
    return np.exp(-0.5 * (centered_time / width) ** 2)


def _fit_candidate(
    time: np.ndarray,
    flux: np.ndarray,
    sigma: np.ndarray,
    center: float,
    width: float,
    window_widths: float,
) -> PeakDetection | None:
    half_window = window_widths * width
    in_window = np.abs(time - center) <= half_window
    if np.count_nonzero(in_window) < MIN_LOCAL_FIT_POINTS:
        return None

    local_time = time[in_window]
    local_flux = flux[in_window]
    local_sigma = sigma[in_window]
    centered_time = local_time - center
    template = _gaussian_template(centered_time, width)

    # Start from the local median rather than the candidate height so the first
    # baseline guess is not biased by the peak we are trying to test.
    intercept0 = np.median(local_flux)
    slope0 = 0.0
    amplitude0 = max(float(local_flux.max() - intercept0), 0.0)

    def null_residual(params):
        intercept, slope = params
        model = intercept + slope * centered_time
        return (local_flux - model) / local_sigma

    def peak_residual(params):
        intercept, slope, amplitude = params
        model = intercept + slope * centered_time + amplitude * template
        return (local_flux - model) / local_sigma

    null_fit = optimize.least_squares(null_residual, x0=np.array([intercept0, slope0]))
    peak_fit = optimize.least_squares(
        peak_residual,
        x0=np.array([intercept0, slope0, amplitude0]),
        bounds=([-np.inf, -np.inf, 0.0], [np.inf, np.inf, np.inf]),
    )
    if not null_fit.success or not peak_fit.success:
        return None

    chi2_null = float(np.sum(null_fit.fun**2))
    chi2_peak = float(np.sum(peak_fit.fun**2))
    delta_chi2 = max(chi2_null - chi2_peak, 0.0)

    amplitude = float(peak_fit.x[2])
    amplitude_err = np.nan
    amplitude_snr = np.nan
    try:
        # ``least_squares`` returns the Jacobian of normalized residuals, so the
        # inverse normal matrix estimates the parameter covariance when the
        # supplied sigmas are meaningful. The pseudo-inverse keeps nearly
        # singular local windows from crashing candidate evaluation.
        cov = np.linalg.pinv(peak_fit.jac.T @ peak_fit.jac)
        amplitude_err = float(np.sqrt(cov[2, 2]))
        if amplitude_err > 0:
            amplitude_snr = amplitude / amplitude_err
    except np.linalg.LinAlgError:
        pass

    # The p-value treats the center and width as fixed and uses one degree of
    # freedom for the extra amplitude parameter. Because the detector scans many
    # candidates and widths, use simulations or bootstrap max-statistics for
    # final global false-alarm probabilities.
    p_value = float(stats.chi2.sf(delta_chi2, df=1))

    center_index = int(np.argmin(np.abs(time - center)))
    return PeakDetection(
        index=center_index,
        time=float(time[center_index]),
        width=float(width),
        amplitude=amplitude,
        amplitude_err=amplitude_err,
        amplitude_snr=float(amplitude_snr),
        delta_chi2=delta_chi2,
        p_value=p_value,
        baseline_intercept=float(peak_fit.x[0]),
        baseline_slope=float(peak_fit.x[1]),
        window_start=float(local_time.min()),
        window_end=float(local_time.max()),
        n_points=int(local_time.size),
    )


def dedupe_overlapping_peaks(
    detections: list[PeakDetection],
) -> list[PeakDetection]:
    """Keep the widest detection from each overlapping set of peak windows."""

    if not detections:
        return []

    # Sort by interval start and maintain the rightmost end of the current group.
    # This merges transitive overlaps too: A overlaps B and B overlaps C means all
    # three are one candidate event even if A does not directly overlap C.
    sorted_detections = sorted(detections, key=lambda fit: fit.window_start)
    deduped: list[PeakDetection] = []
    current_group = [sorted_detections[0]]
    current_end = sorted_detections[0].window_end

    for detection in sorted_detections[1:]:
        if detection.window_start <= current_end:
            current_group.append(detection)
            current_end = max(current_end, detection.window_end)
            continue

        deduped.append(
            max(current_group, key=lambda fit: (fit.width, fit.delta_chi2))
        )
        current_group = [detection]
        current_end = detection.window_end

    deduped.append(max(current_group, key=lambda fit: (fit.width, fit.delta_chi2)))
    return sorted(deduped, key=lambda fit: fit.time)


def detect_peaks(
    time,
    flux,
    sigma=None,
    *,
    widths=None,
    candidate_prominence=None,
    candidate_distance=None,
    candidate_height=None,
    window_widths: float = DEFAULT_WINDOW_WIDTHS,
    min_delta_chi2: float = DEFAULT_MIN_DELTA_CHI2,
    min_amplitude_snr: float = DEFAULT_MIN_AMPLITUDE_SNR,
    dedupe_overlaps: bool = True,
    max_peaks: int | None = None,
) -> list[PeakDetection]:
    """Detect positive peaks against a local linear baseline.

    Candidate locations are proposed with :func:`scipy.signal.find_peaks`.
    Each candidate is then scored by a likelihood-ratio-style comparison between
    a weighted local linear baseline and the same baseline plus a positive
    Gaussian peak template.

    Parameters
    ----------
    time, flux : array-like
        One-dimensional time and flux arrays. Astropy ``Time`` and quantity-like
        arrays are accepted.
    sigma : array-like, optional
        One-sigma uncertainty for each flux value. If omitted, a robust scatter
        estimate is used for every point.
    widths : array-like, optional
        Gaussian template widths, in the same units as ``time``. By default, the
        scan uses 2, 4, 8, 16, and 32 cadences.
    candidate_prominence, candidate_distance, candidate_height : optional
        Passed through to :func:`scipy.signal.find_peaks`. If prominence is not
        given, a loose ``2.5 * robust_sigma(flux)`` threshold is used.
    window_widths : float
        Half-window size in units of the trial width.
    min_delta_chi2 : float
        Minimum improvement over the local linear null model.
    min_amplitude_snr : float
        Minimum fitted amplitude divided by its linearized uncertainty.
    dedupe_overlaps : bool
        If true, overlapping detection windows are grouped and only the widest
        detection from each group is returned.
    max_peaks : int, optional
        Return only the strongest detections by ``delta_chi2``.
    """

    time = _as_float_array(time, "time")
    flux = _as_float_array(flux, "flux")
    if time.size != flux.size:
        raise ValueError("time and flux must have the same length")
    if time.size < 5:
        return []

    if sigma is None:
        sigma = np.full_like(flux, _robust_sigma(flux))
    else:
        sigma = _as_float_array(sigma, "sigma")
        if sigma.size != flux.size:
            raise ValueError("sigma must have the same length as flux")
        if np.any(sigma <= 0):
            raise ValueError("sigma values must be positive")

    widths = _default_widths(time) if widths is None else _as_float_array(widths, "widths")
    if np.any(widths <= 0):
        raise ValueError("widths must be positive")
    if window_widths <= 0:
        raise ValueError("window_widths must be positive")

    if candidate_prominence is None:
        candidate_prominence = DEFAULT_PROMINENCE_SIGMA * _robust_sigma(flux)

    candidate_indices, _ = signal.find_peaks(
        flux,
        height=candidate_height,
        prominence=candidate_prominence,
        distance=candidate_distance,
    )

    detections: list[PeakDetection] = []
    for candidate_index in candidate_indices:
        center = time[candidate_index]
        fits = [
            fit
            for width in widths
            if (
                fit := _fit_candidate(
                    time=time,
                    flux=flux,
                    sigma=sigma,
                    center=center,
                    width=float(width),
                    window_widths=window_widths,
                )
            )
            is not None
        ]
        if not fits:
            continue

        # The same local maximum may fit several template widths. Keep the width
        # with the strongest local likelihood-ratio improvement, then let the
        # optional overlap deduper collapse neighbouring candidate maxima.
        best_fit = max(fits, key=lambda fit: fit.delta_chi2)
        if (
            best_fit.delta_chi2 >= min_delta_chi2
            and best_fit.amplitude_snr >= min_amplitude_snr
        ):
            detections.append(best_fit)

    if dedupe_overlaps:
        detections = dedupe_overlapping_peaks(detections)

    detections.sort(key=lambda fit: fit.delta_chi2, reverse=True)
    if max_peaks is not None:
        detections = detections[:max_peaks]
    return sorted(detections, key=lambda fit: fit.time)
