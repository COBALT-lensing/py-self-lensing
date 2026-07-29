"""Matched-filter searches over a bank of light-curve templates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy import signal


@dataclass(frozen=True)
class MatchedFilterTemplate:
    """A sampled template for matched-filter searches.

    The template is resampled from ``time`` onto the median cadence of the target
    light curve before filtering.
    """

    flux: np.ndarray
    time: np.ndarray


@dataclass(frozen=True)
class MatchedFilterMatch:
    """A matched-filter detection mapped back onto the input light curve."""

    template_index: int
    peak_index: int
    start_index: int
    end_index: int
    peak_time: float
    start_time: float
    end_time: float
    score: float
    prominence: float


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


def _normalise_signal(values: np.ndarray, name: str) -> np.ndarray:
    centred = values - np.mean(values)
    energy = np.sqrt(np.sum(centred**2))
    if energy <= 0:
        raise ValueError(f"{name} has zero energy after mean subtraction")
    return centred / energy


def _median_cadence(time: np.ndarray, name: str) -> float:
    diffs = np.diff(time)
    if not np.all(diffs > 0):
        raise ValueError(f"{name} must be strictly increasing")
    return float(np.median(diffs))


def _resample_template_to_cadence(
    template_flux: np.ndarray,
    template_time,
    target_cadence: float,
    template_label: str,
) -> np.ndarray:
    template_time_array = _as_float_array(template_time, f"{template_label}.time")
    if len(template_time_array) != len(template_flux):
        raise ValueError(
            f"{template_label}.time and {template_label}.flux differ in length"
        )

    template_cadence = _median_cadence(template_time_array, f"{template_label}.time")
    if np.isclose(template_cadence, target_cadence, rtol=1e-6, atol=0.0):
        return template_flux

    resampled_time = np.arange(
        template_time_array[0],
        template_time_array[-1] + (0.5 * target_cadence),
        target_cadence,
    )
    if len(resampled_time) < 2:
        raise ValueError(f"{template_label} is too short to resample")
    return np.interp(resampled_time, template_time_array, template_flux)


def _robust_scatter(values: np.ndarray) -> float:
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    if mad > 0:
        return 1.4826 * mad
    std = np.std(values)
    return float(std) if std > 0 else 1.0


def _default_prominence(values: np.ndarray, sigma: float) -> float:
    return sigma * _robust_scatter(values)


def _default_height(values: np.ndarray, sigma: float) -> float:
    return float(np.median(values) + sigma * _robust_scatter(values))


def _dedupe_overlapping_matches(
    matches: list[MatchedFilterMatch],
) -> list[MatchedFilterMatch]:
    if not matches:
        return []

    sorted_matches = sorted(matches, key=lambda match: match.start_index)
    deduped: list[MatchedFilterMatch] = []
    current_group = [sorted_matches[0]]
    current_end = sorted_matches[0].end_index

    for match in sorted_matches[1:]:
        if match.start_index <= current_end:
            current_group.append(match)
            current_end = max(current_end, match.end_index)
            continue

        deduped.append(max(current_group, key=lambda match: match.score))
        current_group = [match]
        current_end = match.end_index

    deduped.append(max(current_group, key=lambda match: match.score))
    return sorted(deduped, key=lambda match: match.peak_time)


def detect_matched_filter_peaks(
    time,
    flux,
    templates: Iterable[MatchedFilterTemplate],
    *,
    prominence: float | None = None,
    prominence_sigma: float = 5.0,
    height: float | None = None,
    min_score_sigma: float | None = 8.0,
    distance: int | None = None,
    dedupe_overlaps: bool = True,
    max_matches: int | None = None,
) -> list[MatchedFilterMatch]:
    """Find template-like peaks in a light curve.

    The input flux and each template are mean-subtracted. Templates with their
    own time grid are resampled to the median light-curve cadence. Templates are
    then scaled to unit energy before a valid cross-correlation is computed.
    Peaks in each correlation trace are proposed with
    :func:`scipy.signal.find_peaks`.

    Parameters
    ----------
    time, flux : array-like
        One-dimensional light-curve time and flux arrays. ``astropy.time.Time``
        and quantity-like arrays are accepted.
    templates : iterable
        Template objects with flux and time arrays. Each template is resampled
        to the median cadence of the input light curve.
    prominence : float, optional
        Direct prominence threshold for ``find_peaks``. If omitted, a robust
        per-template threshold is estimated from the matched-filter output.
    prominence_sigma : float
        Robust-sigma multiplier used when ``prominence`` is omitted.
    height : optional
        Direct matched-filter score threshold passed to
        ``scipy.signal.find_peaks``.
    min_score_sigma : float, optional
        Robust-sigma score threshold used when ``height`` is omitted. Set to
        ``None`` to disable the default score threshold.
    distance : optional
        Passed through to ``scipy.signal.find_peaks``.
    dedupe_overlaps : bool
        If true, overlapping matches across the template bank are collapsed to
        the highest-scoring match.
    max_matches : int, optional
        If provided, return only the highest-scoring matches across all
        templates.
    """

    time_array = _as_float_array(time, "time")
    flux_array = _as_float_array(flux, "flux")
    if len(time_array) != len(flux_array):
        raise ValueError("time and flux must have the same length")
    target_cadence = _median_cadence(time_array, "time")

    search_flux = flux_array - np.mean(flux_array)
    matches: list[MatchedFilterMatch] = []

    for template_index, template in enumerate(templates):
        if not isinstance(template, MatchedFilterTemplate):
            raise TypeError("templates must contain MatchedFilterTemplate objects")
        raw_template_flux = _as_float_array(
            template.flux,
            f"templates[{template_index}].flux",
        )
        raw_template_flux = _resample_template_to_cadence(
            raw_template_flux,
            template.time,
            target_cadence,
            f"templates[{template_index}]",
        )
        template_len = len(raw_template_flux)
        if template_len > len(search_flux):
            raise ValueError(
                f"templates[{template_index}] is longer than the light curve"
            )
        template_flux = _normalise_signal(
            raw_template_flux,
            f"templates[{template_index}].flux",
        )

        matched_filter_output = signal.correlate(
            search_flux,
            template_flux,
            mode="valid",
            method="auto",
        )
        peak_prominence = (
            prominence
            if prominence is not None
            else _default_prominence(matched_filter_output, prominence_sigma)
        )
        peak_height = (
            height
            if height is not None or min_score_sigma is None
            else _default_height(matched_filter_output, min_score_sigma)
        )
        peak_indices, properties = signal.find_peaks(
            matched_filter_output,
            prominence=peak_prominence,
            height=peak_height,
            distance=distance,
        )
        prominences = properties.get("prominences", np.zeros_like(peak_indices))
        template_center = template_len // 2

        for peak_index, peak_prominence_value in zip(peak_indices, prominences):
            start_index = int(peak_index)
            end_index = start_index + template_len - 1
            center_index = start_index + template_center
            matches.append(
                MatchedFilterMatch(
                    template_index=template_index,
                    peak_index=center_index,
                    start_index=start_index,
                    end_index=end_index,
                    peak_time=float(time_array[center_index]),
                    start_time=float(time_array[start_index]),
                    end_time=float(time_array[end_index]),
                    score=float(matched_filter_output[peak_index]),
                    prominence=float(peak_prominence_value),
                )
            )

    if dedupe_overlaps:
        matches = _dedupe_overlapping_matches(matches)

    matches = sorted(matches, key=lambda match: match.score, reverse=True)
    if max_matches is not None:
        matches = matches[:max_matches]
    return sorted(matches, key=lambda match: match.peak_time)
