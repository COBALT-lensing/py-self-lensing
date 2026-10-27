"""Estimating self-lensing model parameters directly from observables."""


import jax
import jax.numpy as jnp
import numpy
import pandas
from selflensing.models import DEFAULT_LIMITS

from selflensing.batch import make_batched_light_curve_generator
from selflensing.gp.PhysicalRelations import effective_temperature_K, stellar_radius_Rsun
from selflensing.lightcurves import PARAMETER_ORDER
from selflensing.orbits import (
    AU_M,
    DAY_S,
    M_SUN_KG,
    R_SUN_M,
    einstein_radius_m,
    orbital_period_days,
)


OBSERVABLE_SPACE_SAMPLE_SIZE = 10_000
OBSERVABLE_SPACE_RANDOM_SEED = 0


def weighted_mean_and_std(values, weights):
    """Return the weighted mean and local weighted dispersion of values."""
    mean = numpy.average(values, weights=weights)
    std = numpy.sqrt(numpy.average((values - mean) ** 2, weights=weights))
    return mean, std


def companion_stellar_properties(
    companion_mass_msun,
    companion_radius_rsun=None,
    companion_teff_k=None,
):
    """Return companion properties, deriving main-sequence values when omitted."""
    if companion_radius_rsun is None:
        companion_radius_rsun = float(stellar_radius_Rsun(companion_mass_msun))
    if companion_teff_k is None:
        companion_teff_k = float(effective_temperature_K(companion_mass_msun))
    return companion_mass_msun, companion_radius_rsun, companion_teff_k


def calc_self_lensing_observable_space(
    sample_size=OBSERVABLE_SPACE_SAMPLE_SIZE,
    random_seed=OBSERVABLE_SPACE_RANDOM_SEED,
    *,
    companion_mass_msun=1.0,
    companion_radius_rsun=None,
    companion_teff_k=None,
):
    """Sample self-lensing systems for the supplied companion-star properties."""
    (
        companion_mass_msun,
        companion_radius_rsun,
        companion_teff_k,
    ) = companion_stellar_properties(
        companion_mass_msun,
        companion_radius_rsun,
        companion_teff_k,
    )

    key = jax.random.key(random_seed)
    keys = jax.random.split(key, 5)

    def uniform(draw_key, low, high):
        return jax.random.uniform(draw_key, (sample_size,), minval=low, maxval=high)

    def log_uniform(draw_key, low, high):
        return 10 ** uniform(draw_key, numpy.log10(low), numpy.log10(high))

    semi_major_axis_au = log_uniform(keys[0], *DEFAULT_LIMITS["a"])
    lens_mass_msun = log_uniform(keys[1], *DEFAULT_LIMITS["Macc"])
    companion_mass = jnp.full(sample_size, companion_mass_msun)
    companion_radius = jnp.full(sample_size, companion_radius_rsun)
    companion_teff = jnp.full(sample_size, companion_teff_k)
    eccentricity = uniform(keys[2], *DEFAULT_LIMITS["e"])
    periapsis_phase = uniform(keys[3], 0.0, 2.0 * numpy.pi)
    impact_fraction = uniform(keys[4], 0.0, 0.95)

    semi_major_axis_m = semi_major_axis_au * AU_M
    lens_mass_kg = lens_mass_msun * M_SUN_KG
    companion_mass_kg = companion_mass * M_SUN_KG
    companion_radius_m = companion_radius * R_SUN_M

    edge_on_einstein_radius_m = einstein_radius_m(
        semi_major_axis_m,
        jnp.ones(sample_size),
        lens_mass_kg,
        eccentricity,
        periapsis_phase,
    )
    maximum_impact = 1.0 + companion_radius_m / edge_on_einstein_radius_m
    impact_einstein_radii = impact_fraction * maximum_impact
    cos_inclination = jnp.clip(
        impact_einstein_radii * edge_on_einstein_radius_m / semi_major_axis_m,
        0.0,
        0.2,
    )
    inclination = jnp.arccos(cos_inclination)

    param_matrix = jnp.stack(
        [
            {
                "a": semi_major_axis_au,
                "i": inclination,
                "Macc": lens_mass_msun,
                "Mcomp": companion_mass,
                "Rcomp": companion_radius,
                "Teff": companion_teff,
                "e": eccentricity,
                "periapsis_phase": periapsis_phase,
            }[name]
            for name in PARAMETER_ORDER
        ],
        axis=-1,
    )

    sin_inclination = jnp.sin(inclination)
    orbital_period_days_value = orbital_period_days(
        semi_major_axis_m,
        lens_mass_kg + companion_mass_kg,
    )
    orbital_period_seconds = orbital_period_days_value * DAY_S
    einstein_radius_m_value = einstein_radius_m(
        semi_major_axis_m,
        sin_inclination,
        lens_mass_kg,
        eccentricity,
        periapsis_phase,
    )
    impact_parameter = jnp.abs(
        semi_major_axis_m * jnp.cos(inclination) / einstein_radius_m_value
    )
    maximum_impact = 1.0 + companion_radius_m / einstein_radius_m_value
    crossing_time_days = (
        (orbital_period_seconds * (einstein_radius_m_value + companion_radius_m))
        / (jnp.pi * semi_major_axis_m * sin_inclination)
        * jnp.sqrt(
            jnp.clip(1.0 - (impact_parameter / maximum_impact) ** 2, 0.0, 1.0)
        )
    ) / DAY_S

    generate = make_batched_light_curve_generator(
        jnp.array([0.0]),
        peak_times=0.0,
        sl=True,
        ev=False,
        db=False,
        device=jax.devices()[0],
    )
    peak_flux = numpy.asarray(generate(param_matrix).block_until_ready())[:, 0]
    peak_amplification_ppm = 1e6 * (peak_flux - 1.0)
    crossing_time_hr = numpy.asarray(crossing_time_days) * 24.0
    orbital_period_yr = numpy.asarray(orbital_period_days_value) / 365.25
    semi_major_axis = numpy.asarray(semi_major_axis_au)
    lens_mass = numpy.asarray(lens_mass_msun)
    eccentricity = numpy.asarray(eccentricity)
    periapsis_phase = numpy.asarray(periapsis_phase)
    impact_fraction = numpy.asarray(impact_fraction)
    inclination = numpy.asarray(inclination)

    valid = (
        numpy.isfinite(peak_amplification_ppm)
        & numpy.isfinite(crossing_time_hr)
        & numpy.isfinite(orbital_period_yr)
        & (peak_amplification_ppm > 1.0)
        & (crossing_time_hr > 0.0)
        & (orbital_period_yr > 0.0)
        & (inclination >= DEFAULT_LIMITS["i"][0])
        & (inclination <= DEFAULT_LIMITS["i"][1])
    )
    if not valid.any():
        raise ValueError("No valid self-lensing observable-space samples were generated")

    return {
        "sample_size": sample_size,
        "random_seed": random_seed,
        "companion_mass_msun": companion_mass_msun,
        "companion_radius_rsun": companion_radius_rsun,
        "companion_teff_k": companion_teff_k,
        "peak_amplification_ppm": peak_amplification_ppm,
        "crossing_time_hr": crossing_time_hr,
        "orbital_period_yr": orbital_period_yr,
        "semi_major_axis_au": semi_major_axis,
        "lens_mass_msun": lens_mass,
        "eccentricity": eccentricity,
        "periapsis_phase_rad": periapsis_phase,
        "impact_fraction": impact_fraction,
        "inclination_rad": inclination,
        "valid": valid,
    }


def nearest_self_lensing_observable_space(
    peak_amplification_ppm,
    crossing_time_hr,
    *,
    n_neighbors=50,
    sample_size=OBSERVABLE_SPACE_SAMPLE_SIZE,
    random_seed=OBSERVABLE_SPACE_RANDOM_SEED,
    companion_mass_msun=1.0,
    companion_radius_rsun=None,
    companion_teff_k=None,
):
    """Return the sampled systems nearest an observed self-lensing signal.

    Distances are Euclidean in log10 peak-amplification and log10 crossing-time
    space, giving both observable axes equal weight across their dynamic ranges.
    """
    if not numpy.isfinite(peak_amplification_ppm) or not numpy.isfinite(crossing_time_hr) or peak_amplification_ppm <= 0.0 or crossing_time_hr <= 0.0:
        raise ValueError("Peak amplification and crossing time must be positive")
    if n_neighbors < 1:
        raise ValueError("n_neighbors must be at least one")

    estimates = calc_self_lensing_observable_space(
        sample_size=sample_size,
        random_seed=random_seed,
        companion_mass_msun=companion_mass_msun,
        companion_radius_rsun=companion_radius_rsun,
        companion_teff_k=companion_teff_k,
    )
    valid_indices = numpy.flatnonzero(estimates["valid"])
    neighbor_count = min(n_neighbors, len(valid_indices))
    log_distance = numpy.hypot(
        numpy.log10(estimates["peak_amplification_ppm"][valid_indices])
        - numpy.log10(peak_amplification_ppm),
        numpy.log10(estimates["crossing_time_hr"][valid_indices])
        - numpy.log10(crossing_time_hr),
    )
    nearest_indices = valid_indices[numpy.argsort(log_distance)[:neighbor_count]]

    neighbor_columns = (
        "peak_amplification_ppm",
        "crossing_time_hr",
        "semi_major_axis_au",
        "lens_mass_msun",
        "orbital_period_yr",
        "eccentricity",
        "periapsis_phase_rad",
        "impact_fraction",
        "inclination_rad",
    )
    neighbors = pandas.DataFrame(
        {column: estimates[column][nearest_indices] for column in neighbor_columns}
    )
    neighbors.insert(0, "observable_space_index", nearest_indices)
    neighbors["log_observable_distance"] = numpy.sort(log_distance)[:neighbor_count]
    neighbors.attrs.update(
        {
            "target_peak_amplification_ppm": peak_amplification_ppm,
            "target_crossing_time_hr": crossing_time_hr,
            "random_seed": random_seed,
            "companion_mass_msun": companion_mass_msun,
            "companion_radius_rsun": companion_radius_rsun,
            "companion_teff_k": companion_teff_k,
        }
    )
    return neighbors



def _seed_features(candidates):
    """Coordinates for clustering observable-space candidates into modes."""
    phase = candidates["periapsis_phase_rad"].to_numpy(dtype=float)
    features = numpy.column_stack(
        (
            numpy.log(candidates["semi_major_axis_au"].to_numpy(dtype=float)),
            candidates["inclination_rad"].to_numpy(dtype=float),
            numpy.log(candidates["lens_mass_msun"].to_numpy(dtype=float)),
            candidates["eccentricity"].to_numpy(dtype=float),
            numpy.sin(phase),
            numpy.cos(phase),
        )
    )
    scale = numpy.maximum(numpy.std(features, axis=0), 1.0e-8)
    return (features - numpy.mean(features, axis=0)) / scale


def _observable_seed_modes(candidates, count):
    """Return medoid candidates from distinct physically plausible regions."""
    count = min(int(count), len(candidates))
    if count < 1:
        raise ValueError("Observable-space seeding found no candidate systems")
    features = _seed_features(candidates)
    # Farthest-point starts avoid collapsing all walkers onto the nearest kNN
    # cloud; Lloyd iterations then identify dense parameter-space modes.
    centres = [0]
    minimum_distance = numpy.sum((features - features[0]) ** 2, axis=1)
    for _ in range(1, count):
        next_index = int(numpy.argmax(minimum_distance))
        centres.append(next_index)
        minimum_distance = numpy.minimum(
            minimum_distance, numpy.sum((features - features[next_index]) ** 2, axis=1)
        )
    centres = features[numpy.asarray(centres)]
    labels = numpy.zeros(len(features), dtype=int)
    for _ in range(20):
        distances = numpy.sum((features[:, None, :] - centres[None, :, :]) ** 2, axis=2)
        labels = numpy.argmin(distances, axis=1)
        next_centres = numpy.array(
            [
                features[labels == index].mean(axis=0)
                if numpy.any(labels == index) else centres[index]
                for index in range(count)
            ]
        )
        if numpy.allclose(next_centres, centres):
            break
        centres = next_centres
    medoids = []
    for index in range(count):
        members = numpy.flatnonzero(labels == index)
        if not len(members):
            continue
        medoids.append(members[numpy.argmin(numpy.sum((features[members] - centres[index]) ** 2, axis=1))])
    return candidates.iloc[medoids].reset_index(drop=True)



def observable_seed_positions(peak_amplification_ppm, crossing_time_hr, *,
                              nwalkers=64, n_neighbors=256, mode_count=8,
                              sample_size=10000, random_seed=0,
                              jitter_fraction=.001, **stellar_parameters):
    """Seed distinct physical modes near event amplitude and duration estimates.

    Observables guide initialization only; they do not impose a likelihood or
    prior. Returns physical (a, i, Macc, e, periapsis_phase) coordinates.
    """
    if nwalkers < 1 or mode_count < 1 or not numpy.isfinite(jitter_fraction) or jitter_fraction < 0:
        raise ValueError('Walker/mode counts must be positive and jitter nonnegative')
    candidates = nearest_self_lensing_observable_space(
        peak_amplification_ppm, crossing_time_hr, n_neighbors=n_neighbors,
        sample_size=sample_size, random_seed=random_seed, **stellar_parameters)
    modes = _observable_seed_modes(candidates, mode_count)
    coords = modes[["semi_major_axis_au", "inclination_rad", "lens_mass_msun",
                    "eccentricity", "periapsis_phase_rad"]].to_numpy(dtype=float)
    names = ('a', 'i', 'Macc', 'e', 'periapsis_phase')
    limits = numpy.array([DEFAULT_LIMITS[name] for name in names])
    rng = numpy.random.default_rng(random_seed)
    positions = coords[numpy.arange(nwalkers) % len(coords)].copy()
    positions += rng.normal(size=positions.shape) * jitter_fraction * (limits[:, 1] - limits[:, 0])
    positions = numpy.clip(positions, limits[:, 0], limits[:, 1])
    return positions
