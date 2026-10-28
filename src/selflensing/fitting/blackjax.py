"""BlackJAX fitting backends."""

import logging
from dataclasses import dataclass

import jax
import jax.numpy as jax_numpy
import numpy
from astropy import units
from astropy.constants import G
from quaxed import numpy as jnp
from tqdm import tqdm

from ..generators import SelfLensingLightcurveGenerator
from ..models import SelfLensingSystem, DEFAULT_LIMITS
from ._utils import _select_fit_device
from .geometry import (b_source_scale, physical_to_geometry, geometry_to_physical, validate_geometry)
from .likelihood import (
    LIKELIHOOD_CONVENTION,
    make_batched_numeric_combined_chi2_vec_jax,
    make_combined_chi2_vec_jax,
)


@dataclass
class BlackJAXFitState:
    coords: numpy.ndarray
    parameterization: str = "semi_major_axis"
    refine_initialization: bool = False
    inclination_parameterization: str = "inclination"


class BlackJAXSamplerResult:
    """Small emcee-like adapter for downstream diagnostics."""

    def __init__(self, chain, log_prob, acceptance=None, chain_start_step=0):
        self.chain = numpy.asarray(chain)
        self.log_prob = numpy.asarray(log_prob)
        self.acceptance = None if acceptance is None else numpy.asarray(acceptance)
        self.chain_start_step = int(chain_start_step)
        self.likelihood_convention = LIKELIHOOD_CONVENTION
        self.warmup_acceptance = ()
        self.proposal_scale = None
        self.inclination_parameterization = "inclination"

    @property
    def iteration(self):
        return len(self.chain)

    @property
    def acceptance_fraction(self):
        if self.acceptance is None or self.acceptance.size == 0:
            return numpy.full(self.chain.shape[1], numpy.nan)
        return numpy.mean(self.acceptance, axis=0)

    def get_chain(self, flat=False, discard=0, thin=1):
        discard = max(0, int(discard) - self.chain_start_step)
        chain = self.chain[discard::thin]
        if flat:
            return chain.reshape((-1, chain.shape[-1]))
        return chain

    def get_log_prob(self, flat=False, discard=0, thin=1):
        discard = max(0, int(discard) - self.chain_start_step)
        log_prob = self.log_prob[discard::thin]
        if flat:
            return log_prob.reshape((-1,))
        return log_prob


DEFAULT_PROPOSAL_REFERENCE_OBSERVATIONS = 13.0
DEFAULT_POSTERIOR_SUMMARY_TOP_K = 256
PHASE_INDEX = 4
DEFAULT_WARMUP_MAX_STEPS = 500
DEFAULT_WARMUP_ROUNDS = 8
# The optimal random-walk Metropolis acceptance rate is near 0.234 in a
# moderately correlated multi-parameter posterior.
DEFAULT_TARGET_ACCEPTANCE = 0.234


def _initial_walker_positions(
    nwalkers, initial_walker_state=None, peak_time_prior_sigma_days=None, rng=None
):
    rng = numpy.random if rng is None else rng
    ndim = 6 if peak_time_prior_sigma_days is not None else 5
    if initial_walker_state is not None:
        initial_walker_coords = getattr(
            initial_walker_state,
            "coords",
            initial_walker_state,
        )
        initial_walker_coords = numpy.asarray(initial_walker_coords)
        if (peak_time_prior_sigma_days is not None
                and initial_walker_coords.ndim == 2
                and initial_walker_coords.shape[1] == 5):
            offsets = rng.normal(
                0.0, peak_time_prior_sigma_days * 0.25,
                size=(initial_walker_coords.shape[0], 1),
            )
            initial_walker_coords = numpy.concatenate(
                (initial_walker_coords, offsets), axis=1
            )
        if initial_walker_coords.ndim != 2 or initial_walker_coords.shape[1] != ndim:
            raise ValueError(
                "initial_walker_state must have coordinates with shape "
                f"(nwalkers, {ndim})"
            )
        return initial_walker_coords

    p0 = [
        rng.uniform(bounds[0], bounds[1], nwalkers)
        for bounds in (
            DEFAULT_LIMITS["i"],
            DEFAULT_LIMITS["Macc"],
            DEFAULT_LIMITS["e"],
            DEFAULT_LIMITS["periapsis_phase"],
        )
    ]
    p0.insert(
        0,
        jnp.logspace(
            jnp.log10(DEFAULT_LIMITS["a"][0]),
            jnp.log10(DEFAULT_LIMITS["a"][1]),
            nwalkers,
        ),
    )
    positions = numpy.asarray(list(zip(*p0)), dtype=float)
    if peak_time_prior_sigma_days is not None:
        offsets = rng.normal(
            0.0, peak_time_prior_sigma_days * 0.25, size=(nwalkers, 1)
        )
        positions = numpy.concatenate((positions, offsets), axis=1)
    return positions


def _initial_batched_walker_positions(nfits, nwalkers, initial_walker_state=None, rng=None):
    ndim = 5
    if initial_walker_state is not None:
        initial_walker_coords = getattr(
            initial_walker_state,
            "coords",
            initial_walker_state,
        )
        initial_walker_coords = numpy.asarray(initial_walker_coords)
        if initial_walker_coords.ndim == 2:
            if initial_walker_coords.shape[1] != ndim:
                raise ValueError(
                    "initial_walker_state must have coordinates with shape "
                    f"(nwalkers, {ndim}) or (nfits, nwalkers, {ndim})"
                )
            if initial_walker_coords.shape[0] != nwalkers:
                nwalkers = initial_walker_coords.shape[0]
            return numpy.broadcast_to(
                initial_walker_coords[None, :, :],
                (nfits, nwalkers, ndim),
            ).copy()
        if initial_walker_coords.ndim == 3:
            if initial_walker_coords.shape[0] != nfits or initial_walker_coords.shape[2] != ndim:
                raise ValueError(
                    "initial_walker_state must have coordinates with shape "
                    f"(nwalkers, {ndim}) or (nfits, nwalkers, {ndim})"
                )
            return initial_walker_coords
        raise ValueError(
            "initial_walker_state must have coordinates with shape "
            f"(nwalkers, {ndim}) or (nfits, nwalkers, {ndim})"
        )

    return numpy.stack(
        [_initial_walker_positions(nwalkers, rng=rng) for _ in range(nfits)],
        axis=0,
    )


def _default_blackjax_proposal_scale(
    scale_fraction=0.02, peak_time_prior_sigma_days=None
):
    scales = [
            scale_fraction * (DEFAULT_LIMITS["a"][1] - DEFAULT_LIMITS["a"][0]),
            scale_fraction * (DEFAULT_LIMITS["i"][1] - DEFAULT_LIMITS["i"][0]),
            scale_fraction
            * (DEFAULT_LIMITS["Macc"][1] - DEFAULT_LIMITS["Macc"][0]),
            scale_fraction * (DEFAULT_LIMITS["e"][1] - DEFAULT_LIMITS["e"][0]),
            scale_fraction
            * (
                DEFAULT_LIMITS["periapsis_phase"][1]
                - DEFAULT_LIMITS["periapsis_phase"][0]
            ),
        ]
    if peak_time_prior_sigma_days is not None:
        scales.append(0.25 * peak_time_prior_sigma_days)
    return jnp.asarray(scales, dtype=float)


def _coordinate_normal_random_walk(proposal_scale):
    """Build a symmetric proposal that changes one coordinate per step.

    A simultaneous diagonal random walk is especially inefficient when one
    parameter is constrained by a hard physical boundary. Selecting the moved
    coordinate uniformly preserves proposal symmetry and therefore the usual
    Metropolis acceptance rule.
    """
    proposal_scale = jnp.asarray(proposal_scale)

    def propose(rng_key, position):
        step_key, coordinate_key = jax.random.split(rng_key)
        coordinate = jax.random.randint(
            coordinate_key, (), 0, position.shape[-1]
        )
        step = jax.random.normal(step_key, shape=position.shape) * proposal_scale
        return step * jax.nn.one_hot(coordinate, position.shape[-1])

    return propose


def _count_observations(observed, mask=None):
    if observed is None:
        return None
    if mask is None:
        return numpy.asarray(observed).shape[-1]
    return numpy.sum(numpy.asarray(mask, dtype=bool), axis=-1)


def _proposal_observation_counts(observed_rv, rv_mask, observed_lc, lc_mask):
    rv_count = _count_observations(observed_rv, rv_mask)
    if rv_count is not None:
        return rv_count
    lc_count = _count_observations(observed_lc, lc_mask)
    if lc_count is not None:
        return lc_count
    return 1


def _rv_precision_scale(observed_rv_err, rv_mask):
    if observed_rv_err is None:
        return None
    rv_err = numpy.asarray(observed_rv_err, dtype=float)
    valid = numpy.isfinite(rv_err) & (rv_err > 0)
    if rv_mask is not None:
        valid &= numpy.asarray(rv_mask, dtype=bool)
    inverse_variance = numpy.where(valid, 1.0 / numpy.square(rv_err), 0.0)
    valid_count = numpy.sum(valid, axis=-1)
    with numpy.errstate(invalid="ignore", divide="ignore"):
        precision = numpy.sum(inverse_variance, axis=-1) / valid_count
    finite_precision = precision[numpy.isfinite(precision) & (precision > 0)]
    if finite_precision.size == 0:
        return None
    reference_precision = numpy.min(finite_precision)
    scale = numpy.where(
        numpy.isfinite(precision) & (precision > 0),
        precision / reference_precision,
        1.0,
    )
    return numpy.maximum(scale, 1.0)


def _scaled_proposal_scale(
    proposal_scale,
    observed_rv,
    observed_rv_err,
    rv_mask,
    observed_lc,
    lc_mask,
    reference_observations=DEFAULT_PROPOSAL_REFERENCE_OBSERVATIONS,
):
    counts = numpy.asarray(
        _proposal_observation_counts(observed_rv, rv_mask, observed_lc, lc_mask),
        dtype=float,
    )
    counts = numpy.maximum(counts, 1.0)
    if reference_observations is None:
        finite_counts = counts[numpy.isfinite(counts)]
        reference_observations = numpy.min(finite_counts) if finite_counts.size else 1.0
    reference_observations = max(float(reference_observations), 1.0)
    precision_scale = _rv_precision_scale(observed_rv_err, rv_mask)
    effective_counts = counts if precision_scale is None else counts * precision_scale
    factor = numpy.minimum(
        1.0,
        numpy.sqrt(reference_observations / effective_counts),
    )
    proposal_scale = numpy.asarray(proposal_scale, dtype=float)
    if factor.ndim == 0:
        return proposal_scale * float(factor)
    return factor[:, None, None] * proposal_scale[None, None, :]


def _posterior_summary_from_flat_samples(flat_samples, flat_log_prob, fallback_position):
    finite = jnp.isfinite(flat_log_prob)
    count = jnp.sum(finite)
    weights = finite.astype(float)
    safe_count = jnp.maximum(count, 1)
    regular_sum = jnp.sum(flat_samples * weights[:, None], axis=0)
    summary = regular_sum / safe_count
    phase = jnp.mod(
        jnp.arctan2(
            jnp.sum(jnp.sin(flat_samples[:, PHASE_INDEX]) * weights),
            jnp.sum(jnp.cos(flat_samples[:, PHASE_INDEX]) * weights),
        ),
        2 * jnp.pi,
    )
    summary = summary.at[PHASE_INDEX].set(phase)
    return jnp.where(count > 0, summary, fallback_position)


def _posterior_summaries_from_flat_samples(flat_positions, flat_log_prob, fallback_position):
    finite = jnp.isfinite(flat_log_prob)
    counts = jnp.sum(finite, axis=1)
    weights = finite.astype(float)
    safe_counts = jnp.maximum(counts, 1)
    regular_sum = jnp.sum(flat_positions * weights[:, :, None], axis=1)
    summaries = regular_sum / safe_counts[:, None]
    phases = jnp.mod(
        jnp.arctan2(
            jnp.sum(jnp.sin(flat_positions[:, :, PHASE_INDEX]) * weights, axis=1),
            jnp.sum(jnp.cos(flat_positions[:, :, PHASE_INDEX]) * weights, axis=1),
        ),
        2 * jnp.pi,
    )
    summaries = summaries.at[:, PHASE_INDEX].set(phases)
    return jnp.where(counts[:, None] > 0, summaries, fallback_position)


def _update_top_posterior_samples(
    top_positions,
    top_log_prob,
    positions,
    log_prob,
    keep_steps,
):
    flat_positions = positions.reshape((-1, positions.shape[-1]))
    flat_log_prob = jnp.where(
        keep_steps[:, None],
        log_prob,
        -jnp.inf,
    ).reshape((-1,))
    combined_positions = jnp.concatenate((top_positions, flat_positions), axis=0)
    combined_log_prob = jnp.concatenate((top_log_prob, flat_log_prob), axis=0)
    next_log_prob, next_indices = jax.lax.top_k(combined_log_prob, top_log_prob.shape[0])
    next_positions = combined_positions[next_indices]
    return next_positions, next_log_prob


def _update_batched_top_posterior_samples(
    top_positions,
    top_log_prob,
    positions,
    log_prob,
    keep_steps,
):
    flat_positions = jnp.transpose(positions, (1, 0, 2, 3)).reshape(
        (top_positions.shape[0], -1, positions.shape[-1])
    )
    flat_log_prob = jnp.where(
        keep_steps[None, :, None],
        jnp.transpose(log_prob, (1, 0, 2)),
        -jnp.inf,
    ).reshape((top_positions.shape[0], -1))
    combined_positions = jnp.concatenate((top_positions, flat_positions), axis=1)
    combined_log_prob = jnp.concatenate((top_log_prob, flat_log_prob), axis=1)
    next_log_prob, next_indices = jax.lax.top_k(
        combined_log_prob,
        top_log_prob.shape[1],
    )
    next_positions = jnp.take_along_axis(
        combined_positions,
        next_indices[:, :, None],
        axis=1,
    )
    return next_positions, next_log_prob


def _best_fit_system_from_samples(flat_samples, flat_log_prob, Rcomp, Mcomp, Teff, d):
    res = flat_samples[jnp.argmax(flat_log_prob)]
    out_a, out_i, out_Macc, out_e, out_periapsis_phase = res[:5]
    return SelfLensingSystem(
        a=out_a * SelfLensingSystem.param_units["a"],
        i=out_i * SelfLensingSystem.param_units["i"],
        Macc=out_Macc * SelfLensingSystem.param_units["Macc"],
        Rcomp=Rcomp,
        Mcomp=Mcomp,
        Teff=Teff,
        e=out_e * SelfLensingSystem.param_units["e"],
        periapsis_phase=out_periapsis_phase
        * SelfLensingSystem.param_units["periapsis_phase"],
        d=d,
    )


def _best_fit_systems_from_positions(positions, Rcomp, Mcomp, Teff, d):
    systems = []
    for index, res in enumerate(numpy.asarray(positions)):
        out_a, out_i, out_Macc, out_e, out_periapsis_phase = res
        systems.append(
            SelfLensingSystem(
                a=out_a * SelfLensingSystem.param_units["a"],
                i=out_i * SelfLensingSystem.param_units["i"],
                Macc=out_Macc * SelfLensingSystem.param_units["Macc"],
                Rcomp=Rcomp[index],
                Mcomp=Mcomp[index],
                Teff=None if Teff is None else Teff[index],
                e=out_e * SelfLensingSystem.param_units["e"],
                periapsis_phase=out_periapsis_phase
                * SelfLensingSystem.param_units["periapsis_phase"],
                d=d[index],
            )
        )
    return systems


@units.quantity_input
def self_lensing_system_from_fit_blackjax(
    Mcomp: units.kg,
    peak_time,
    rv_times=None,
    observed_rv=None,
    observed_rv_err=None,
    lc_times=None,
    observed_lc=None,
    observed_lc_err=None,
    gp_model=None,
    Rcomp: units.m = None,
    Teff: units.K = None,
    tau_eff: units.day = None,
    d: units.pc = SelfLensingSystem.DEFAULT_D,
    false_alarm_probability=0.1,
    detect_periods=False,
    periods=[],
    period_min: units.day = None,
    period_max: units.day = None,
    weight_lc_peak=False,
    nwalkers=100,
    nsteps=5000,
    burn=500,
    lensing_method=None,
    center_lc_model=False,
    progress=True,
    progress_desc=None,
    prefer_gpu=True,
    device=None,
    initial_walker_state=None,
    proposal_scale=None,
    proposal_scale_fraction=0.02,
    proposal_reference_observations=DEFAULT_PROPOSAL_REFERENCE_OBSERVATIONS,
    random_seed=0,
    progress_chunk_size=100,
    posterior_summary_top_k=DEFAULT_POSTERIOR_SUMMARY_TOP_K,
    return_sampler=True,
    return_walker_state=True,
    peak_time_prior_sigma: units.day = None,
    parameterization="semi_major_axis",
    rv_source_ids=None,
    profile_lc_baseline=False,
    inclination_parameterization="inclination",
    warmup_steps=None,
    warmup_rounds=None,
    adapt_covariance=False,
    warmup_restart=False,
    robust_covariance=False,
    recovery_probability=0.,
    recovery_df=3.,
    recovery_centres_count=16,
    warmup_log_density_cut=20.,
    observable_seed=None,
    independence_proposal=None,
    independence_chunk_size=512,
    independence_coordinates="sampling",
):
    """Fit using a BlackJAX random-walk Metropolis-Hastings backend.

    This is an additive, JAX-native alternative to self_lensing_system_from_fit().
    It keeps walker positions and log-probabilities on the selected JAX device
    during sampling and returns a small emcee-like result adapter.

    Warm-up tunes each coordinate separately (or an optional full covariance).
    ``warmup_steps`` defaults to ``burn``; warm-up is unrecorded. The transform
    is frozen throughout the returned chain. ``warmup_restart`` optionally
    seeds each warm-up block from high-probability states, which can lose modes.

    Limitations:
    - Rcomp must be provided. The default SelfLensingSystem path infers Rcomp
      from Mcomp using Astropy quantities, which is not safe inside the traced
      JAX likelihood.
    - gp_model is not supported yet because the GP likelihood setup still
      expects Python/object-level state that is not carried through this
      device-native sampler path.
    - detect_periods=True is not supported yet. Period detection is a CPU
      preprocessing step in the emcee fitter and has not been split out for
      this backend.
    - There is no stop_on_convergence equivalent yet. BlackJAX runs exactly
      nsteps and leaves convergence checks to downstream diagnostics.
    - This uses independent random-walk Metropolis-Hastings chains. It is the
      smallest safe move away from emcee, not a tuned HMC/NUTS implementation.
    - Set return_sampler=False for production fits that do not need chain
      diagnostics. In that mode a small high-posterior sample reservoir is
      tracked on device and the full chain is not copied back to host memory.
    - Set return_walker_state=False when the caller does not need to seed a
      later fit from the final walker positions. This avoids copying the final
      walker coordinates back from the accelerator.
    """
    import blackjax

    if observed_rv is None and observed_lc is None:
        raise RuntimeError("You must provide at least one of observed_rv, observed_ts")
    if Rcomp is None:
        # SelfLensingSystem can infer Rcomp from Mcomp outside JAX tracing, but
        # that inference multiplies traced values by Astropy units and fails
        # inside the compiled likelihood. Keep this explicit until the model
        # constructor has a fully unitless/JAX-native path.
        raise NotImplementedError(
            "BlackJAX fitting requires Rcomp to be provided; inferring Rcomp from "
            "Mcomp inside the traced likelihood is not JAX-safe"
        )
    if gp_model is not None:
        # The GP path builds generator/GP objects during likelihood evaluation.
        # That object state has not been made into a JAX pytree, so keep it out
        # of the device-native sampler for now.
        raise NotImplementedError("BlackJAX fitting does not currently support gp_model")
    if detect_periods:
        # The emcee fitter performs Lomb-Scargle detection before sampler setup.
        # This backend currently accepts explicit `periods` only, so callers can
        # still do CPU period detection separately if they need those priors.
        raise NotImplementedError(
            "BlackJAX fitting does not currently support detect_periods=True"
        )

    fit_device = _select_fit_device(device=device, prefer_gpu=prefer_gpu)
    if lensing_method is None:
        lensing_method = (
            SelfLensingLightcurveGenerator.default_lensing_method_for_device(fit_device)
        )
    logging.info(
        "blackjax fit: using JAX %s device %s with lensing_method=%s",
        fit_device.platform,
        fit_device,
        lensing_method,
    )

    nwalkers = int(nwalkers)
    nsteps = int(nsteps)
    burn = int(burn)
    peak_time_prior_sigma_days = (
        None
        if peak_time_prior_sigma is None
        else float(peak_time_prior_sigma.to_value(units.day))
    )
    if peak_time_prior_sigma_days is not None and peak_time_prior_sigma_days <= 0:
        raise ValueError("peak_time_prior_sigma must be positive")
    p0 = _initial_walker_positions(
        nwalkers, initial_walker_state, peak_time_prior_sigma_days,
        rng=numpy.random.default_rng(random_seed),
    )
    if observable_seed is not None:
        if initial_walker_state is not None:
            raise ValueError('observable_seed cannot be combined with initial_walker_state')
        if parameterization != 'semi_major_axis':
            raise ValueError('observable_seed requires semi_major_axis parameterization')
        from .initialization import observable_seed_positions
        physical_seeds = observable_seed_positions(
            **observable_seed, nwalkers=nwalkers, random_seed=random_seed,
            companion_mass_msun=float(Mcomp.to_value(units.M_sun)),
            companion_radius_rsun=float(Rcomp.to_value(units.R_sun)),
            companion_teff_k=None if Teff is None else float(Teff.to_value(units.K)),
        )
        p0[:, :5] = physical_seeds
    nwalkers = p0.shape[0]
    if parameterization not in ("semi_major_axis", "period"):
        raise ValueError("Unknown parameterization")
    state_parameterization = getattr(initial_walker_state, "parameterization", "semi_major_axis")
    if initial_walker_state is not None and state_parameterization != parameterization:
        raise ValueError("Initial walker state parameterization does not match sampler")
    validate_geometry(inclination_parameterization, DEFAULT_LIMITS['i'])
    state_geometry = getattr(initial_walker_state, 'inclination_parameterization', 'inclination')
    if initial_walker_state is not None and state_geometry != inclination_parameterization:
        raise ValueError('Initial walker geometry does not match sampler')
    kepler_factor = G.to_value(units.au**3 / (units.M_sun * units.day**2)) / (4 * numpy.pi**2)
    companion_mass = float(Mcomp.to_value(units.M_sun))
    radius_rsun = float(Rcomp.to_value(units.R_sun))

    def orbital_position(position):
        if parameterization == 'semi_major_axis':
            return position
        a = (kepler_factor * (companion_mass + position[2]) * position[0]**2)**(1 / 3)
        return position.at[0].set(a)

    def physical_and_logjac(position):
        physical = orbital_position(position)
        if inclination_parameterization == 'b_source':
            return geometry_to_physical(physical, radius_rsun, DEFAULT_LIMITS['i'])
        return physical, 0.

    def physical_position(position):
        return physical_and_logjac(position)[0]

    if parameterization == "period":
        if period_min is None or period_max is None:
            raise ValueError("Period parameterization requires period_min and period_max")
        period_low = float(period_min.to_value(units.day))
        period_high = float(period_max.to_value(units.day))
        if not 0 < period_low < period_high:
            raise ValueError("Period bounds must be positive and ordered")
        if initial_walker_state is None:
            p0[:, 0] = numpy.random.default_rng(random_seed).uniform(period_low, period_high, len(p0))

    if inclination_parameterization == 'b_source' and initial_walker_state is None:
        physical_starts = jax.vmap(orbital_position)(jax_numpy.asarray(p0))
        p0 = numpy.array(p0, copy=True)
        p0[:, 1] = numpy.asarray(physical_to_geometry(physical_starts, radius_rsun))[:, 1]

    Mcomp = Mcomp.to(SelfLensingSystem.param_units["Mcomp"])
    if Rcomp is not None:
        Rcomp = Rcomp.to(SelfLensingSystem.param_units["Rcomp"])
    if observed_rv is not None:
        observed_rv = observed_rv.to(units.m / units.s)
    if observed_rv_err is not None:
        observed_rv_err = observed_rv_err.to(units.m / units.s)
    if observed_lc is not None and hasattr(observed_lc, "to"):
        observed_lc = observed_lc.to(units.dimensionless_unscaled)
    if observed_lc_err is not None and hasattr(observed_lc_err, "to"):
        observed_lc_err = observed_lc_err.to(units.dimensionless_unscaled)

    default_sls = SelfLensingSystem(
        0 * SelfLensingSystem.param_units["Macc"],
        0 * SelfLensingSystem.param_units["Mcomp"],
        0 * SelfLensingSystem.param_units["Rcomp"],
        0 * SelfLensingSystem.param_units["a"],
    )
    detected_periods = jnp.array(periods) if len(periods) > 0 else jnp.array([])
    logprob_vec = make_combined_chi2_vec_jax(
        Mcomp.value,
        Rcomp.value if Rcomp is not None else None,
        tau_eff.value if tau_eff is not None else None,
        d.value if d is not None else None,
        peak_time,
        rv_times,
        observed_rv.value if observed_rv is not None else None,
        observed_rv_err.value if observed_rv_err is not None else None,
        lc_times,
        observed_lc.value if hasattr(observed_lc, "value") else observed_lc,
        observed_lc_err.value if hasattr(observed_lc_err, "value") else observed_lc_err,
        gp_model,
        detected_periods,
        lensing_method,
        center_lc_model,
        weight_lc_peak,
        period_min,
        period_max,
        default_sls,
        fit_device,
        peak_time_prior_sigma_days,
        rv_source_ids=rv_source_ids,
        profile_lc_baseline=profile_lc_baseline,
    )

    def physical_scores(positions, scores):
        if inclination_parameterization != 'b_source' and parameterization != 'period':
            return scores
        flat = positions.reshape((-1, positions.shape[-1]))
        physical, jacobian = jax.vmap(physical_and_logjac)(flat)
        if parameterization == 'period':
            jacobian = jacobian + jnp.log(2*physical[:, 0]/(3*flat[:, 0]))
        jacobian = jnp.broadcast_to(jacobian, (len(flat),)).reshape(scores.shape)
        return jnp.where(jnp.isfinite(jacobian), scores-jacobian, -jnp.inf)

    def logprob(position):
        physical, geometry_logjac = physical_and_logjac(position)
        value = logprob_vec(physical[jnp.newaxis, :])[0] + geometry_logjac
        if parameterization == "period":
            value = value + jnp.log(2 * physical[0] / (3 * position[0]))
            value = jnp.where((position[0] >= period_low) & (position[0] <= period_high), value, -jnp.inf)
        return value

    if proposal_scale is None:
        proposal_scale = _default_blackjax_proposal_scale(
            proposal_scale_fraction, peak_time_prior_sigma_days
        )
        if inclination_parameterization == 'b_source':
            proposal_scale = proposal_scale.at[1].set(.05)
    if parameterization == "period" and proposal_scale is None:
        raise AssertionError("Missing proposal scales")
    if parameterization == "period":
        proposal_scale = numpy.array(proposal_scale, copy=True)
        proposal_scale[0] = 0.1 * (period_high - period_low)
    proposal_scale = _scaled_proposal_scale(
        proposal_scale,
        observed_rv.value if observed_rv is not None else None,
        observed_rv_err.value if observed_rv_err is not None else None,
        None,
        observed_lc.value if hasattr(observed_lc, "value") else observed_lc,
        None,
        proposal_reference_observations,
    )
    proposal_scale = jax.device_put(jax_numpy.asarray(proposal_scale), fit_device)

    # Select between independent constrained starts and one local proposal
    # around each of them. This is deterministic for a fixed random_seed and
    # does not alter explicitly supplied resume states.
    should_refine_starts = (
        initial_walker_state is None
        or getattr(initial_walker_state, "refine_initialization", False)
    )
    if should_refine_starts:
        initial_positions = jax.device_put(jax_numpy.asarray(p0), fit_device)
        initial_log_prob = numpy.asarray(
            jax.device_get(jax.vmap(logprob)(initial_positions))
        )
        start_rng = numpy.random.default_rng(random_seed + 1)
        local_scale = 0.25 * numpy.asarray(jax.device_get(proposal_scale))
        local_positions = numpy.asarray(p0, dtype=float) + start_rng.normal(
            scale=local_scale, size=numpy.asarray(p0).shape
        )
        first_bounds = (
            (period_low, period_high)
            if parameterization == "period"
            else DEFAULT_LIMITS["a"]
        )
        local_positions[:, 0] = numpy.clip(local_positions[:, 0], *first_bounds)
        if inclination_parameterization == 'inclination':
            local_positions[:, 1] = numpy.clip(local_positions[:, 1], *DEFAULT_LIMITS['i'])
        local_positions[:, 2] = numpy.clip(local_positions[:, 2], *DEFAULT_LIMITS["Macc"])
        local_positions[:, 3] = numpy.clip(local_positions[:, 3], *DEFAULT_LIMITS["e"])
        local_positions[:, 4] = numpy.mod(
            local_positions[:, 4], 2 * numpy.pi
        )
        if inclination_parameterization == 'b_source':
            physical = numpy.asarray(jax.vmap(orbital_position)(jax_numpy.asarray(local_positions)))
            scale = numpy.asarray(b_source_scale(physical[:, 0], radius_rsun, physical[:, 3], physical[:, 4]))
            low = scale * numpy.cos(float(DEFAULT_LIMITS['i'][1]))
            high = scale * numpy.cos(float(DEFAULT_LIMITS['i'][0]))
            local_positions[:, 1] = numpy.clip(local_positions[:, 1], low, high * (1-1e-12))
        local_log_prob = numpy.asarray(jax.device_get(jax.vmap(logprob)(
            jax.device_put(jax_numpy.asarray(local_positions), fit_device)
        )))
        improve = numpy.isfinite(local_log_prob) & (
            ~numpy.isfinite(initial_log_prob) | (local_log_prob > initial_log_prob)
        )
        p0 = numpy.where(improve[:, None], local_positions, p0)
        logging.info(
            "blackjax fit: locally refined %d/%d starting walkers",
            int(improve.sum()), nwalkers,
        )
    from .recovery import mixture_candidate
    from .adaptation import tune_proposals, proposal_step
    warmup = tune_proposals(
        lambda positions: jax.vmap(logprob)(positions[0])[None, :],
        jax.device_put(jax_numpy.asarray(p0)[None, :, :], fit_device), proposal_scale,
        steps=burn if warmup_steps is None else warmup_steps, rounds=warmup_rounds,
        random_seed=random_seed + 2, covariance=adapt_covariance,
        restart=warmup_restart, robust_covariance=robust_covariance,
        score_transform=physical_scores, log_density_cut=warmup_log_density_cut,
        recovery_probability=recovery_probability, recovery_df=recovery_df,
        recovery_centres_count=recovery_centres_count, periodic_index=4,
    )
    transform = warmup.transform[0]
    proposal_scale = jnp.diagonal(transform)
    warmup_acceptance = warmup.acceptance[:, 0, :]

    def proposal(key, position):
        increment, _ = proposal_step(key, position[None, :], transform, adapt_covariance)
        return increment[0]

    random_walk = blackjax.additive_step_random_walk(logprob, proposal)
    state = jax.vmap(random_walk.init)(warmup.positions[0])

    @jax.jit
    def run_steps(current_state, rng_keys):
        def one_step(state, key):
            if recovery_probability:
                proposal_key, accept_key = jax.random.split(key)
                candidate, correction, _, recovery = mixture_candidate(
                    proposal_key, state.position[None], transform[None], adapt_covariance,
                    warmup.recovery_centres, warmup.recovery_transform,
                    recovery_probability, recovery_df)
                score = jax.vmap(logprob)(candidate[0])
                accepted = jnp.log(jax.random.uniform(accept_key, score.shape)) < score-state.logdensity+correction[0]
                state = state._replace(position=jnp.where(accepted[:,None],candidate[0],state.position),
                                       logdensity=jnp.where(accepted,score,state.logdensity))
                recovery = recovery[0]
            else:
                keys = jax.random.split(key, nwalkers)
                state, info = jax.vmap(random_walk.step)(keys, state)
                accepted = info.is_accepted
                recovery = jnp.zeros_like(accepted)
            return state, (state.position, state.logdensity, accepted, recovery)
        return jax.lax.scan(one_step, current_state, rng_keys)

    logging.info('blackjax fit: coordinate warm-up acceptance %s; final proposal scale %s',
                 warmup_acceptance, numpy.asarray(proposal_scale))
    rng_key = jax.random.PRNGKey(random_seed + 3)

    chain_chunks = []
    log_prob_chunks = []
    acceptance_chunks = []
    recovery_chunks = []
    posterior_summary_top_k = int(posterior_summary_top_k)
    if posterior_summary_top_k < 1:
        raise ValueError("posterior_summary_top_k must be at least 1")
    top_positions = jnp.zeros((posterior_summary_top_k, state.position.shape[-1]))
    top_log_prob = jnp.full((posterior_summary_top_k,), -jnp.inf)
    if not return_sampler and burn >= nsteps:
        raise ValueError("burn must be smaller than nsteps when return_sampler=False")
    remaining_steps = nsteps
    completed_steps = 0
    if progress:
        progress_iter = tqdm(
            total=nsteps,
            desc=progress_desc,
            disable=not progress,
        )
    else:
        progress_iter = None

    mixture_sampling_chain = None
    mixture_sampling_log_prob = None
    try:
        if independence_proposal is not None:
            if not return_sampler:
                raise ValueError('Frozen mixture sampling currently requires return_sampler=True')
            from .mixture import sample_independence
            if independence_proposal.centres.shape[-1] != state.position.shape[-1]:
                raise ValueError('Mixture dimensions must match sampler coordinates')
            last_progress = [0]
            def mixture_progress(done, total, acceptance):
                if progress_iter is not None:
                    progress_iter.update(done-last_progress[0])
                last_progress[0] = done
            if independence_coordinates not in ('sampling', 'pulse'):
                raise ValueError('independence_coordinates must be sampling or pulse')
            if independence_coordinates == 'pulse':
                if parameterization != 'semi_major_axis' or inclination_parameterization != 'inclination':
                    raise ValueError('Pulse mixture sampling reports physical a/i coordinates; use semi_major_axis and inclination')
                from .pulse import physical_to_pulse, pulse_to_physical
                def mixture_target(z):
                    physical, jacobian = pulse_to_physical(z, companion_mass, radius_rsun)
                    return logprob_vec(physical) + jacobian
                mixture_target = jax.jit(mixture_target)
                mixture_initial = physical_to_pulse(state.position, companion_mass, radius_rsun)
            else:
                mixture_target = lambda x: jax.vmap(logprob)(x)
                mixture_initial = state.position
            positions, log_prob, accepted = sample_independence(
                mixture_target, independence_proposal,
                numpy.asarray(mixture_initial), steps=nsteps, seed=random_seed,
                chunk=independence_chunk_size, report=mixture_progress)
            if independence_coordinates == 'pulse':
                mixture_sampling_chain = positions
                mixture_sampling_log_prob = log_prob
                decoded, jacobian = pulse_to_physical(jnp.asarray(positions.reshape(-1, positions.shape[-1])),
                                                     companion_mass, radius_rsun)
                positions = numpy.asarray(decoded).reshape(positions.shape)
                log_prob = log_prob - numpy.asarray(jacobian).reshape(log_prob.shape)
            chain_chunks.append(positions)
            log_prob_chunks.append(log_prob)
            acceptance_chunks.append(accepted)
            state = state._replace(position=jnp.asarray(positions[-1]),
                                   logdensity=jnp.asarray(log_prob[-1]))
        else:
            chunk_size = nsteps if not progress else max(1, int(progress_chunk_size))
            while remaining_steps > 0:
                current_chunk_size = min(chunk_size, remaining_steps)
                # Keep every adaptation block entirely within discarded burn-in.
                if completed_steps < burn:
                    current_chunk_size = min(current_chunk_size, burn - completed_steps)
                rng_key, sample_key = jax.random.split(rng_key)
                step_keys = jax.random.split(sample_key, current_chunk_size)
                if return_sampler:
                    state, (positions, log_prob, accepted, recovery) = run_steps(state, step_keys)
                    chain_chunks.append(positions)
                    log_prob_chunks.append(log_prob)
                    acceptance_chunks.append(accepted)
                    recovery_chunks.append(recovery)
                else:
                    state, (positions, log_prob, accepted, recovery) = run_steps(state, step_keys)
                    keep_steps = (
                        jnp.arange(completed_steps, completed_steps + current_chunk_size)
                        >= burn
                    )
                    top_positions, top_log_prob = _update_top_posterior_samples(
                        top_positions,
                        top_log_prob,
                        positions,
                        physical_scores(positions, log_prob),
                        keep_steps,
                    )
                remaining_steps -= current_chunk_size
                completed_steps += current_chunk_size
                if progress_iter is not None:
                    progress_iter.update(current_chunk_size)
    finally:
        if progress_iter is not None:
            progress_iter.close()

    final_walker_state = (
        BlackJAXFitState(coords=jax.device_get(state.position), parameterization=parameterization,
                         inclination_parameterization=inclination_parameterization)
        if return_walker_state
        else None
    )
    if return_sampler:
        chain = jax.device_get(jnp.concatenate(chain_chunks, axis=0))
        log_prob = jax.device_get(jnp.concatenate(log_prob_chunks, axis=0))
        acceptance = jax.device_get(jnp.concatenate(acceptance_chunks, axis=0))
        sampler = BlackJAXSamplerResult(chain, log_prob, acceptance=acceptance)
        sampler.recovery_attempted = numpy.asarray(jnp.concatenate(recovery_chunks)) if recovery_probability and recovery_chunks else None
        sampler.proposal_kind = 'local_mixture' if independence_proposal is not None else 'random_walk'
        sampler.independence_proposal = independence_proposal
        sampler.independence_coordinates = independence_coordinates
        sampler.sampling_chain = mixture_sampling_chain
        sampler.sampling_log_prob = mixture_sampling_log_prob
        _set_proposal_metadata(sampler, warmup, 0, recovery_probability, recovery_df, robust_covariance, warmup_log_density_cut)
        sampler.warmup_acceptance = numpy.asarray(warmup_acceptance)
        sampler.proposal_scale = numpy.asarray(jax.device_get(proposal_scale))
        sampler.proposal_transform = numpy.asarray(jax.device_get(transform))
        sampler.inclination_parameterization = inclination_parameterization
        sampler.parameterization = parameterization
        sampler.parameter_names = ["period" if parameterization == "period" else "a", "i", "Macc", "e", "periapsis_phase"]
        sampler.parameter_units = ["day" if parameterization == "period" else "au", "rad", "solMass", "", "rad"]
        if inclination_parameterization == 'b_source':
            sampler.parameter_names[1] = 'b_source'
            sampler.parameter_units[1] = ''
        if chain.shape[-1] == 6:
            sampler.parameter_names.append("peak_time_offset")
            sampler.parameter_units.append("day")
        flat_samples = jnp.asarray(sampler.get_chain(flat=True, discard=burn))
        flat_log_prob = jnp.asarray(sampler.get_log_prob(flat=True, discard=burn))
        flat_log_prob = physical_scores(flat_samples, flat_log_prob)
        summary_top_k = min(posterior_summary_top_k, flat_log_prob.shape[0])
        top_log_prob, top_indices = jax.lax.top_k(flat_log_prob, summary_top_k)
        top_positions = flat_samples[top_indices]
        # A mean of high-scoring samples need not itself be a good solution.
        # Report the actual retained sample with the highest log probability.
        summary_position = top_positions[0]
    else:
        sampler = None
        # A mean of high-scoring samples need not itself be a good solution.
        # Report the actual retained sample with the highest log probability.
        summary_position = top_positions[0]
    sls = _best_fit_system_from_samples(
        jax.device_get(physical_position(summary_position)[jnp.newaxis, :]),
        jax.device_get(jnp.asarray([0.0])),
        Rcomp,
        Mcomp,
        Teff,
        d,
    )
    sls.fit_peak_time_offset_days = (
        0.0 if peak_time_prior_sigma_days is None else float(summary_position[5])
    )
    return sls, sampler, final_walker_state


def _batched_quantity(quantity, unit, nfits, name):
    if quantity is None:
        return None
    quantity = quantity.to(unit)
    values = numpy.asarray(quantity.value)
    if values.ndim == 0:
        values = numpy.full(nfits, values.item())
    if values.shape[0] != nfits:
        raise ValueError(f"{name} must have length {nfits}; got shape {values.shape}")
    return values * unit


def _batched_observation_block(times, observed, observed_err, mask, nfits):
    if observed is None:
        shape = (nfits, 1)
        return (
            numpy.zeros(shape, dtype=float),
            numpy.zeros(shape, dtype=float),
            numpy.ones(shape, dtype=float),
            numpy.zeros(shape, dtype=bool),
        )
    if mask is None:
        mask = numpy.ones(numpy.asarray(observed).shape, dtype=bool)
    return times, observed, observed_err, mask


def _batched_periods(periods, nfits):
    if hasattr(periods, "to"):
        periods = periods.to(units.day).value
    if len(periods) == 0:
        return numpy.full((nfits, 0), numpy.nan, dtype=float)

    period_array = numpy.asarray(periods, dtype=object)
    if period_array.ndim == 1 and all(numpy.isscalar(value) for value in period_array):
        period_values = numpy.asarray(periods, dtype=float)
        return numpy.tile(period_values, (nfits, 1))

    rows = []
    for row in periods:
        if hasattr(row, "to"):
            row = row.to(units.day).value
        values = numpy.ravel(numpy.asarray(row, dtype=float))
        values = values[numpy.isfinite(values)]
        rows.append(values)
    if len(rows) != nfits:
        raise ValueError("Batched periods must have one row per fit")

    width = max((len(row) for row in rows), default=0)
    padded = numpy.full((nfits, width), numpy.nan, dtype=float)
    for index, row in enumerate(rows):
        padded[index, : len(row)] = row
    return padded


@units.quantity_input
def self_lensing_systems_from_batched_fit_blackjax(
    Mcomp: units.kg,
    peak_time,
    rv_times=None,
    observed_rv=None,
    observed_rv_err=None,
    rv_mask=None,
    lc_times=None,
    observed_lc=None,
    observed_lc_err=None,
    lc_mask=None,
    gp_model=None,
    Rcomp: units.m = None,
    Teff: units.K = None,
    tau_eff: units.day = None,
    d: units.pc = SelfLensingSystem.DEFAULT_D,
    false_alarm_probability=0.1,
    detect_periods=False,
    periods=[],
    period_min: units.day = None,
    period_max: units.day = None,
    weight_lc_peak=False,
    nwalkers=100,
    nsteps=5000,
    burn=500,
    lensing_method=None,
    center_lc_model=False,
    progress=True,
    progress_desc=None,
    prefer_gpu=True,
    device=None,
    initial_walker_state=None,
    proposal_scale=None,
    proposal_scale_fraction=0.02,
    proposal_reference_observations=None,
    random_seed=0,
    progress_chunk_size=100,
    posterior_summary_top_k=DEFAULT_POSTERIOR_SUMMARY_TOP_K,
    return_sampler=False,
    return_walker_state=False,
    posterior_sample_count=None,
    rv_source_ids=None,
    profile_lc_baseline=False,
    inclination_parameterization="inclination",
    warmup_steps=None,
    warmup_rounds=None,
    adapt_covariance=False,
    warmup_restart=False,
    robust_covariance=False,
    recovery_probability=0.,
    recovery_df=3.,
    recovery_centres_count=16,
    warmup_log_density_cut=20.,
):
    """Fit many independent systems in one batched BlackJAX-style run.

    Observation arrays must have a leading fit dimension. Ragged RV/LC series
    should be padded to a common length and accompanied by boolean masks. The
    sampler tracks a small high-posterior post-burn sample reservoir for each
    fit on device and copies posterior summaries back once at the end. When
    ``return_sampler`` is enabled, the full chain is copied back to host memory
    and returned as one ``BlackJAXSamplerResult`` per fit. Set
    ``posterior_sample_count`` with ``return_sampler=True`` to retain at most
    that many evenly spaced post-burn walker samples per fit instead of the
    complete chain on the accelerator.
    """
    if observed_rv is None and observed_lc is None:
        raise RuntimeError("You must provide at least one of observed_rv, observed_lc")
    if Rcomp is None:
        raise NotImplementedError(
            "Batched BlackJAX fitting requires Rcomp to be provided"
        )
    if gp_model is not None:
        raise NotImplementedError("Batched BlackJAX fitting does not support gp_model")
    if detect_periods:
        raise NotImplementedError(
            "Batched BlackJAX fitting does not currently support detect_periods=True"
        )

    fit_device = _select_fit_device(device=device, prefer_gpu=prefer_gpu)
    if lensing_method is None:
        lensing_method = (
            SelfLensingLightcurveGenerator.default_lensing_method_for_device(fit_device)
        )
    logging.info(
        "batched blackjax fit: using JAX %s device %s with lensing_method=%s",
        fit_device.platform,
        fit_device,
        lensing_method,
    )

    Mcomp = Mcomp.to(SelfLensingSystem.param_units["Mcomp"])
    nfits = int(numpy.asarray(Mcomp.value).shape[0])
    Rcomp = _batched_quantity(Rcomp, SelfLensingSystem.param_units["Rcomp"], nfits, "Rcomp")
    Mcomp = _batched_quantity(Mcomp, SelfLensingSystem.param_units["Mcomp"], nfits, "Mcomp")
    Teff = _batched_quantity(Teff, SelfLensingSystem.param_units["Teff"], nfits, "Teff")
    d = _batched_quantity(d, SelfLensingSystem.param_units["d"], nfits, "d")
    if tau_eff is not None:
        tau_eff = _batched_quantity(
            tau_eff, SelfLensingSystem.param_units["tau_eff"], nfits, "tau_eff"
        )
    if observed_rv is not None:
        observed_rv = observed_rv.to(units.m / units.s)
    if observed_rv_err is not None:
        observed_rv_err = observed_rv_err.to(units.m / units.s)
    if observed_lc is not None and hasattr(observed_lc, "to"):
        observed_lc = observed_lc.to(units.dimensionless_unscaled)
    if observed_lc_err is not None and hasattr(observed_lc_err, "to"):
        observed_lc_err = observed_lc_err.to(units.dimensionless_unscaled)

    rv_times, observed_rv, observed_rv_err, rv_mask = _batched_observation_block(
        rv_times,
        observed_rv.value if observed_rv is not None else None,
        observed_rv_err.value if observed_rv_err is not None else None,
        rv_mask,
        nfits,
    )
    lc_times, observed_lc, observed_lc_err, lc_mask = _batched_observation_block(
        lc_times,
        observed_lc.value if hasattr(observed_lc, "value") else observed_lc,
        observed_lc_err.value if hasattr(observed_lc_err, "value") else observed_lc_err,
        lc_mask,
        nfits,
    )

    detected_periods = jnp.array(_batched_periods(periods, nfits))
    logprob_vec = make_batched_numeric_combined_chi2_vec_jax(
        Mcomp.value,
        Rcomp.value,
        tau_eff.value if tau_eff is not None else None,
        peak_time,
        rv_times,
        observed_rv,
        observed_rv_err,
        lc_times,
        observed_lc,
        observed_lc_err,
        detected_periods,
        center_lc_model,
        period_min,
        period_max,
        rv_mask,
        lc_mask,
        fit_device,
        rv_source_ids=rv_source_ids,
        profile_lc_baseline=profile_lc_baseline,
    )

    nwalkers = int(nwalkers)
    nsteps = int(nsteps)
    burn = int(burn)
    if posterior_sample_count is not None:
        posterior_sample_count = int(posterior_sample_count)
        if posterior_sample_count < 1:
            raise ValueError("posterior_sample_count must be at least 1")
        if not return_sampler:
            raise ValueError("posterior_sample_count requires return_sampler=True")
        if burn >= nsteps:
            raise ValueError("burn must be smaller than nsteps")
    p0 = _initial_batched_walker_positions(nfits, nwalkers, initial_walker_state,
                                           rng=numpy.random.default_rng(random_seed))
    if initial_walker_state is not None and getattr(initial_walker_state, 'parameterization', 'semi_major_axis') != 'semi_major_axis':
        raise ValueError('Batched fitting requires semi_major_axis walker coordinates')
    validate_geometry(inclination_parameterization, DEFAULT_LIMITS['i'])
    if initial_walker_state is not None and getattr(initial_walker_state, 'inclination_parameterization', 'inclination') != inclination_parameterization:
        raise ValueError('Initial walker geometry does not match sampler')
    if inclination_parameterization == 'b_source':
        radius_rsun = Rcomp.to_value(units.R_sun)[:, None]
        if initial_walker_state is None:
            p0 = numpy.asarray(physical_to_geometry(p0, radius_rsun))
        physical_logprob_vec = logprob_vec
        def logprob_vec(position):
            physical, jacobian = geometry_to_physical(position, radius_rsun, DEFAULT_LIMITS['i'])
            return physical_logprob_vec(physical) + jacobian
    def physical_scores(positions, scores):
        if inclination_parameterization != 'b_source':
            return scores
        _, jacobian = geometry_to_physical(positions, radius_rsun, DEFAULT_LIMITS['i'])
        return jnp.where(jnp.isfinite(jacobian), scores - jacobian, -jnp.inf)
    nwalkers = p0.shape[1]
    position = jax.device_put(jax_numpy.asarray(p0), fit_device)
    log_prob = logprob_vec(position)

    if proposal_scale is None:
        proposal_scale = _default_blackjax_proposal_scale(proposal_scale_fraction)
        if inclination_parameterization == 'b_source':
            proposal_scale = proposal_scale.at[1].set(.05)
    proposal_scale = _scaled_proposal_scale(
        proposal_scale,
        observed_rv,
        observed_rv_err,
        rv_mask,
        observed_lc,
        lc_mask,
        proposal_reference_observations,
    )
    proposal_scale = jax.device_put(jax_numpy.asarray(proposal_scale), fit_device)

    from .recovery import mixture_candidate
    from .adaptation import tune_proposals, proposal_step
    warmup = tune_proposals(
        logprob_vec, position, proposal_scale,
        steps=burn if warmup_steps is None else warmup_steps, rounds=warmup_rounds,
        random_seed=random_seed + 2, covariance=adapt_covariance,
        restart=warmup_restart, robust_covariance=robust_covariance,
        score_transform=physical_scores, log_density_cut=warmup_log_density_cut,
        recovery_probability=recovery_probability, recovery_df=recovery_df,
        recovery_centres_count=recovery_centres_count, periodic_index=4,
    )
    position = warmup.positions
    log_prob = logprob_vec(position)
    transform = warmup.transform

    def one_step(carry, rng_key):
        position, log_prob = carry
        proposal_key, accept_key = jax.random.split(rng_key)
        if recovery_probability:
            proposal, correction, _, recovery = mixture_candidate(
                proposal_key, position, transform, adapt_covariance,
                warmup.recovery_centres, warmup.recovery_transform,
                recovery_probability, recovery_df)
        else:
            increment, _ = proposal_step(proposal_key, position, transform, adapt_covariance)
            proposal = position + increment
            correction = 0.
            recovery = jnp.zeros_like(log_prob, dtype=bool)
        proposal_log_prob = logprob_vec(proposal)
        log_accept_ratio = proposal_log_prob - log_prob + correction
        accepted = jnp.log(
            jax.random.uniform(accept_key, shape=log_prob.shape)
        ) < log_accept_ratio
        new_position = jnp.where(accepted[..., None], proposal, position)
        new_log_prob = jnp.where(accepted, proposal_log_prob, log_prob)
        return (new_position, new_log_prob), (new_position, new_log_prob, accepted, recovery)

    @jax.jit
    def run_steps(position, log_prob, rng_keys):
        return jax.lax.scan(one_step, (position, log_prob), rng_keys)

    def advance_one_step(carry, rng_key):
        next_carry, _ = one_step(carry, rng_key)
        return next_carry, None

    @jax.jit
    def advance_steps(position, log_prob, rng_keys):
        return jax.lax.scan(advance_one_step, (position, log_prob), rng_keys)[0]

    rng_key = jax.random.PRNGKey(random_seed)
    chain_chunks = []
    log_prob_chunks = []
    acceptance_chunks = []
    recovery_chunks = []
    posterior_summary_top_k = int(posterior_summary_top_k)
    if posterior_summary_top_k < 1:
        raise ValueError("posterior_summary_top_k must be at least 1")
    top_positions = jnp.zeros((nfits, posterior_summary_top_k, position.shape[-1]))
    top_log_prob = jnp.full((nfits, posterior_summary_top_k), -jnp.inf)
    if not return_sampler and burn >= nsteps:
        raise ValueError("burn must be smaller than nsteps when return_sampler=False")
    remaining_steps = nsteps
    completed_steps = 0
    if progress:
        progress_iter = tqdm(
            total=nsteps,
            desc=progress_desc,
            disable=not progress,
        )
    else:
        progress_iter = None

    try:
        if posterior_sample_count is not None:
            retained_walkers = min(nwalkers, posterior_sample_count)
            retained_steps = min(
                max(1, posterior_sample_count // retained_walkers), nsteps - burn
            )
            segment_lengths = numpy.full(
                retained_steps, (nsteps - burn) // retained_steps, dtype=int
            )
            segment_lengths[: (nsteps - burn) % retained_steps] += 1
            if burn:
                rng_key, sample_key = jax.random.split(rng_key)
                position, log_prob = advance_steps(
                    position, log_prob, jax.random.split(sample_key, burn)
                )
                if progress_iter is not None:
                    progress_iter.update(burn)
            for segment_length in segment_lengths:
                rng_key, sample_key = jax.random.split(rng_key)
                position, log_prob = advance_steps(
                    position,
                    log_prob,
                    jax.random.split(sample_key, int(segment_length)),
                )
                chain_chunks.append(position[:, :retained_walkers, :])
                log_prob_chunks.append(log_prob[:, :retained_walkers])
                if progress_iter is not None:
                    progress_iter.update(int(segment_length))
        else:
            chunk_size = nsteps if not progress else max(1, int(progress_chunk_size))
            while remaining_steps > 0:
                current_chunk_size = min(chunk_size, remaining_steps)
                rng_key, sample_key = jax.random.split(rng_key)
                step_keys = jax.random.split(sample_key, current_chunk_size)
                if return_sampler:
                    (position, log_prob), (positions, chunk_log_prob, accepted, recovery) = run_steps(
                        position, log_prob, step_keys
                    )
                    chain_chunks.append(positions)
                    log_prob_chunks.append(chunk_log_prob)
                    acceptance_chunks.append(accepted)
                    recovery_chunks.append(recovery)
                else:
                    (position, log_prob), (positions, chunk_log_prob, _accepted, _recovery) = run_steps(
                        position, log_prob, step_keys
                    )
                    keep_steps = (
                        jnp.arange(completed_steps, completed_steps + current_chunk_size)
                        >= burn
                    )
                    top_positions, top_log_prob = _update_batched_top_posterior_samples(
                        top_positions,
                        top_log_prob,
                        positions,
                        physical_scores(positions, chunk_log_prob),
                        keep_steps,
                    )
                remaining_steps -= current_chunk_size
                completed_steps += current_chunk_size
                if progress_iter is not None:
                    progress_iter.update(current_chunk_size)
    finally:
        if progress_iter is not None:
            progress_iter.close()

    final_walker_state = (
        BlackJAXFitState(coords=jax.device_get(position), inclination_parameterization=inclination_parameterization)
        if return_walker_state
        else None
    )
    if return_sampler:
        if posterior_sample_count is None:
            chain_all = jnp.concatenate(chain_chunks, axis=0)
            log_prob_all = jnp.concatenate(log_prob_chunks, axis=0)
        else:
            chain_all = jnp.stack(chain_chunks, axis=0)
            log_prob_all = jnp.stack(log_prob_chunks, axis=0)
        sampled_chain_start = burn if posterior_sample_count is not None else 0
        post_burn_chain = chain_all if posterior_sample_count is not None else chain_all[burn:]
        post_burn_log_prob = (
            log_prob_all if posterior_sample_count is not None else log_prob_all[burn:]
        )
        flat_positions = jnp.transpose(post_burn_chain, (1, 0, 2, 3)).reshape(
            (nfits, -1, chain_all.shape[-1])
        )
        flat_log_prob = jnp.transpose(post_burn_log_prob, (1, 0, 2)).reshape(
            (nfits, -1)
        )
        flat_log_prob = physical_scores(flat_positions, flat_log_prob)
        summary_top_k = min(posterior_summary_top_k, flat_log_prob.shape[1])
        top_log_prob, top_indices = jax.lax.top_k(flat_log_prob, summary_top_k)
        top_positions = jnp.take_along_axis(
            flat_positions,
            top_indices[:, :, None],
            axis=1,
        )
        # Keep each reported system on an actual retained high-probability sample.
        summary_position = top_positions[:, 0, :]
    else:
        # Keep each reported system on an actual retained high-probability sample.
        summary_position = top_positions[:, 0, :]
    physical_summary_position = summary_position
    if inclination_parameterization == 'b_source':
        physical_summary_position, _ = geometry_to_physical(summary_position, radius_rsun[:, 0], DEFAULT_LIMITS['i'])
    systems = _best_fit_systems_from_positions(
        jax.device_get(physical_summary_position),
        Rcomp,
        Mcomp,
        Teff,
        d,
    )
    if return_sampler:
        chain = jax.device_get(chain_all)
        log_prob = jax.device_get(log_prob_all)
        acceptance = (
            jax.device_get(jnp.concatenate(acceptance_chunks, axis=0))
            if acceptance_chunks
            else None
        )
        samplers = [
            BlackJAXSamplerResult(
                chain[:, fit_index, :, :],
                log_prob[:, fit_index, :],
                acceptance=None if acceptance is None else acceptance[:, fit_index, :],
                chain_start_step=sampled_chain_start,
            )
            for fit_index in range(nfits)
        ]
        for fit_index, sampler in enumerate(samplers):
            sampler.inclination_parameterization = inclination_parameterization
            sampler.parameter_names = ['a', 'b_source' if inclination_parameterization == 'b_source' else 'i', 'Macc', 'e', 'periapsis_phase']
            sampler.parameter_units = ['au', '' if inclination_parameterization == 'b_source' else 'rad', 'solMass', '', 'rad']
            sampler.recovery_attempted = (numpy.asarray(jnp.concatenate(recovery_chunks))[:,fit_index]
                if recovery_probability and recovery_chunks else None)
            _set_proposal_metadata(sampler, warmup, fit_index, recovery_probability, recovery_df, robust_covariance, warmup_log_density_cut)
            sampler.warmup_acceptance = warmup.acceptance[:, fit_index, :]
            sampler.proposal_transform = numpy.asarray(transform[fit_index])
            sampler.proposal_scale = numpy.diag(sampler.proposal_transform)
    else:
        samplers = None
    return systems, samplers, final_walker_state


def _set_proposal_metadata(sampler, warmup, index, probability, df, robust, cut):
    """Record the frozen proposal and discarded warm-up diagnostics."""
    sampler.recovery_probability = probability
    sampler.recovery_df = df
    sampler.robust_covariance = robust
    sampler.warmup_log_density_cut = cut
    sampler.warmup_eligible_fraction = numpy.asarray(warmup.eligible_fraction)[:,index]
    sampler.warmup_recovery_acceptance = numpy.asarray(warmup.recovery_acceptance)[:,index]
    sampler.recovery_centres = None if warmup.recovery_centres is None else numpy.asarray(warmup.recovery_centres[index])
    sampler.recovery_transform = None if warmup.recovery_transform is None else numpy.asarray(warmup.recovery_transform[index])
