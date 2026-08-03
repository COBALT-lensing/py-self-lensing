"""BlackJAX fitting backends."""

import logging
from dataclasses import dataclass

import jax
import jax.numpy as jax_numpy
import numpy
from astropy import units
from quaxed import numpy as jnp
from tqdm import tqdm

from ..generators import SelfLensingLightcurveGenerator
from ..models import SelfLensingSystem, DEFAULT_LIMITS
from ._utils import _select_fit_device
from .likelihood import make_combined_chi2_vec_jax, make_batched_combined_chi2_vec_jax


@dataclass
class BlackJAXFitState:
    coords: numpy.ndarray


class BlackJAXSamplerResult:
    """Small emcee-like adapter for downstream diagnostics."""

    def __init__(self, chain, log_prob, acceptance=None):
        self.chain = numpy.asarray(chain)
        self.log_prob = numpy.asarray(log_prob)
        self.acceptance = None if acceptance is None else numpy.asarray(acceptance)

    @property
    def iteration(self):
        return len(self.chain)

    @property
    def acceptance_fraction(self):
        if self.acceptance is None or self.acceptance.size == 0:
            return numpy.full(self.chain.shape[1], numpy.nan)
        return numpy.mean(self.acceptance, axis=0)

    def get_chain(self, flat=False, discard=0, thin=1):
        chain = self.chain[discard::thin]
        if flat:
            return chain.reshape((-1, chain.shape[-1]))
        return chain

    def get_log_prob(self, flat=False, discard=0, thin=1):
        log_prob = self.log_prob[discard::thin]
        if flat:
            return log_prob.reshape((-1,))
        return log_prob


def _initial_walker_positions(nwalkers, initial_walker_state=None):
    ndim = 5
    if initial_walker_state is not None:
        initial_walker_coords = getattr(
            initial_walker_state,
            "coords",
            initial_walker_state,
        )
        initial_walker_coords = numpy.asarray(initial_walker_coords)
        if initial_walker_coords.ndim != 2 or initial_walker_coords.shape[1] != ndim:
            raise ValueError(
                "initial_walker_state must have coordinates with shape "
                f"(nwalkers, {ndim})"
            )
        return initial_walker_coords

    p0 = [
        numpy.random.uniform(bounds[0], bounds[1], nwalkers)
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
    return numpy.asarray(list(zip(*p0)), dtype=float)


def _initial_batched_walker_positions(nfits, nwalkers, initial_walker_state=None):
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
        [_initial_walker_positions(nwalkers) for _ in range(nfits)],
        axis=0,
    )


def _default_blackjax_proposal_scale(scale_fraction=0.02):
    return jnp.asarray(
        [
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
        ],
        dtype=float,
    )


def _best_fit_system_from_samples(flat_samples, flat_log_prob, Rcomp, Mcomp, Teff, d):
    res = flat_samples[jnp.argmax(flat_log_prob)]
    out_a, out_i, out_Macc, out_e, out_periapsis_phase = res
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
    random_seed=0,
    progress_chunk_size=100,
    return_sampler=True,
    return_walker_state=True,
):
    """Fit using a BlackJAX random-walk Metropolis-Hastings backend.

    This is an additive, JAX-native alternative to self_lensing_system_from_fit().
    It keeps walker positions and log-probabilities on the selected JAX device
    during sampling and returns a small emcee-like result adapter.

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
      diagnostics. In that mode the best sample is tracked on device and the
      full chain is not copied back to host memory.
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
    p0 = _initial_walker_positions(nwalkers, initial_walker_state)
    nwalkers = p0.shape[0]

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
    )

    def logprob(position):
        return logprob_vec(position[jnp.newaxis, :])[0]

    if proposal_scale is None:
        proposal_scale = _default_blackjax_proposal_scale(proposal_scale_fraction)
    proposal_scale = jax.device_put(jax_numpy.asarray(proposal_scale), fit_device)
    random_walk = blackjax.additive_step_random_walk(
        logprob,
        blackjax.mcmc.random_walk.normal(proposal_scale),
    )
    state = jax.vmap(random_walk.init)(jax.device_put(jax_numpy.asarray(p0), fit_device))

    def one_step(state, rng_key):
        walker_keys = jax.random.split(rng_key, nwalkers)
        new_state, info = jax.vmap(random_walk.step)(walker_keys, state)
        return new_state, (new_state.position, new_state.logdensity, info.is_accepted)

    @jax.jit
    def run_steps(state, rng_keys):
        return jax.lax.scan(one_step, state, rng_keys)

    def one_best_step(carry, inputs):
        state, best_position, best_log_prob = carry
        rng_key, keep_sample = inputs
        state, _ = one_step(state, rng_key)
        step_best_index = jnp.argmax(state.logdensity)
        step_best_log_prob = state.logdensity[step_best_index]
        step_best_position = state.position[step_best_index]
        use_step_best = keep_sample & (step_best_log_prob > best_log_prob)
        best_log_prob = jnp.where(use_step_best, step_best_log_prob, best_log_prob)
        best_position = jnp.where(use_step_best, step_best_position, best_position)
        return (state, best_position, best_log_prob), None

    @jax.jit
    def run_steps_best(state, best_position, best_log_prob, rng_keys, keep_steps):
        (state, best_position, best_log_prob), _ = jax.lax.scan(
            one_best_step,
            (state, best_position, best_log_prob),
            (rng_keys, keep_steps),
        )
        return state, best_position, best_log_prob

    rng_key = jax.random.PRNGKey(random_seed)
    chain_chunks = []
    log_prob_chunks = []
    acceptance_chunks = []
    best_position = jnp.zeros_like(state.position[0])
    best_log_prob = jnp.asarray(-jnp.inf)
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
        chunk_size = nsteps if not progress else max(1, int(progress_chunk_size))
        while remaining_steps > 0:
            current_chunk_size = min(chunk_size, remaining_steps)
            rng_key, sample_key = jax.random.split(rng_key)
            step_keys = jax.random.split(sample_key, current_chunk_size)
            if return_sampler:
                state, (positions, log_prob, accepted) = run_steps(state, step_keys)
                chain_chunks.append(positions)
                log_prob_chunks.append(log_prob)
                acceptance_chunks.append(accepted)
            else:
                keep_steps = (
                    jnp.arange(completed_steps, completed_steps + current_chunk_size)
                    >= burn
                )
                state, best_position, best_log_prob = run_steps_best(
                    state,
                    best_position,
                    best_log_prob,
                    step_keys,
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
        BlackJAXFitState(coords=jax.device_get(state.position))
        if return_walker_state
        else None
    )
    if return_sampler:
        chain = jax.device_get(jnp.concatenate(chain_chunks, axis=0))
        log_prob = jax.device_get(jnp.concatenate(log_prob_chunks, axis=0))
        acceptance = jax.device_get(jnp.concatenate(acceptance_chunks, axis=0))
        sampler = BlackJAXSamplerResult(chain, log_prob, acceptance=acceptance)
        flat_samples = sampler.get_chain(flat=True, discard=burn)
        flat_log_prob = sampler.get_log_prob(flat=True, discard=burn)
        sls = _best_fit_system_from_samples(
            flat_samples, flat_log_prob, Rcomp, Mcomp, Teff, d
        )
    else:
        sampler = None
        sls = _best_fit_system_from_samples(
            jax.device_get(best_position[jnp.newaxis, :]),
            jax.device_get(best_log_prob[jnp.newaxis]),
            Rcomp,
            Mcomp,
            Teff,
            d,
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
    random_seed=0,
    progress_chunk_size=100,
    return_sampler=False,
    return_walker_state=False,
):
    """Fit many independent systems in one batched BlackJAX-style run.

    Observation arrays must have a leading fit dimension. Ragged RV/LC series
    should be padded to a common length and accompanied by boolean masks. The
    sampler tracks the best post-burn sample for each fit on device and copies
    the batched best positions back once at the end. When ``return_sampler`` is
    enabled, the full chain is copied back to host memory and returned as one
    ``BlackJAXSamplerResult`` per fit.
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

    default_sls = SelfLensingSystem(
        0 * SelfLensingSystem.param_units["Macc"],
        0 * SelfLensingSystem.param_units["Mcomp"],
        0 * SelfLensingSystem.param_units["Rcomp"],
        0 * SelfLensingSystem.param_units["a"],
    )
    detected_periods = jnp.array(_batched_periods(periods, nfits))
    logprob_vec = make_batched_combined_chi2_vec_jax(
        Mcomp.value,
        Rcomp.value,
        tau_eff.value if tau_eff is not None else None,
        d.value,
        peak_time,
        rv_times,
        observed_rv,
        observed_rv_err,
        lc_times,
        observed_lc,
        observed_lc_err,
        gp_model,
        detected_periods,
        lensing_method,
        center_lc_model,
        weight_lc_peak,
        period_min,
        period_max,
        default_sls,
        rv_mask,
        lc_mask,
        fit_device,
    )

    nwalkers = int(nwalkers)
    nsteps = int(nsteps)
    burn = int(burn)
    p0 = _initial_batched_walker_positions(nfits, nwalkers, initial_walker_state)
    nwalkers = p0.shape[1]
    position = jax.device_put(jax_numpy.asarray(p0), fit_device)
    log_prob = logprob_vec(position)

    if proposal_scale is None:
        proposal_scale = _default_blackjax_proposal_scale(proposal_scale_fraction)
    proposal_scale = jax.device_put(jax_numpy.asarray(proposal_scale), fit_device)

    def one_step(carry, rng_key):
        position, log_prob = carry
        proposal_key, accept_key = jax.random.split(rng_key)
        proposal = position + jax.random.normal(
            proposal_key, shape=position.shape
        ) * proposal_scale
        proposal_log_prob = logprob_vec(proposal)
        log_accept_ratio = proposal_log_prob - log_prob
        accepted = jnp.log(
            jax.random.uniform(accept_key, shape=log_prob.shape)
        ) < log_accept_ratio
        new_position = jnp.where(accepted[..., None], proposal, position)
        new_log_prob = jnp.where(accepted, proposal_log_prob, log_prob)
        return (new_position, new_log_prob), (new_position, new_log_prob, accepted)

    @jax.jit
    def run_steps(position, log_prob, rng_keys):
        return jax.lax.scan(one_step, (position, log_prob), rng_keys)

    def one_best_step(carry, inputs):
        position, log_prob, best_position, best_log_prob = carry
        rng_key, keep_sample = inputs
        (position, log_prob), _ = one_step((position, log_prob), rng_key)
        best_walker_indices = jnp.argmax(log_prob, axis=1)
        fit_indices = jnp.arange(nfits)
        step_best_log_prob = log_prob[fit_indices, best_walker_indices]
        step_best_position = position[fit_indices, best_walker_indices]
        use_step_best = keep_sample & (step_best_log_prob > best_log_prob)
        best_log_prob = jnp.where(use_step_best, step_best_log_prob, best_log_prob)
        best_position = jnp.where(
            use_step_best[:, None],
            step_best_position,
            best_position,
        )
        return (position, log_prob, best_position, best_log_prob), None

    @jax.jit
    def run_steps_best(position, log_prob, best_position, best_log_prob, rng_keys, keep_steps):
        (position, log_prob, best_position, best_log_prob), _ = jax.lax.scan(
            one_best_step,
            (position, log_prob, best_position, best_log_prob),
            (rng_keys, keep_steps),
        )
        return position, log_prob, best_position, best_log_prob

    rng_key = jax.random.PRNGKey(random_seed)
    chain_chunks = []
    log_prob_chunks = []
    acceptance_chunks = []
    best_position = jnp.zeros_like(position[:, 0, :])
    best_log_prob = jnp.full((nfits,), -jnp.inf)
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
        chunk_size = nsteps if not progress else max(1, int(progress_chunk_size))
        while remaining_steps > 0:
            current_chunk_size = min(chunk_size, remaining_steps)
            rng_key, sample_key = jax.random.split(rng_key)
            step_keys = jax.random.split(sample_key, current_chunk_size)
            if return_sampler:
                (position, log_prob), (positions, chunk_log_prob, accepted) = run_steps(
                    position, log_prob, step_keys
                )
                chain_chunks.append(positions)
                log_prob_chunks.append(chunk_log_prob)
                acceptance_chunks.append(accepted)
            else:
                keep_steps = (
                    jnp.arange(completed_steps, completed_steps + current_chunk_size)
                    >= burn
                )
                position, log_prob, best_position, best_log_prob = run_steps_best(
                    position,
                    log_prob,
                    best_position,
                    best_log_prob,
                    step_keys,
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
        BlackJAXFitState(coords=jax.device_get(position))
        if return_walker_state
        else None
    )
    systems = _best_fit_systems_from_positions(
        jax.device_get(best_position),
        Rcomp,
        Mcomp,
        Teff,
        d,
    )
    if return_sampler:
        chain = jax.device_get(jnp.concatenate(chain_chunks, axis=0))
        log_prob = jax.device_get(jnp.concatenate(log_prob_chunks, axis=0))
        acceptance = jax.device_get(jnp.concatenate(acceptance_chunks, axis=0))
        samplers = [
            BlackJAXSamplerResult(
                chain[:, fit_index, :, :],
                log_prob[:, fit_index, :],
                acceptance=acceptance[:, fit_index, :],
            )
            for fit_index in range(nfits)
        ]
    else:
        samplers = None
    return systems, samplers, final_walker_state
