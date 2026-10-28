"""Warm-up utilities shared by single-system and batched Metropolis fits.

Adaptation and optional elite restarts are initialization only. Callers must
freeze the returned transform before collecting posterior samples.
"""
from dataclasses import dataclass
import numpy as np
import jax
import jax.numpy as jnp


@dataclass
class WarmupResult:
    positions: object
    transform: object
    acceptance: np.ndarray
    recovery_centres: object = None
    recovery_transform: object = None
    eligible_fraction: object = None
    recovery_acceptance: object = None


def proposal_step(key, position, transform, covariance=False):
    """Symmetric increments, with independent coordinate choices per walker."""
    noise_key, coordinate_key = jax.random.split(key)
    noise = jax.random.normal(noise_key, position.shape)
    if covariance:
        return jnp.einsum('...ij,...wj->...wi', transform, noise), None
    coordinate = jax.random.randint(coordinate_key, position.shape[:-1], 0, position.shape[-1])
    mask = jax.nn.one_hot(coordinate, position.shape[-1])
    return noise * jnp.diagonal(transform, axis1=-2, axis2=-1)[..., None, :] * mask, mask


def tune_proposals(logprob, positions, scale, *, steps=400, rounds=None,
                   random_seed=0, covariance=False, restart=False,
                   target_acceptance=None, robust_covariance=False,
                   score_transform=None, log_density_cut=20.,
                   recovery_probability=0., recovery_df=3., recovery_centres_count=16,
                   periodic_index=None):
    """Tune per-fit, per-coordinate scales; optionally learn full covariance.

    Positions have shape (nfits, nwalkers, ndim). Coordinate acceptance uses
    only attempted moves of that coordinate, so unconstrained directions do
    not inflate the scales of constrained ones. No production chain is returned.
    Elite restarts are opt-in because they may discard separated modes.
    """
    from .recovery import select_centres, mixture_candidate
    if not 0 <= recovery_probability <= 1 or not np.isfinite(recovery_df) or recovery_df <= 0:
        raise ValueError('Recovery probability must be in [0,1] and degrees of freedom positive')
    if int(recovery_centres_count) < 1 or not np.isfinite(log_density_cut) or log_density_cut <= 0:
        raise ValueError('Centre count and physical log-density cut must be positive')
    if robust_covariance and not covariance:
        raise ValueError('robust_covariance requires covariance proposals')
    position = jnp.asarray(positions)
    if position.ndim != 3 or min(position.shape) < 1:
        raise ValueError('positions must have shape (nfits, nwalkers, ndim)')
    steps = int(steps)
    # Longer warm-up needs more adjustments, not merely longer pilot blocks.
    rounds = max(8, int(np.ceil(steps / 250))) if rounds is None else int(rounds)
    if steps < 0 or rounds < 1:
        raise ValueError('steps must be nonnegative and rounds positive')
    nfits, nwalkers, ndim = position.shape
    scale = jnp.asarray(scale)
    if scale.ndim == 3:
        scale = scale[:, 0, :]
    scale = jnp.broadcast_to(scale, (nfits, ndim))
    if not np.all(np.isfinite(scale)) or np.any(np.asarray(scale) <= 0):
        raise ValueError('proposal scales must be positive and finite')
    transform = jax.vmap(jnp.diag)(scale)
    target = (.234 if covariance else .44) if target_acceptance is None else target_acceptance
    if not 0 < target < 1:
        raise ValueError('target_acceptance must lie between zero and one')
    probability = logprob(position)
    if not np.all(np.any(np.isfinite(np.asarray(probability)), axis=1)):
        raise ValueError('Every fit requires at least one finite initial walker')

    def physical_scores(positions, scores):
        return scores if score_transform is None else score_transform(positions,scores)

    def centres_for(positions, scores):
        flat = jnp.transpose(positions,(1,0,2,3)).reshape(nfits,-1,ndim)
        physical = jnp.transpose(physical_scores(positions,scores),(1,0,2)).reshape(nfits,-1)
        return select_centres(flat,physical,scale,count=recovery_centres_count,
                              log_density_cut=log_density_cut,periodic_index=periodic_index)

    if recovery_probability:
        recovery_centres = centres_for(position[None],probability[None])
    else:
        recovery_centres = jnp.zeros((nfits,1,ndim))
    recovery_transform = transform

    def one_step(carry, key, transform, centres, recovery_transform):
        position, probability = carry
        proposal_key, accept_key = jax.random.split(key)
        if recovery_probability:
            proposal, correction, attempted, recovery = mixture_candidate(
                proposal_key,position,transform,covariance,centres,recovery_transform,
                recovery_probability,recovery_df)
        else:
            increment, mask = proposal_step(proposal_key, position, transform, covariance)
            proposal = position + increment
            correction = 0.
            attempted = jnp.ones_like(position) if covariance else mask
            recovery = jnp.zeros_like(probability,dtype=bool)
        next_probability = logprob(proposal)
        accepted = jnp.log(jax.random.uniform(accept_key, probability.shape)) < next_probability - probability + correction
        position = jnp.where(accepted[..., None], proposal, position)
        probability = jnp.where(accepted, next_probability, probability)
        return (position, probability), (position, probability, accepted[..., None] * attempted,
                                         attempted, recovery, accepted & recovery)

    @jax.jit
    def run(position, probability, keys, transform, centres, recovery_transform):
        return jax.lax.scan(lambda carry, key: one_step(carry, key, transform,centres,recovery_transform),
                            (position, probability), keys)

    history = []
    eligible_history = []
    recovery_history = []
    covariance_scale = jnp.ones((nfits, ndim))
    key = jax.random.PRNGKey(random_seed)
    if steps:
        rounds = min(rounds, steps)
        for index in range(rounds):
            key, block_key = jax.random.split(key)
            length = steps // rounds + (index < steps % rounds)
            (position, probability), (chain, scores, accepted, attempted, recovery, recovery_accepted) = run(
                position, probability, jax.random.split(block_key, length), transform,
                recovery_centres,recovery_transform)
            rec_attempts = jnp.sum(recovery,axis=(0,2))
            recovery_history.append(np.asarray(jnp.where(rec_attempts>0,
                jnp.sum(recovery_accepted,axis=(0,2))/jnp.maximum(rec_attempts,1),jnp.nan)))
            attempts = jnp.sum(attempted, axis=(0, 2))
            rate = jnp.sum(accepted, axis=(0, 2)) / jnp.maximum(attempts, 1)
            history.append(np.asarray(rate))
            # Diminishing gain; keep changes bounded in each warm-up block.
            factor = jnp.exp(jnp.clip(2 / np.sqrt(index + 1) * (rate - target), -np.log(2), np.log(2)))
            factor = jnp.where(attempts > 0, factor, 1.)
            if covariance:
                # Robust selection changes the proposal estimate only; states are
                # neither deleted nor resampled. Use physical, not transformed,
                # log density so the rule is invariant to sampler coordinates.
                flat = jnp.transpose(chain, (1, 0, 2, 3)).reshape(nfits, -1, ndim)
                if robust_covariance:
                    physical = jnp.transpose(physical_scores(chain,scores),(1,0,2)).reshape(nfits,-1)
                    finite = jnp.isfinite(physical) & jnp.all(jnp.isfinite(flat),axis=-1)
                    maximum = jnp.max(jnp.where(finite,physical,-jnp.inf),axis=1,keepdims=True)
                    eligible = finite & (physical >= maximum-log_density_cut)
                    weight = eligible.astype(flat.dtype)
                    safe = jnp.where(eligible[...,None],flat,0.)
                    count = jnp.sum(weight,axis=1)
                    mean = jnp.sum(safe,axis=1)/jnp.maximum(count[:,None],1)
                    centred = jnp.where(eligible[...,None],safe-mean[:,None,:],0.)
                    empirical = jnp.einsum('fki,fkj->fij',centred,centred)/jnp.maximum(count-1,1)[:,None,None]
                    eligible_history.append(np.asarray(eligible.mean(axis=1)))
                else:
                    centred = flat - jnp.mean(flat, axis=1, keepdims=True)
                    empirical = jnp.einsum('fki,fkj->fij', centred, centred) / max(flat.shape[1] - 1, 1)
                    eligible_history.append(np.ones(nfits))
                # Keep shape estimation separate from accumulated step-size
                # tuning: rebuilding covariance must not reset scale learning.
                shape_transform = transform / covariance_scale[..., None]
                prior_cov = shape_transform @ jnp.swapaxes(shape_transform, -1, -2)
                if robust_covariance:
                    empirical = jnp.where((count>1)[:,None,None],empirical,prior_cov)
                cov = .9 * empirical + .1 * prior_cov
                transform = jax.vmap(jnp.linalg.cholesky)(cov * (2.38**2 / ndim))
                covariance_scale = covariance_scale * factor
                transform = transform * covariance_scale[..., None]
            else:
                transform = transform * factor[..., None]
            if recovery_probability:
                recovery_centres = centres_for(chain,scores)
                recovery_transform = transform
            if restart:
                flat = jnp.transpose(chain, (1, 0, 2, 3)).reshape(nfits, -1, ndim)
                scores = jnp.transpose(scores, (1, 0, 2)).reshape(nfits, -1)
                _, indices = jax.lax.top_k(jnp.where(jnp.isfinite(scores), scores, -jnp.inf), nwalkers)
                position = jnp.take_along_axis(flat, indices[..., None], axis=1)
                probability = logprob(position)
    return WarmupResult(position, transform, np.asarray(history).reshape(-1, nfits, ndim),
        recovery_centres if recovery_probability else None,
        recovery_transform if recovery_probability else None,
        np.asarray(eligible_history).reshape(-1,nfits),
        np.asarray(recovery_history).reshape(-1,nfits))
