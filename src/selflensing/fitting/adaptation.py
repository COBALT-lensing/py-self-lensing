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


def proposal_step(key, position, transform, covariance=False):
    """Symmetric increments, with independent coordinate choices per walker."""
    noise_key, coordinate_key = jax.random.split(key)
    noise = jax.random.normal(noise_key, position.shape)
    if covariance:
        return jnp.einsum('...ij,...wj->...wi', transform, noise), None
    coordinate = jax.random.randint(coordinate_key, position.shape[:-1], 0, position.shape[-1])
    mask = jax.nn.one_hot(coordinate, position.shape[-1])
    return noise * jnp.diagonal(transform, axis1=-2, axis2=-1)[..., None, :] * mask, mask


def tune_proposals(logprob, positions, scale, *, steps=400, rounds=8,
                   random_seed=0, covariance=False, restart=False,
                   target_acceptance=None):
    """Tune per-fit, per-coordinate scales; optionally learn full covariance.

    Positions have shape (nfits, nwalkers, ndim). Coordinate acceptance uses
    only attempted moves of that coordinate, so unconstrained directions do
    not inflate the scales of constrained ones. No production chain is returned.
    Elite restarts are opt-in because they may discard separated modes.
    """
    position = jnp.asarray(positions)
    if position.ndim != 3 or min(position.shape) < 1:
        raise ValueError('positions must have shape (nfits, nwalkers, ndim)')
    steps, rounds = int(steps), int(rounds)
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

    def one_step(carry, key, transform):
        position, probability = carry
        proposal_key, accept_key = jax.random.split(key)
        increment, mask = proposal_step(proposal_key, position, transform, covariance)
        proposal = position + increment
        next_probability = logprob(proposal)
        accepted = jnp.log(jax.random.uniform(accept_key, probability.shape)) < next_probability - probability
        position = jnp.where(accepted[..., None], proposal, position)
        probability = jnp.where(accepted, next_probability, probability)
        attempted = jnp.ones_like(position) if covariance else mask
        return (position, probability), (position, probability, accepted[..., None] * attempted, attempted)

    @jax.jit
    def run(position, probability, keys, transform):
        return jax.lax.scan(lambda carry, key: one_step(carry, key, transform),
                            (position, probability), keys)

    history = []
    key = jax.random.PRNGKey(random_seed)
    if steps:
        rounds = min(rounds, steps)
        for index in range(rounds):
            key, block_key = jax.random.split(key)
            length = steps // rounds + (index < steps % rounds)
            (position, probability), (chain, scores, accepted, attempted) = run(
                position, probability, jax.random.split(block_key, length), transform)
            attempts = jnp.sum(attempted, axis=(0, 2))
            rate = jnp.sum(accepted, axis=(0, 2)) / jnp.maximum(attempts, 1)
            history.append(np.asarray(rate))
            # Diminishing gain; keep changes bounded in each warm-up block.
            factor = jnp.exp(jnp.clip(2 / np.sqrt(index + 1) * (rate - target), -np.log(2), np.log(2)))
            factor = jnp.where(attempts > 0, factor, 1.)
            if covariance:
                # Learn from all finite warm-up states, not just high scores.
                flat = jnp.transpose(chain, (1, 0, 2, 3)).reshape(nfits, -1, ndim)
                centred = flat - jnp.mean(flat, axis=1, keepdims=True)
                empirical = jnp.einsum('fki,fkj->fij', centred, centred) / max(flat.shape[1] - 1, 1)
                prior_cov = transform @ jnp.swapaxes(transform, -1, -2)
                cov = .9 * empirical + .1 * prior_cov
                transform = jax.vmap(jnp.linalg.cholesky)(cov * (2.38**2 / ndim))
                transform = transform * factor[..., None]
            else:
                transform = transform * factor[..., None]
            if restart:
                flat = jnp.transpose(chain, (1, 0, 2, 3)).reshape(nfits, -1, ndim)
                scores = jnp.transpose(scores, (1, 0, 2)).reshape(nfits, -1)
                _, indices = jax.lax.top_k(jnp.where(jnp.isfinite(scores), scores, -jnp.inf), nwalkers)
                position = jnp.take_along_axis(flat, indices[..., None], axis=1)
                probability = logprob(position)
    return WarmupResult(position, transform, np.asarray(history).reshape(-1, nfits, ndim))
