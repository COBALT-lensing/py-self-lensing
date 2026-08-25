"""Experimental independent-walker, GPU-batched MALA transitions."""

from typing import NamedTuple

import jax
import jax.numpy as jnp


class BatchedMALAState(NamedTuple):
    position: jax.Array
    logprob: jax.Array
    gradient: jax.Array


def initialise(position, logprob_and_grad):
    """Initialise state for ``(fit, walker, parameter)`` positions."""
    logprob, gradient = logprob_and_grad(position)
    return BatchedMALAState(position, logprob, gradient)


def step(rng_key, state, logprob_and_grad, step_size, metric_scale):
    """Perform independent preconditioned MALA transitions for every walker."""
    step_size = jnp.asarray(step_size)
    metric_scale = jnp.asarray(metric_scale)
    if metric_scale.ndim == 1:
        metric_scale = jnp.broadcast_to(metric_scale, state.position.shape[::2])
    metric_variance = metric_scale**2
    proposal_key, accept_key = jax.random.split(rng_key)
    drift = step_size[:, None, None] * metric_variance[:, None, :] * state.gradient
    proposal = state.position + drift + (
        jnp.sqrt(2.0 * step_size)[:, None, None]
        * metric_scale[:, None, :]
        * jax.random.normal(proposal_key, state.position.shape)
    )
    proposal_logprob, proposal_gradient = logprob_and_grad(proposal)
    forward = proposal - state.position - drift
    reverse = state.position - proposal - (
        step_size[:, None, None] * metric_variance[:, None, :] * proposal_gradient
    )
    denominator = 4.0 * step_size[:, None, None] * metric_variance[:, None, :]
    log_acceptance = (
        proposal_logprob - state.logprob
        + jnp.sum(forward**2 / denominator, axis=-1)
        - jnp.sum(reverse**2 / denominator, axis=-1)
    )
    accepted = jnp.log(jax.random.uniform(accept_key, state.logprob.shape)) < log_acceptance
    return (
        BatchedMALAState(
            jnp.where(accepted[..., None], proposal, state.position),
            jnp.where(accepted, proposal_logprob, state.logprob),
            jnp.where(accepted[..., None], proposal_gradient, state.gradient),
        ),
        accepted,
    )


def run(rng_key, state, logprob_and_grad, step_size, metric_scale, nsteps):
    """JIT-compatible scan of batched MALA steps."""
    keys = jax.random.split(rng_key, nsteps)
    transition = lambda carry, key: step(
        key, carry, logprob_and_grad, step_size, metric_scale
    )
    return jax.lax.scan(transition, state, keys)
