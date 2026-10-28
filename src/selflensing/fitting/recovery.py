"""Frozen Student-t mixture independence proposals and exact MH correction.

These are proposal distributions, never additional likelihoods or priors.
Mixture centres may be learned during discarded warm-up only.
"""
import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular
from jax.scipy.special import logsumexp


def select_centres(positions, scores, scale, *, count=16, log_density_cut=20., periodic_index=None):
    """Select diverse high-physical-density positions, retaining distinct regions.

    Arrays have shape (fit, draw, parameter). Starts at the highest score and
    uses farthest-point selection in standardized coordinates. Duplicates are
    permitted when fewer than count eligible points exist; their mixture weights
    are still included exactly in the proposal density.
    """
    finite = jnp.isfinite(scores) & jnp.all(jnp.isfinite(positions), axis=-1)
    best = jnp.max(jnp.where(finite, scores, -jnp.inf), axis=1, keepdims=True)
    eligible = finite & (scores >= best-log_density_cut)
    safe = jnp.where(eligible[..., None], positions, 0.)
    n = jnp.maximum(eligible.sum(axis=1, keepdims=True), 1)
    mean = safe.sum(axis=1)/n
    variance = jnp.sum(jnp.where(eligible[..., None], (safe-mean[:,None,:])**2, 0.), axis=1)/n
    units = jnp.maximum(jnp.sqrt(variance), jnp.asarray(scale))
    if periodic_index is not None:
        units = units.at[:,periodic_index].set(jnp.pi)
    index = jnp.argmax(jnp.where(eligible, scores, -jnp.inf), axis=1)
    selected = jnp.take_along_axis(positions, index[:,None,None], axis=1)[:,0,:]
    minimum = jnp.full(scores.shape, jnp.inf)

    def body(carry, _):
        selected, minimum = carry
        delta = positions-selected[:,None,:]
        if periodic_index is not None:
            phase = delta[...,periodic_index]
            delta = delta.at[...,periodic_index].set(jnp.arctan2(jnp.sin(phase),jnp.cos(phase)))
        distance = jnp.sum((delta/units[:,None,:])**2, axis=-1)
        minimum = jnp.minimum(minimum, jnp.where(eligible, distance, -jnp.inf))
        next_index = jnp.argmax(minimum, axis=1)
        next_selected = jnp.take_along_axis(positions, next_index[:,None,None], axis=1)[:,0,:]
        return (next_selected, minimum), selected

    _, centres = jax.lax.scan(body, (selected, minimum), None, length=int(count))
    return jnp.swapaxes(centres,0,1)


def mixture_log_density(position, centres, transform, df=3.):
    """Log Student-t mixture density up to a common per-fit constant.

    The omitted normalizer (including determinant) is identical for x and y
    because the transform is fixed within a kernel and shared by all centres.
    """
    nfits, nwalkers, ndim = position.shape
    count = centres.shape[1]
    delta = position[:,:,None,:]-centres[:,None,:,:]
    def whiten(matrix, differences):
        flat = differences.reshape(-1,ndim)
        return solve_triangular(matrix,flat.T,lower=True).T.reshape(nwalkers,count,ndim)
    whitened = jax.vmap(whiten)(transform,delta)
    kernels = -.5*(df+ndim)*jnp.log1p(jnp.sum(whitened**2,axis=-1)/df)
    return logsumexp(kernels,axis=-1)-jnp.log(float(count))


def independence_candidate(key, position, centres, transform, df=3.):
    """Draw y independently and return log q(x)-log q(y)."""
    index_key, normal_key, gamma_key = jax.random.split(key,3)
    index = jax.random.randint(index_key,position.shape[:-1],0,centres.shape[1])
    centre = jnp.take_along_axis(centres,index[...,None],axis=1)
    normal = jax.random.normal(normal_key,position.shape)
    chi2 = 2*jax.random.gamma(gamma_key,df/2,shape=position.shape[:-1])
    noise = normal*jnp.sqrt(df/jnp.maximum(chi2,jnp.finfo(normal.dtype).tiny))[...,None]
    proposal = centre+jnp.einsum('fij,fwj->fwi',transform,noise)
    correction = mixture_log_density(position,centres,transform,df)-mixture_log_density(proposal,centres,transform,df)
    return proposal, correction


def mixture_candidate(key, position, transform, covariance, centres, recovery_transform,
                      probability=.1, df=3.):
    """Mix a symmetric local kernel with a corrected independence kernel.

    Component selection is independent of state. Each component has its own
    valid MH acceptance rule; their constant-weight mixture preserves the target.
    """
    from .adaptation import proposal_step
    local_key, recovery_key, choose_key = jax.random.split(key,3)
    increment, local_mask = proposal_step(local_key,position,transform,covariance)
    independent, correction = independence_candidate(recovery_key,position,centres,recovery_transform,df)
    recovery = jax.random.uniform(choose_key,position.shape[:-1]) < probability
    candidate = jnp.where(recovery[...,None],independent,position+increment)
    attempted = jnp.ones_like(position) if covariance else local_mask
    # Local-scale tuning uses local attempts only, never recovery acceptance.
    attempted = attempted*(~recovery)[...,None]
    return candidate,jnp.where(recovery,correction,0.),attempted,recovery
