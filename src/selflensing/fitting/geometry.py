"""Exact sampler-coordinate transforms at the model's longitude-zero epoch.

b_source is in stellar radii, not Einstein radii, and is not a minimization
of projected distance over longitude. The valid inclination branch is [0, pi/2].
"""
import numpy as np
import jax.numpy as jnp
from ..orbits import AU_M, R_SUN_M


def b_source_scale(a_au, radius_rsun, eccentricity, periapsis, *, xp=jnp):
    """r(longitude=0)/Rstar, the exact scale multiplying cos(i)."""
    return a_au * AU_M / (radius_rsun * R_SUN_M) * (1-eccentricity)*(1+eccentricity) / (1+eccentricity*xp.cos(periapsis))


def inclination_from_b_source(a_au, b_source, radius_rsun, eccentricity, periapsis, *, xp=jnp):
    scale = b_source_scale(a_au, radius_rsun, eccentricity, periapsis, xp=xp)
    return xp.arccos(xp.clip(b_source / scale, -1., 1.))


def physical_to_geometry(positions, radius_rsun):
    """Convert physical (a[AU], i[rad], ...) arrays to b_source coordinates."""
    position = jnp.asarray(positions)
    scale = b_source_scale(position[..., 0], radius_rsun, position[..., 3], position[..., 4])
    return position.at[..., 1].set(scale*jnp.cos(position[..., 1]))


def geometry_to_physical(positions, radius_rsun, inclination_limits):
    """Return physical coordinates and log|di/db_source|, or -inf off support.

    The i=0 endpoint has a singular, integrable density and is excluded as a
    measure-zero boundary; it must not introduce +inf log probabilities.
    All non-crossing systems within the original inclination prior are retained.
    """
    position = jnp.asarray(positions)
    scale = b_source_scale(position[..., 0], radius_rsun, position[..., 3], position[..., 4])
    cosine = position[..., 1] / jnp.where(scale > 0, scale, 1.)
    valid = (scale > 0) & (cosine >= 0) & (cosine < 1)
    safe_cosine = jnp.clip(cosine, 0., jnp.nextafter(jnp.array(1.), jnp.array(0.)))
    inclination = jnp.arccos(safe_cosine)
    valid &= (inclination >= inclination_limits[0]) & (inclination <= inclination_limits[1])
    log_jacobian = -jnp.log(jnp.where(scale > 0, scale, 1.)) - .5*jnp.log1p(-safe_cosine**2)
    return position.at[..., 1].set(inclination), jnp.where(valid, log_jacobian, -jnp.inf)


def validate_geometry(name, limits):
    if name not in ('inclination', 'b_source'):
        raise ValueError('inclination_parameterization must be inclination or b_source')
    if name == 'b_source' and not (0 <= limits[0] < limits[1] <= np.pi/2):
        raise ValueError('b_source requires inclination limits within [0, pi/2]')


def log_abs_di_db_source(a_au, b_source, radius_rsun, eccentricity, periapsis, *, xp=jnp):
    """Interior Jacobian for valid saved draws; supports NumPy postprocessing."""
    scale = b_source_scale(a_au, radius_rsun, eccentricity, periapsis, xp=xp)
    return -xp.log(scale) - .5*xp.log1p(-(b_source/scale)**2)
