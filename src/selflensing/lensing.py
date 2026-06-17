import numpy
import rustlens
import jax

from jax import numpy as jnp
from scipy import special


RADIUS_LIM = 1e-5
ANNULUS_WIDTH = 1e-4
LD_COEFF = 0.6
LD_LINEAR_INT = 2.5132741228717936
ELLIPTIC_QUADRATURE_POINTS = 96
BRIGHTNESS_QUADRATURE_POINTS = 96

_LEGENDRE_NODES, _LEGENDRE_WEIGHTS = special.roots_legendre(
    ELLIPTIC_QUADRATURE_POINTS
)
_THETA = numpy.asarray(0.25 * numpy.pi * (_LEGENDRE_NODES + 1.0))
_THETA_WEIGHTS = numpy.asarray(0.25 * numpy.pi * _LEGENDRE_WEIGHTS)
_BRIGHTNESS_NODES, _BRIGHTNESS_WEIGHTS = special.roots_legendre(
    BRIGHTNESS_QUADRATURE_POINTS
)
_BRIGHTNESS_R = numpy.asarray(0.5 * (_BRIGHTNESS_NODES + 1.0))
_BRIGHTNESS_WEIGHTS = numpy.asarray(0.5 * _BRIGHTNESS_WEIGHTS)


def wm_callback(
    offsets: numpy.ndarray, R_s_val: float, R_E_val: float
) -> numpy.ndarray:
    """
    Safe wrapper around rustlens.integrated_witt_mao_magnification.
    Offsets are passed in units of source radius (dimensionless).
    Returns unity profile on failure to keep GP stable.
    """
    # bs = offsets / source_radius
    if (
        numpy.isnan(R_E_val)
        or numpy.isnan(R_s_val)
        or numpy.any(numpy.isnan(offsets))
        or numpy.isinf(R_E_val)
        or numpy.isinf(R_s_val)
        or numpy.any(numpy.isinf(offsets))
    ):
        return numpy.ones_like(offsets)
    try:
        r = numpy.array(
            rustlens.integrated_witt_mao_magnification(
                offsets.tolist(),  # dimensionless impact parameters
                float(R_E_val),  # Einstein radius [m]
                float(R_s_val),  # source radius [m]
            )
        )
        # print("Witt & Mao result:", min(r), max(r), r)
        return r
    # except Exception:
    #     return np.ones_like(offsets)
    except BaseException:
        print("PANIC", R_E_val, R_s_val)  # , bs)
        raise


def _theta_grid(dtype):
    return jnp.asarray(_THETA, dtype=dtype), jnp.asarray(_THETA_WEIGHTS, dtype=dtype)


def _brightness_grid(dtype):
    return (
        jnp.asarray(_BRIGHTNESS_R, dtype=dtype),
        jnp.asarray(_BRIGHTNESS_WEIGHTS, dtype=dtype),
    )


def _complete_ellipk(m):
    theta, weights = _theta_grid(m.dtype)
    sin2 = jnp.sin(theta) ** 2
    return jnp.sum(weights / jnp.sqrt(1.0 - m[..., None] * sin2), axis=-1)


def _complete_ellipe(m):
    theta, weights = _theta_grid(m.dtype)
    sin2 = jnp.sin(theta) ** 2
    return jnp.sum(weights * jnp.sqrt(1.0 - m[..., None] * sin2), axis=-1)


def _complete_ellippi(n, m):
    theta, weights = _theta_grid(m.dtype)
    sin2 = jnp.sin(theta) ** 2
    return jnp.sum(
        weights
        / (
            (1.0 - n[..., None] * sin2)
            * jnp.sqrt(1.0 - m[..., None] * sin2)
        ),
        axis=-1,
    )


def _point_source_magnification(u):
    safe_u = jnp.where(u == 0.0, jnp.finfo(u.dtype).tiny, u)
    return (safe_u**2 + 2.0) / (safe_u * jnp.sqrt(safe_u**2 + 4.0))


def _witt_mao_magnification_scaled(l_scaled, rstar_scaled):
    rstar_scaled2 = rstar_scaled**2
    point_value = _point_source_magnification(l_scaled)

    l_r_diff = l_scaled - rstar_scaled
    l_r_sum = l_scaled + rstar_scaled
    edge_lim = jnp.maximum(RADIUS_LIM, 10.0 * ANNULUS_WIDTH * jnp.abs(rstar_scaled))
    edge_case = jnp.abs(l_r_diff) < edge_lim

    edge_value = (
        (2.0 / rstar_scaled)
        + ((1.0 + rstar_scaled2) / rstar_scaled2)
        * (
            (jnp.pi / 2.0)
            + jnp.arcsin((rstar_scaled2 - 1.0) / (rstar_scaled2 + 1.0))
        )
    ) / jnp.pi

    kernel1 = l_r_diff**2
    kernel2 = jnp.sqrt(4.0 + kernel1)

    safe_l_r_sum = jnp.where(l_r_sum == 0.0, jnp.finfo(l_scaled.dtype).eps, l_r_sum)
    elliptic_n = 4.0 * rstar_scaled * l_scaled / (safe_l_r_sum**2)
    elliptic_k = jnp.sqrt(4.0 * elliptic_n) / kernel2
    elliptic_m = elliptic_k**2

    first_term = (l_r_sum * kernel2) / (2.0 * rstar_scaled2)
    second_term = (
        l_r_diff
        * (4.0 + (0.5 * (l_scaled**2 - rstar_scaled2)))
        / (kernel2 * rstar_scaled2)
    )
    third_term = (
        2.0
        * kernel1
        * (1.0 + rstar_scaled2)
        / (rstar_scaled2 * safe_l_r_sum * kernel2)
    )

    formula_value = (
        _complete_ellipe(elliptic_m) * first_term
        - _complete_ellipk(elliptic_m) * second_term
        + _complete_ellippi(elliptic_n, elliptic_m) * third_term
    ) / jnp.pi

    finite_source_value = jnp.where(edge_case, edge_value, formula_value)
    return jnp.where(rstar_scaled <= 0.0, point_value, finite_source_value)


def witt_mao_magnification_jax(l, re, rstar):
    """JAX-compatible approximation of RustLens' Witt-Mao magnification."""
    l = jnp.asarray(l)
    dtype = l.dtype
    re = jnp.asarray(re, dtype=dtype)
    rstar = jnp.asarray(rstar, dtype=dtype)

    rstar_scaled = rstar / re
    l_scaled = l * rstar_scaled
    formula_value = _witt_mao_magnification_scaled(l_scaled, rstar_scaled)

    valid = (
        jnp.isfinite(l)
        & jnp.isfinite(re)
        & jnp.isfinite(rstar)
        & (re > 0.0)
        & (rstar > 0.0)
    )
    return jnp.where(valid, formula_value, jnp.nan)


def integrated_witt_mao_magnification_jax(l, re, rstar):
    """JAX-native integrated Witt-Mao magnification."""
    l = jnp.asarray(l)
    dtype = l.dtype
    re = jnp.asarray(re, dtype=dtype)
    rstar = jnp.asarray(rstar, dtype=dtype)

    rstar_scaled = rstar / re
    l_scaled = l * rstar_scaled
    mu, weights = _brightness_grid(dtype)
    r = jnp.sqrt(1.0 - mu**2)

    def integrate_one(l_scaled_i):
        cumulative_flux = (
            r**2 * _witt_mao_magnification_scaled(l_scaled_i, rstar_scaled * r)
        )
        full_flux = _witt_mao_magnification_scaled(l_scaled_i, rstar_scaled)
        mag_int = jnp.pi * (
            (1.0 - LD_COEFF) * full_flux
            + LD_COEFF * jnp.sum(weights * cumulative_flux)
        )
        return mag_int / LD_LINEAR_INT

    valid = (
        jnp.isfinite(l)
        & jnp.isfinite(re)
        & jnp.isfinite(rstar)
        & (re > 0.0)
        & (rstar > 0.0)
    )
    return jnp.where(valid, jax.vmap(integrate_one)(l_scaled), jnp.nan)
