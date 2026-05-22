"""Shared helpers for fitting backends."""

import jax
import jax.numpy as jax_numpy
from quaxed import numpy as jnp


def _jax_devices(platform):
    try:
        return jax.devices(platform)
    except RuntimeError:
        return []


def _select_fit_device(device=None, prefer_gpu=True):
    if device is not None and hasattr(device, "platform"):
        return device

    if device is None or str(device).lower() == "auto":
        gpu_devices = _jax_devices("gpu") if prefer_gpu else []
        return gpu_devices[0] if gpu_devices else jax.devices()[0]

    device_name = str(device).lower()
    platform, _, device_id = device_name.partition(":")
    if platform == "cuda":
        platform = "gpu"

    devices = _jax_devices(platform)
    if not devices:
        raise RuntimeError(f"No JAX {platform!r} devices are available")

    if device_id:
        try:
            return devices[int(device_id)]
        except (ValueError, IndexError) as exc:
            raise RuntimeError(
                f"No JAX {platform!r} device with index {device_id}"
            ) from exc

    return devices[0]


def _extract_value(obj):
    """Extract unitless value from astropy Quantity or Time object.

    This function is called at Python level (before JIT tracing) to convert
    astropy objects to plain NumPy arrays, enabling efficient GPU computation.
    """
    if obj is None:
        return None

    # Handle astropy Time objects
    if hasattr(obj, "jd") and hasattr(obj, "iso"):
        return obj.jd

    # Handle astropy Quantity objects
    if hasattr(obj, "value"):
        return obj.value

    # Already a plain array or scalar
    return obj


def _device_array(value, device):
    if value is None:
        return None
    return jax.device_put(jax_numpy.asarray(value), device)


def gauss_lnprior(x, mu, sigma=None):
    if sigma is None:
        sigma = mu / 10
    # https://stackoverflow.com/questions/49810234/using-emcee-with-gaussian-priors
    gauss_norm = jnp.log(1.0 / (jnp.sqrt(2 * jnp.pi) * sigma))
    return gauss_norm - 0.5 * (x - mu) ** 2 / sigma**2
