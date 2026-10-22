import numpy as np
import jax
import jax.numpy as jnp
from astropy import constants as const
from tinygp import GaussianProcess, kernels

# LensingGP.py, Line 15 (CORRECTED)
from tinygp.kernels.quasisep import SHO
import rustlens

# ---- Physics (single source of truth for R(M); SI meters) ----
from .PhysicalRelations import stellar_radius_m  # R(M) in meters
from ..lensing import wm_callback

# Astrophysical constants (SI)
M_sun = const.M_sun.value
R_sun = const.R_sun.value
G = const.G.value
c = const.c.value
PI = jnp.pi
TWOPI = 2.0 * PI


# ----------------------------------------------------------------------------
# RustLens callback for self-lensing magnification with JAX compatibility
# ----------------------------------------------------------------------------
# def _run_nucs_safe(
#     offsets: np.ndarray, source_radius: float, R_s_val: float, R_E_val: float
# ) -> np.ndarray:
#     """
#     Safe wrapper around rustlens.integrated_witt_mao_magnification.
#     Offsets are passed in units of source radius (dimensionless).
#     Returns unity profile on failure to keep GP stable.
#     """
#     bs = offsets / source_radius
#     if (
#         np.isnan(R_E_val)
#         or np.isnan(R_s_val)
#         or np.any(np.isnan(bs))
#         or np.isinf(R_E_val)
#         or np.isinf(R_s_val)
#         or np.any(np.isinf(bs))
#     ):
#         return np.ones_like(offsets)
#     try:
#         return np.array(
#             rustlens.integrated_witt_mao_magnification(
#                 bs.tolist(),  # dimensionless impact parameters
#                 float(R_E_val),  # Einstein radius [m]
#                 float(R_s_val),  # source radius [m]
#             )
#         )
#     # except Exception:
#     #     return np.ones_like(offsets)
#     except BaseException:
#         print("PANIC", R_E_val, R_s_val)  # , bs)
#         raise


@jax.custom_jvp
def run_nucs_callback(
    offsets: jnp.ndarray, source_radius: float, R_s_val: float, R_E_val: float
) -> jnp.ndarray:
    """
    JAX-compatible wrapper for Einstein-radius offsets and source size.

    ``offsets`` is d/RE, ``source_radius`` is Rstar/RE, and ``R_s_val`` and
    ``R_E_val`` are physical radii in matching units. Convert the offsets to
    source radii before calling the three-argument RustLens wrapper.
    Note: we treat it as non-differentiable (zero tangent below).
    """
    out_shape = jax.ShapeDtypeStruct(offsets.shape, offsets.dtype)
    return jax.pure_callback(
        wm_callback,
        out_shape,
        offsets / source_radius,
        R_s_val,
        R_E_val,
        vmap_method="sequential",
    )


@run_nucs_callback.defjvp
def run_nucs_jvp(primals, tangents):
    primal_out = run_nucs_callback(*primals)
    tangent_out = jnp.zeros_like(primal_out)  # non-differentiable w.r.t. inputs
    return primal_out, tangent_out


# ----------------------------------------------------------------------------
# Self-lensing mean function: μ taken from theta (log10 μ as last element)
# ----------------------------------------------------------------------------
@jax.jit
def periodic_lensing_mean(pars: jnp.ndarray, T: jnp.ndarray, *,
                          eccentricity=0., periapsis_phase=0.) -> jnp.ndarray:
    """Physical self-lensing mean, with optional fixed eccentricity/periapsis.

    Legacy layout: [Ms, Mco, log10(P seconds), inclination, epoch_phase, logmu].
    epoch_phase shifts the longitude-zero epoch by epoch_phase*P/(2*pi), not
    true anomaly linearly in time. Omitted eccentricity explicitly means e=0.
    """
    from ..lightcurves import self_lensing_amplification
    from .PhysicalRelations import effective_temperature_K
    Ms, Mco, logP, inc, epoch_phase = pars[:5]
    period = 10.**logP
    a = (G*(Mco+Ms)*M_sun*(period/(2*jnp.pi))**2)**(1/3)
    row = jnp.stack([a/const.au.value, inc, Mco, Ms,
                     stellar_radius_m(Ms)/R_sun, effective_temperature_K(Ms),
                     jnp.asarray(eccentricity), jnp.asarray(periapsis_phase)])
    epoch = T[0]+epoch_phase*period/(2*jnp.pi)
    return 10.**pars[-1]*self_lensing_amplification((T-epoch)/86400, row, 0.)


# ----------------------------------------------------------------------------
# Kernels (parameters in **Hz** for frequencies; internal conversion to ω = 2πν)
# ----------------------------------------------------------------------------
def Gran_kernel(pars: jnp.ndarray) -> kernels.Kernel:
    # pars = [log10(nu_gran_Hz), log10(sigma)]
    nu_gran = 10.0 ** pars[0]  # Hz
    omega = TWOPI * nu_gran  # rad/s
    Q_gran = 1.0 / jnp.sqrt(2.0)
    sigma = 10.0 ** pars[1]
    # use keyword args to avoid order bugs
    return SHO(sigma=sigma, omega=omega, quality=Q_gran)


def Seis_kernel(pars: jnp.ndarray) -> kernels.Kernel:
    # pars = [log10(nu_s_Hz), Q_s, log10(sigma)]
    nu_s = 10.0 ** pars[0]  # Hz
    omega = TWOPI * nu_s  # rad/s
    Q_s = jnp.clip(pars[1], 1e-6, 1e6)  # ensure positive
    sigma = 10.0 ** pars[2]
    return SHO(sigma=sigma, omega=omega, quality=Q_s)


def Ellip_kernel(pars: jnp.ndarray) -> kernels.Kernel:
    """
    *** MODIFIED KERNEL ***
    Now a SHO kernel for quasi-periodicity.
    pars = [log10(nu_e_Hz), Q_e, log10(sigma)]
    """
    nu_e = 10.0 ** pars[0]  # Hz
    omega = TWOPI * nu_e  # rad/s
    Q_e = jnp.clip(pars[1], 1e-6, 1e6)  # Quality factor
    sigma = 10.0 ** pars[2]
    return SHO(sigma=sigma, omega=omega, quality=Q_e)


def Flare_kernel(pars: jnp.ndarray):
    """
    *** MODIFIED KERNEL ***
    Now a simple exponential decay (Matérn 1/2), scaled by variance.
    pars = [log10(tau_d [s]), log10(sigma)]
    """
    log_tau, log_sig = pars
    tau = 10.0**log_tau  # decay timescale in seconds
    sigma = 10.0**log_sig
    variance = sigma**2  # Calculate variance
    # k(t) = variance * exp(-|t|/scale)
    return variance * kernels.Exp(scale=tau)


# ----------------------------------------------------------------------------
# Multi-component kernel construction
# ----------------------------------------------------------------------------
def kernel_pieces(theta: jnp.ndarray, kernel_type: str) -> dict:
    """
    Split full theta into subarrays for each kernel letter.
    *** MODIFIED PACKING CONVENTION ***
      G: [log10(nu_gran_Hz), log10(sigma)]
      S: [log10(nu_seis_Hz), Q,             log10(sigma)]
      E: [log10(nu_ellip_Hz), Q,             log10(sigma)]
      F: [log10(tau_d [s]),                 log10(sigma)]
    """
    pieces = {}
    idx = 0
    if "G" in kernel_type:
        pieces["G"] = theta[idx : idx + 2]
        idx += 2
    if "S" in kernel_type:
        pieces["S"] = theta[idx : idx + 3]
        idx += 3
    if "E" in kernel_type:
        pieces["E"] = theta[idx : idx + 3]
        idx += 3  # <-- *** THIS LINE IS CHANGED ***
    if "F" in kernel_type:
        pieces["F"] = theta[idx : idx + 2]
        idx += 2
    return pieces


def make_kernel_fn(kernel_type: str):
    """
    Return a function mapping kernel hyperparameters -> TinyGP kernel sum.
    Inputs must follow the packing documented in kernel_pieces (Hz-based).
    Internally each builder converts ν [Hz] -> ω [rad/s] with ω = 2π ν.
    """

    def kernel_fn(params: jnp.ndarray) -> kernels.Kernel:
        comps = kernel_pieces(params, kernel_type)
        k = None
        if "G" in comps:
            k = Gran_kernel(comps["G"]) if k is None else k + Gran_kernel(comps["G"])
        if "S" in comps:
            k = Seis_kernel(comps["S"]) if k is None else k + Seis_kernel(comps["S"])
        if "E" in comps:
            k = Ellip_kernel(comps["E"]) if k is None else k + Ellip_kernel(comps["E"])
        if "F" in comps:
            k = Flare_kernel(comps["F"]) if k is None else k + Flare_kernel(comps["F"])
        if k is None:
            raise ValueError(f"No kernel components for '{kernel_type}'")
        return k

    return kernel_fn


# ----------------------------------------------------------------------------
# GP model builder (consumes scalar μ; last element in theta is log10 μ)
# ----------------------------------------------------------------------------
def build_model_additive(
    theta: jnp.ndarray,
    nhyp: int,
    kernel_fn,
    T: jnp.ndarray,
    include_lens_mean: int = 1,
    noise_measure: float = 0.0,
    mean_vals: jnp.ndarray | None = None,
    lens_eccentricity=0.,
    lens_periapsis_phase=0.,
) -> GaussianProcess:
    """
    theta layout reminder (for any kernel_type):
      [ kernel_hyperparameters (size = nhyp) |
        Ms, Mco, logP, inc, delta | log10(μ) ]

    - All kernel *frequency* hyperparameters are in log10(ν [Hz]) and are
      converted to ω [rad/s] internally in the kernel builders.
    - The self-lensing mean uses:
        P = 10**logP [s], inc [rad], delta [rad], masses in Msun.
      stellar_radius_m(Ms) supplies R_s [m]. Fixed eccentricity and periapsis
      are supplied through lens_eccentricity and lens_periapsis_phase; the
      legacy five-parameter layout defaults explicitly to circular.

    noise_measure:
      - Interpreted as fractional white noise relative to μ (dimensionless).
      - The GP diagonal is set to (noise_measure * μ)^2.
    """
    # Kernel from hyperparameters (in Hz space)
    kern_params = theta[:nhyp]
    kern = kernel_fn(kern_params)

    # Baseline μ from trailing log10 term
    logmu = theta[-1]
    mu = 10.0**logmu

    if include_lens_mean == 1:
        # Expect lens params immediately after kernel params:
        # [Ms, Mco, logP, inc, delta]
        lens_core = theta[nhyp : nhyp + 5]
        # Append logmu so periodic_lensing_mean can fetch μ consistently
        lens_params = jnp.concatenate([lens_core, jnp.atleast_1d(logmu)])
        mean_vals = periodic_lensing_mean(lens_params, T, eccentricity=lens_eccentricity,
                                           periapsis_phase=lens_periapsis_phase)
    elif mean_vals is None:
        # No lensing mean; flat baseline
        mean_vals = mu * jnp.ones_like(T)

    # White noise scaled to μ (same convention as before)
    noise_var = noise_measure**2

    return GaussianProcess(
        kernel=kern, X=T, diag=noise_var, mean=None, mean_value=mean_vals
    )


# JIT with the same static args as before
build_model_additive = jax.jit(
    build_model_additive, static_argnames=["nhyp", "kernel_fn", "include_lens_mean"]
)
