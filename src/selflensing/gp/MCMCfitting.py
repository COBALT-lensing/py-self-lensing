import numpy as np
import jax
import jax.numpy as jnp

from . import LensingGP as lgp
from . import PriorsSamplers as ps


# =============================================================================
# 0) Hyperparameter bookkeeping
# =============================================================================


def count_nhyp(kernel_type: str) -> int:
    """
    Number of kernel hyperparameters for a given kernel_type string.

    *** FINAL CORRECTED PACKING ***
      G → [log10 ν_g, log10 σ_g]                        (2)
      S → [log10 ν_s, Q, log10 σ_s]                    (3)
      E → [log10 ν_e, Q, log10 σ_e]                    (3)
      F → [log10 τ_d [s], log10 σ_f]                   (2)
    """
    n = 0
    for c in kernel_type:
        if c == "G":
            n += 2
        elif c == "S":
            n += 3
        elif c == "E":
            n += 3
        elif c == "F":
            n += 2
        else:
            raise ValueError(
                f"Unknown kernel component '{c}' in kernel_type='{kernel_type}'"
            )
    return n


# =============================================================================
# 1) Walker initialisation
# =============================================================================


def distribute_walkers_uniform(
    param_bounds: list[tuple[float, float]] | tuple[tuple[float, float], ...],
    n_walkers: int,
    random_state=None,
) -> np.ndarray:
    """
    Purely uniform initial walker positions inside the hyper-rectangle.
    Returns: (n_walkers, ndim) float64
    """
    ndim = len(param_bounds)
    if isinstance(random_state, (int, np.integer)):
        rng = np.random.default_rng(int(random_state))
    elif isinstance(random_state, np.random.Generator):
        rng = random_state
    else:
        rng = np.random.default_rng()

    b = np.asarray(param_bounds, dtype=float)
    lows, highs = b[:, 0], b[:, 1]
    return rng.uniform(lows, highs, size=(n_walkers, ndim)).astype(float)


def init_walkers(
    n_walkers: int,
    *,
    param_bounds: list[tuple[float, float]] | tuple[tuple[float, float], ...],
    kernel_type: str,
    include_lens_mean: bool,
    # prior-informed options:
    use_kde: bool = False,
    kde3d_pack: tuple | None = None,  # 6-tuple from ps.build_kde_3d
    phys_scale: bool = False,  # True => physics-tied kernel seeds
    # optional data for σ/μ anchor & data-informed seeds:
    T: np.ndarray | None = None,
    y: np.ndarray | None = None,
    cadence_s: float | None = None,
    # mass-distance → μ (if lens included):
    ms_array: np.ndarray | None = None,
    d_array: np.ndarray | None = None,
    # optional partially specified θ (finite entries preserved):
    P0: np.ndarray | None = None,
) -> np.ndarray:
    """
    Initialise walker positions by sampling **once** from your prior logic
    (deterministic seeds via ps.sample_theta_array_simple), then jitter slightly
    within the bounds to diversify walkers.

    This is a *convenience* alternative to distribute_walkers_uniform when you
    want smarter starts.

    NOTE: This function depends on 'ps.sample_theta_array_simple'
    which you must add to PriorsSamplers.py (as provided in the previous step).
    """
    bounds = tuple(map(tuple, param_bounds))
    nhyp = count_nhyp(kernel_type)
    ndim = len(bounds)

    # base θ seed (deterministic; preserves P0 finite entries)
    # This function MUST be added to PriorsSamplers.py
    theta0 = ps.sample_theta_array_simple(
        kernel_type=kernel_type,
        nhyp=nhyp,
        combined_bounds=bounds,
        include_lens_mean=int(bool(include_lens_mean)),
        use_kde=bool(use_kde),
        phys=bool(phys_scale),
        P0=P0,
        T=T,
        y=y,
        cadence_s=cadence_s,
        kde3d_pack=kde3d_pack,
        ms_array=ms_array,
        d_array=d_array,
    )

    # small Gaussian jitter (1% of box width) clipped to bounds
    b = np.asarray(bounds, float)
    width = np.clip(b[:, 1] - b[:, 0], 1e-12, np.inf)
    jitter = 0.01 * width

    rng = np.random.default_rng()
    p0 = np.tile(theta0, (n_walkers, 1))
    p0 += rng.normal(0.0, jitter, size=p0.shape)
    p0 = np.clip(p0, b[:, 0], b[:, 1])
    return p0.astype(float)


# =============================================================================
# 2) Log-posterior (JAX) and factories
# =============================================================================


def _log_posterior_single(
    theta: jnp.ndarray,
    T: jnp.ndarray,
    y: jnp.ndarray,
    *,
    # --- Static args passed by factory ---
    kernel_fn,  # from lgp.make_kernel_fn
    log_prior_phys_fn,  # from ps.make_log_prior_physical_jit (or None)
    kde_lp_fn,  # from ps.make_kde3d_logprior_jit (or None)
    nhyp: int,
    noise_measure: float,
    combined_bounds: tuple[tuple[float, float], ...],
    include_lens_mean: int,
    mean_vals: jnp.ndarray | None = None,
    lens_eccentricity=0.,
    lens_periapsis_phase=0.,
) -> jnp.ndarray:
    """
    Single-point log-posterior: prior + GP likelihood.
    This function is JIT-compiled by the factory.
    """
    theta = jnp.asarray(theta)
    T = jnp.asarray(T)
    y = jnp.asarray(y)

    # ---- 1) Base box prior (always active) ----
    lp_bounds = ps.log_prior_bounds(theta, combined_bounds)
    # jax.debug.print("lp_bounds {x}", x=list(zip(theta, combined_bounds)))

    # ---- 2) Define the "True" branch for jax.lax.cond ----
    # This function is *only* called if lp_bounds is finite
    def compute_full_logprob(_):
        # Start with the finite bounds prior
        lp = lp_bounds

        # Add conditional priors (JIT will prune if fn is None)
        if log_prior_phys_fn is not None:
            # Extract anchors for physical prior
            Ms_est = theta[nhyp] if include_lens_mean == 1 else jnp.nan
            logP_est = theta[nhyp + 2] if include_lens_mean == 1 else jnp.nan
            logmu = theta[-1]
            lp += log_prior_phys_fn(theta, Ms_est, logP_est, logmu)

        if kde_lp_fn is not None:
            lp += kde_lp_fn(theta)

        # Add Likelihood
        gp = lgp.build_model_additive(
            theta=theta,
            nhyp=nhyp,
            kernel_fn=kernel_fn,
            T=T,
            include_lens_mean=include_lens_mean,
            noise_measure=noise_measure,
            mean_vals=mean_vals,
            lens_eccentricity=lens_eccentricity,
            lens_periapsis_phase=lens_periapsis_phase,
        )
        return lp + gp.log_probability(y)

    # ---- 3) Define the "False" branch for jax.lax.cond ----
    def return_neg_inf(_):
        return -jnp.inf

    # ---- 4) JAX-native conditional branch ----
    # This correctly branches based on the *traced* value of jnp.isfinite
    return jax.lax.cond(
        jnp.isfinite(lp_bounds),
        compute_full_logprob,
        return_neg_inf,
        operand=None,
    )


def make_logposterior_fn(
    *,
    kernel_type: str,
    nhyp: int,
    noise_measure: float,
    param_bounds: list[tuple[float, float]] | tuple[tuple[float, float], ...],
    kde3d_args: tuple | None = None,  # 6-tuple from PriorsSamplers.build_kde_3d
    phys_scale: bool = False,
    include_lens_mean: bool = True,
    mean_vals: jnp.ndarray | None = None,
    lens_eccentricity=0.,
    lens_periapsis_phase=0.,
):
    """
    Factory returning a **JIT-compiled** single-point log-posterior:
        f(theta, T, y) -> scalar (jnp.ndarray)

    Fixed eccentricity/periapsis may be supplied through lens_eccentricity
    and lens_periapsis_phase; their defaults explicitly specify a circular mean.
    The five legacy lens coordinates do not sample these two fixed parameters.

    This factory creates all static components (kernel, priors) *outside*
    the JIT boundary and passes them to the JIT-compiled function.
    """
    # --- Create all static components ---

    # 1. Kernel
    kernel_fn = lgp.make_kernel_fn(kernel_type)

    # 2. Bounds (must be tuple-of-tuples to be a static arg)
    combined_bounds = tuple(map(tuple, param_bounds))

    # 3. Physical prior (if requested)
    if phys_scale:
        # This function must exist in PriorsSamplers.py
        log_prior_phys_fn, _ = ps.make_log_prior_physical_jit(
            kernel_type, eccentricity=lens_eccentricity if include_lens_mean else None)
    else:
        log_prior_phys_fn = None

    # 4. KDE prior (if requested)
    if kde3d_args is not None and include_lens_mean:
        # This function must exist in PriorsSamplers.py
        # kde3d_args is the 6-tuple: (kde, inv_stds, grid_Ms, grid_Mco, grid_logP, log_pdf3d)
        grid_Ms, grid_Mco, grid_logP, log_pdf3d = kde3d_args[2:6]
        kde_lp_fn = ps.make_kde3d_logprior_jit(
            grid_Ms=grid_Ms,
            grid_Mco=grid_Mco,
            grid_logP=grid_logP,
            log_pdf3d=log_pdf3d,
            nhyp=nhyp,
        )
    else:
        kde_lp_fn = None

    # --- Wrap the JIT-compiled function ---

    @jax.jit
    def _wrapped(theta, T, y):
        # Call the core function, passing all computed components
        # as static arguments (captured by closure).
        return _log_posterior_single(
            theta=jnp.asarray(theta),
            T=jnp.asarray(T),
            y=jnp.asarray(y),
            # --- Static args ---
            kernel_fn=kernel_fn,
            log_prior_phys_fn=log_prior_phys_fn,
            kde_lp_fn=kde_lp_fn,
            nhyp=nhyp,
            noise_measure=jnp.array(noise_measure, float),
            combined_bounds=combined_bounds,
            include_lens_mean=int(bool(include_lens_mean)),
            mean_vals=mean_vals,
            lens_eccentricity=lens_eccentricity,
            lens_periapsis_phase=lens_periapsis_phase,
        )

    return _wrapped


def make_batched_logposterior_fn(
    *,
    kernel_type: str,
    nhyp: int,
    noise_measure: float,
    param_bounds: list[tuple[float, float]] | tuple[tuple[float, float], ...],
    kde3d_args: tuple | None = None,
    phys_scale: bool = False,
    include_lens_mean: bool = True,
    mean_vals: jnp.ndarray | None = None,
    lens_eccentricity=0.,
    lens_periapsis_phase=0.,
):
    """
    Returns a **JIT-compiled** function evaluating the log-posterior for a *batch*
    of walkers at once: f(thetas, T, y) -> (n_walkers,) (jnp.ndarray)
    """
    # This now correctly uses the fixed factory function
    single = make_logposterior_fn(
        kernel_type=kernel_type,
        nhyp=nhyp,
        noise_measure=noise_measure,
        param_bounds=param_bounds,
        kde3d_args=kde3d_args,
        phys_scale=phys_scale,
        include_lens_mean=include_lens_mean,
        mean_vals=mean_vals,
        lens_eccentricity=lens_eccentricity,
        lens_periapsis_phase=lens_periapsis_phase,
    )

    @jax.jit
    def batched(thetas, T, y):
        thetas = jnp.asarray(thetas)  # (n_walkers, ndim)
        T = jnp.asarray(T)
        y = jnp.asarray(y)
        return jax.vmap(lambda th: single(th, T, y))(thetas)

    return batched


def make_emcee_vectorized_logprob(
    *,
    kernel_type: str,
    nhyp: int,
    noise_measure: float,
    param_bounds: list[tuple[float, float]] | tuple[tuple[float, float], ...],
    kde3d_args: tuple | None = None,
    phys_scale: bool = False,
    include_lens_mean: bool = True,
    mean_vals: jnp.ndarray | None = None,
    lens_eccentricity=0.,
    lens_periapsis_phase=0.,
):
    """
    Convenience wrapper for emcee(vectorize=True).
    Returns a NumPy function: logprob(thetas, T, y) -> np.ndarray (n_walkers,)
    """
    # This also now correctly uses the fixed batched function
    batched = make_batched_logposterior_fn(
        kernel_type=kernel_type,
        nhyp=nhyp,
        noise_measure=noise_measure,
        param_bounds=param_bounds,
        kde3d_args=kde3d_args,
        phys_scale=phys_scale,
        include_lens_mean=include_lens_mean,
        mean_vals=mean_vals,
        lens_eccentricity=lens_eccentricity,
        lens_periapsis_phase=lens_periapsis_phase,
    )

    def logprob(thetas, T, y):
        # emcee feeds/consumes NumPy; keep compute on device, then bring back
        return np.asarray(batched(thetas, T, y))

    return logprob


# =============================================================================
# 3) Small utilities for sanity checks / summaries
# =============================================================================


def check_bounds_match_dim(param_bounds, ndim: int):
    """Raise if bounds length doesn't match ndim."""
    L = len(param_bounds)
    if L != ndim:
        raise ValueError(f"Bounds length {L} != ndim {ndim}")
    return True


def summarize_chain(chain: np.ndarray, logp: np.ndarray) -> dict:
    """
    Quick-and-dirty summary dict (MAP-ish, means, medians).
    chain: (N, ndim)  flat array
    logp : (N,)
    """
    best_idx = int(np.argmax(logp))
    theta_hat = chain[best_idx]
    return {
        "best_index": best_idx,
        "theta_hat": theta_hat,
        "logp_best": float(logp[best_idx]),
        "logp_mean": float(np.mean(logp)),
        "logp_median": float(np.median(logp)),
    }
