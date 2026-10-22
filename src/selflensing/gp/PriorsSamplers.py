import numpy as np
import jax
import jax.numpy as jnp
import pandas as pd
from jax.scipy.stats import norm as jnorm
from sklearn.neighbors import KernelDensity
from astropy.timeseries import LombScargle

from . import PhysicalRelations as physr
from .FluxFinderSensitivity import FluxModel


# =============================================================================
# BOUNDS + UNIFORM PRIOR + sampler
# =============================================================================


def build_combined_bounds(
    *,
    kernel_type: str,
    kernel_bounds_dict: dict,
    lens_bounds: list[tuple[float, float]],  # [Ms, Mco, logP, inc, delta]
    mu_bounds: list[tuple[float, float]],  # [log10 μ]
    include_lens_mean: int = 1,
) -> list[tuple[float, float]]:
    """
    Return bounds in exactly the θ order:
      θ = [kernel hypers..., Ms, Mco, logP, inc, delta, log10 μ]  (include_lens_mean==1)
      θ = [kernel hypers..., log10 μ]                             (include_lens_mean==0)
    """
    bounds = []
    for c in kernel_type:
        if c not in kernel_bounds_dict:
            raise ValueError(f"kernel_bounds_dict missing '{c}'")
        part = list(kernel_bounds_dict[c])
        if any((len(b) != 2) for b in part):
            raise ValueError(f"kernel_bounds_dict['{c}'] must be [(low, high), ...]")
        bounds += part

    if include_lens_mean == 1:
        if len(lens_bounds) != 5:
            raise ValueError(
                "lens_bounds must be 5 tuples: [Ms, Mco, logP, inc, delta]"
            )
        if len(mu_bounds) != 1:
            raise ValueError("mu_bounds must be 1 tuple: [log10 μ]")
        bounds += list(lens_bounds) + list(mu_bounds)
    else:
        if len(mu_bounds) != 1:
            raise ValueError("mu_bounds must be 1 tuple: [log10 μ]")
        bounds += list(mu_bounds)

    return bounds


@jax.jit
def log_prior_bounds(
    theta: jnp.ndarray, bounds: list[tuple[float, float]]
) -> jnp.ndarray:
    """
    Uniform box prior:
      0 if all parameters are within their [low, high] bounds,
      -inf otherwise.
    """
    # Note: bounds list must be converted to tuple-of-tuples to be a static arg for JIT
    lows = jnp.asarray([lo for lo, _ in bounds], dtype=theta.dtype)
    highs = jnp.asarray([hi for _, hi in bounds], dtype=theta.dtype)
    inside_all = jnp.all((theta >= lows) & (theta <= highs))
    return jnp.where(inside_all, jnp.array(0.0, dtype=theta.dtype), -jnp.inf)


def sample_uniform(
    bounds: list[tuple[float, float]],
    n: int | None = None,
    rng: np.random.Generator | int | None = None,
    dtype=float,
) -> np.ndarray:
    """
    Uniformly sample within rectangular bounds.
      n=None → (D,)
      n=int  → (n, D)
    """
    lows = np.asarray([lo for lo, _ in bounds], dtype=float)
    highs = np.asarray([hi for _, hi in bounds], dtype=float)

    if (
        not np.all(np.isfinite(lows))
        or not np.all(np.isfinite(highs))
        or np.any(highs <= lows)
    ):
        raise ValueError("Invalid bounds passed to sample_uniform.")

    if isinstance(rng, (int, np.integer)):
        rng = np.random.default_rng(int(rng))
    elif rng is None:
        rng = np.random.default_rng()

    D = len(bounds)
    if n is None:
        out = rng.uniform(lows, highs).astype(dtype, copy=False)
    else:
        out = rng.uniform(lows, highs, size=(int(n), D)).astype(dtype, copy=False)
    return out


# =============================================================================
# PRIOR USING INITIAL GUESS (P0) + sampler
# =============================================================================


@jax.jit
def log_prior_P0(
    theta: jnp.ndarray,
    P0: jnp.ndarray,
    guess_scale: float,
) -> jnp.ndarray:
    """
    Gaussian pulls around user guesses:
      add N(theta_i | mean=P0_i, sd=guess_scale) if P0_i != 0
      add 0                                      if P0_i == 0
    """
    theta = jnp.asarray(theta, dtype=jnp.float32)
    P0 = jnp.asarray(P0, dtype=theta.dtype)

    mask = (P0 != 0.0).astype(theta.dtype)  # 1.0 where we have a guess, 0.0 else
    s = jnp.asarray(guess_scale, dtype=theta.dtype)
    s = jnp.maximum(s, jnp.finfo(theta.dtype).tiny)  # numeric safety

    z = (theta - P0) / s
    # log N = -0.5 z^2 - log(s) - 0.5 log(2π)
    log_norm_const = -jnp.log(s) - 0.5 * jnp.log(2.0 * jnp.pi)
    lp_i = -0.5 * z**2 + log_norm_const

    # only count where we actually have a guess
    return jnp.sum(mask * lp_i)


def sample_P0(
    key: jax.Array,
    P0: jnp.ndarray,
    guess_scale: float,
    bounds: list[tuple[float, float]],
) -> jnp.ndarray:
    """
    Draw θ given user guesses and bounds:
      - if P0[i] != 0: θ_i ~ N(P0[i], guess_scale)
      - if P0[i] == 0: θ_i := midpoint(bounds_i)
    Then clip to bounds.
    """
    P0 = jnp.asarray(P0, dtype=jnp.float32)

    lows = jnp.asarray([lo for lo, _ in bounds], dtype=P0.dtype)
    highs = jnp.asarray([hi for _, hi in bounds], dtype=P0.dtype)
    mid = 0.5 * (lows + highs)

    s = jnp.array(guess_scale, dtype=P0.dtype)
    s = jnp.maximum(s, jnp.finfo(P0.dtype).tiny)

    mask = P0 != 0.0
    eps = jax.random.normal(key, shape=P0.shape)
    draws = P0 + s * eps

    theta = jnp.where(mask, draws, mid)
    return jnp.clip(theta, lows, highs)


# =============================================================================
# PHYSICAL PRIORS + sampler
# =============================================================================


def ppm_to_logfrac(ppm: float) -> float:
    """
    Convert parts-per-million into log10 fractional amplitude.
    Example: 1000 ppm = 1e-3 -> log10(1e-3) = -3
    """
    return np.log10(ppm / 1e6)


# Fractional σ/μ brackets (log10 of a fraction). Used to anchor absolute σ via μ.
FRAC_RANGES = {
    "G": (ppm_to_logfrac(50), ppm_to_logfrac(300)),  # granulation
    "S": (ppm_to_logfrac(100), ppm_to_logfrac(1000)),  # solar-like osc
    "E": (ppm_to_logfrac(1000), ppm_to_logfrac(10000)),  # ellipsoidal
    "F": (ppm_to_logfrac(500), ppm_to_logfrac(50000)),  # flare-like
}


# Broad coherent-p-mode modelling prior, not a calibrated population law.
# Q ~= nu / linewidth for a narrow SHO resonance; observed linewidths vary
# with temperature (https://arxiv.org/abs/1112.3295).
SEISMIC_Q_BOUNDS = (100.0, 10_000.0)


def sample_quality_factor(rng, *, bounds=SEISMIC_Q_BOUNDS, size=None):
    """Draw seismic Q uniformly in log10(Q), using a caller-owned RNG."""
    low, high = bounds
    if not np.isfinite(low) or not np.isfinite(high) or low <= 0.5 or high <= low:
        raise ValueError("Quality bounds must be finite, increasing, and above 0.5")
    return 10.0 ** rng.uniform(np.log10(low), np.log10(high), size=size)


def sample_log_sigma(logmu, tag, rng, *, ppm=None, size=None):
    """Draw an amplitude uniformly in linear ppm, then convert to log counts.

    Supply ppm to use an externally constrained component amplitude instead.
    An observed total RMS alone cannot separate the GP components.
    """
    if ppm is None:
        lo, hi = FRAC_RANGES[tag]
        ppm = rng.uniform(1e6 * 10.0**lo, 1e6 * 10.0**hi, size=size)
    if not np.all(np.isfinite(ppm)) or np.any(np.asarray(ppm) <= 0):
        raise ValueError("Component amplitudes must be positive finite ppm")
    return expected_log_sigma(logmu, tag, ppm=ppm)


def expected_log_sigma(
    logmu: jnp.ndarray, tag: str, ppm: float | None = None
) -> jnp.ndarray:
    """
    Expected log10 σ (σ in e-/cadence) given log10 μ and a fractional band for `tag`.
    If `ppm` is provided, use that; else geometric mean of FRAC_RANGES[tag].
    """
    mu_flux = jnp.power(10.0, logmu)  # μ in e-/cadence
    if ppm is not None:
        frac = jnp.asarray(ppm / 1e6)
    else:
        lo, hi = FRAC_RANGES[tag]  # Python floats known at trace time
        frac = jnp.power(10.0, 0.5 * (lo + hi))
    sigma_abs = jnp.maximum(mu_flux * frac, 1e-30)
    return jnp.log10(sigma_abs)


def _kernel_hyper_indices(kernel_type: str):
    """
    Map `kernel_type` to integer indices inside theta[:nhyp].
      G = [lnu_g,       lsig_g]                     -> use both
      S = [lnu_s, Q,    lsig_s]                     -> use lnu_s & lsig_s (skip Q)
      E = [lnu_e, Q,    lsig_e]                     -> *** THIS COMMENT IS NOW CORRECT ***
      F = [ltau_f,      lsig_f]                     -> use lsig_f only
    Returns (idx_G, idx_S, idx_E, idx_F, nhyp), each idx_* is (i_lnu_like, i_lsig_like) or (-1, -1).
    """
    i = 0
    idx_G = (-1, -1)
    idx_S = (-1, -1)
    idx_E = (-1, -1)
    idx_F = (-1, -1)

    for c in kernel_type:
        if c == "G":
            idx_G = (i + 0, i + 1)
            i += 2
        elif c == "S":
            idx_S = (i + 0, i + 2)
            i += 3
        elif c == "E":
            idx_E = (i + 0, i + 2)
            i += 3  # Corrected logic
        elif c == "F":
            idx_F = (i + 0, i + 1)
            i += 2
        else:
            raise ValueError(
                f"Unknown kernel component '{c}' in kernel_type='{kernel_type}'"
            )

    return idx_G, idx_S, idx_E, idx_F, i


def make_log_prior_physical_jit(kernel_type: str, *, scale: float = 0.3, eccentricity=None):
    """
    Factory returning a **JIT-compiled** physical prior:

        log_prior_phys(theta, Ms_est, logP_est, logmu) -> scalar log-prior

    Notes:
      * No dicts or Python branching inside JIT; all layout resolved beforehand.
      * Uses jnp.where with finiteness guards so NaNs disable a term.
      * Wraps PhysicalRelations functions in tiny jitted helpers.
      * The ellipsoidal 2/P frequency prior is enabled only when the caller
        explicitly supplies eccentricity=0; unknown/eccentric cases skip it.
    """
    idx_G, idx_S, idx_E, idx_F, nhyp = _kernel_hyper_indices(kernel_type)
    has_G = idx_G[0] >= 0
    has_S = idx_S[0] >= 0
    has_E = idx_E[0] >= 0
    has_F = idx_F[0] >= 0
    s = float(scale)

    @jax.jit
    def _nu_g(Ms):
        return jnp.asarray(physr.granulation_frequency_relation(Ms))

    @jax.jit
    def _nu_s(Ms):
        return jnp.asarray(physr.asteroseismic_frequency_relation(Ms))

    @jax.jit
    def log_prior_phys(
        theta: jnp.ndarray,
        Ms_est: jnp.ndarray,
        logP_est: jnp.ndarray,
        logmu: jnp.ndarray,
    ) -> jnp.ndarray:
        th = jnp.asarray(theta)
        dtype = th.dtype
        zero = jnp.array(0.0, dtype)
        tiny = jnp.array(1e-30, dtype)
        sc = jnp.array(s, dtype)

        ms_ok = jnp.isfinite(Ms_est)
        logP_ok = jnp.isfinite(logP_est)
        mu_ok = jnp.isfinite(logmu)

        lp = jnp.array(0.0, dtype)

        # ---- G
        if has_G:
            i_lnu_g, i_lsig_g = idx_G
            lnu_g = th[i_lnu_g]
            lsig_g = th[i_lsig_g]
            lp += jnp.where(
                ms_ok,
                jnorm.logpdf(
                    lnu_g, loc=jnp.log10(jnp.maximum(_nu_g(Ms_est), tiny)), scale=sc
                ),
                zero,
            )
            lp += jnp.where(
                mu_ok,
                jnorm.logpdf(lsig_g, loc=expected_log_sigma(logmu, "G"), scale=sc),
                zero,
            )

        # ---- S (skip Q)
        if has_S:
            i_lnu_s, i_lsig_s = idx_S
            lnu_s = th[i_lnu_s]
            lsig_s = th[i_lsig_s]
            lp += jnp.where(
                ms_ok,
                jnorm.logpdf(
                    lnu_s, loc=jnp.log10(jnp.maximum(_nu_s(Ms_est), tiny)), scale=sc
                ),
                zero,
            )
            lp += jnp.where(
                mu_ok,
                jnorm.logpdf(lsig_s, loc=expected_log_sigma(logmu, "S"), scale=sc),
                zero,
            )

        # ---- E
        if has_E:
            i_lnu_e, i_lsig_e = idx_E
            lnu_e = th[i_lnu_e]
            lsig_e = th[i_lsig_e]
            loc_e = jnp.log10(jnp.array(2.0, dtype)) - logP_est
            lp += jnp.where(logP_ok & (eccentricity is not None and eccentricity == 0),
                            jnorm.logpdf(lnu_e, loc=loc_e, scale=sc), zero)
            lp += jnp.where(
                mu_ok,
                jnorm.logpdf(lsig_e, loc=expected_log_sigma(logmu, "E"), scale=sc),
                zero,
            )

        # ---- F (only amplitude anchor)
        if has_F:
            _i_ltau_f, i_lsig_f = idx_F
            lsig_f = th[i_lsig_f]
            lp += jnp.where(
                mu_ok,
                jnorm.logpdf(lsig_f, loc=expected_log_sigma(logmu, "F"), scale=sc),
                zero,
            )

        return lp

    return log_prior_phys, nhyp


def sample_physical(
    kernel_type: str,
    *,
    logmu: float = np.nan,
    Ms_est: float = np.nan,
    logP_est: float = np.nan,
    eccentricity: float | None = None,
) -> np.ndarray:
    """
    Deterministic kernel seeds placed at the *centers* of the physical priors.
    The 2/P ellipsoidal anchor is used only for explicitly known e=0.
    Unknown/eccentric systems get an unanchored frequency seed.
    """
    out = []
    for c in kernel_type:
        if c == "G":
            lnu = (
                np.log10(
                    max(float(physr.granulation_frequency_relation(Ms_est)), 1e-30)
                )
                if np.isfinite(Ms_est)
                else -3.5
            )
            lsig = (
                float(expected_log_sigma(jnp.asarray(logmu), "G"))
                if np.isfinite(logmu)
                else -3.0
            )
            out += [lnu, lsig]

        elif c == "S":
            lnu = (
                np.log10(
                    max(float(physr.asteroseismic_frequency_relation(Ms_est)), 1e-30)
                )
                if np.isfinite(Ms_est)
                else -3.5
            )
            Q = np.sqrt(np.prod(SEISMIC_Q_BOUNDS))
            lsig = (
                float(expected_log_sigma(jnp.asarray(logmu), "S"))
                if np.isfinite(logmu)
                else -3.0
            )
            out += [lnu, Q, lsig]

        elif c == "E":
            lnu = ((np.log10(2.0)-float(logP_est))
                   if np.isfinite(logP_est) and eccentricity == 0 else -4.0)
            Q = 50.0
            lsig = (
                float(expected_log_sigma(jnp.asarray(logmu), "E"))
                if np.isfinite(logmu)
                else -3.0
            )
            out += [lnu, Q, lsig]

        elif c == "F":
            ltau = np.log10(2.0 * 3600.0)  # ~2 hr
            lsig = (
                float(expected_log_sigma(jnp.asarray(logmu), "F"))
                if np.isfinite(logmu)
                else -3.0
            )
            out += [ltau, lsig]

        else:
            raise ValueError(f"Unknown kernel component '{c}'")

    return np.asarray(out, dtype=float)


# =============================================================================
# KDE PRIOR + sampler
# =============================================================================


def build_kde_1d(
    csv_paths,
    column: str = "Ms",
    bandwidth: float | None = 0.2,
    grid_size: int = 500,
    expand_frac: float = 0.0,
):
    """
    1D KDE for a single column.
    If column == 'Porb' (days), converts to logP = log10(P [s]) before fitting.
    """
    if isinstance(csv_paths, (list, tuple)):
        df = pd.concat([pd.read_csv(p) for p in csv_paths], ignore_index=True)
    else:
        df = pd.read_csv(csv_paths)

    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found. Available: {list(df.columns)}")

    data = df[column].to_numpy(copy=True)
    if column == "Porb":
        data = np.log10(data * 86400.0)  # days -> seconds -> log10
    data = data[np.isfinite(data)]
    if data.size == 0:
        raise ValueError(f"Column '{column}' has no finite data.")

    if bandwidth is None:
        std = max(np.std(data), 1e-6)
        N = max(int(data.size), 2)
        bw = 1.06 * std * N ** (-1 / 5)  # Silverman
    else:
        bw = float(bandwidth)
    bw = max(bw, 1e-6)

    grid_size = max(int(grid_size), 2)
    expand_frac = max(float(expand_frac), 0.0)
    lo, hi = np.min(data), np.max(data)
    span = hi - lo
    lo -= expand_frac * span
    hi += expand_frac * span
    grid = np.linspace(lo, hi, grid_size)

    kde = KernelDensity(kernel="gaussian", bandwidth=bw).fit(data.reshape(-1, 1))
    log_pdf = kde.score_samples(grid.reshape(-1, 1))
    return kde, grid, log_pdf


def build_kde_3d(
    csv_paths,
    bandwidth: float | None = None,
    grid_sizes: tuple[int, int, int] = (50, 50, 50),
    expand_frac: float = 0.0,
):
    """
    3D KDE over (Ms, Mco='Ml', logP), with per-dimension std scaling.
    Returns: kde_obj, inv_stds, grid_Ms, grid_Mco, grid_logP, log_pdf3d
    """
    if isinstance(csv_paths, (list, tuple)):
        df = pd.concat([pd.read_csv(p) for p in csv_paths], ignore_index=True)
    else:
        df = pd.read_csv(csv_paths)

    for col in ("Ms", "Ml", "Porb"):
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found. Available: {list(df.columns)}")

    Ms = df["Ms"].to_numpy(copy=True)
    Mco = df["Ml"].to_numpy(copy=True)
    Porb = df["Porb"].to_numpy(copy=True)  # days
    logP = np.log10(Porb * 86400.0)

    X = np.vstack([Ms, Mco, logP]).T
    X = X[np.all(np.isfinite(X), axis=1)]
    if X.shape[0] == 0:
        raise ValueError("No finite data for 3D KDE (check Ms/Ml/Porb columns).")

    stds = np.std(X, axis=0)
    stds = np.where(stds <= 1e-6, 1e-6, stds)
    inv_stds = 1.0 / stds
    Xs = X * inv_stds

    if bandwidth is None:
        N, d = Xs.shape[0], 3
        silver = (4.0 / (d + 2.0)) ** (1.0 / (d + 4.0))
        bw = silver * (N ** (-1.0 / (d + 4.0)))
    else:
        bw = float(bandwidth)
    bw = max(bw, 1e-6)

    grid_sizes = tuple(max(int(g), 2) for g in grid_sizes)
    expand_frac = max(float(expand_frac), 0.0)
    mins, maxs = X.min(axis=0), X.max(axis=0)
    spans = maxs - mins
    mins -= expand_frac * spans
    maxs += expand_frac * spans

    grid_Ms = np.linspace(mins[0], maxs[0], grid_sizes[0])
    grid_Mco = np.linspace(mins[1], maxs[1], grid_sizes[1])
    grid_logP = np.linspace(mins[2], maxs[2], grid_sizes[2])

    kde = KernelDensity(kernel="gaussian", bandwidth=bw).fit(Xs)

    gMs_s = grid_Ms * inv_stds[0]
    gMco_s = grid_Mco * inv_stds[1]
    gP_s = grid_logP * inv_stds[2]

    mMs, mMco, mP = np.meshgrid(gMs_s, gMco_s, gP_s, indexing="ij")
    coords_s = np.vstack([mMs.ravel(), mMco.ravel(), mP.ravel()]).T
    log_pdf3d = kde.score_samples(coords_s).reshape(grid_sizes)

    return kde, inv_stds, grid_Ms, grid_Mco, grid_logP, log_pdf3d


def sample_kde_1d(kde_obj, n_samples: int = 1000, seed: int | None = None):
    """
    Draw samples from a fitted sklearn KernelDensity in 1D.
    """
    return kde_obj.sample(n_samples, random_state=seed).ravel()


def sample_kde_3d(
    kde_obj_3d,
    inv_stds_3d,
    n_samples: int = 1000,
    seed: int | None = None,
    clip_bounds=None,
):
    """
    Draw samples from a fitted 3D KDE in scaled space, unscale back to (Ms, Mco, logP).
    """
    w = kde_obj_3d.sample(n_samples, random_state=seed)
    inv_stds_3d = np.asarray(inv_stds_3d, float)
    samples = w / inv_stds_3d

    Ms = samples[:, 0].copy()
    Mco = samples[:, 1].copy()
    logP = samples[:, 2].copy()

    if clip_bounds is not None:
        (lo_Ms, hi_Ms), (lo_Mco, hi_Mco), (lo_logP, hi_logP) = clip_bounds
        Ms = np.clip(Ms, lo_Ms, hi_Ms)
        Mco = np.clip(Mco, lo_Mco, hi_Mco)
        logP = np.clip(logP, lo_logP, hi_logP)

    return Ms, Mco, logP


def sample_lens_from_kde_mode(
    kde3d_pack: tuple,
    inc_bounds: tuple[float, float],
    delta_bounds: tuple[float, float],
) -> tuple[float, float, float, float, float]:
    """
    Deterministic KDE-mode for (Ms, Mco, logP) + midpoints for (inc, delta).
    kde3d_pack: output of build_kde_3d (kde, inv_stds, grid_Ms, grid_Mco, grid_logP, log_pdf3d)
    """
    if kde3d_pack is None or len(kde3d_pack) != 6:
        raise ValueError("kde3d_pack must be the 6-tuple from build_kde_3d.")
    _, _, grid_Ms, grid_Mco, grid_logP, log_pdf3d = kde3d_pack
    idx = np.unravel_index(np.argmax(log_pdf3d), log_pdf3d.shape)
    Ms = float(grid_Ms[idx[0]])
    Mco = float(grid_Mco[idx[1]])
    logP = float(grid_logP[idx[2]])
    inc = 0.5 * (inc_bounds[0] + inc_bounds[1])
    delta = 0.5 * (delta_bounds[0] + delta_bounds[1])
    return Ms, Mco, logP, inc, delta


def make_kde3d_logprior_jit(
    grid_Ms: np.ndarray,
    grid_Mco: np.ndarray,
    grid_logP: np.ndarray,
    log_pdf3d: np.ndarray,
    nhyp: int,
):
    """
    Return a JIT-compiled callable:
        kde_lp(theta) -> scalar log prior contribution
    Assumes Ms/Mco/logP live at theta[nhyp:nhyp+3].
    """
    gMs = jnp.asarray(grid_Ms)
    gMco = jnp.asarray(grid_Mco)
    glogP = jnp.asarray(grid_logP)
    lpdf = jnp.asarray(log_pdf3d)

    @jax.jit
    def kde_lp(theta: jnp.ndarray) -> jnp.ndarray:
        Ms = theta[nhyp + 0]
        Mco = theta[nhyp + 1]
        logP = theta[nhyp + 2]

        i = jnp.clip(jnp.searchsorted(gMs, Ms), 0, gMs.size - 1).astype(jnp.int32)
        j = jnp.clip(jnp.searchsorted(gMco, Mco), 0, gMco.size - 1).astype(jnp.int32)
        k = jnp.clip(jnp.searchsorted(glogP, logP), 0, glogP.size - 1).astype(jnp.int32)

        return lpdf[i, j, k]

    return kde_lp


# =============================================================================
# μ (logmu) PRIORS + samplers
# =============================================================================


def sample_lens_uniform(lens_bounds: list[tuple[float, float]]) -> np.ndarray:
    """
    Deterministic midpoint draw for [Ms, Mco, logP, inc, delta] from provided bounds.
    Expects: lens_bounds = [(lo,hi), ...] length 5, in that order.
    """
    b = np.asarray(lens_bounds, float)
    if b.ndim != 2 or b.shape[1] != 2 or b.shape[0] != 5:
        raise ValueError(
            "lens_bounds must be shape (5, 2) = [(Ms_lo,Ms_hi), (Mco_lo,Mco_hi), (logP_lo,logP_hi), (inc_lo,inc_hi), (delta_lo,delta_hi)]."
        )
    if np.any(~np.isfinite(b)) or np.any(b[:, 1] <= b[:, 0]):
        raise ValueError("Non-finite or invalid lens bounds.")
    return 0.5 * (b[:, 0] + b[:, 1])


def sample_logmu_from_bounds(logmu_bounds: tuple[float, float]) -> float:
    """
    Deterministic log μ from the midpoint of its bounds.
    Expects: logmu_bounds = (lo, hi) with hi > lo.
    """
    lo, hi = float(logmu_bounds[0]), float(logmu_bounds[1])
    if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
        raise ValueError("Invalid logμ bounds.")
    return 0.5 * (lo + hi)


def sample_logmu_from_data_median(y: np.ndarray) -> float:
    """
    Approximate μ (mean counts per cadence) from the light-curve level.
    If y are already counts/cadence, μ ≈ median(y); we return log10(μ).
    Ignores NaNs and guards against non-positive medians.
    """
    y = np.asarray(y, float)
    y = y[np.isfinite(y)]
    if y.size == 0:
        raise ValueError("Cannot estimate μ from data: no finite samples in y.")
    mu = max(float(np.median(y)), 1e-30)
    return float(np.log10(mu))


def sample_logmu_from_mass_distance(
    Ms: float,
    ms_array: np.ndarray,
    d_array: np.ndarray,
    *,
    cadence_s: float = 120.0,
    flux_model: FluxModel | None = None,
) -> float:
    """
    Convert (Ms, nearest-neighbour distance) → apparent mag → flux/cadence → log μ.

    Parameters
    ----------
    Ms          : stellar mass estimate (scalar)
    ms_array    : catalog of masses to look up a proxy distance (same length as d_array)
    d_array     : matching catalog of distances (same length as ms_array)
    cadence_s   : cadence in seconds (must be > 0)
    flux_model  : FluxModel instance providing mass_to_mag(Ms, dist);
                  must be provided (no global fallback here, to keep dependency explicit).

    Returns
    -------
    log10 μ where μ is electrons per cadence (e-/cadence).
    """
    if flux_model is None:
        raise ValueError(
            "sample_logmu_from_mass_distance requires a FluxModel instance (flux_model=...)."
        )

    ms_array = np.asarray(ms_array, float)
    d_array = np.asarray(d_array, float)

    if ms_array.shape != d_array.shape or ms_array.ndim != 1 or ms_array.size == 0:
        raise ValueError(
            "ms_array and d_array must be 1D arrays of the same non-zero length."
        )

    # keep only finite, positive distances for the lookup
    mask = np.isfinite(ms_array) & np.isfinite(d_array) & (d_array > 0.0)
    if not np.any(mask):
        raise ValueError(
            "No finite/positive (mass, distance) pairs available for lookup."
        )

    Ms = float(Ms)
    diffs = np.abs(ms_array[mask] - Ms)

    # pick a small neighbourhood around the nearest mass to reduce outlier distances
    k = min(20, diffs.size)
    idxs_sorted = np.argsort(diffs)[:k]
    dist = float(np.median(d_array[mask][idxs_sorted]))

    # magnitude -> flux/sec -> flux/cadence
    m_app, _ = flux_model.mass_to_mag(
        Ms, dist
    )  # implementation-specific, returns apparent mag
    # Use the model's own zeropoint: flux/sec = zp_counts * 10**(-0.4 * m_app)
    flux_per_sec = float(flux_model.zp_counts * (10.0 ** (-0.4 * m_app)))
    if not np.isfinite(flux_per_sec) or flux_per_sec <= 0.0:
        raise ValueError("Flux model returned a non-finite or non-positive flux.")

    cadence_s = float(cadence_s)
    if not np.isfinite(cadence_s) or cadence_s <= 0.0:
        raise ValueError("cadence_s must be a positive, finite number of seconds.")

    mu_cad = max(flux_per_sec * cadence_s, 1e-30)  # e-/cadence
    return float(np.log10(mu_cad))


# =============================================================================
# DATA-BASED ANCHORS FOR KERNEL HYPERS (uneven sampling aware via Lomb–Scargle)
# =============================================================================


def _compute_psd_ls(T, y, cadence_s=None, ofac=5):
    """
    Lomb–Scargle PSD for uneven sampling.
    Returns:
      f        : frequency grid [Hz]
      Pxx      : power spectral density (astropy normalization='psd')
      nu_nyq   : effective Nyquist for median sampling [Hz]
      nu_res   : frequency resolution ~ 1/Tspan [Hz]
      cadence  : adopted cadence [s] (median dt if not provided)
    """
    T = np.asarray(T, float)
    y = np.asarray(y, float)
    t = T - np.nanmin(T)
    m = np.isfinite(t) & np.isfinite(y)
    t, y = t[m], y[m]
    if t.size < 3:
        return np.array([]), np.array([]), 1.0, 1.0, (cadence_s if cadence_s else 1.0)

    med_dt = np.median(np.diff(np.sort(t)))
    if not (
        isinstance(cadence_s, (int, float)) and np.isfinite(cadence_s) and cadence_s > 0
    ):
        cadence_s = float(med_dt)

    nu_res = 1.0 / (t.ptp() + 1e-12)
    nu_nyq = 1.0 / (2.0 * med_dt + 1e-12)

    # Oversampled frequency grid (guarded minimum length)
    n_est = max(int(ofac * (t.ptp() / cadence_s)), 2048)
    f = np.linspace(nu_res, nu_nyq, n_est)

    ls = LombScargle(t, y - np.nanmedian(y))
    Pxx = ls.power(f, normalization="psd")
    return f, Pxx, nu_nyq, nu_res, cadence_s


def _robust_lsig_from_data(y):
    """log10 amplitude anchor from robust RMS when logmu is unknown."""
    y = np.asarray(y, float)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return -3.0
    med = np.median(y)
    mad = np.median(np.abs(y - med)) + 1e-30
    robust_std = 1.4826 * mad
    return np.log10(max(robust_std, 1e-30))


def _pick_granulation_knee(f, Pxx, nu_nyq):
    """Median-energy frequency below ~0.3 Nyquist → log10 ν_g."""
    if f.size == 0:
        return -3.5
    band = (f > 0) & (f < 0.3 * nu_nyq)
    if not np.any(band):
        return -3.5
    fb, Pb = f[band], Pxx[band]
    cdf = np.cumsum(Pb) / (Pb.sum() + 1e-300)
    i = min(np.searchsorted(cdf, 0.5), fb.size - 1)
    return np.log10(max(fb[i], 1e-12))


def _pick_seismic_center(f, Pxx, lnu_g, nu_res, nu_nyq):
    """
    Power-weighted center for S above the granulation knee.
    Avoids <~2 bins above resolution and <~0.6 Nyquist.
    Returns log10 ν_s.
    """
    if f.size == 0:
        return lnu_g + 0.5
    nu_g = 10**lnu_g
    band = (f > max(1.5 * nu_g, 2.0 * nu_res)) & (f < 0.6 * nu_nyq)
    if not np.any(band):
        return lnu_g + 0.5
    fb, Pb = f[band], Pxx[band]
    nu_s = float((fb * Pb).sum() / (Pb.sum() + 1e-300))
    return np.log10(max(nu_s, 1.5 * nu_g, 1e-12))


def _pick_narrow_peak(f, Pxx, nu_res, nu_nyq):
    """
    Strongest narrowband line (for E); avoid very low f and very high f.
    This is robust to uneven sampling because LS handles it.
    Returns log10 ν_e.
    """
    if f.size == 0:
        return -4.0
    band = (f > 2.0 * nu_res) & (f < 0.8 * nu_nyq)
    if not np.any(band):
        return -4.0
    fb, Pb = f[band], Pxx[band]
    j = int(np.argmax(Pb))
    return np.log10(max(fb[j], 1e-12))


def make_data_anchors_from_psd(
    T,
    y,
    kernel_type: str,
    *,
    cadence_s: float | None = None,
    logmu: float | None = None,
) -> np.ndarray:
    """
    Build a full (nhyp,) anchor vector for the given kernel_type using the data PSD:
      G → [log10 ν_g(knee), log10 σ_G]
      S → [log10 ν_s(center), Q(prior geometric center), log10 σ_S]
      E → [log10 ν_e(peak),  log10 σ_E]
      F → [log10 τ_f ~ 2 hr, log10 σ_F]   (τ_f is a weak anchor, amplitude from μ/data)

    Amplitude anchors:
      - If logmu is provided, σ anchors use expected_log_sigma(logmu, tag).
      - Else, all σ use a shared robust log-std from the light curve.

    Returns
    -------
    anchors : np.ndarray, shape (nhyp,)
      Aligned exactly with the θ layout implied by `kernel_type`.
      These can be fed to make_log_prior_data_jit(anchors, scale).
    """
    f, Pxx, nu_nyq, nu_res, _ = _compute_psd_ls(T, y, cadence_s=cadence_s, ofac=5)

    # amplitude anchors
    if isinstance(logmu, (int, float)) and np.isfinite(logmu):
        lsig_G = float(expected_log_sigma(jnp.asarray(logmu), "G"))
        lsig_S = float(expected_log_sigma(jnp.asarray(logmu), "S"))
        lsig_E = float(expected_log_sigma(jnp.asarray(logmu), "E"))
        lsig_F = float(expected_log_sigma(jnp.asarray(logmu), "F"))
    else:
        lsig = _robust_lsig_from_data(y)
        lsig_G = lsig_S = lsig_E = lsig_F = lsig

    # frequency anchors
    lnu_g = _pick_granulation_knee(f, Pxx, nu_nyq)
    lnu_s = _pick_seismic_center(f, Pxx, lnu_g, nu_res, nu_nyq)
    lnu_e = _pick_narrow_peak(f, Pxx, nu_res, nu_nyq)
    ltau_f = np.log10(2.0 * 3600.0)  # ~2 hr

    # pack in the exact hyper layout
    anchors = []
    for c in kernel_type:
        if c == "G":
            anchors += [lnu_g, lsig_G]
        elif c == "S":
            anchors += [lnu_s, np.sqrt(np.prod(SEISMIC_Q_BOUNDS)), lsig_S]
        elif c == "E":
            anchors += [lnu_e, 20.0, lsig_E]
        elif c == "F":
            anchors += [ltau_f, lsig_F]
        else:
            raise ValueError(f"Unknown kernel component '{c}'")
    return np.asarray(anchors, float)


# =============================================================================
# full prior and Samplers
# =============================================================================

# =============================================================================
# ADD THIS ENTIRE FUNCTION TO THE END OF PriorsSamplers.py
# (Type hint removed from flux_model)
# =============================================================================


def sample_theta_array_simple(
    *,
    kernel_type: str,
    nhyp: int,
    combined_bounds: list[tuple[float, float]] | tuple[tuple[float, float], ...],
    include_lens_mean: int,
    use_kde: bool = False,
    phys: bool = False,
    P0: np.ndarray | None = None,
    # Data anchors (passed to sample_physical or make_data_anchors)
    T: np.ndarray | None = None,
    y: np.ndarray | None = None,
    cadence_s: float | None = None,
    # KDE pack (passed to sample_lens_from_kde_mode)
    kde3d_pack: tuple | None = None,
    # Mu estimation (passed to sample_logmu_...)
    ms_array: np.ndarray | None = None,
    d_array: np.ndarray | None = None,
    flux_model=None,  # <--- TYPE HINT REMOVED
) -> np.ndarray:
    """
    Deterministic seed (1D array) for theta, using prior centers.
    This is the missing function called by mcf.init_walkers.
    """
    bounds = np.asarray(combined_bounds, dtype=float)
    ndim = len(bounds)
    theta = np.zeros(ndim, dtype=float)

    # ---- 1) Get data-based estimates for anchoring ----
    logmu_est = np.nan
    Ms_est = np.nan
    logP_est = np.nan

    if y is not None:
        logmu_est = sample_logmu_from_data_median(
            y
        )  # Uses function already in PriorsSamplers.py

    if include_lens_mean:
        # Use midpoint of bounds as a rough Ms/logP guess
        Ms_est = 0.5 * (bounds[nhyp + 0][0] + bounds[nhyp + 0][1])
        logP_est = 0.5 * (bounds[nhyp + 2][0] + bounds[nhyp + 2][1])

    # ---- 2) Kernel hypers (physical or data-anchored) ----
    if phys:
        # Use physical relations, anchored by Ms/mu estimates
        hypers = sample_physical(  # Uses function already in PriorsSamplers.py
            kernel_type, logmu=logmu_est, Ms_est=Ms_est, logP_est=logP_est
        )
    elif T is not None and y is not None:
        # Use data PSD anchors
        hypers = (
            make_data_anchors_from_psd(  # Uses function already in PriorsSamplers.py
                T, y, kernel_type, cadence_s=cadence_s, logmu=logmu_est
            )
        )
    else:
        # Fallback: midpoint of bounds
        hypers = 0.5 * (bounds[:nhyp, 0] + bounds[:nhyp, 1])

    theta[:nhyp] = hypers

    # ---- 3) Lens + Mu params ----
    if include_lens_mean:
        lens_params = np.zeros(5, dtype=float)
        if use_kde and kde3d_pack is not None:
            # Use KDE mode for (Ms, Mco, logP)
            lens_params[0:3] = (
                sample_lens_from_kde_mode(  # Uses function already in PriorsSamplers.py
                    kde3d_pack, tuple(bounds[nhyp + 3]), tuple(bounds[nhyp + 4])
                )[0:3]
            )
            lens_params[3] = 0.5 * (bounds[nhyp + 3][0] + bounds[nhyp + 3][1])  # inc
            lens_params[4] = 0.5 * (bounds[nhyp + 4][0] + bounds[nhyp + 4][1])  # delta
        else:
            # Fallback: uniform midpoints
            lens_params = 0.5 * (
                bounds[nhyp : nhyp + 5, 0] + bounds[nhyp : nhyp + 5, 1]
            )
        theta[nhyp : nhyp + 5] = lens_params

        # Estimate mu
        if np.isfinite(logmu_est):
            theta[-1] = logmu_est
        else:
            # Use mass-distance if available, else bounds midpoint
            if (
                flux_model is not None
                and ms_array is not None
                and d_array is not None
                and np.isfinite(Ms_est)
                and cadence_s is not None
            ):
                try:
                    theta[-1] = (
                        sample_logmu_from_mass_distance(  # Uses function already in PriorsSamplers.py
                            Ms_est,
                            ms_array,
                            d_array,
                            cadence_s=cadence_s,
                            flux_model=flux_model,
                        )
                    )
                except ValueError:  # Handle cases where lookup fails
                    theta[-1] = 0.5 * (bounds[-1][0] + bounds[-1][1])
            else:
                theta[-1] = 0.5 * (bounds[-1][0] + bounds[-1][1])  # fallback
    else:
        # No lens mean, just mu
        if np.isfinite(logmu_est):
            theta[-1] = logmu_est
        else:
            theta[-1] = 0.5 * (bounds[-1][0] + bounds[-1][1])  # fallback

    # ---- 4) Override with P0 if provided ----
    if P0 is not None:
        P0 = np.asarray(P0, dtype=float)
        mask = np.isfinite(P0)
        theta[mask] = P0[mask]

    # Final clip to ensure bounds
    return np.clip(theta, bounds[:, 0], bounds[:, 1])
