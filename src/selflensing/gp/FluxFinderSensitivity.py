# FluxFinderSensitivity.py
# Instrument + photometric conversions for the lensing project.
# All stellar physics (R–L–Teff) is delegated to PhysicalRelations.
# JAX is used for the band integration & fast μ interpolation.

import jax
import jax.numpy as jnp
from jax.scipy.integrate import trapezoid as trapz_jax  # <- use this everywhere
import pandas as pd
import numpy as np
from astropy import constants as const

import gzip
import os

# ---- Physics (single source of truth) ----
from .PhysicalRelations import (
    luminosity_W,  # L(M) [W]
    stellar_radius_m,  # R(M) [m]
    effective_temperature_from_LR,  # Teff(L,R) [K]
    effective_temperature_K,  # Teff(M) [K]
)

# ---- Defaults (edit paths to your environment) ----
PWD = os.path.dirname(os.path.abspath(__file__))
DEFAULT_NOISE_MODEL_PATH = f"{PWD}/data/selected_noise_model_good_coords.csv.gz"
DEFAULT_RESPONSE_DATA_PATH = f"{PWD}/data/TESS_TESS.Red.dat.gz"
DEFAULT_QE_DATA_PATH = None  # CSV with 'wavelength' [Å] & 'QE' columns

# ---- Physical constants (SI) ----
C_LIGHT = const.c.value  # [m/s]
H_PLANCK = const.h.value  # [J*s]
K_BOLTZMANN = const.k_B.value  # [J/K]
SIGMA_SB = const.sigma_sb.value  # [W m^-2 K^-4]
R_SUN = const.R_sun.value  # [m]
L_SUN = const.L_sun.value  # [W]
PI = jnp.pi
KPC_TO_M = 3.086e19  # [m per kpc]

# ---- Built-in QE fallback (TESS-like, coarse) ----
QE_WAVELENGTHS_BUILTIN = (
    jnp.array([638, 698, 749, 797, 828, 848, 879, 904, 939, 978, 999, 1063]) * 1e-10
)  # m
QE_VALUES_BUILTIN = jnp.array(
    [0.918, 0.934, 0.958, 0.959, 0.957, 0.945, 0.916, 0.842, 0.685, 0.404, 0.287, 0.047]
)


# ---- I/O (host-side; not jitted) ----
def load_noise_model(path=DEFAULT_NOISE_MODEL_PATH):
    df = pd.read_csv(path)
    return jnp.array(df["T_mag"].values, dtype=float), jnp.array(
        df["noise"].values, dtype=float
    )


def load_response_data(path=DEFAULT_RESPONSE_DATA_PATH):
    lam, resp = [], []
    with gzip.open(path, "r") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                lam.append(float(parts[0]) * 1e-10)  # Å -> m
                resp.append(float(parts[1]))
    return jnp.array(lam, dtype=float), jnp.array(resp, dtype=float)


def load_quantum_efficiency(wavelengths, qe_path=DEFAULT_QE_DATA_PATH):
    if qe_path:
        df = pd.read_csv(qe_path)
        wl = jnp.asarray(df["wavelength"].values * 1e-10, dtype=float)  # Å -> m
        qe = jnp.asarray(df["QE"].values, dtype=float)
    else:
        wl = QE_WAVELENGTHS_BUILTIN
        qe = QE_VALUES_BUILTIN
    # clip + interp to ensure defined across band
    return jnp.interp(jnp.clip(wavelengths, wl[0], wl[-1]), wl, qe)


# ---- JITted core photometry ----
@jax.jit
def zero_point_counts(wavelengths, response, qe):
    """
    Zeropoint count rate [e-/s] for AB mag=0 (f_nu=3631 Jy),
    integrated over the instrument bandpass.
    """
    f_nu = 3631e-26  # [W m^-2 Hz^-1]
    f_lambda = f_nu * C_LIGHT / wavelengths**2
    phi_lambda = f_lambda * wavelengths / (H_PLANCK * C_LIGHT)  # photons s^-1 m^-2 m^-1
    y = phi_lambda * response * qe
    return trapz_jax(y, wavelengths)


# --- Physics delegations (kept for API stability; jitted for speed) ---
@jax.jit
def find_L(M):
    """Luminosity [W]."""
    return luminosity_W(M)


@jax.jit
def find_R(M):
    """Radius [m] (uses R ∝ M^0.8 via PhysicalRelations)."""
    return stellar_radius_m(M)


@jax.jit
def find_Teff(L, R):
    """Effective temperature [K] from (L,R)."""
    return effective_temperature_from_LR(L, R)


@jax.jit
def find_Teff_from_M(M):
    """Effective temperature [K] from mass directly."""
    return effective_temperature_K(M)


@jax.jit
def find_counts(Teff, Rstar, D_kpc, wavelengths, response, qe):
    """
    Photon count rate [e-/s] at Earth for a blackbody star (Teff, Rstar) at D_kpc,
    integrated over the bandpass (wavelengths/response/qe).
    """
    D_m = D_kpc * KPC_TO_M
    exponent = H_PLANCK * C_LIGHT / (wavelengths * K_BOLTZMANN * Teff)
    B_lambda = (2 * H_PLANCK * C_LIGHT**2 / wavelengths**5) / (jnp.exp(exponent) - 1)
    F_lambda = PI * B_lambda  # surface flux [W m^-2 m^-1]
    F_lambda_earth = F_lambda * (Rstar / D_m) ** 2  # flux at Earth
    phi_lambda = (
        F_lambda_earth * wavelengths / (H_PLANCK * C_LIGHT)
    )  # photons s^-1 m^-2 m^-1
    y = phi_lambda * response * qe
    return trapz_jax(y, wavelengths)


@jax.jit
def find_app_mag(counts, zp_counts):
    """Apparent magnitude from counts and zeropoint counts."""
    EPS = 1e-30
    return -2.5 * jnp.log10((counts + EPS) / (zp_counts + EPS))


@jax.jit
def app_mag_to_flux(m_app, zero_point=20.44):
    """Convert magnitude to relative flux using instrument zeropoint."""
    return 10 ** ((m_app - zero_point) / -2.5)


@jax.jit
def flux_to_app_mag(flux, zero_point=20.44):
    """Convert relative flux to magnitude using instrument zeropoint."""
    return -2.5 * jnp.log10(flux) + zero_point


# ---- Host wrapper combining loading and JIT compute ----
def conv_mass_to_mag(
    M,
    D_kpc,
    noise_model_path=DEFAULT_NOISE_MODEL_PATH,
    response_data_path=DEFAULT_RESPONSE_DATA_PATH,
    qe_data_path=DEFAULT_QE_DATA_PATH,
):
    """
    Convenience wrapper:
      - loads instrument curves,
      - computes zeropoint counts,
      - returns (m_app, noise_level) for (M, D_kpc).
    """
    mags, noise = load_noise_model(noise_model_path)
    wavelengths, response = load_response_data(response_data_path)
    qe = load_quantum_efficiency(wavelengths, qe_data_path)

    zp = zero_point_counts(wavelengths, response, qe)
    L = find_L(M)
    R = find_R(M)
    Teff = find_Teff(L, R)
    counts = find_counts(Teff, R, D_kpc, wavelengths, response, qe)
    m_app = find_app_mag(counts, zp)
    noise_level = jnp.interp(m_app, mags, noise)
    return m_app, noise_level


# ---- Utility conversions ----
def counts_from_mag(m_app, zero_point=20.44):
    """Detector count rate [e-/s] from an instrument magnitude."""
    return 10 ** ((m_app - zero_point) / -2.5)


def counts_per_cadence_from_mag(m_app, cadence_s=120.0, zero_point=20.44):
    """Counts per cadence given a magnitude and cadence length (default 120 s)."""
    return counts_from_mag(m_app, zero_point) * cadence_s


# ---- FluxModel cache + μ helpers ----
class FluxModel:
    """
    Cached instrument curves + zeropoint, with convenience conversions.
    """

    def __init__(
        self,
        noise_model_path: str = DEFAULT_NOISE_MODEL_PATH,
        response_data_path: str = DEFAULT_RESPONSE_DATA_PATH,
        qe_data_path: str = DEFAULT_QE_DATA_PATH,
    ):
        self.mags, self.noise = load_noise_model(noise_model_path)
        self.wavelengths, self.response = load_response_data(response_data_path)
        self.qe = load_quantum_efficiency(self.wavelengths, qe_data_path)
        self.zp_counts = zero_point_counts(self.wavelengths, self.response, self.qe)

    def mass_to_mag(self, M: float, D_kpc: float):
        """
        Convert (M, D_kpc) -> (m_app, noise_level) using:
          - Physics: PhysicalRelations
          - Instrument: this model's curves
        """
        L = find_L(M)
        R = find_R(M)
        Teff = find_Teff(L, R)
        counts = find_counts(Teff, R, D_kpc, self.wavelengths, self.response, self.qe)
        EPS = 1e-30
        m_app = -2.5 * jnp.log10((counts + EPS) / (self.zp_counts + EPS))
        noise_level = jnp.interp(m_app, self.mags, self.noise)
        return m_app, noise_level


def mu_exact(Ms: float, D_kpc: float, flux_model: "FluxModel" = None) -> float:
    """
    Exact baseline flux (counts/s) using full bandpass integration (host-side).
    """
    fm = flux_model or FluxModel()
    m_app, _ = fm.mass_to_mag(Ms, D_kpc)
    return 10 ** (-0.4 * m_app) * fm.zp_counts


def precompute_flux_grid(flux_model: "FluxModel", Ms_vals, D_vals):
    """
    Precompute a 2D grid of μ(Ms, D) in counts/s. Returns jnp arrays.
    """
    Ms_vals = np.asarray(Ms_vals, dtype=float)
    D_vals = np.asarray(D_vals, dtype=float)
    grid = np.zeros((len(Ms_vals), len(D_vals)), dtype=float)
    for i, M in enumerate(Ms_vals):
        for j, D in enumerate(D_vals):
            m_app, _ = flux_model.mass_to_mag(M, D)
            grid[i, j] = 10 ** (-0.4 * m_app) * flux_model.zp_counts
    return jnp.asarray(Ms_vals), jnp.asarray(D_vals), jnp.asarray(grid)


@jax.jit
def mu_fast(
    Ms: float,
    logD: float,
    Ms_vals: jnp.ndarray,
    D_vals: jnp.ndarray,
    flux_grid: jnp.ndarray,
) -> float:
    """
    Fast, JAX-native μ via two 1D jnp.interp calls (bilinear).
    Inputs:
      Ms        : stellar mass [Msun]
      logD      : log10 distance [kpc]
      Ms_vals   : grid masses [Msun]
      D_vals    : grid distances [kpc]
      flux_grid : μ grid [counts/s]
    """
    D = 10.0**logD
    nM = Ms_vals.shape[0]
    Ms_idx = jnp.interp(Ms, Ms_vals, jnp.arange(nM, dtype=jnp.float32))
    i0 = jnp.clip(jnp.floor(Ms_idx).astype(int), 0, nM - 2)
    i1 = i0 + 1
    row0 = flux_grid[i0, :]
    row1 = row0 if (i1 == i0) else flux_grid[i1, :]  # guard degenerate edge
    f0 = jnp.interp(D, D_vals, row0)
    f1 = jnp.interp(D, D_vals, row1)
    Ms_pair = jnp.array([Ms_vals[i0], Ms_vals[i1]])
    f_pair = jnp.array([f0, f1])
    return jnp.interp(Ms, Ms_pair, f_pair)
