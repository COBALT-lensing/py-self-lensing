
# References:
# - ν_max scaling (ν_max ∝ g / sqrt(T_eff), ν_max,⊙ ≈ 3090 μHz = 0.00309 Hz):
#     Kjeldsen & Bedding (1995, 2011); Huber et al. (2011); Chaplin & Miglio (2013)
# - Granulation characteristic frequency tracks ν_max (factor ~0.3–0.5):
#     Kallinger et al. (2014)
# - Ellipsoidal variability scaling:
#     Morris (1985); Zucker et al. (2007)

import jax.numpy as jnp
from astropy import constants as const

# ---------- Constants (SI / solar) ----------
T_SUN = 5772.0               # K
NU_MAX_SUN_HZ = 3090e-6      # 3090 μHz = 0.00309 Hz
L_SUN = const.L_sun.value    # W
R_SUN = const.R_sun.value    # m
SIGMA = const.sigma_sb.value # W m^-2 K^-4

# small helper: ensure float array dtype under jit/vmap/grad
def _af(x):
    return jnp.asarray(x, dtype=jnp.result_type(0.0, x))

# =========================
# Mass–R–L–Teff relations
# =========================
def stellar_radius_Rsun(M_s):
    """Dimensionless radius: R/Rsun = (M/Msun)^0.8 (scalar or array)."""
    M_s = _af(M_s)
    return M_s ** 0.8

def stellar_radius_m(M_s):
    """Radius in meters."""
    return stellar_radius_Rsun(M_s) * R_SUN

def luminosity_W(M_s):
    """Luminosity in watts: L = Lsun * (M/Msun)^3.5."""
    M_s = _af(M_s)
    return (M_s ** 3.5) * L_SUN

def effective_temperature_from_LR(L_W, R_m):
    """T_eff [K] from L and R: (L / (4πσ R^2))^0.25."""
    L_W = _af(L_W)
    R_m = _af(R_m)
    return (L_W / (4.0 * jnp.pi * SIGMA * (R_m ** 2))) ** 0.25

def effective_temperature_K(M_s):
    """T_eff(M) in Kelvin via L(M), R(M)."""
    return effective_temperature_from_LR(luminosity_W(M_s), stellar_radius_m(M_s))

# ================================
# Asteroseismology / granulation
# ================================
def asteroseismic_frequency_relation(M_s):
    """
    ν_max [Hz] = ν_max,⊙ * (g/g_⊙) * (T_eff/T_⊙)^(-1/2),
    with g/g_⊙ = (M/M_⊙) / (R/R_⊙)^2 using stellar_radius_Rsun.
    (Kjeldsen & Bedding 1995/2011; Huber+2011; Chaplin & Miglio 2013)
    """
    M_s = _af(M_s)
    R_over_Rsun = stellar_radius_Rsun(M_s)
    g_ratio = M_s / (R_over_Rsun ** 2)
    Teff = effective_temperature_K(M_s)
    return NU_MAX_SUN_HZ * g_ratio * (Teff / T_SUN) ** (-0.5)

GRAN_OVER_NUMAX = 0.30  # conventional; Kallinger+2014 shows ~0.3–0.5
def granulation_frequency_relation(M_s):
    """Granulation characteristic frequency in **Hz**."""
    return GRAN_OVER_NUMAX * asteroseismic_frequency_relation(M_s)

# ==========================
# Ellipsoidal variability
# ==========================
def ellipsoidal_period_relation(logP):
    """P_ellip = P/2. Input logP = log10(P [s]); output in seconds."""
    P = 10.0 ** _af(logP)
    return P / 2.0

def ellipsoidal_amplitude_relation(M_s, M_co, logP, inc_rad):
    """
    Dimensionless scaling: A ∝ (M_co * sin^2 i) / (P^{2/3} * M_s^{1/3})
    (Morris 1985; Zucker et al. 2007)
    """
    M_s = _af(M_s)
    M_co = _af(M_co)
    P = 10.0 ** _af(logP)
    inc_rad = _af(inc_rad)
    return (M_co * jnp.sin(inc_rad) ** 2) / (P ** (2.0 / 3.0) * M_s ** (1.0 / 3.0))
