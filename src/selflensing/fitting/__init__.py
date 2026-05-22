"""Bayesian fitting for self-lensing binaries.

This package preserves the historical ``selflensing.fitting`` public API while
keeping backend-specific implementation details in smaller modules.
"""

import warnings

warnings.filterwarnings("ignore", module="erfa.core")

from ._utils import gauss_lnprior
from .blackjax import (
    BlackJAXFitState,
    BlackJAXSamplerResult,
    self_lensing_system_from_fit_blackjax,
    self_lensing_systems_from_batched_fit_blackjax,
)
from .emcee import self_lensing_system_from_fit
from .likelihood import (
    chi2,
    chi2_masked,
    combined_chi2,
    make_batched_combined_chi2_vec_jax,
    make_combined_chi2_vec,
    make_combined_chi2_vec_jax,
)

__all__ = [
    "BlackJAXFitState",
    "BlackJAXSamplerResult",
    "chi2",
    "chi2_masked",
    "combined_chi2",
    "gauss_lnprior",
    "make_batched_combined_chi2_vec_jax",
    "make_combined_chi2_vec",
    "make_combined_chi2_vec_jax",
    "self_lensing_system_from_fit",
    "self_lensing_system_from_fit_blackjax",
    "self_lensing_systems_from_batched_fit_blackjax",
]
