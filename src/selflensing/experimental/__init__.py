"""Experimental differentiable fitting components.

These APIs are intentionally separate from the stable fitting interface while
their numerical equivalence and GPU memory behaviour are validated.
"""

from .photometry import (
    PhotometryRow,
    batched_photometric_logprob_and_grad,
    photometric_logprob,
)
from .mala import BatchedMALAState, initialise as initialise_mala, run as run_mala

__all__ = (
    "PhotometryRow",
    "batched_photometric_logprob_and_grad",
    "photometric_logprob",
    "BatchedMALAState",
    "initialise_mala",
    "run_mala",
)
