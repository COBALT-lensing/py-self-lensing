"""Preserve physical posteriors under both recovery coordinate Jacobians."""
import unittest
from unittest.mock import patch
import test_impact_geometry as geometry_tests
from selflensing.fitting.blackjax import self_lensing_systems_from_batched_fit_blackjax


def with_recovery(fitter):
    def call(*args, **kwargs):
        kwargs.update(adapt_covariance=True, robust_covariance=True, recovery_probability=.1)
        return fitter(*args, **kwargs)
    return call


class RecoveryGeometryTest(unittest.TestCase):
    def test_physical_gaussian_in_inclination_impact_and_period_coordinates(self):
        fitter=geometry_tests.self_lensing_system_from_fit_blackjax
        with patch.object(geometry_tests,'self_lensing_system_from_fit_blackjax',side_effect=with_recovery(fitter)):
            geometry_tests.ImpactGeometryTest().test_fitter_recovers_same_physical_gaussian_posterior_and_state_metadata()

    def test_batched_physical_gaussian_with_distinct_stellar_radii(self):
        with patch('selflensing.fitting.blackjax.self_lensing_systems_from_batched_fit_blackjax',
                   side_effect=with_recovery(self_lensing_systems_from_batched_fit_blackjax)):
            geometry_tests.ImpactGeometryTest().test_batch_geometry_and_resume_metadata()
