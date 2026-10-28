"""End-to-end adaptive fitting of a synthetic self-lensing pulse."""
import unittest
import jax
jax.config.update('jax_enable_x64', True)
import numpy as np
from astropy import units
from astropy.time import Time
from selflensing.models import SelfLensingSystem
from selflensing.generators import LightcurveGenerator
from selflensing.fitting.blackjax import (
    BlackJAXFitState, self_lensing_system_from_fit_blackjax,
    self_lensing_systems_from_batched_fit_blackjax,
)


class RecoveryFitTest(unittest.TestCase):
    def test_single_and_batched_recover_synthetic_event(self):
        self.check_event('inclination')

    def test_impact_coordinates_recover_synthetic_event(self):
        self.check_event('b_source')

    def check_event(self, geometry):
        truth = np.array([1., 1.5707, 9., .2, .4])
        times = Time(2457000. + np.linspace(-.3, .3, 31), format='jd')
        system = SelfLensingSystem(a=truth[0]*units.au, i=truth[1]*units.rad,
            Macc=truth[2]*units.M_sun, Mcomp=1*units.M_sun, Rcomp=1*units.R_sun,
            e=truth[3], periapsis_phase=truth[4]*units.rad)
        flux = LightcurveGenerator(system, times=times, peak_time=times[15],
            lensing_method='IntegratedWittMaoJax').flux_array
        rng = np.random.default_rng(4)
        error = np.full(len(times), 1e-4)
        observed = flux + .02 + rng.normal(size=len(times))*error
        start = truth + rng.normal(size=(16, 5))*[.005, .00001, .05, .005, .005]
        if geometry == 'b_source':
            from selflensing.fitting.geometry import physical_to_geometry
            start = np.asarray(physical_to_geometry(start, 1.))
        shared = dict(inclination_parameterization=geometry, Mcomp=1*units.M_sun, Rcomp=1*units.R_sun,
            nwalkers=16, nsteps=300, burn=100, warmup_steps=150,
            proposal_scale=np.array([.005, .006 if geometry == "b_source" else .00003, .1, .01, .01]),
            profile_lc_baseline=True, progress=False, prefer_gpu=False,
            adapt_covariance=True, robust_covariance=True, recovery_probability=.1,
            return_sampler=True)
        fitted, sampler, _ = self_lensing_system_from_fit_blackjax(
            peak_time=times[15], lc_times=times,
            observed_lc=observed*units.dimensionless_unscaled,
            observed_lc_err=error*units.dimensionless_unscaled,
            initial_walker_state=BlackJAXFitState(start, inclination_parameterization=geometry), **shared)
        prediction = LightcurveGenerator(fitted, times=times, peak_time=times[15],
            lensing_method='IntegratedWittMaoJax').flux_array
        residual = observed - prediction
        self.assertLess(np.sum(((residual-residual.mean())/error)**2)/len(error), 2.)
        self.assertEqual(sampler.recovery_attempted.shape,(300,16))
        self.assertEqual(sampler.recovery_centres.shape,(16,5))
        self.assertTrue(np.isfinite(sampler.recovery_transform).all())
        shared = dict(shared, Mcomp=[1.]*units.M_sun)
        systems, samplers, _ = self_lensing_systems_from_batched_fit_blackjax(
            peak_time=np.array([times[15].jd]), lc_times=times.jd[None, :],
            observed_lc=observed[None]*units.dimensionless_unscaled,
            observed_lc_err=error[None]*units.dimensionless_unscaled,
            initial_walker_state=BlackJAXFitState(start[None], inclination_parameterization=geometry), **shared)
        prediction = LightcurveGenerator(systems[0], times=times, peak_time=times[15],
            lensing_method='IntegratedWittMaoJax').flux_array
        residual = observed - prediction
        self.assertLess(np.sum(((residual-residual.mean())/error)**2)/len(error), 2.)
        self.assertEqual(samplers[0].recovery_attempted.shape,(300,16))
        self.assertEqual(samplers[0].recovery_centres.shape,(16,5))
