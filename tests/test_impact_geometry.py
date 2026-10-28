"""Prior-preserving impact coordinates, boundaries and posterior recovery."""
import unittest
from unittest.mock import patch
import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from scipy.integrate import quad
from astropy import units as u
from astropy.time import Time
from selflensing.fitting.geometry import (physical_to_geometry, geometry_to_physical,
    log_abs_di_db_source, b_source_scale)
from selflensing.fitting.blackjax import BlackJAXFitState, self_lensing_system_from_fit_blackjax


class ImpactGeometryTest(unittest.TestCase):
    def test_roundtrip_matches_eccentric_model_definition_and_non_crossings(self):
        from selflensing.models import SelfLensingSystem
        positions = np.array([[.1, 1.4, 9., .0, .4], [2., .3, 9., .9, 2.],
                              [200., 1.5707, 50., .99, 6.]])
        coords = physical_to_geometry(positions, 1.2)
        physical, jac = geometry_to_physical(coords, 1.2, (0., np.pi/2))
        np.testing.assert_allclose(physical, positions, atol=1e-13)
        self.assertTrue(np.all(np.isfinite(jac)))
        for p, b in zip(positions, np.asarray(coords)[:, 1]):
            system = SelfLensingSystem(a=p[0]*u.au, i=p[1]*u.rad, Macc=p[2]*u.M_sun,
                Mcomp=1*u.M_sun, Rcomp=1.2*u.R_sun, e=p[3], periapsis_phase=p[4]*u.rad)
            self.assertAlmostEqual(b, float(system.b_source.value), places=9)
        self.assertGreater(float(coords[1, 1]), 1.)

    def test_jacobian_derivative_and_integrated_original_prior(self):
        for a in [.3, 20., 180.]:
            p = jnp.array([a, 1.2, 9., .6, 2.])
            b = physical_to_geometry(p, 1.3)[1]
            derivative = jax.grad(lambda b: geometry_to_physical(p.at[1].set(b), 1.3, (0., np.pi/2))[0][1])(b)
            jac = log_abs_di_db_source(a, b, 1.3, .6, 2.)
            np.testing.assert_allclose(np.log(abs(derivative)), jac, atol=1e-12)
            scale = float(b_source_scale(a, 1.3, .6, 2.))
            total = quad(lambda b: np.exp(log_abs_di_db_source(a, b, 1.3, .6, 2., xp=np)),
                         scale*np.cos(1.5), scale*np.cos(.1))[0]
            self.assertAlmostEqual(total, 1.4, places=8)

    def test_invalid_support_and_singular_boundary_are_rejected(self):
        p = jnp.array([2., 0., 9., .2, .4])
        scale = float(b_source_scale(2., 1., .2, .4))
        for b in [-1., scale, scale*1.01]:
            _, logjac = geometry_to_physical(p.at[1].set(b), 1., (0., np.pi/2))
            self.assertTrue(np.isneginf(logjac))
        physical, jac = geometry_to_physical(p, 1., (0., np.pi/2))
        self.assertAlmostEqual(float(physical[1]), np.pi/2)
        self.assertTrue(np.isfinite(jac))
        _, jac = geometry_to_physical(p, 1., (.1, 1.4))
        self.assertTrue(np.isneginf(jac))

    def test_fitter_recovers_same_physical_gaussian_posterior_and_state_metadata(self):
        centre = np.array([2., 1.3, 9., .2, .4])
        widths = np.array([.2, .025, .3, .02, .02])
        rng = np.random.default_rng(5)
        starts = centre + rng.normal(size=(24, 5))*widths
        results = []
        for separation, geometry in [('semi_major_axis', 'inclination'), ('semi_major_axis', 'b_source'), ('period', 'b_source')]:
            coords = starts if geometry == 'inclination' else np.asarray(physical_to_geometry(starts, 1.))
            if separation == 'period':
                from astropy.constants import G
                factor = G.to_value(u.au**3 / (u.M_sun*u.day**2))/(4*np.pi**2)
                coords = np.array(coords, copy=True)
                coords[:, 0] = np.sqrt(starts[:, 0]**3 / (factor*(starts[:, 2]+1.)))
                centre_period = np.sqrt(centre[0]**3 / (factor*(centre[2]+1.)))
            scales = np.array([.15, .025, .2, .015, .015])
            if geometry == 'b_source':
                scales[1] *= float(b_source_scale(2., 1., .2, .4))
            with patch('selflensing.fitting.blackjax.make_combined_chi2_vec_jax',
                return_value=lambda p: -.5*jnp.sum(((p-centre)/widths)**2, axis=-1)):
                system, sampler, state = self_lensing_system_from_fit_blackjax(
                    Mcomp=1*u.M_sun, Rcomp=1*u.R_sun, peak_time=0.,
                    lc_times=Time([2457000.], format='jd'), observed_lc=np.ones(1),
                    nwalkers=24, nsteps=8000, burn=2000, warmup_steps=2000,
                    progress=False, prefer_gpu=False, proposal_scale=scales,
                    inclination_parameterization=geometry, parameterization=separation,
                    period_min=centre_period*.5*u.day if separation == 'period' else None,
                    period_max=centre_period*1.5*u.day if separation == 'period' else None,
                    initial_walker_state=BlackJAXFitState(coords, parameterization=separation,
                                                        inclination_parameterization=geometry))
            self.assertEqual(state.inclination_parameterization, geometry)
            chain = sampler.get_chain(discard=2000)
            if separation == 'period':
                chain = np.array(chain, copy=True)
                chain[:, :, 0] = (factor*(chain[:, :, 2]+1.)*chain[:, :, 0]**2)**(1/3)
            if geometry == 'b_source':
                chain, _ = geometry_to_physical(chain, 1., (0., np.pi/2))
            flat = np.asarray(chain).reshape(-1, 5)
            np.testing.assert_allclose(flat.mean(axis=0), centre, atol=.04)
            np.testing.assert_allclose(flat.std(axis=0), widths, rtol=.2)
            lo, hi = np.quantile(flat, [.16, .84], axis=0)
            self.assertTrue(np.all((lo < centre) & (centre < hi)))
            self.assertLess(abs(float(system.i.value)-centre[1]), .1)
            results.append(flat.mean(axis=0))
        np.testing.assert_allclose(results[0], results[1], atol=.04)


    def test_batch_geometry_and_resume_metadata(self):
        from selflensing.fitting.blackjax import self_lensing_systems_from_batched_fit_blackjax
        centre = np.array([2., 1.3, 9., .2, .4])
        widths = np.array([.2, .025, .3, .02, .02])
        starts = centre + np.random.default_rng(6).normal(size=(2, 24, 5))*widths
        coords = np.asarray(physical_to_geometry(starts, jnp.array([[1.], [1.2]])))
        with patch('selflensing.fitting.blackjax.make_batched_numeric_combined_chi2_vec_jax',
            return_value=lambda p: -.5*jnp.sum(((p-centre)/widths)**2, axis=-1)):
            systems, samplers, state = self_lensing_systems_from_batched_fit_blackjax(
                Mcomp=[1.,1.]*u.M_sun, Rcomp=[1.,1.2]*u.R_sun,
                peak_time=np.zeros(2), lc_times=np.ones((2, 1)), observed_lc=np.ones((2, 1)),
                nwalkers=24, nsteps=6000, burn=1500, warmup_steps=1500,
                progress=False, prefer_gpu=False, return_sampler=True, return_walker_state=True,
                proposal_scale=np.array([.15, 8., .2, .015, .015]), inclination_parameterization='b_source',
                initial_walker_state=BlackJAXFitState(coords, inclination_parameterization='b_source'))
        self.assertEqual(state.inclination_parameterization, 'b_source')
        for index, sampler in enumerate(samplers):
            self.assertEqual(sampler.parameter_names[1], 'b_source')
            physical, _ = geometry_to_physical(sampler.get_chain(discard=1500),
                                              [1., 1.2][index], (0., np.pi/2))
            flat = np.asarray(physical).reshape(-1, 5)
            np.testing.assert_allclose(flat.mean(axis=0), centre, atol=.06)
            np.testing.assert_allclose(flat.std(axis=0), widths, rtol=.25)
            np.testing.assert_allclose(state.coords[index], sampler.chain[-1])
            self.assertLess(abs(float(systems[index].i.value)-1.3), .1)

    def test_resume_rejects_mismatched_geometry(self):
        with self.assertRaisesRegex(ValueError, 'geometry does not match'):
            self_lensing_system_from_fit_blackjax(Mcomp=1*u.M_sun, Rcomp=1*u.R_sun,
                peak_time=0., observed_lc=np.ones(1), inclination_parameterization='b_source',
                initial_walker_state=BlackJAXFitState(np.tile([2.,1.3,9.,.2,.4], (2,1))),
                nsteps=1, burn=0, warmup_steps=0, progress=False, prefer_gpu=False)

    def test_internal_physical_starts_are_converted_once(self):
        from selflensing.fitting.blackjax import _initial_walker_positions
        physical = _initial_walker_positions(4, rng=np.random.default_rng(0))
        expected = physical_to_geometry(physical, 1.)
        with patch('selflensing.fitting.blackjax.make_combined_chi2_vec_jax',
                   return_value=lambda p: -.5*jnp.sum(p*p, axis=-1)):
            _, sampler, _ = self_lensing_system_from_fit_blackjax(
                Mcomp=1*u.M_sun, Rcomp=1*u.R_sun, peak_time=0., observed_lc=np.ones(1),
                nwalkers=4, nsteps=1, burn=0, warmup_steps=0, progress=False, prefer_gpu=False,
                proposal_scale=np.full(5, 1e-25), inclination_parameterization='b_source')
        np.testing.assert_allclose(sampler.chain[0], expected, atol=1e-12)
