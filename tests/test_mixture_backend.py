"""Native mixture sampler preserves physical posterior and geometry Jacobians."""
import unittest
from unittest.mock import patch
import numpy as np
from astropy.constants import G
from astropy import units as u
import jax
jax.config.update('jax_enable_x64',True)
from selflensing.fitting.mixture import LocalMixture
from selflensing.fitting.geometry import physical_to_geometry
import test_impact_geometry as geometry_tests


class MixtureBackendTest(unittest.TestCase):
    def test_native_mixture_in_physical_impact_and_period_coordinates(self):
        original=geometry_tests.self_lensing_system_from_fit_blackjax
        def fit(*args,**kwargs):
            physical=np.array([2.,1.3,9.,.2,.4])+np.random.default_rng(12).normal(size=(10000,5))*[.2,.025,.3,.02,.02]
            coords=physical.copy()
            if kwargs['inclination_parameterization']=='b_source':
                coords=np.asarray(physical_to_geometry(coords,1.))
            if kwargs['parameterization']=='period':
                factor=G.to_value(u.au**3/(u.M_sun*u.day**2))/(4*np.pi**2)
                coords=np.array(coords,copy=True)
                coords[:,0]=np.sqrt(physical[:,0]**3/(factor*(physical[:,2]+1.)))
            proposal=LocalMixture.fit(coords,components=8)
            kwargs.update(independence_proposal=proposal,warmup_steps=0,nsteps=4000)
            result=original(*args,**kwargs)
            self.assertEqual(result[1].proposal_kind,'local_mixture')
            self.assertGreater(result[1].acceptance.mean(),.3)
            return result
        with patch.object(geometry_tests,'self_lensing_system_from_fit_blackjax',side_effect=fit):
            geometry_tests.ImpactGeometryTest().test_fitter_recovers_same_physical_gaussian_posterior_and_state_metadata()

    def test_pulse_sampling_reports_physical_chain_and_density(self):
        import jax.numpy as jnp
        from astropy.time import Time
        from selflensing.fitting.pulse import physical_to_pulse,pulse_to_physical
        from selflensing.fitting.blackjax import BlackJAXFitState,self_lensing_system_from_fit_blackjax
        centre=np.array([2.,1.3,9.,.2,.4,.025]);width=np.array([.2,.025,.3,.02,.02,.003])
        samples=centre+np.random.default_rng(6).normal(size=(12000,6))*width
        proposal=LocalMixture.fit(np.asarray(physical_to_pulse(samples,1.,1.)),components=8)
        with patch('selflensing.fitting.blackjax.make_combined_chi2_vec_jax',
                   return_value=lambda p:-.5*jnp.sum(((p-centre)/width)**2,axis=-1)):
            system,sampler,state=self_lensing_system_from_fit_blackjax(
                Mcomp=1*u.M_sun,Rcomp=1*u.R_sun,peak_time=0.,lc_times=Time([2457000.],format='jd'),
                observed_lc=np.ones(1),nwalkers=24,nsteps=4000,burn=500,warmup_steps=0,
                progress=False,prefer_gpu=False,peak_time_prior_sigma=.5*u.day,
                initial_walker_state=BlackJAXFitState(samples[:24]),independence_proposal=proposal,
                independence_coordinates='pulse')
        flat=sampler.get_chain(discard=500,flat=True)
        np.testing.assert_allclose((flat.mean(axis=0)-centre)/width,0.,atol=.05)
        np.testing.assert_allclose(flat.std(axis=0)/width,1.,atol=.05)
        expected=-.5*np.sum(((sampler.chain-centre)/width)**2,axis=-1)
        np.testing.assert_allclose(sampler.log_prob,expected,atol=1e-9)
        np.testing.assert_allclose(state.coords,sampler.chain[-1])
        np.testing.assert_allclose(pulse_to_physical(sampler.sampling_chain,1.,1.)[0],sampler.chain,atol=1e-10)
        self.assertEqual(sampler.parameterization,'semi_major_axis')
