"""Independent checks of local mixture densities and stationary distributions."""
import unittest
import numpy as np
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
from scipy.stats import multivariate_t
from scipy.special import logsumexp
from selflensing.fitting.mixture import LocalMixture,sample_independence


class LocalMixtureTest(unittest.TestCase):
    def test_density_matches_scipy_for_distinct_covariances_and_weights(self):
        mixture=LocalMixture(np.array([.8,.2]),np.array([[-2.,1.],[3.,-1.]]),np.array([[[.3,0.],[.2,1.]],[[2.,0.],[-.4,.5]]]))
        x=np.random.default_rng(1).normal(size=(19,2))
        expected=logsumexp([np.log(w)+multivariate_t.logpdf(x,loc=c,shape=L@L.T,df=5) for w,c,L in zip(mixture.weights,mixture.centres,mixture.transforms)],axis=0)
        np.testing.assert_allclose(mixture.logpdf(x),expected,atol=1e-11)

    def test_curved_posterior_and_frozen_proposal(self):
        rng=np.random.default_rng(1)
        truth=rng.normal(size=(15000,2));truth[:,1]=.8*(truth[:,0]**2-1)+.1*truth[:,1]
        proposal=LocalMixture.fit(truth,components=24)
        frozen=[x.copy() for x in (proposal.weights,proposal.centres,proposal.transforms)]
        target=jax.jit(lambda z:-.5*(z[:,0]**2+((z[:,1]-.8*(z[:,0]**2-1))/.1)**2))
        chain,p,a=sample_independence(target,proposal,truth[:32],steps=4000,seed=2)
        flat=chain[500:].reshape(-1,2)
        np.testing.assert_allclose(flat.mean(axis=0),0.,atol=.04)
        np.testing.assert_allclose(np.cov(flat.T),np.diag([1.,1.29]),atol=.08)
        for old,new in zip(frozen,(proposal.weights,proposal.centres,proposal.transforms)):
            np.testing.assert_array_equal(old,new)
        self.assertGreater(a.mean(),.6)

    def test_unequal_modes_despite_wrong_proposal_weights(self):
        proposal=LocalMixture(np.array([.95,.05]),np.array([[-4.],[4.]]),np.array([[[.5]],[[.7]]]))
        target=jax.jit(lambda z:jax.scipy.special.logsumexp(jnp.log(jnp.array([.3,.7]))-.5*((z-jnp.array([-4.,4.]))/.5)**2,axis=-1))
        chain,p,a=sample_independence(target,proposal,np.tile([[-4.],[4.]],(16,1)),steps=10000,seed=3)
        self.assertAlmostEqual(float((chain[1000:,:,0]>0).mean()),.7,delta=.035)

    def test_invalid_mixture_rejected(self):
        with self.assertRaises(ValueError):
            LocalMixture(np.array([2.]),np.zeros((1,1)),np.ones((1,1,1)))
        with self.assertRaises(ValueError):
            LocalMixture(np.array([1.]),np.zeros((1,1)),np.zeros((1,1,1)))
