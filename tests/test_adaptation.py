"""Sampling recovery, coverage and isolation of tuning from production."""
import unittest
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
from selflensing.fitting.adaptation import tune_proposals, proposal_step


def production(logprob, warmup, steps=2000, covariance=False):
    def step(carry, key):
        position, score = carry
        move_key, accept_key = jax.random.split(key)
        delta, _ = proposal_step(move_key, position, warmup.transform, covariance)
        proposal = position + delta
        new_score = logprob(proposal)
        accept = jnp.log(jax.random.uniform(accept_key, score.shape)) < new_score - score
        position = jnp.where(accept[..., None], proposal, position)
        return (position, jnp.where(accept, new_score, score)), position
    return np.asarray(jax.jit(lambda: jax.lax.scan(step,
        (warmup.positions, logprob(warmup.positions)),
        jax.random.split(jax.random.PRNGKey(31), steps))[1])())


class AdaptationTest(unittest.TestCase):
    def test_anisotropic_targets_recover_means_and_posterior_widths(self):
        centres = jnp.array([[1., -2.], [3., 1.]])
        widths = jnp.array([[.02, 2.], [.2, .5]])
        logprob = lambda p: -.5 * jnp.sum(((p - centres[:, None]) / widths[:, None])**2, axis=-1)
        initial = np.asarray(centres[:, None] + jax.random.normal(jax.random.PRNGKey(1), (2, 32, 2)) * widths[:, None] * 3)
        warmup = tune_proposals(logprob, initial, [1., 1.], steps=1600, rounds=16)
        scales = np.diagonal(np.asarray(warmup.transform), axis1=-2, axis2=-1)
        self.assertLess(scales[0, 0], scales[0, 1] / 10)
        frozen = np.asarray(warmup.transform).copy()
        chain = production(logprob, warmup)[500:]
        np.testing.assert_allclose(chain.mean(axis=(0, 2)), centres, atol=.12)
        np.testing.assert_allclose(chain.std(axis=(0, 2)), widths, rtol=.15)
        # Known target centres fall inside each marginal 68% interval.
        lo, hi = np.percentile(chain, [16, 84], axis=(0, 2))
        self.assertTrue(np.all((lo < centres) & (centres < hi)))
        np.testing.assert_array_equal(warmup.transform, frozen)

    def test_covariance_proposal_recovers_correlated_gaussian(self):
        cov = np.array([[1., .9], [.9, 1.]])
        inverse = jnp.asarray(np.linalg.inv(cov))
        logprob = lambda p: -.5 * jnp.einsum('fwi,ij,fwj->fw', p, inverse, p)
        initial = np.random.default_rng(2).multivariate_normal([0., 0.], cov, size=(1, 32))
        warmup = tune_proposals(logprob, initial, [.3, .3], steps=800, covariance=True)
        chain = production(logprob, warmup, covariance=True)[500:].reshape(-1, 2)
        np.testing.assert_allclose(chain.mean(axis=0), 0., atol=.12)
        np.testing.assert_allclose(np.cov(chain.T), cov, atol=.15)

    def test_initialization_uses_explicit_rng_without_changing_global_state(self):
        from selflensing.fitting.blackjax import _initial_walker_positions
        np.random.seed(10)
        expected = np.random.uniform()
        np.random.seed(10)
        first = _initial_walker_positions(16, peak_time_prior_sigma_days=.5,
                                         rng=np.random.default_rng(7))
        self.assertEqual(np.random.uniform(), expected)
        second = _initial_walker_positions(16, peak_time_prior_sigma_days=.5,
                                          rng=np.random.default_rng(7))
        np.testing.assert_array_equal(first, second)

    def test_invalid_initialization_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'finite initial walker'):
            tune_proposals(lambda p: jnp.full(p.shape[:-1], -jnp.inf), np.zeros((1, 2, 1)), [1.])


    def test_covariance_rebuild_preserves_accumulated_scale_tuning(self):
        # Two narrow separated modes have a broad pooled covariance. A scale
        # reset on every rebuild leaves acceptance close to zero indefinitely.
        initial = np.tile(np.array([-5., 5.]), 8).reshape(1, 16, 1)
        logprob = lambda p: -.5*jnp.sum(((jnp.abs(p)-5.)/.01)**2, axis=-1)
        warmup = tune_proposals(logprob, initial, [1.], steps=10000,
                                rounds=80, covariance=True)
        self.assertGreater(warmup.acceptance[-10:].mean(), .1)
        self.assertLess(float(warmup.transform[0, 0, 0]), .2)
