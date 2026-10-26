"""Per-source RV offsets must be profiled during every likelihood evaluation."""
import unittest

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose

from selflensing.fitting.likelihood import (
    profiled_rv_chi2, _prepare_rv_source_ids,
    make_combined_chi2_vec, make_combined_chi2_vec_jax,
    make_batched_combined_chi2_vec_jax, make_batched_numeric_combined_chi2_vec_jax,
)


class RVSourceTest(unittest.TestCase):
    def test_weighted_offsets_and_source_shift_invariance(self):
        observed = jnp.array([1., 5., 100., 106.])
        error = jnp.array([1., 2., 1., 2.])
        groups = jnp.array([0, 0, 1, 1])
        evaluate = jax.jit(lambda obs: profiled_rv_chi2(obs, error, jnp.zeros(4), source_ids=groups))
        # Weighted means are 1.8 and 101.2, giving chi squared 3.2 + 7.2.
        self.assertAlmostEqual(float(evaluate(observed)), -10.4)
        self.assertAlmostEqual(float(evaluate(observed + jnp.array([40., 40., -20., -20.]))), -10.4)
        self.assertLess(float(profiled_rv_chi2(observed, error, jnp.zeros(4))), -1000.)

    def test_singletons_masks_empty_groups_and_gradients(self):
        obs = jnp.array([3., 9., jnp.nan, 40.])
        err = jnp.array([1., 2., 0., 1.])
        groups = jnp.array([0, 0, 1, 2])
        mask = jnp.array([True, True, False, True])
        evaluate = jax.jit(lambda model: profiled_rv_chi2(obs, err, model, mask, groups))
        self.assertAlmostEqual(float(evaluate(jnp.zeros(4))), -7.2)
        self.assertTrue(np.isfinite(jax.grad(evaluate)(jnp.zeros(4))).all())
        self.assertEqual(float(profiled_rv_chi2(obs, err, jnp.zeros(4), jnp.zeros(4, dtype=bool), groups)), 0.)
        assert_allclose(profiled_rv_chi2(jnp.array([3.]), jnp.array([1.]), jnp.array([22.]), source_ids=jnp.array([0])), 0.)

    def test_single_source_matches_legacy_and_validates_shape(self):
        obs = jnp.array([3., 5., 8.]); err = jnp.array([1., 2., 3.])
        assert_allclose(profiled_rv_chi2(obs, err, jnp.zeros(3)),
                        profiled_rv_chi2(obs, err, jnp.zeros(3), source_ids=jnp.zeros(3, dtype=int)))
        with self.assertRaisesRegex(ValueError, 'same shape'):
            _prepare_rv_source_ids(['A', 'B'], obs)
        with self.assertRaisesRegex(ValueError, 'same shape'):
            _prepare_rv_source_ids(['A'], None)
        assert_allclose(_prepare_rv_source_ids(['B', 'B', 'A'], obs), [1, 1, 0])

    def test_single_and_batched_builders_agree_and_ignore_source_shifts(self):
        params = jnp.array([[1., 1.5707, 9., .2, .4], [1.1, 1.55, 8., .1, .6]])
        times = jnp.linspace(-.2, .2, 4)
        obs = jnp.array([1., 4., 103., 108.])
        err = jnp.array([1., 2., 1., 2.]) * 100.
        labels = np.array(['A', 'A', 'B', 'B'])
        common = dict(Mcomp=1.1, Rcomp=1.2, tau_eff=None, peak_time=0., rv_times=times,
                      observed_rv=obs, observed_rv_err=err, lc_times=None,
                      observed_lc=None, observed_lc_err=None, detected_periods=jnp.array([]))
        for builder in (make_combined_chi2_vec, make_combined_chi2_vec_jax):
            kwargs = dict(common, d=1., gp=None, lensing_method='IntegratedWittMaoJax', rv_source_ids=labels)
            actual = np.asarray(builder(**kwargs)(params))
            shifted = dict(kwargs, observed_rv=obs+jnp.array([500., 500., -200., -200.]))
            assert_allclose(builder(**shifted)(params), actual, rtol=1e-10, atol=1e-10)
            for batched in (make_batched_combined_chi2_vec_jax, make_batched_numeric_combined_chi2_vec_jax):
                for tau in (None, jnp.array([jnp.nan, jnp.nan])):
                    kw = {key: (None if value is None else jnp.stack([value, value])) for key,value in common.items()}
                    kw.update(tau_eff=tau, rv_source_ids=np.stack([labels, ['C', 'C', 'D', 'D']]),
                              rv_mask=jnp.ones((2,4), dtype=bool))
                    if batched is make_batched_combined_chi2_vec_jax:
                        kw.update(d=jnp.ones(2), gp=None, lensing_method='IntegratedWittMaoJax')
                    result = batched(**kw)(jnp.stack([params,params]))
                    assert_allclose(result, np.stack([actual,actual]), rtol=1e-10, atol=1e-10)

    def test_source_labels_reach_single_and_batched_samplers(self):
        from types import SimpleNamespace
        from astropy import units
        from astropy.time import Time
        from selflensing.fitting import (
            self_lensing_system_from_fit_blackjax,
            self_lensing_systems_from_batched_fit_blackjax,
        )
        coords = np.array([[1., 1.5707, 9., .2, .4], [1.01, 1.5706, 9.1, .21, .41]])
        kwargs = dict(
            Mcomp=1.1 * units.M_sun, Rcomp=1.2 * units.R_sun,
            peak_time=Time(2457000., format='jd'),
            rv_times=Time(2457000. + np.array([0., .1, 1., 1.1]), format='jd'),
            observed_rv=np.array([1., 2., 101., 102.]) * units.km / units.s,
            observed_rv_err=np.ones(4) * units.km / units.s,
            rv_source_ids=['A', 'A', 'B', 'B'],
            nwalkers=2, nsteps=3, burn=1, prefer_gpu=False, progress=False,
            initial_walker_state=SimpleNamespace(coords=coords),
            return_sampler=True, random_seed=1,
        )
        _, sampler, _ = self_lensing_system_from_fit_blackjax(**kwargs)
        self.assertTrue(np.isfinite(sampler.get_log_prob()).all())
        kwargs.update(
            Mcomp=np.array([1.1, 1.1]) * units.M_sun,
            Rcomp=np.array([1.2, 1.2]) * units.R_sun,
            peak_time=Time([2457000., 2457000.], format='jd'),
            rv_times=Time(np.tile(kwargs['rv_times'].jd, (2, 1)), format='jd'),
            observed_rv=np.tile(kwargs['observed_rv'].value, (2, 1)) * units.km / units.s,
            observed_rv_err=np.ones((2, 4)) * units.km / units.s,
            rv_source_ids=[['A', 'A', 'B', 'B'], ['C', 'C', 'D', 'D']],
            rv_mask=np.array([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=bool),
            initial_walker_state=SimpleNamespace(coords=np.stack([coords, coords])),
        )
        systems, samplers, _ = self_lensing_systems_from_batched_fit_blackjax(**kwargs)
        self.assertEqual(len(systems), 2)
        for sampler in samplers:
            self.assertTrue(np.isfinite(sampler.get_log_prob()).all())


if __name__ == '__main__':
    unittest.main()
