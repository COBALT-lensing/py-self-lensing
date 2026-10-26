"""Regression coverage for analytically profiled photometric baselines."""
import unittest
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose
from selflensing.fitting.likelihood import (
    profiled_lc_chi2, combined_chi2, numeric_combined_chi2,
    make_combined_chi2_vec_jax, make_batched_numeric_combined_chi2_vec_jax,
)
from selflensing.fitting.numeric import deterministic_models, parameter_row


class ProfileBaselineTest(unittest.TestCase):
    def test_weighted_offset_and_masked_invalid_padding(self):
        obs = jnp.array([2., 4., jnp.nan])
        model = jnp.array([1., 1., jnp.nan])
        error = jnp.array([1., 2., 0.])
        mask = jnp.array([True, True, False])
        # Weighted residual mean = (1 + 3/4)/(1 + 1/4) = 1.4.
        expected = -(.4**2 + 1.6**2 / 4)
        evaluate = jax.jit(profiled_lc_chi2)
        assert_allclose(evaluate(obs, error, model, mask), expected)
        assert_allclose(evaluate(obs + 17, error, model, mask), expected)
        assert_allclose(evaluate(obs, error, model, jnp.zeros(3, bool)), 0.)

    def test_likelihood_paths_and_batched_builder_are_shift_invariant(self):
        params = jnp.array([1., 1.5707, 9., .2, .4])
        times = jnp.linspace(-.2, .2, 9)
        model, _ = deterministic_models(times, times, parameter_row(params, 1.1, 1.2), 0.)
        error = jnp.linspace(.001, .002, 9)
        obs = model + .03 + jnp.linspace(-.001, .001, 9)
        expected = profiled_lc_chi2(obs, error, model)
        shared = dict(Mcomp=1.1, Rcomp=1.2, tau_eff=None, peak_time=0.,
                      rv_times=times, observed_rv=None, observed_rv_err=None,
                      lc_times=times, observed_lc=obs, observed_lc_err=error,
                      detected_periods=jnp.array([]), profile_lc_baseline=True)
        for centered in (False, True):
            for evaluate in (combined_chi2, numeric_combined_chi2):
                kwargs = dict(shared, center_lc_model=centered)
                if evaluate is combined_chi2:
                    kwargs.update(d=1., gp_model=None, lensing_method="IntegratedWittMaoJax")
                actual = evaluate(params=params, **kwargs)
                shifted = evaluate(params=params, **dict(kwargs, observed_lc=obs + 2.))
                assert_allclose(actual, expected, atol=1e-9)
                assert_allclose(shifted, expected, atol=1e-9)
        builder_kwargs = dict(shared)
        builder_kwargs.update(d=1., gp=None, lensing_method="IntegratedWittMaoJax")
        single = make_combined_chi2_vec_jax(**builder_kwargs)
        assert_allclose(single(params[None]), [expected], atol=1e-9)
        batch = {key: (None if value is None else jnp.asarray(value)[None])
                 for key, value in shared.items() if key != "profile_lc_baseline"}
        batch["profile_lc_baseline"] = True
        batched = make_batched_numeric_combined_chi2_vec_jax(**batch)
        assert_allclose(batched(params[None, None]), [[expected]], atol=1e-9)
        fixed = numeric_combined_chi2(params=params, **dict(shared, profile_lc_baseline=False))
        self.assertLess(float(fixed), float(expected) - 100.)


if __name__ == "__main__":
    unittest.main()
