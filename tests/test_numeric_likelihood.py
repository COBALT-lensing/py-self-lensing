import unittest

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from astropy import units
from numpy.testing import assert_allclose

from selflensing.fitting.blackjax import self_lensing_systems_from_batched_fit_blackjax
from selflensing.fitting.likelihood import (
    combined_chi2,
    make_batched_numeric_combined_chi2_vec_jax,
    numeric_combined_chi2,
)
from selflensing.fitting.numeric import parameter_row, physical_properties
from selflensing.models import SelfLensingSystem
from selflensing.fitting.blackjax import _initial_walker_positions


class NumericLikelihoodTest(unittest.TestCase):
    def setUp(self):
        self.params = jnp.array([1.0, 1.5707, 9.0, 0.2, 0.4])
        self.mcomp = 1.1
        self.rcomp = 1.2
        self.times = jnp.linspace(-0.2, 0.2, 9)
        self.observed_rv = jnp.linspace(-40.0, 30.0, self.times.size)
        self.observed_lc = jnp.linspace(0.999, 1.001, self.times.size)
        self.rv_error = jnp.full(self.times.shape, 100.0)
        self.lc_error = jnp.full(self.times.shape, 0.01)

    def _combined(self, params, tau_eff, detected_periods, rv_mask, lc_mask):
        return combined_chi2(
            params,
            self.mcomp,
            self.rcomp,
            tau_eff,
            1.0,
            0.0,
            self.times,
            self.observed_rv,
            self.rv_error,
            self.times,
            self.observed_lc,
            self.lc_error,
            None,
            detected_periods,
            "IntegratedWittMaoJax",
            True,
            None,
            None,
            None,
            rv_mask,
            lc_mask,
        )

    def _numeric(self, params, tau_eff, detected_periods, rv_mask, lc_mask):
        return numeric_combined_chi2(
            params,
            self.mcomp,
            self.rcomp,
            tau_eff,
            0.0,
            self.times,
            self.observed_rv,
            self.rv_error,
            self.times,
            self.observed_lc,
            self.lc_error,
            detected_periods,
            center_lc_model=True,
            rv_mask=rv_mask,
            lc_mask=lc_mask,
        )

    def test_matches_combined_chi2_with_masks_and_priors(self):
        model_tau_eff = physical_properties(
            parameter_row(self.params, self.mcomp, self.rcomp)
        )[2]
        tau_eff = model_tau_eff * 1.01
        detected_periods = jnp.array([100.0, jnp.nan])
        rv_mask = jnp.array([True, False, True, True, False, True, True, True, False])
        lc_mask = jnp.array([True, True, False, True, True, False, True, True, True])

        assert_allclose(
            self._numeric(self.params, tau_eff, detected_periods, rv_mask, lc_mask),
            self._combined(self.params, tau_eff, detected_periods, rv_mask, lc_mask),
            rtol=1e-12,
            atol=1e-12,
        )

    def test_non_lensing_orientations_have_finite_likelihoods(self):
        # Exercise the compiled quantity-aware and numeric paths, including face-on.
        rows = jnp.stack([self.params.at[1].set(i) for i in
                          (0., np.pi / 6, np.pi / 3, 1.5707)])
        values = []
        for evaluator in (self._combined, self._numeric):
            evaluate = jax.jit(jax.vmap(
                lambda row: evaluator(row, None, jnp.array([]), None, None)))
            actual = np.asarray(evaluate(rows))
            self.assertTrue(np.isfinite(actual).all(), actual)
            values.append(actual)
        assert_allclose(values[0], values[1], rtol=1e-12, atol=1e-12)
        system = SelfLensingSystem(a=1.*units.au, i=np.pi/3*units.rad,
            Macc=9.*units.M_sun, Mcomp=self.mcomp*units.M_sun,
            Rcomp=self.rcomp*units.R_sun, e=.2*units.one,
            periapsis_phase=.4*units.rad)
        self.assertFalse(bool(system.sl_possible))

    def test_non_lensing_still_respects_explicit_duration_and_bounds(self):
        params = self.params.at[1].set(np.pi/3)
        for evaluator in (self._combined, self._numeric):
            with self.subTest(evaluator=evaluator.__name__):
                # A measured lensing duration explicitly presupposes a lensing event.
                self.assertTrue(np.isneginf(evaluator(params, .1, jnp.array([]), None, None)))
                self.assertTrue(np.isfinite(evaluator(params, jnp.nan, jnp.array([]), None, None)))
                for invalid in (params.at[1].set(-.1), params.at[1].set(np.pi/2+.1),
                                params.at[2].set(-1.), params.at[3].set(1.1)):
                    self.assertTrue(np.isneginf(evaluator(invalid, None, jnp.array([]), None, None)))
        for evaluator in (combined_chi2, numeric_combined_chi2):
            kwargs = dict(params=params, Mcomp=self.mcomp, Rcomp=self.rcomp,
                tau_eff=None, peak_time=0., rv_times=self.times,
                observed_rv=self.observed_rv, observed_rv_err=self.rv_error,
                lc_times=self.times, observed_lc=self.observed_lc,
                observed_lc_err=self.lc_error, detected_periods=jnp.array([]))
            if evaluator is combined_chi2:
                kwargs.update(d=1., gp_model=None, lensing_method='IntegratedWittMaoJax')
            period = float(physical_properties(parameter_row(params,self.mcomp,self.rcomp))[1])
            self.assertTrue(np.isfinite(evaluator(**kwargs,period_min=period*.9,period_max=period*1.1)))
            self.assertTrue(np.isneginf(evaluator(**kwargs,period_max=period*.9)))
            self.assertTrue(np.isneginf(evaluator(**kwargs,period_min=period*1.1)))

    def test_default_walkers_explore_non_edge_on_orientations(self):
        state = np.random.get_state()
        try:
            np.random.seed(12)
            inclinations = _initial_walker_positions(64)[:,1]
        finally:
            np.random.set_state(state)
        self.assertLess(inclinations.min(), np.pi/6)
        self.assertGreater(inclinations.max(), np.pi/3)
        self.assertTrue(np.all((inclinations >= 0) & (inclinations <= np.pi/2)))

    def test_batched_numeric_likelihood_matches_combined_chi2_per_row(self):
        walker_params = jnp.array(
            [
                [self.params, self.params.at[0].set(1.1)],
                [self.params.at[0].set(1.2), self.params.at[2].set(8.0)],
            ]
        )
        mcomp = jnp.array([1.1, 1.2])
        rcomp = jnp.array([1.2, 1.3])
        times = jnp.broadcast_to(self.times, (2, self.times.size))
        observed_rv = jnp.broadcast_to(self.observed_rv, (2, self.times.size))
        observed_lc = jnp.broadcast_to(self.observed_lc, (2, self.times.size))
        rv_error = jnp.broadcast_to(self.rv_error, (2, self.times.size))
        lc_error = jnp.broadcast_to(self.lc_error, (2, self.times.size))
        rv_mask = jnp.ones((2, self.times.size), dtype=bool)
        lc_mask = jnp.ones((2, self.times.size), dtype=bool)
        periods = jnp.full((2, 0), jnp.nan)

        logprob = make_batched_numeric_combined_chi2_vec_jax(
            mcomp,
            rcomp,
            None,
            jnp.zeros(2),
            times,
            observed_rv,
            rv_error,
            times,
            observed_lc,
            lc_error,
            periods,
            rv_mask=rv_mask,
            lc_mask=lc_mask,
        )
        actual = np.asarray(logprob(walker_params))
        expected = np.asarray(
            [
                [
                    combined_chi2(
                        walker,
                        float(mcomp[fit]),
                        float(rcomp[fit]),
                        None,
                        1.0,
                        0.0,
                        times[fit],
                        observed_rv[fit],
                        rv_error[fit],
                        times[fit],
                        observed_lc[fit],
                        lc_error[fit],
                        None,
                        periods[fit],
                        "IntegratedWittMaoJax",
                        False,
                        None,
                        None,
                        None,
                        rv_mask[fit],
                        lc_mask[fit],
                    )
                    for walker in walker_params[fit]
                ]
                for fit in range(2)
            ]
        )
        assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_batched_blackjax_uses_numeric_likelihood_with_retained_samples(self):
        systems, samplers, walker_state = self_lensing_systems_from_batched_fit_blackjax(
            Mcomp=np.array([1.1, 1.2]) * units.M_sun,
            Rcomp=np.array([1.2, 1.3]) * units.R_sun,
            peak_time=np.zeros(2),
            rv_times=np.broadcast_to(np.linspace(-0.1, 0.1, 5), (2, 5)),
            observed_rv=np.zeros((2, 5)) * units.m / units.s,
            observed_rv_err=np.ones((2, 5)) * 100.0 * units.m / units.s,
            initial_walker_state=np.array(
                [
                    [[1.0, 1.5707, 9.0, 0.2, 0.4]],
                    [[1.1, 1.5707, 8.0, 0.2, 0.4]],
                ]
            ),
            nwalkers=1,
            nsteps=4,
            burn=1,
            prefer_gpu=False,
            progress=False,
            return_sampler=True,
            return_walker_state=True,
            posterior_sample_count=2,
        )

        self.assertEqual(len(systems), 2)
        self.assertEqual(len(samplers), 2)
        self.assertEqual(samplers[0].chain.shape, (2, 1, 5))
        self.assertEqual(samplers[0].get_chain(flat=True, discard=1).shape, (2, 5))
        self.assertEqual(walker_state.coords.shape, (2, 1, 5))


if __name__ == "__main__":
    unittest.main()
