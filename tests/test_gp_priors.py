import unittest
from types import SimpleNamespace
from unittest.mock import patch

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np

from selflensing.gp import TESSGP, PriorsSamplers as ps


class StellarNoisePriorTest(unittest.TestCase):
    def test_sampling_distributions(self):
        rng = np.random.default_rng(42)
        quality = ps.sample_quality_factor(rng, size=20000)
        self.assertTrue(np.all((quality >= 100) & (quality <= 10000)))
        self.assertAlmostEqual(float(np.mean(np.log10(quality))), 3., delta=.02)
        for tag, bounds in [('G', (50., 300.)), ('S', (100., 1000.))]:
            ppm = 1e6 * 10.**(np.asarray(ps.sample_log_sigma(6., tag, rng, size=20000)) - 6.)
            low, high = bounds
            self.assertTrue(np.all((ppm >= low) & (ppm <= high)))
            self.assertAlmostEqual(float(np.mean((ppm-low)/(high-low))), .5, delta=.01)

    def make_gp(self, **kwargs):
        source = SimpleNamespace(Mcomp=SimpleNamespace(value=1.), d=SimpleNamespace(value=1.))
        with patch('selflensing.gp.FluxFinderSensitivity.conv_mass_to_mag',
                   return_value=(jnp.asarray(10.), jnp.asarray(.001))):
            return TESSGP(source, lc_times=jnp.arange(16.) * 600., **kwargs)

    def test_seeded_gp_draws_are_stable_and_reproducible(self):
        first = self.make_gp(rng=123)
        second = self.make_gp(rng=123)
        np.testing.assert_array_equal(first.theta, second.theta)
        before = np.asarray(first.theta).copy()
        draw = first.generate()
        np.testing.assert_allclose(draw, second.generate())
        self.assertTrue(np.all(np.isfinite(draw)))
        np.testing.assert_array_equal(first.theta, before)
        self.assertFalse(np.array_equal(draw, first.generate()))
        self.assertFalse(np.array_equal(first.theta, self.make_gp(rng=124).theta))
        self.assertLessEqual(first.kernel_bounds['S'][1][0], first.Q_s)
        self.assertGreaterEqual(first.kernel_bounds['S'][1][1], first.Q_s)

    def test_measured_amplitudes_and_custom_quality_bounds(self):
        gp = self.make_gp(rng=3, component_ppm={'G': 80., 'S': 250.},
                          seismic_quality_bounds=(200., 300.))
        self.assertTrue(200. <= gp.Q_s <= 300.)
        self.assertAlmostEqual(float(1e6 * 10.**(gp.lsig_g-gp.mu_test)), 80.)
        self.assertAlmostEqual(float(1e6 * 10.**(gp.lsig_s-gp.mu_test)), 250.)
        for bounds in [(5., 5.), (0., 10.), (100., np.inf)]:
            with self.assertRaises(ValueError):
                ps.sample_quality_factor(np.random.default_rng(0), bounds=bounds)
        with self.assertRaises(ValueError):
            ps.sample_log_sigma(6., 'G', np.random.default_rng(0), ppm=-1.)


if __name__ == '__main__':
    unittest.main()
