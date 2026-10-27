import unittest
import numpy as np
from selflensing.models import DEFAULT_LIMITS
from selflensing.fitting.initialization import observable_seed_positions


class ObservableInitializationTest(unittest.TestCase):
    def test_seeds_are_reproducible_physical_and_diverse(self):
        kwargs = dict(nwalkers=16, sample_size=512, n_neighbors=32,
                      mode_count=4, random_seed=3)
        first = observable_seed_positions(10000., 5., **kwargs)
        second = observable_seed_positions(10000., 5., **kwargs)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, (16, 5))
        self.assertTrue(np.all(np.isfinite(first)))
        limits = np.array([DEFAULT_LIMITS[k] for k in ('a', 'i', 'Macc', 'e', 'periapsis_phase')])
        self.assertTrue(np.all((first >= limits[:, 0]) & (first <= limits[:, 1])))
        self.assertGreater(np.ptp(first[:, 0]), .001)
