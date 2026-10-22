"""Regression checks against angular momentum and an independent Kepler solve."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np
from astropy import constants, units
from numpy.testing import assert_allclose
from scipy.optimize import brentq

from selflensing.fitting.numeric import physical_properties
from selflensing.models import SelfLensingSystem
from selflensing.orbits import conjunction_crossing_time_factor


class CrossingTimeTest(unittest.TestCase):
    def system(self, e, peri, inclination=np.pi / 2):
        return SelfLensingSystem(
            a=10 * units.au, i=inclination * units.rad,
            Macc=10 * units.M_sun, Mcomp=1 * units.M_sun,
            Rcomp=1 * units.R_sun, Teff=5772 * units.K,
            e=e * units.dimensionless_unscaled,
            periapsis_phase=peri * units.rad, d=1 * units.kpc,
        )

    def test_duration_matches_chord_over_kepler_transverse_speed(self):
        for e, peri in [(0., 0.), (.8, 0.), (.8, np.pi),
                        (.99, .7), (.9999, 0.), (.999, np.pi)]:
            with self.subTest(e=e, peri=peri):
                model = self.system(e, peri)
                a = 10 * constants.au.value
                mu = constants.G.value * 11 * constants.M_sun.value
                r = a * (1-e*e) / (1+e*np.cos(peri))
                h = np.sqrt(mu*a*(1-e*e))
                transverse_speed = h/r
                radius = float(model.rE().to(units.m).value) + constants.R_sun.value
                expected = 2*radius/transverse_speed/86400
                assert_allclose(model.tau_eff.to(units.day).value, expected, rtol=2e-12)

    def test_circular_limit_and_apsidal_speed_ratio(self):
        assert_allclose(conjunction_crossing_time_factor(0., 1.3), 1.)
        for e in [.2, .9, .9999]:
            assert_allclose(conjunction_crossing_time_factor(e, 0.),
                            np.sqrt((1-e)/(1+e)), rtol=1e-12)
            assert_allclose(conjunction_crossing_time_factor(e, np.pi),
                            np.sqrt((1+e)/(1-e)), rtol=1e-12)

    def test_correction_matches_independent_projected_position_derivative(self):
        # Bracket Kepler independently, then differentiate Cartesian x=r sin(longitude).
        for e, peri in [(.7, .8), (.95, 0.), (.95, np.pi)]:
            E0 = 2*np.arctan2(np.sqrt(1-e)*np.sin(-peri/2),
                             np.sqrt(1+e)*np.cos(-peri/2))
            M0 = E0-e*np.sin(E0)
            step = 1e-7*(1-e)**1.5
            def projected_position(mean):
                E = brentq(lambda x: x-e*np.sin(x)-mean, -2*np.pi, 2*np.pi,
                           xtol=1e-15)
                return ((np.cos(E)-e)*np.sin(peri)
                        + np.sqrt(1-e*e)*np.sin(E)*np.cos(peri))
            speed_over_na = (projected_position(M0+step)-projected_position(M0-step))/(2*step)
            assert_allclose(1/speed_over_na, conjunction_crossing_time_factor(e, peri),
                            rtol=2e-6)

    def test_quantity_numeric_and_compiled_batch_paths_agree(self):
        rows, expected = [], []
        for e, peri in [(0., .3), (.8, 0.), (.8, np.pi), (.99, .7)]:
            model = self.system(e, peri, np.pi/2-1e-5)
            rows.append([10., np.pi/2-1e-5, 10., 1., 1., 5772., e, peri])
            expected.append(float(model.tau_eff.to(units.day).value))
        feasible, _, duration = jax.jit(jax.vmap(physical_properties))(jnp.asarray(rows))
        self.assertTrue(np.asarray(feasible).all())
        assert_allclose(duration, expected, rtol=2e-12)
        derivative = jax.grad(lambda e: physical_properties(jnp.asarray(rows[0]).at[6].set(e))[2])(.4)
        self.assertTrue(np.isfinite(derivative))


if __name__ == '__main__':
    unittest.main()
