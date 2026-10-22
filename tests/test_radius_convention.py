"""Physical handoff checks independent of the finite-source implementations."""

import unittest

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from astropy import constants, units
from numpy.testing import assert_allclose
from scipy.integrate import quad

from selflensing.batch import batched_light_curve_jit
from selflensing.generators import SelfLensingLightcurveGenerator
from selflensing.gp.LensingGP import run_nucs_callback
from selflensing.lightcurves import self_lensing_amplification
from selflensing.models import SelfLensingSystem


def disc_magnification(separation, re_over_rstar, gamma=0.6):
    """Integrate point-lens magnification over the physical brightness map.

    Lens-centred polar coordinates use source-radius units. The radial
    Jacobian cancels the integrable point-lens singularity at radius zero.
    This does not call RustLens or either Witt–Mao implementation.
    """
    s, q = separation, re_over_rstar
    angle_limit = np.pi if s <= 1 else np.arcsin(1 / s)

    def angular(theta):
        projection = s * np.cos(theta)
        half_chord = np.sqrt(max(0., 1 - s*s*np.sin(theta)**2))
        lower = max(0., projection - half_chord)
        upper = max(0., projection + half_chord)

        def radial(radius):
            mu = np.sqrt(max(0., 1 - s*s - radius*radius
                             + 2*s*radius*np.cos(theta)))
            intensity = 1 - gamma + gamma*mu
            return intensity * (radius*radius + 2*q*q) / np.sqrt(radius*radius + 4*q*q)

        return quad(radial, lower, upper, epsabs=1e-10, epsrel=1e-10)[0]

    flux = 2 * quad(angular, 0., angle_limit, epsabs=1e-10, epsrel=1e-10)[0]
    return flux / (np.pi * (1 - gamma/3))


def circular_case(a_au, lens_mass, radius_rsun):
    """Choose known off-centre separations without package orbit helpers."""
    a = a_au * constants.au.value
    radius = radius_rsun * constants.R_sun.value
    inclination = np.arccos(0.5 * radius / a)
    re = np.sqrt(4 * constants.G.value * lens_mass * constants.M_sun.value
                 * a * np.sin(inclination) / constants.c.value**2)
    period_days = 2*np.pi*np.sqrt(
        a**3 / (constants.G.value * (lens_mass + 1) * constants.M_sun.value)
    ) / 86400
    separations = np.array([0.5, 0.75, 1.5])
    phase = np.arcsin(np.sqrt(separations**2 - 0.5**2) * radius / (a*np.sin(inclination)))
    times = phase * period_days / (2*np.pi)
    params = np.array([a_au, inclination, lens_mass, 1., radius_rsun, 5800., 0., 0.])
    return params, times, separations, re*np.sqrt(np.cos(phase))/radius


class RadiusConventionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Exercise Einstein radii below and above the source radius.
        cls.cases = [circular_case(*values) for values in
                     [(1., 10., 1.), (10., 10., 0.5), (30., 50., 1.)]]
        cls.expected = np.array([
            [disc_magnification(s, qi) for s, qi in zip(separations, q)]
            for _, _, separations, q in cls.cases
        ])

    def test_independent_reference_matches_aligned_uniform_disc(self):
        for q in [0.05, 0.1, 0.2, 1.]:
            with self.subTest(re_over_rstar=q):
                assert_allclose(disc_magnification(0., q, gamma=0.),
                                np.sqrt(1 + 4*q*q), rtol=1e-12, atol=1e-12)

    def test_shared_and_batched_models_match_offcentre_disc_integral(self):
        params = np.stack([case[0] for case in self.cases])
        times = np.stack([case[1] for case in self.cases])
        single = np.stack([
            np.asarray(self_lensing_amplification(t, p))
            for p, t in zip(params, times)
        ])
        batched = np.asarray(batched_light_curve_jit(times, params, ev=False, db=False))
        # Fixed 96-point JAX quadrature has finite error; offsets avoid the limb.
        # This tolerance is far smaller than the missing-conversion error.
        assert_allclose(single, self.expected, rtol=2e-5, atol=2e-5)
        assert_allclose(batched, self.expected, rtol=2e-5, atol=2e-5)

    def test_legacy_native_generators_match_offcentre_disc_integral(self):
        for case, expected in zip(self.cases, self.expected):
            params, times, separations, q = case
            system = SelfLensingSystem(
                a=params[0]*units.au, i=params[1]*units.rad,
                Macc=params[2]*units.M_sun, Mcomp=params[3]*units.M_sun,
                Rcomp=params[4]*units.R_sun, Teff=params[5]*units.K,
            )
            with self.subTest(re_over_rstar=q):
                generator = SelfLensingLightcurveGenerator(
                    system, times=times, peak_time=0., lensing_method="IntegratedWittMao")
                assert_allclose(np.asarray(generator.amplification), expected,
                                rtol=2e-5, atol=2e-5)
                # The uniform-disc native method has no limb darkening.
                uniform = [disc_magnification(s, qi, gamma=0.) for s, qi in zip(separations, q)]
                assert_allclose(generator.witt_mao_amplification, uniform,
                                rtol=1e-8, atol=1e-8)

    def test_gp_callback_converts_einstein_offsets_and_preserves_zero_tangent(self):
        separations = np.array([0.5, 0.75, 1.5])
        evaluate = jax.jit(run_nucs_callback)
        for q in [0.1, 0.5, 2.]:
            with self.subTest(re_over_rstar=q):
                actual = np.asarray(evaluate(jnp.asarray(separations/q), 1/q, 1., q))
                expected = [disc_magnification(s, q) for s in separations]
                assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
        _, tangent = jax.jvp(
            lambda offsets: run_nucs_callback(offsets, 10., 1., 0.1),
            (jnp.asarray(separations/0.1),), (jnp.ones(3),))
        assert_allclose(np.asarray(tangent), 0., rtol=0., atol=0.)


if __name__ == "__main__":
    unittest.main()
