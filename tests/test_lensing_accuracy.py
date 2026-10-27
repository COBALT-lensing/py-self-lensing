"""Independent references for the Witt–Mao numerical corrections.

Heyrovsky (2003), eqs. (1)–(4): integrate a radial ring kernel using
SciPy's Carlson elliptic functions, independent of the JAX disk-area integral.
"""
import unittest

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose
from scipy.integrate import quad
from scipy.special import elliprf, elliprj

from selflensing.lensing import (
    LD_COEFF, native_magnification_callback,
    witt_mao_magnification_jax as uniform,
    integrated_witt_mao_magnification_jax as integrated,
)


def ring_reference(l, r, e):
    if l == 0.0 or r == 0.0:
        u = max(l, r)/e
        return (u*u+2.0)/(u*np.hypot(u, 2.0))
    d, s = abs(l-r), l+r
    h = np.hypot(d, 2.0*e)
    p = (d/s)**2
    kc2 = p*(s*s+4.0*e*e)/(h*h)
    n = 4.0*l*r/(s*s)
    return 2.0/(np.pi*s*h) * (
        (2.0*e*e+d*d)*elliprf(0.0, kc2, 1.0)
        + d*d*n*elliprj(0.0, kc2, 1.0, p)/3.0
    )


def reference(l, e, limb_darkened):
    c = LD_COEFF if limb_darkened else 0.0
    def f(r):
        return r*(1.0-c+c*np.sqrt(1.0-r*r))*ring_reference(l, r, e)
    points = [l] if 0.0 < l < 1.0 else None
    return quad(f, 0.0, 1.0, points=points, epsabs=1e-9,
                epsrel=1e-10, limit=300)[0] / ((1.0-c/3.0)/2.0)


class LensingAccuracyTest(unittest.TestCase):
    def test_against_independent_ring_integration(self):
        positions = np.array([0., 1e-7, .2, .8, .9999, 1., 1.0001, 2., 20.])
        for e in [.01, .25, 1., 10.]:
            for fn, ld in [(uniform, False), (integrated, True)]:
                with self.subTest(e=e, ld=ld):
                    expected = [reference(l, e, ld) for l in positions]
                    assert_allclose(fn(positions, e, 1.), expected, rtol=2e-7, atol=2e-8)

    def test_analytic_limits_and_no_limb_plateau(self):
        for e in [1e-6, .01, .25, 1., 100., 1e6]:
            assert_allclose(uniform(0., e, 1.), np.hypot(1., 2.*e), rtol=2e-13)
            rho = 1./e
            edge = (2./rho + 2.*(1.+1./rho**2)*np.arctan(rho))/np.pi
            assert_allclose(uniform(1., e, 1.), edge, rtol=2e-13)
        values = np.asarray(uniform(np.array([.9999, 1., 1.0001]), .25, 1.))
        self.assertTrue(np.all(np.diff(values) < 0.))
        assert_allclose(uniform(1e4, .25, 1.), 1., atol=1e-12)

    def test_inputs_broadcasting_and_float32(self):
        for fn in [uniform, integrated]:
            assert_allclose(fn([0, 1, 2], .25, 1), fn(jnp.array([0., 1., 2.]), .25, 1.))
            invalid = np.asarray(jax.jit(fn)(jnp.array([-.1, 0., 1.]), jnp.array([1., 0., np.nan]), 1.))
            self.assertTrue(np.isnan(invalid).all())
            assert_allclose(fn(jnp.array([0., .5, 1., 2.]), .25, 1.),
                            fn(jnp.array([0., .5, 1., 2.]), 2.5, 10.), rtol=1e-13)
            assert_allclose(fn(jnp.array([0., .5]), jnp.array([.1, .25]), 1.),
                            [fn(0., .1, 1.), fn(.5, .25, 1.)], rtol=1e-13)
            positions = jnp.array([0., 1e-7, .5, .9999, 1., 1.0001, 2.], dtype=jnp.float32)
            values = jax.jit(fn)(positions, jnp.float32(.25), jnp.float32(1.))
            self.assertTrue(np.isfinite(values).all())
            assert_allclose(values, fn(positions.astype(jnp.float64), .25, 1.), rtol=2e-6)
        assert_allclose(native_magnification_callback(np.array([0, 1]), 1., 0.), [1., 1.])
        self.assertTrue(np.isnan(native_magnification_callback(np.array([-1.]), 1., 0.)).all())

    def test_parameter_gradients_away_from_exact_limb(self):
        # The uniform disk has a divergent separation derivative at exact contact;
        # do not mistake finite quadrature derivatives there for an analytic limit.
        for fn in [uniform, integrated]:
            grad = jax.jit(jax.grad(lambda x: fn(x[0], x[1], x[2])))
            for l in [0., .5, .9999, 1.0001, 2.]:
                x = np.array([l, .25, 1.])
                actual = np.asarray(grad(jnp.asarray(x)))
                self.assertTrue(np.isfinite(actual).all())
                for k in range(3):
                    if l == 0. and k == 0:
                        assert_allclose(actual[k], 0., atol=1e-12)
                        continue
                    dx = np.zeros(3); dx[k] = 1e-7
                    expected = (fn(*(x+dx))-fn(*(x-dx)))/(2e-7)
                    assert_allclose(actual[k], expected, rtol=3e-5, atol=3e-8)


if __name__ == "__main__":
    unittest.main()
