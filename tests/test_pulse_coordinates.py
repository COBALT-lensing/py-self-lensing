"""Exact inverse and analytic prior-Jacobian validation."""
import unittest
import numpy as np
import jax
jax.config.update('jax_enable_x64',True)
from selflensing.fitting.pulse import physical_to_pulse,pulse_to_physical


class PulseCoordinatesTest(unittest.TestCase):
    def test_roundtrip_and_full_jacobian_for_two_stars(self):
        rng=np.random.default_rng(123)
        x=np.column_stack([np.exp(rng.uniform(np.log(.1),np.log(200),500)),rng.uniform(.01,np.pi/2,500),rng.uniform(.1,150,500),rng.uniform(.00001,.9999,500),rng.uniform(0,2*np.pi,500),rng.normal(size=500)*.1])
        for companion,radius in ((1.,1.),(2.3,1.7)):
            z=physical_to_pulse(x,companion,radius)
            back,jac=pulse_to_physical(z,companion,radius)
            np.testing.assert_allclose(back,x,rtol=2e-9,atol=1e-10)
            J=jax.vmap(jax.jacfwd(lambda p:physical_to_pulse(p,companion,radius)))(x)
            _,det=np.linalg.slogdet(np.asarray(J))
            np.testing.assert_allclose(jac,-det,rtol=1e-8,atol=1e-8)

    def test_negative_impact_is_off_support(self):
        z=physical_to_pulse(np.array([[2.,1.3,9.,.2,.4]]),1.,1.)
        z=z.at[...,1].set(-1.)
        self.assertTrue(np.isneginf(pulse_to_physical(z,1.,1.)[1]).all())
