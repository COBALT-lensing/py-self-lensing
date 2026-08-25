import unittest

import jax
import jax.numpy as jnp
import numpy

from selflensing.experimental.photometry import (
    PhotometryRow,
    _raw_photometric_logprob,
    batched_photometric_logprob_and_grad,
    photometric_logprob,
)
from selflensing.experimental.mala import initialise, run


class TestExperimentalPhotometry(unittest.TestCase):
    def setUp(self):
        self.row = PhotometryRow(
            times_days=jnp.linspace(0.0, 4.0, 8),
            flux=jnp.zeros(8),
            flux_err=jnp.full(8, 1.0e-3),
            mask=jnp.ones(8, dtype=bool),
            known=jnp.asarray((1.0, 1.0, 6000.0)),
            peak_time_days=jnp.asarray(2.0),
        )
        self.params = jnp.asarray((1.0, jnp.pi / 2, 5.0, 0.1, 0.0))

    def test_custom_gradient_matches_jvp_gradient(self):
        custom = jax.grad(photometric_logprob)(self.params, self.row)
        reference = jax.jacfwd(lambda value: _raw_photometric_logprob(value, self.row))(
            self.params
        )
        numpy.testing.assert_allclose(custom, reference, rtol=1e-7, atol=1e-8)

    def test_batched_api_has_fit_and_walker_axes(self):
        rows = jax.tree.map(lambda value: jnp.stack((value, value)), self.row)
        params = jnp.stack((jnp.stack((self.params, self.params)),) * 2)
        values, gradients = batched_photometric_logprob_and_grad(params, rows)
        self.assertEqual(values.shape, (2, 2))
        self.assertEqual(gradients.shape, (2, 2, 5))

    def test_batched_mala_kernel_runs_with_photometric_gradient(self):
        rows = jax.tree.map(lambda value: value[jnp.newaxis], self.row)
        positions = jnp.broadcast_to(self.params, (1, 2, 5))
        likelihood = lambda value: batched_photometric_logprob_and_grad(value, rows)
        state = initialise(positions, likelihood)
        state, accepted = run(
            jax.random.PRNGKey(2), state, likelihood,
            step_size=jnp.asarray((1.0e-4,)),
            metric_scale=jnp.ones((1, 5)),
            nsteps=3,
        )
        self.assertEqual(state.position.shape, (1, 2, 5))
        self.assertEqual(accepted.shape, (3, 1, 2))


if __name__ == "__main__":
    unittest.main()
