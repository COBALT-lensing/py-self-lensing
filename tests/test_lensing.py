import unittest

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose
import rustlens

from selflensing.lensing import (
    integrated_witt_mao_magnification_jax,
    witt_mao_magnification_jax,
)


class WittMaoJaxTest(unittest.TestCase):
    def test_witt_mao_matches_rustlens(self):
        offsets = np.array(
            [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 0.9, 0.99, 1.0, 1.01, 1.1, 2.0]
        )

        for re, rstar in [(0.02, 1.0), (0.1, 1.0), (0.5, 1.0), (0.1, 0.5)]:
            with self.subTest(re=re, rstar=rstar):
                rustlens_values = np.asarray(
                    rustlens.witt_mao_magnification(offsets.tolist(), re, rstar)
                )
                jax_values = np.asarray(witt_mao_magnification_jax(offsets, re, rstar))

                assert_allclose(jax_values, rustlens_values, rtol=1e-6, atol=1e-6)

    def test_integrated_witt_mao_matches_rustlens(self):
        offsets = np.array([0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 0.9, 1.0, 1.1, 2.0])

        for re, rstar in [(0.02, 1.0), (0.1, 1.0), (0.5, 1.0), (0.1, 0.5)]:
            with self.subTest(re=re, rstar=rstar):
                rustlens_values = np.asarray(
                    rustlens.integrated_witt_mao_magnification(
                        offsets.tolist(), re, rstar
                    )
                )
                jax_values = np.asarray(
                    integrated_witt_mao_magnification_jax(offsets, re, rstar)
                )

                assert_allclose(jax_values, rustlens_values, rtol=1e-4, atol=1e-4)

    def test_witt_mao_can_be_jitted(self):
        offsets = jnp.array([0.0, 0.1, 0.5, 1.0, 2.0], dtype=jnp.float64)
        re = 0.1
        rstar = 1.0

        rustlens_values = np.asarray(
            rustlens.integrated_witt_mao_magnification(
                np.asarray(offsets).tolist(), re, rstar
            )
        )
        jitted_values = np.asarray(
            jax.jit(integrated_witt_mao_magnification_jax)(offsets, re, rstar)
        )

        assert_allclose(jitted_values, rustlens_values, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
