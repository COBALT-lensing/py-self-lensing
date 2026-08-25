import unittest

import numpy as np
from numpy.testing import assert_allclose

from selflensing.fitting.blackjax import BlackJAXSamplerResult, _scaled_proposal_scale


class BlackJAXProposalScaleTest(unittest.TestCase):
    def test_retained_post_burn_chain_honours_original_discard(self):
        sampler = BlackJAXSamplerResult(
            chain=np.arange(2 * 3 * 5).reshape(2, 3, 5),
            log_prob=np.arange(2 * 3).reshape(2, 3),
            chain_start_step=200,
        )

        assert_allclose(sampler.get_chain(discard=200), sampler.chain)
        assert_allclose(sampler.get_log_prob(discard=200), sampler.log_prob)
        self.assertEqual(sampler.get_chain(flat=True, discard=200).shape, (6, 5))

    def test_scale_shrinks_with_more_observations(self):
        proposal_scale = np.ones(5)
        observed_rv = np.ones((2, 4))
        rv_mask = np.array(
            [
                [True, True, False, False],
                [True, True, True, True],
            ]
        )

        scaled = _scaled_proposal_scale(
            proposal_scale,
            observed_rv,
            None,
            rv_mask,
            None,
            None,
            reference_observations=None,
        )

        assert_allclose(scaled[:, 0, 0], [1.0, np.sqrt(0.5)])

    def test_scale_shrinks_with_smaller_rv_errors_at_fixed_count(self):
        proposal_scale = np.ones(5)
        observed_rv = np.ones((3, 4))
        observed_rv_err = np.array(
            [
                [100.0, 100.0, 100.0, 100.0],
                [10.0, 10.0, 10.0, 10.0],
                [1.0, 1.0, 1.0, 1.0],
            ]
        )
        rv_mask = np.ones_like(observed_rv, dtype=bool)

        scaled = _scaled_proposal_scale(
            proposal_scale,
            observed_rv,
            observed_rv_err,
            rv_mask,
            None,
            None,
            reference_observations=None,
        )

        assert_allclose(scaled[:, 0, 0], [1.0, 0.1, 0.01])


if __name__ == "__main__":
    unittest.main()
