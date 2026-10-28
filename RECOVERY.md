# Optional robust covariance and recovery proposals

Both single and batched BlackJAX fitters accept `robust_covariance=True` (requires `adapt_covariance=True`) and `recovery_probability=0.1`. Both default off. They preserve the existing likelihood and priors.

Robust covariance estimates the proposal shape from warm-up states within `warmup_log_density_cut=20` physical log-density units of the best state in each warm-up block. Coordinate Jacobians are removed for this selection only; MH acceptance always uses the full sampling-coordinate target. The existing regularization and diminishing scale adaptation remain. No walkers or production states are deleted or replaced.

Recovery mixes the local proposal with an independence proposal from a fixed, equally weighted mixture of `recovery_centres_count=16` multivariate Student-t kernels (`recovery_df=3`). Diverse high-density warm-up states supply the centres; the final warm-up transform supplies their shared scale. Every recovery move uses `log q(x)-log q(y)` in its MH ratio. Mixture-component selection has constant probability independent of the current state. The Student-t mixture has full support; proposals outside physical limits are rejected by the original target. Centres and scales are frozen throughout production.

Recovery can move a stranded walker to a better region, but does not guarantee discovery of unvisited modes or convergence along a curved degeneracy. The density cut affects proposal efficiency, not the production target. Always inspect convergence, mode occupancy and effective sample size.

The result adapter records `recovery_attempted`, the frozen `recovery_centres` and `recovery_transform`, `warmup_eligible_fraction`, and `warmup_recovery_acceptance`. Acceptance is still recorded for every move. Sparse batched retention does not retain per-step acceptance or recovery masks.

Tests check a correlated Gaussian, a negligible distant mode with a trapped initial walker, unequal posterior mode weights under deliberately mismatched proposal weights, and the Student-t density ratio against SciPy. Synthetic light-curve tests exercise both geometries through single and batched fitting.
