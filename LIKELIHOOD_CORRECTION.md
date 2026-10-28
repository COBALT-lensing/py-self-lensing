# Gaussian likelihood correction and observable-constraint audit (2026-10-28)

The shared selflensing Gaussian data terms now contribute `-chi_squared/2` to the log target. Previously they contributed `-chi_squared`. Fixed Gaussian normalization constants are omitted as before. Explicit Gaussian priors (including duration and epoch) and native GP log probabilities are not rescaled. The RV paper's separate `src/rv_only_no_prior.py` target is corrected consistently. The experimental shared photometric likelihood and its custom derivative now use the same convention. Negative-chi-squared diagnostic helpers retain their original definitions.

New shared BlackJAX posterior exports identify this target as `gaussian_chi2_half_v2`. Existing chains and stored inference results retain their previous target and are not corrected by reloading the code. Paper fits and derived figures must be recomputed before reporting corrected inference. Stage-1 schedules derived from the old photometric posterior also reflect the old inference; this patch does not regenerate the production paper pipeline or overwrite its results.

## Amplitude/crossing-time constraints in self_lensing_radial_velocity

- `main.tex`, Photometric MCMC fit: amplitude and crossing time determine initial walker estimates; it explicitly specifies uniform physical parameter priors.
- `src/estimation.py`: nearest neighbours in log peak-amplitude / log crossing-time space estimate mass and period distributions. The weighted mass/period summaries are used in the observable-space analysis notebook; they are not additional Gaussian mass/period factors in the production MCMC target.
- `src/prefit.py`: `observable_seed_walker_positions` clusters those neighbouring systems to supply initial walkers. They can move outside those initial regions.
- Both single and batched stage-1 fits pass `tau_eff` from simulated truth to the shared fitter. The shared likelihood applies `Normal(tau_eff, tau_eff/100)` to the model duration, a 1% constraint. Thus stage-1 inference includes trusted duration information beyond the photometric fluxes. The manuscript describes the idealization of known stellar properties and crossing time, but should make the implemented 1% duration term explicit.
- The review candidate workflow estimates amplitude and FWHM to guide initial walkers. It does not impose neighbour-derived mass/period priors, and it does not pass a known `tau_eff`. The FWHM estimate is not the same quantity as the model's effective crossing time. Pulse-coordinate sampling and learned proposal mixtures preserve the physical prior through Jacobian/MH corrections; they are not observational priors.

Do not copy the simulation's 1% true-duration constraint onto a real candidate's rough FWHM. Adding a prior derived from the same photometry that is already fitted risks double-counting those data; a summary-observable likelihood would need to replace the overlapping information or account for dependence/uncertainty.

A separate manuscript/code discrepancy remains outside this normalization correction: the manuscript states a hard self-lensing geometry restriction, while the current shared fitter allows non-lensing orientations. This should be reconciled when updating the methods/results.

Validation includes absolute Gaussian densities against SciPy for single/batched photometry, RV and joint paths; masked data; preserved duration/period/epoch priors; native GP and Gaussian fallback; experimental gradients; and an independent nonzero-residual test of the local RV-only target. An older RV equivalence fixture was passing kg/metres into the current public solar-mass/solar-radius API; its inputs were corrected, and the noiseless model is now explicitly required to have zero residual likelihood.
