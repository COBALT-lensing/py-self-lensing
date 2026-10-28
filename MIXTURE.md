# Frozen local-mixture sampling and exact pulse coordinates

`self_lensing_system_from_fit_blackjax` accepts an optional `independence_proposal`, a `LocalMixture` trained on discarded pilot/warm-up draws. Each component has its own covariance, weight and normalization. Components are multivariate Student-t kernels with 5 degrees of freedom by default. The full independence MH correction is applied; proposals and likelihoods are evaluated in GPU batches, then the exact MH decisions are scanned in time order. All repeated states are retained.

```python
from selflensing.fitting.mixture import LocalMixture
from selflensing.fitting.pulse import physical_to_pulse

# pilot_physical: discarded pilot draws in columns a, i, Macc, e, omega, [epoch offset]
proposal = LocalMixture.fit(
    physical_to_pulse(pilot_physical, companion_mass_msun, radius_rsun),
    components=64,
)
proposal.save('proposal.npz')
# Pass to the usual fitter:
# independence_proposal=proposal, independence_coordinates='pulse',
# warmup_steps=0, return_sampler=True
```

A supplied mixture freezes before production. `burn` still discards the declared initial recorded steps for posterior summaries. `independence_chunk_size=512` controls batching and GPU memory, not the target distribution. Mixture sampling currently requires `return_sampler=True` and is implemented for the single-system fitter. The batched fitter retains its existing kernels.

With `independence_coordinates='sampling'`, train in the selected fitter coordinates. With `'pulse'`, use `parameterization='semi_major_axis'` and `inclination_parameterization='inclination'`. The pulse mode returns **physical a/i/M/e/omega samples and physical log probabilities** through the ordinary adapter, so confidence intervals, best-fit selection, plotting and resumed physical walker states retain their existing conventions. Raw pulse samples and transformed log probabilities are also available as `sampling_chain` and `sampling_log_prob`.

## Exact transform and original prior

Let R be the stellar radius in AU, mu = G in AU^3/(solar mass day^2), r the separation at longitude zero, f = 1+e cos(omega), and F=(1-e^2)/f. Sample

- log(kappa), where kappa = RE0^2/R^2 = 4 mu M r sin(i)/(c^2 R^2);
- b = r cos(i)/R;
- log(s), where s = vt0/R and vt0^2 = mu (M+Mstar) f/r;
- the original eccentricity, periapsis angle and optional epoch offset.

The forward orbit has r=aF. The inverse solves the unique positive-mass root

`(A r - Mstar) sqrt(r^2 - (b R)^2) = kappa/K`,

where A=(sR)^2/(mu f), K=4mu/(c^2 R^2), then reconstructs M, i and a. A safeguarded Newton solve is checked for residual accuracy. The inverse Jacobian is

`|d(a,i,M)/d(log kappa,b,log s)| = 2 R sin(i) M(M+Mstar) / [F (M + (M+Mstar)sin(i)^2)]`.

It is added to the original full physical log target. All original physical bounds and priors remain; the full eccentric orbital light curve is evaluated. No straight-track light-curve approximation or additional transit/lensing prior is introduced. The transform is validated against independently differentiated forward equations and round trips across wide orbital ranges and two different stars; known physical posterior tests also verify the adapter's reported density.

## Notebook workflow

The review workflow accepts optional `BLACKJAX_INDEPENDENCE_PROPOSAL_PATH` and `BLACKJAX_INDEPENDENCE_COORDINATES`. A path loads the frozen proposal once; it does not retrain during production. Leave the path unset to use the existing sampler. Pulse proposals require the physical inclination setting described above. Retrain a proposal on discarded pilot draws when needed for sampling efficiency after changing the data/model; the proposal itself never changes the physical target.
