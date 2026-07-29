# py-self-lensing

A Python module for generating and fitting light curves and radial velocity time series for self-lensing binaries. Includes three effects in light curves: self-lensing, Doppler boosting, and ellipsoidal variations. 

The effect of eccentricity is included in both RV and light curves, **except** for the effect this has to change the magnification during self-lensing. Eccentricity is correctly considered when working out the distance between the lens and the source at the mid-point of the event, but at the moment we don't vary this distance in the calculations even though it would technically change continuously in an eccentric system (in other words the mid-point distance is used for the whole event). The distance is varied continuously for ellipsoidal variations, however.

## Installing

For now, this can be installed with `pip` directly from GitHub:

```shell
$ pip install git+https://github.com/COBALT-lensing/py-self-lensing.git
```

Though while the repo is private you'll likely need to use SSH:

```shell
pip install git+ssh://git@github.com/COBALT-lensing/py-self-lensing.git
```

### GPU support

Fitting uses JAX for the vectorised likelihood calculation. A normal install uses
JAX's CPU backend, but if a GPU-enabled JAX backend is installed the fitter will
prefer it automatically.

For development installs with uv, sync one of the optional GPU extras:

```shell
uv sync --extra cuda
```

`cuda` is an alias for the current recommended NVIDIA CUDA 13 wheels. Other
available extras are:

```shell
uv sync --extra cuda13
uv sync --extra cuda12
uv sync --extra cuda13-local
uv sync --extra cuda12-local
uv sync --extra rocm
```

Use the `*-local` groups only when CUDA and cuDNN are already installed on the
machine. Otherwise prefer `cuda`/`cuda13`, which install the NVIDIA CUDA
libraries from Python wheels.

For pip installs, request the matching package extra:

```shell
pip install "git+https://github.com/COBALT-lensing/py-self-lensing.git#egg=selflensing[cuda]"
```

Check which backend JAX can see with:

```shell
uv run python -c "import jax; print(jax.devices())"
```

The fitting function also accepts explicit device controls:

```python
self_lensing_system_from_fit(..., prefer_gpu=True)  # default
self_lensing_system_from_fit(..., device="gpu:0")
self_lensing_system_from_fit(..., prefer_gpu=False)  # force CPU fallback
```

## Generating light curves and RV

The first step is to define the properties of your self-lensing system:

```python
from astropy import units
from selflensing.models import SelfLensingSystem

sls = SelfLensingSystem(
    a=1 * units.AU,
    b=0 * units.dimensionless_unscaled,
    Macc=9*units.M_sun,
    Mcomp=1*units.M_sun,
    Rcomp=10*units.R_sun
)
```

Then you can use the generators to produce simulated data:

```python
from astropy.time import Time
from selflensing.generators import LightcurveGenerator, RVGenerator

start_time = Time("2025-01-01")
end_time = Time("2026-01-01")
peak_time = Time("2025-06-01")

lcg = LightcurveGenerator(sls, start_time=start_time, end_time=end_time, peak_time=peak_time)
rvg = RVGenerator(sls, start_time=start_time, end_time=end_time, peak_time=peak_time)
```

```python
from matplotlib import pyplot

pyplot.scatter(lcg.times.jd, lcg.flux_array)
```

![alt text](https://github.com/COBALT-lensing/py-self-lensing/blob/main/images/lc.png?raw=true)

```python
pyplot.scatter(rvg.times.jd, rvg.rv)
```

![alt text](https://github.com/COBALT-lensing/py-self-lensing/blob/main/images/rv.png?raw=true)

## Matched-filter peak searches

The `selflensing.matched_filtering` module searches a light curve with a bank of
pre-sampled templates. It subtracts the mean from the input flux, subtracts and
unit-energy normalises each template, computes a valid cross-correlation, and
uses `scipy.signal.find_peaks` to identify candidate matches. By default, a
candidate must pass both a robust prominence threshold and a robust absolute
matched-filter score threshold, which helps reject low-score continuum
structure. Overlapping matches across the template bank are collapsed to the
highest-scoring match by default.

Templates carry their own time grid and are resampled to the median cadence of
the light curve being searched:

```python
from selflensing.matched_filtering import (
    MatchedFilterTemplate,
    detect_matched_filter_peaks,
)

templates = [
    MatchedFilterTemplate(template_flux_1, template_time_1),
    MatchedFilterTemplate(template_flux_2, template_time_2),
]

matches = detect_matched_filter_peaks(
    time,
    flux,
    templates,
    prominence_sigma=5.0,
    min_score_sigma=8.0,
    dedupe_overlaps=True,
    max_matches=10,
)

for match in matches:
    print(match.start_time, match.peak_time, match.end_time, match.template_index)
```

Each result is a `MatchedFilterMatch` with the matching `template_index`,
correlation score, peak prominence, and the start, peak, and end
indices/timestamps mapped back onto the original light curve. See
`notebooks/matched_filtering.ipynb` for a demo that simulates a TESS-like light
curve, builds a small template bank from a range of `SelfLensingSystem`
parameters, and plots the returned windows.

## Fitting light curves and RV

The `selflensing.fitting` package provides Bayesian MCMC fitters for recovering
self-lensing binary parameters from light curves, radial velocities, or both.
The default public entry point uses `emcee`, with the model likelihood evaluated
through JAX:

```python
from astropy import units
from selflensing.fitting import self_lensing_system_from_fit

sls_fit, sampler, walker_state = self_lensing_system_from_fit(
    Mcomp=1 * units.M_sun,
    Rcomp=10 * units.R_sun,
    Teff=5000 * units.K,
    d=1 * units.kpc,
    peak_time=peak_time,
    lc_times=lcg.times,
    observed_lc=observed_flux,
    observed_lc_err=observed_flux_err,
    rv_times=rvg.times,
    observed_rv=observed_rv,
    observed_rv_err=observed_rv_err,
    nwalkers=100,
    nsteps=5000,
    burn=500,
)
```

At least one of `observed_lc` or `observed_rv` must be supplied. If both are
provided, the fitter uses the combined likelihood. Light-curve fluxes should be
dimensionless and centred on the same convention as the generated model; pass
`center_lc_model=True` when fitting data that has already had a unit baseline
subtracted. RV values and errors should be Astropy quantities convertible to
`m / s`.

The fitter samples five parameters:

- `a`: semi-major axis, in AU.
- `i`: inclination, in radians.
- `Macc`: compact-object/accretor mass, in solar masses.
- `e`: eccentricity.
- `periapsis_phase`: periapsis phase, in radians.

The companion properties (`Mcomp`, and usually `Rcomp` and `Teff`), distance
`d`, and the reference `peak_time` are treated as fixed inputs. The returned
`sls_fit` is a `SelfLensingSystem` constructed from the highest log-probability
post-burn-in sample. The `sampler` is the underlying `emcee.EnsembleSampler`,
and `walker_state` can be passed back as `initial_walker_state` to continue or
refine a run:

```python
sls_fit, sampler, walker_state = self_lensing_system_from_fit(
    ...,
    nsteps=2000,
    burn=200,
)

sls_refined, sampler_refined, walker_state = self_lensing_system_from_fit(
    ...,
    initial_walker_state=walker_state,
    nsteps=3000,
    burn=0,
)
```

The sampler is useful for diagnostics and posterior summaries:

```python
samples = sampler.get_chain(flat=True, discard=500)
log_prob = sampler.get_log_prob(flat=True, discard=500)
print(samples.shape, log_prob.max())
```

Optional period information can be included in the likelihood. Use `periods` for
known candidate periods in days, or set `detect_periods=True` to run a
Lomb-Scargle search on the supplied light curve and/or RV before fitting:

```python
sls_fit, sampler, walker_state = self_lensing_system_from_fit(
    ...,
    periods=[365.25],
    period_min=300 * units.day,
    period_max=450 * units.day,
)
```

You can also pass `tau_eff` to add a Gaussian prior on the effective
self-lensing duration. The period and duration terms are soft likelihood terms;
they guide the sampler without replacing the photometry/RV fit.

For fully JAX-native sampling, use the BlackJAX backend:

```python
from selflensing.fitting import self_lensing_system_from_fit_blackjax

sls_fit, sampler, walker_state = self_lensing_system_from_fit_blackjax(
    Mcomp=1 * units.M_sun,
    Rcomp=10 * units.R_sun,
    peak_time=peak_time,
    lc_times=lcg.times,
    observed_lc=observed_flux,
    observed_lc_err=observed_flux_err,
    nwalkers=128,
    nsteps=5000,
    burn=500,
    random_seed=1,
)
```

This backend keeps walker positions and log-probabilities on the selected JAX
device during sampling. It currently requires `Rcomp`, does not support
`gp_model` or `detect_periods=True`, and runs a fixed number of random-walk
Metropolis-Hastings steps. Set `return_sampler=False` for large production runs
when you only need the best-fit system and final walker state.

To fit many independent systems at once, use the batched BlackJAX helper. Inputs
must have a leading fit dimension; pad ragged light curves/RV series to a common
length and provide boolean masks:

```python
from selflensing.fitting import self_lensing_systems_from_batched_fit_blackjax

systems, samplers, walker_state = self_lensing_systems_from_batched_fit_blackjax(
    Mcomp=Mcomp_array,
    Rcomp=Rcomp_array,
    peak_time=peak_time_array,
    lc_times=padded_lc_times,
    observed_lc=padded_flux,
    observed_lc_err=padded_flux_err,
    lc_mask=lc_mask,
    nwalkers=128,
    nsteps=3000,
    burn=500,
    return_sampler=False,
)
```

All fitters respect the device controls described in the GPU section above:
`prefer_gpu=True` by default, `device="gpu:0"` to select a specific device, and
`prefer_gpu=False` to force CPU execution.
