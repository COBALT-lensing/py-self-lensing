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

Coming soon!
