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

For development installs with Poetry, install one of the optional GPU groups:

```shell
poetry install --with cuda
```

`cuda` is an alias for the current recommended NVIDIA CUDA 13 wheels. Other
available groups are:

```shell
poetry install --with cuda13
poetry install --with cuda12
poetry install --with cuda13-local
poetry install --with cuda12-local
poetry install --with rocm
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
python -c "import jax; print(jax.devices())"
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

## Fitting light curves and RV

Coming soon!
