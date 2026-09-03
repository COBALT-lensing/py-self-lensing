import numpy
import random
import rustlens

from astropy.timeseries import TimeSeries
from astropy import units, constants
import unxt

import jax
from jax import numpy as jnp

from .lightcurves import (
    doppler_boosting_alpha,
    doppler_boosting_amplification,
    ellipsoidal_variations_amplification,
    light_curve_amplification,
    radial_velocity_m_per_s,
    self_lensing_amplification,
)
from .lensing import wm_callback
from .models import SelfLensingSystem, _quantity_value

# from src.selflensing.gp.LensingGP import _run_nucs_safe


JAX_LENSING_METHODS = {"IntegratedWittMaoJax", "JaxWittMao"}


def _time_values_days(times):
    if hasattr(times, "jd"):
        times = times.jd
    return jnp.asarray(times)


def _peak_time_value_days(peak_time):
    if hasattr(peak_time, "jd"):
        peak_time = peak_time.jd
    return jnp.asarray(peak_time)


def _light_curve_params_from_system(lensing_system):
    return jnp.asarray(
        [
            _quantity_value(lensing_system.a, units.au),
            _quantity_value(lensing_system.i, units.rad),
            _quantity_value(lensing_system.Macc, units.M_sun),
            _quantity_value(lensing_system.Mcomp, units.M_sun),
            _quantity_value(lensing_system.Rcomp, units.R_sun),
            _quantity_value(lensing_system.Teff, units.K),
            _quantity_value(lensing_system.e),
            _quantity_value(lensing_system.periapsis_phase, units.rad),
        ]
    )


class GeneratorBase(object):
    DEFAULT_BINS = 100
    units = units.dimensionless_unscaled

    def __init__(
        self,
        lensing_system,
        times=None,
        start_time=None,
        end_time=None,
        peak_time=None,
        bins=DEFAULT_BINS,
    ):
        self.__set_times(times, start_time, end_time, bins)
        if peak_time is None:
            peak_time = random.choice(list(self.times))
        self.peak_time = peak_time
        self.lensing_system = lensing_system

    def __set_times(self, times, start_time, end_time, bins):
        if times is not None:
            self.times = times
        else:
            if None in (start_time, end_time, bins):
                raise TypeError(
                    "Must initialise with either times or (start_time, end_time, bins)"
                )
            self.start_time = start_time
            self.end_time = end_time
            self.bins = bins
            self.times = numpy.linspace(self.start_time, self.end_time, num=self.bins)

    @property
    def time_offsets(self):
        times = self.times
        peak_time = self.peak_time
        if hasattr(times, "jd"):
            times = times.jd
        if hasattr(peak_time, "jd"):
            peak_time = peak_time.jd
        return unxt.Quantity(jnp.asarray(times) - jnp.asarray(peak_time), units.day)

    @property
    def peak_times(self):
        times = []
        i = 0
        while True:
            new_peak = self.peak_time + (i * self.lensing_system.porb)
            if new_peak > self.times[-1]:
                break
            times.append(new_peak)
            i += 1

        i = 1
        while True:
            new_peak = self.peak_time - (i * self.lensing_system.porb)
            if new_peak < self.times[0]:
                break
            times.append(new_peak)
            i += 1

        return sorted(times)

    @property
    def peak_time_offsets(self):
        return [(t - self.peak_time).jd for t in self.peak_times] * units.day


DEFAULT_BASELINE = 1.0


class FlatLightcurveGenerator(GeneratorBase):
    def __init__(
        self,
        *args,
        baseline=DEFAULT_BASELINE,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if isinstance(self.times, jax.Array):
            self.baseline = unxt.Quantity(
                jnp.full(len(self.times), baseline),
                self.units,
            )
        else:
            self.baseline = numpy.full(len(self.times), baseline) * self.units

    def __str__(self):
        return "Generated Flat Light Curve"

    def fit(self, timeseries=None, flux=None, err2=None, nu=None):
        if flux is None:
            flux = timeseries["flux"].value
        if err2 is None:
            err2 = timeseries["flux_err"].value ** 2
        if nu is None:
            nu = len(timeseries) - 1
        # Chi Squared but also penalise being wrong about maxima and minima
        # return (numpy.nanmedian(flux) - self.flux_array)**2 * numpy.nansum((flux - self.flux_array) ** 2 / err2)
        return numpy.nansum((flux - self.flux_array) ** 2 / err2) / nu

    @property
    def amplification(self):
        return 1.0

    @property
    def flux_array(self):
        return self.amplification * self.baseline

    @property
    def flux_err_array(self):
        return numpy.full(len(self.times), numpy.nan) * self.units

    @property
    def meta(self):
        return {
            "a": float(self.lensing_system.a.value.astype(float)),
            "i": float(self.lensing_system.i.value.astype(float)),
            "Mbh": float(self.lensing_system.Macc.value.astype(float)),
            "Rstar": float(self.lensing_system.Rcomp.value.astype(float)),
            "Mstar": float(self.lensing_system.Mcomp.value.astype(float)),
            "Teff": float(self.lensing_system.Teff.value.astype(float)),
            "e": float(self.lensing_system.e.value.astype(float)),
            "peri": float(self.lensing_system.periapsis_phase.value.astype(float)),
            "d": float(self.lensing_system.d.value.astype(float)),
            "P": float(self.lensing_system.porb.value.astype(float)),
            "b": float(self.lensing_system.b.value.astype(float)),
            "tau_eff": float(self.lensing_system.tau_eff.value.astype(float)),
            "tau_sl": float(self.lensing_system.tau_sl.value.astype(float)),
            "peak_t": self.peak_time.jd,
        }

    # @property
    # def lensing_system(self):
    #     return self._lensing_system

    # @lensing_system.setter
    # def lensing_system(self, v):
    #     self._lensing_system = v
    #     if hasattr(self, "amplification"):
    #         del self.amplification

    @property
    def timeseries(self):
        ts = TimeSeries(
            data={
                "time": self.times,
                "flux": self.flux_array,
                "flux_err": self.flux_err_array,
            }
        )
        ts.primary_key = ("time",)
        ts.meta.update(self.meta)
        return ts


class DopplerBoostingLightcurveGenerator(FlatLightcurveGenerator):
    """
    As defined in Nicholas M. Sorabella et al., ‘Modeling Long-Term Variability in
    Stellar-Compact Object Binary Systems for Mass Determinations’, The Astrophysical
    Journal 936, no. 1 (August 2022): 63, https://doi.org/10.3847/1538-4357/ac82b7.
    """

    obs_freq = (551 * units.nm).to(units.Hz, equivalencies=units.spectral())

    def __str__(self):
        return "Generated Doppler Boosting Light Curve"

    @property
    def alpha(self):
        """
        Average spectral index.

        As defined in Nicholas M. Sorabella et al., ‘Modeling Long-Term Variability in
        Stellar-Compact Object Binary Systems for Mass Determinations’, The Astrophysical
        Journal 936, no. 1 (August 2022): 63, https://doi.org/10.3847/1538-4357/ac82b7.
        """
        return doppler_boosting_alpha(
            _light_curve_params_from_system(self.lensing_system)
        )

    @property
    def amplification(self):
        return doppler_boosting_amplification(
            _time_values_days(self.times),
            _light_curve_params_from_system(self.lensing_system),
            _peak_time_value_days(self.peak_time),
        )

    @property
    def meta(self):
        m = super().meta
        m.update(
            {
                "amp_db": float(max(self.amplification).astype(float)),
            }
        )
        return m

    @property
    def flux_array(self):
        return self.baseline * (1 + self.amplification)


class EllipsoidalVariationsLightcurveGenerator(FlatLightcurveGenerator):
    """
    As defined in Nicholas M. Sorabella et al., ‘Modeling Long-Term Variability in
    Stellar-Compact Object Binary Systems for Mass Determinations’, The Astrophysical
    Journal 936, no. 1 (August 2022): 63, https://doi.org/10.3847/1538-4357/ac82b7.

    With the addition of a variable orbital separation to account for eccentricity.
    """

    # One day we could use the models from here:
    # https://www.aanda.org/articles/aa/full_html/2017/04/aa29705-16/aa29705-16.html
    # But these will do for now.
    g = 0.32  # Gravity darkening co-efficient derived from Lucy 1967 https://ui.adsabs.harvard.edu/abs/1967ZA.....65...89L/abstract
    gamma = 0.51  # Claret (2017) solar-dwarf TESS linear coefficient

    def __str__(self):
        return "Generated Ellipsoidal Variations Light Curve"

    @property
    def amplification(self):
        return ellipsoidal_variations_amplification(
            _time_values_days(self.times),
            _light_curve_params_from_system(self.lensing_system),
            _peak_time_value_days(self.peak_time),
        )

    @property
    def beta(self):
        return 0.15 * (1 + self.g) * (15 + self.gamma) / (3 - self.gamma)

    @property
    def meta(self):
        m = super().meta
        m.update(
            {
                "amp_el": float(max(self.amplification).astype(float)),
            }
        )
        return m

    @property
    def flux_array(self):
        return self.baseline * (1 + self.amplification)


class SelfLensingLightcurveGenerator(FlatLightcurveGenerator):
    # The JAX implementation applies the configured linear limb darkening.
    # The legacy RustLens callback uses its own fixed coefficient.
    DEFAULT_LENSING_METHOD = "IntegratedWittMaoJax"
    DEFAULT_GPU_LENSING_METHOD = "IntegratedWittMaoJax"

    lensing_methods = {
        "IntegratedHeyrovsky": "integrated_heyrovsky_amplification",
        "IntegratedWittMao": "integrated_witt_mao_amplification",
        "IntegratedWittMaoJax": "integrated_witt_mao_jax_amplification",
        "JaxWittMao": "integrated_witt_mao_jax_amplification",
        "WittMao": "witt_mao_amplification",
    }

    @classmethod
    def default_lensing_method_for_device(cls, device=None):
        platform = getattr(device, "platform", None)
        if platform is None and device is not None:
            platform = str(device).split(":", 1)[0].lower()
            if platform == "cuda":
                platform = "gpu"
        if platform == "gpu":
            return cls.DEFAULT_GPU_LENSING_METHOD
        return cls.DEFAULT_LENSING_METHOD

    def __init__(self, *args, lensing_method=DEFAULT_LENSING_METHOD, **kwargs):
        super().__init__(*args, **kwargs)
        self.lensing_method = lensing_method

    def __str__(self):
        return "Generated Self Lensing Light Curve"

    @property
    def meta(self):
        m = super().meta
        m.update(
            {
                "amp_sl": float(max(self.amplification).astype(float)),
            }
        )
        return m

    @property
    def amplification(self):
        if self.lensing_method in JAX_LENSING_METHODS:
            return self.integrated_witt_mao_jax_amplification

        rE = self.lensing_system.rE()
        super_amplification = super().amplification

        # def super_amplification_wrapper(amplification, offsets):
        def super_amplification_wrapper():
            offsets = self.time_offsets
            amplification = super_amplification
            r = jnp.array([amplification] * len(offsets))
            return r

        # super_amplification_callback = jax.pure_callback(
        #     super_amplification_wrapper,
        #     jax.ShapeDtypeStruct(self.time_offsets.shape, self.time_offsets.dtype),
        #     super_amplification,
        #     self.time_offsets,
        # )

        return jax.lax.cond(
            rE is None,
            super_amplification_wrapper,
            lambda: jax.lax.cond(
                jnp.isinf(rE.value),
                super_amplification_wrapper,
                lambda: jax.lax.cond(
                    jnp.isnan(rE.value),
                    super_amplification_wrapper,
                    lambda: jax.lax.cond(
                        jnp.isnan(self.lensing_system.b.value),
                        super_amplification_wrapper,
                        lambda: getattr(
                            self, self.lensing_methods[self.lensing_method]
                        ),
                    ),
                ),
            ),
        )

    @property
    def heyrovsky_amplification(self):
        return numpy.array(
            rustlens.integrated_heyrovsky_magnification(
                self.lensing_system.time_to_l(self.time_offsets),
                self.lensing_system.rE().to(units.m).value,
                self.lensing_system.Rcomp.to(units.m).value,
            )
        )

    @property
    def integrated_witt_mao_amplification(self):
        offsets = self.lensing_system.time_to_b(self.time_offsets)
        out_shape = jax.ShapeDtypeStruct(offsets.shape, offsets.dtype)

        # jax.debug.print("wm pre-callback")
        r = jax.pure_callback(
            wm_callback,
            out_shape,
            offsets,
            self.lensing_system.Rcomp.to(units.m).value,
            self.lensing_system.rE().to(units.m).value,
            vmap_method="sequential",
        )
        # jax.debug.print("wm final result: {r}", r=r)
        return r

        # return jnp.array(
        #     rustlens.integrated_witt_mao_magnification(
        #         self.lensing_system.time_to_b(self.time_offsets),
        #         self.lensing_system.rE().to(units.m).value.astype(float),
        #         self.lensing_system.Rcomp.to(units.m).value.astype(float),
        #     ),
        #     float,
        # )

    @property
    def integrated_witt_mao_jax_amplification(self):
        return self_lensing_amplification(
            _time_values_days(self.times),
            _light_curve_params_from_system(self.lensing_system),
            _peak_time_value_days(self.peak_time),
        )

    @property
    def witt_mao_amplification(self):
        return numpy.array(
            rustlens.witt_mao_magnification(
                self.lensing_system.time_to_b(self.time_offsets),
                self.lensing_system.rE().to(units.m).value,
                self.lensing_system.Rcomp.to(units.m).value,
            )
        )


class LightcurveGenerator(FlatLightcurveGenerator):
    """
    As defined in Nicholas M. Sorabella et al., ‘Modeling Long-Term Variability in
    Stellar-Compact Object Binary Systems for Mass Determinations’, The Astrophysical
    Journal 936, no. 1 (August 2022): 63, https://doi.org/10.3847/1538-4357/ac82b7.
    """

    @classmethod
    def factory(cls, *args, total=None, param_samples=None, **kwargs):
        for lensing_system in SelfLensingSystem.factory(
            total=total, param_samples=param_samples
        ):
            yield cls(*args, lensing_system=lensing_system, **kwargs)

    def __init__(
        self,
        *args,
        sl=True,
        ev=True,
        db=True,
        lensing_method=SelfLensingLightcurveGenerator.DEFAULT_LENSING_METHOD,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.sl_enabled = sl
        self.ev_enabled = ev
        self.db_enabled = db
        self.lensing_method = lensing_method
        # Baseline is deliberately not passed on to these because we don't want to apply
        # it more than once.
        if sl:
            self.sl = SelfLensingLightcurveGenerator(
                self.lensing_system,
                peak_time=self.peak_time,
                times=self.times,
                lensing_method=lensing_method,
            )
        else:
            self.sl = FlatLightcurveGenerator(
                self.lensing_system,
                peak_time=self.peak_time,
                times=self.times,
            )

        if ev:
            self.ev = EllipsoidalVariationsLightcurveGenerator(
                self.lensing_system,
                peak_time=self.peak_time,
                times=self.times,
            )
        else:
            self.ev = FlatLightcurveGenerator(
                self.lensing_system,
                peak_time=self.peak_time,
                times=self.times,
            )

        if db:
            self.db = DopplerBoostingLightcurveGenerator(
                self.lensing_system,
                peak_time=self.peak_time,
                times=self.times,
            )
        else:
            self.db = FlatLightcurveGenerator(
                self.lensing_system,
                peak_time=self.peak_time,
                times=self.times,
            )

    def __str__(self):
        return "Generated Doppler Boosting + Ellipsoidal Variations + Self Lensing Light Curve"

    @property
    def lensing_system(self):
        return self._lensing_system

    @lensing_system.setter
    def lensing_system(self, ls):
        self._lensing_system = ls
        if hasattr(self, "sl"):
            self.sl.lensing_system = ls
        if hasattr(self, "ev"):
            self.ev.lensing_system = ls
        if hasattr(self, "db"):
            self.db.lensing_system = ls

    @property
    def meta(self):
        m = super().meta
        m.update(self.sl.meta)
        m.update(self.ev.meta)
        m.update(self.db.meta)
        return m

    @property
    def peak_time(self):
        return self._peak_time

    @peak_time.setter
    def peak_time(self, pk):
        self._peak_time = pk
        if hasattr(self, "sl"):
            self.sl.peak_time = pk
        if hasattr(self, "ev"):
            self.ev.peak_time = pk
        if hasattr(self, "db"):
            self.db.peak_time = pk

    @property
    def amplification(self):
        if self.lensing_method in JAX_LENSING_METHODS:
            return light_curve_amplification(
                _time_values_days(self.times),
                _light_curve_params_from_system(self.lensing_system),
                _peak_time_value_days(self.peak_time),
                sl=self.sl_enabled,
                ev=self.ev_enabled,
                db=self.db_enabled,
            )

        @jax.jit
        def f():
            return self.sl.amplification * (
                1 + (self.ev.amplification + self.db.amplification)
            )

        return f()


class GPLightcurveGenerator(LightcurveGenerator):
    @classmethod
    def factory(cls, gp, *args, total=None, param_samples=None, **kwargs):
        for lensing_system in SelfLensingSystem.factory(
            total=total, param_samples=param_samples
        ):
            if type(gp) is type:
                gp = gp(sls=lensing_system)
            lcg = cls(lensing_system, gp, *args, **kwargs)
            gp.lensing_system = lensing_system
            gp.lc_times = lcg.time_offsets
            yield lcg

    def __init__(self, lensing_system, gp, *args, kernel_type=None, **kwargs):
        super().__init__(lensing_system, *args, **kwargs)

        self.kernel_type = kernel_type
        self.gp = gp
        self.baseline = 10**self.gp.mu_test
        self.gp.mean_vals = super().flux_array  # * 10**self.gp.mu_test
        self.gp.kernel_type = self.kernel_type

    def __str__(self):
        pass

    @property
    def meta(self):
        m = super().meta
        m["mean_cv"] = float(
            numpy.mean(numpy.sqrt(self.gp.model.variance) / self.gp.model.loc).astype(
                float
            )
        )
        amp_sl = m.get("amp_sl")
        if amp_sl is not None:
            m["vis"] = amp_sl - 1.0 - m["mean_cv"]
        return m

    @property
    def flux_array(self):
        return self.gp.generate()

    @property
    def gp(self):
        return self._gp

    @gp.setter
    def gp(self, v):
        if self.kernel_type is None:
            self.kernel_type = v.default_kernel_type
        if v is None:
            self._gp = None
        else:
            if type(v) is type:

                self._gp = v(
                    sls=self.lensing_system,
                    lc_times=self.time_offsets,
                    kernel_type=self.kernel_type,
                )
            else:
                self._gp = v
                self._gp.kernel_type = self.kernel_type


class RVGenerator(GeneratorBase):
    units = units.m / units.s

    def phases(self):
        return self.lensing_system.time_to_phase(self.time_offsets)

    def _source_polar_velocity_components(self, phases):
        phase_values = _quantity_value(phases, units.rad)
        eccentricity = _quantity_value(self.lensing_system.e)
        periapsis_phase = _quantity_value(
            self.lensing_system.periapsis_phase, units.rad
        )
        true_anomaly = phase_values - periapsis_phase

        semi_major_axis = _quantity_value(self.lensing_system.a, units.m)
        lens_mass = _quantity_value(self.lensing_system.Macc, units.kg)
        total_mass = _quantity_value(self.lensing_system.Mtot, units.kg)
        mu = constants.G.value * total_mass
        angular_momentum = jnp.sqrt(
            mu * semi_major_axis * (1 - eccentricity**2)
        )
        source_scale = lens_mass / total_mass

        velocity_scale = source_scale * (mu / angular_momentum)
        radial_velocity = velocity_scale * eccentricity * jnp.sin(true_anomaly)
        tangential_velocity = velocity_scale * (
            1 + eccentricity * jnp.cos(true_anomaly)
        )
        return radial_velocity, tangential_velocity, velocity_scale

    @property
    def rv(self):
        rv = radial_velocity_m_per_s(
            _time_values_days(self.times),
            _light_curve_params_from_system(self.lensing_system),
            _peak_time_value_days(self.peak_time),
        )
        return unxt.Quantity(rv, self.units)

    @property
    def rv_deriv(self):
        """
        Derivative of radial velocity.
        """
        phases = self.phases()
        phase_values = _quantity_value(phases, units.rad)
        _, _, velocity_scale = self._source_polar_velocity_components(phases)
        return unxt.Quantity(
            velocity_scale
            * jnp.cos(phase_values)
            * _quantity_value(self.lensing_system.sin_i),
            self.units,
        )
