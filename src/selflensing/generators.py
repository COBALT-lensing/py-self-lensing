import numpy
import random
import rustlens

from astropy.timeseries import TimeSeries
from astropy import units, constants

import jax
from jax import numpy as jnp

from .lensing import wm_callback
from .models import SelfLensingSystem

# from src.selflensing.gp.LensingGP import _run_nucs_safe


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
        return (self.times - self.peak_time).jd.astype(numpy.float32) * units.day

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
        x = (
            constants.h.value
            * self.obs_freq.value
            / (constants.k_B.value * self.lensing_system.Teff.value)
        )
        e_x = jnp.exp(x)
        return (e_x * (3 - x) - 3) / (e_x - 1)

    @property
    def amplification(self):
        rvg = RVGenerator(
            self.lensing_system, times=self.times, peak_time=self.peak_time
        )
        rv = rvg.rv.to(units.m / units.s)
        return (
            (3 - self.alpha)
            * (rv.value / constants.c.value)
            * self.lensing_system.sin_i.value
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
    g = 0.4  # Gravity darkening co-efficient
    gamma = 0.6  # Limb darkening co-efficient

    def __str__(self):
        return "Generated Ellipsoidal Variations Light Curve"

    @property
    def amplification(self):
        phases = self.lensing_system.time_to_phase(self.time_offsets * 2)
        return (
            (
                (
                    self.beta
                    * (self.lensing_system.Macc / self.lensing_system.Mcomp).decompose(
                        []
                    )
                    * (
                        self.lensing_system.Rcomp / self.lensing_system.r_sep(phases)
                    ).decompose([])
                    ** 3
                    * self.lensing_system.sin_i**2
                )
                * (jnp.cos(phases.value))
                * -1
            )
            .decompose([])
            .value
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
    DEFAULT_LENSING_METHOD = "IntegratedWittMao"

    lensing_methods = {
        "IntegratedHeyrovsky": "integrated_heyrovsky_amplification",
        "IntegratedWittMao": "integrated_witt_mao_amplification",
        "WittMao": "witt_mao_amplification",
    }

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
        m.update(
            {
                "mean_cv": float(
                    numpy.mean(
                        numpy.sqrt(self.gp.model.variance) / self.gp.model.loc
                    ).astype(float)
                ),
            }
        )
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
        time_offsets = (self.times - self.peak_time).to(units.day)
        return self.lensing_system.time_to_phase(time_offsets)

    @property
    def rv(self):
        phases = self.phases()
        return (
            self.lensing_system.v_comp(phases)
            * jnp.sin(phases.value)
            * self.lensing_system.sin_i
        ).to(self.units)

    @property
    def rv_deriv(self):
        """
        Derivative of radial velocity.
        """
        phases = self.phases()
        return (
            self.lensing_system.v_comp(phases)
            * numpy.cos(phases)
            * self.lensing_system.sin_i
        )
