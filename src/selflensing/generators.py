try:
    from jax import numpy
except ImportError:
    import numpy
import random
import rustlens

from astropy.timeseries import TimeSeries
from astropy import units, constants

from scipy import interpolate

from selflensing import lensing


class GeneratorBase(object):
    DEFAULT_BINS = 100
    units = units.dimensionless_unscaled

    def __init__(
        self,
        lensing_system=None,
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


class FlatLightcurveGenerator(GeneratorBase):
    def __init__(
        self,
        *args,
        baseline=1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.baseline = baseline

    def __str__(self):
        return "Generated Flat Light Curve"

    @property
    def time_offsets(self):
        return (self.times - self.peak_time).jd * units.day

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
    def flux_array(self):
        return numpy.full(len(self.times), self.baseline) * self.units

    @property
    def flux_err_array(self):
        return numpy.full(len(self.times), numpy.nan) * self.units

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
            constants.h * self.obs_freq / (constants.k_B * self.lensing_system.Teff)
        ).value
        e_x = numpy.exp(x)
        return (e_x * (3 - x) - 3) / (e_x - 1)

    @property
    def flux_array(self):
        rvg = RVGenerator(
            self.lensing_system, times=self.times, peak_time=self.peak_time
        )
        rv = rvg.rv
        return (
            super().flux_array
            * ((3 - self.alpha) * (rv / constants.c) * self.lensing_system.sin_i).value
        )


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
    def beta(self):
        return 0.15 * (1 + self.g) * (15 + self.gamma) / (3 - self.gamma)

    @property
    def flux_array(self):
        phases = self.lensing_system.time_to_phase(self.time_offsets * 2)
        return (
            super().flux_array
            * (
                (
                    self.beta
                    * (self.lensing_system.Macc / self.lensing_system.Mcomp).decompose()
                    * (
                        self.lensing_system.Rcomp / self.lensing_system.r_sep(phases)
                    ).decompose()
                    ** 3
                    * self.lensing_system.sin_i**2
                )
                * (numpy.cos(phases))
                * -1
            )
            .decompose()
            .value
        )


class SelfLensingLightcurveGenerator(FlatLightcurveGenerator):
    DEFAULT_LENSING_METHOD = "NonUniformInterpolatedWittMao"

    lensing_methods = {
        "NonUniformWittMao": "non_uniform_witt_mao_flux_array",
        "NonUniformInterpolatedWittMao": "non_uniform_interpolated_witt_mao_flux_array",
        "FiniteWittMao": "finite_witt_mao_flux_array",
        "WittMao": "witt_mao_flux_array",
        "Heyrovsky": "heyrovsky_flux_array",
        "RustWittMao": "rust_witt_mao_flux_array",
        "RustIntegratedWittMao": "rust_integrated_witt_mao_flux_array",
    }

    def __init__(self, *args, lensing_method=DEFAULT_LENSING_METHOD, **kwargs):
        super().__init__(*args, **kwargs)
        self.lensing_method = lensing_method

    def __str__(self):
        return "Generated Self Lensing Light Curve"

    @property
    def flux_array(self):
        return getattr(self, self.lensing_methods[self.lensing_method])

    @property
    def heyrovsky_flux_array(self):
        return (
            numpy.array(
                [
                    lensing.integrated_luminosity(
                        lambda r: lensing.heyrovsky_magnification(
                            l,
                            r,
                            self.lensing_system.rE().to(units.m).value,
                            self.lensing_system.Rcomp.to(units.m).value,
                        ),
                        lensing.ld_linear,
                    )
                    for l in self.lensing_system.time_to_l(self.time_offsets)
                ]
            )
            * super().flux_array
        )

    @property
    def non_uniform_interpolated_witt_mao_flux_array(self):
        xlp = numpy.linspace(-10, 10, 200)
        projectedOffset = numpy.hypot(xlp, self.lensing_system.b)

        # phases = ...need phases of each point in xlp to account for eccentricity during
        # the lensing event. I don't know if it's even worth figuring out how to do that
        # Fixing phase = 0 is probably enough to get roughly the right rE
        rE = self.lensing_system.rE()

        lensing_simulator = lensing.SelfLensingSimulator(
            self.lensing_system.Rcomp.to(units.R_sun).value,
            rE.to(units.R_sun).value,
            projectedOffset.value.tolist(),
        )

        mags = numpy.array(lensing_simulator.magnifications())
        interpolant = interpolate.interp1d(xlp, mags, bounds_error=False, fill_value=1)

        repeating_offsets = self.time_offsets % self.lensing_system.porb
        repeating_offsets = numpy.where(
            repeating_offsets < self.lensing_system.porb / 2,
            repeating_offsets,
            repeating_offsets - self.lensing_system.porb,
        )
        binnedLensPlaneX = (repeating_offsets * self.lensing_system.v_avg) / rE

        flux_offset = interpolant(binnedLensPlaneX.decompose().value)
        return super().flux_array * flux_offset

    @property
    def non_uniform_witt_mao_flux_array(self):
        rE = self.lensing_system.rE()

        repeating_offsets = self.time_offsets % self.lensing_system.porb
        repeating_offsets = numpy.where(
            repeating_offsets < self.lensing_system.porb / 2,
            repeating_offsets,
            repeating_offsets - self.lensing_system.porb,
        )
        binnedLensPlaneX = (repeating_offsets * self.lensing_system.v_avg) / rE
        lensing_simulator = lensing.SelfLensingSimulator(
            self.lensing_system.Rcomp.to(units.R_sun).value,
            rE.to(units.R_sun).value,
            binnedLensPlaneX.decompose().value,
        )

        flux_offset = lensing_simulator.magnifications()
        return super().flux_array * flux_offset

    @property
    def rust_integrated_witt_mao_flux_array(self):
        A = numpy.array(
            [
                rustlens.integrated_witt_mao_magnification(
                    l,
                    self.lensing_system.rE().to(units.m).value,
                    self.lensing_system.Rcomp.to(units.m).value,
                )
                for l in self.lensing_system.time_to_l(self.time_offsets)
            ]
        )
        return lensing.uniform_luminosity(A, super().flux_array)

    @property
    def rust_witt_mao_flux_array(self):
        A = numpy.array(
            [
                rustlens.witt_mao_magnification(
                    l,
                    self.lensing_system.rE().to(units.m).value,
                    self.lensing_system.Rcomp.to(units.m).value,
                )
                for l in self.lensing_system.time_to_l(self.time_offsets)
            ]
        )
        return lensing.uniform_luminosity(A, super().flux_array)

    @property
    def witt_mao_flux_array(self):
        A = numpy.array(
            [
                lensing.witt_mao_magnification(
                    l,
                    self.lensing_system.rE().to(units.m).value,
                    self.lensing_system.Rcomp.to(units.m).value,
                )
                for l in self.lensing_system.time_to_l(self.time_offsets)
            ]
        )
        return lensing.uniform_luminosity(A, super().flux_array)

    @property
    def finite_witt_mao_flux_array(self):
        B_int = lensing.integrated_brightness_profile(lensing.ld_linear)
        A = lensing.finite_witt_mao_magnification(
            self.lensing_system.time_to_l(self.time_offsets),
            self.lensing_system.rE().to(units.R_sun).value,
            self.lensing_system.Rcomp.to(units.R_sun).value,
            lensing.ld_linear,
            B_int,
        )

        return A * super().flux_array

    @property
    def witt_mao_flux_array(self):
        A = numpy.array(
            [
                lensing.witt_mao_magnification(
                    l,
                    self.lensing_system.rE().to(units.m).value,
                    self.lensing_system.Rcomp.to(units.m).value,
                )
                for l in self.lensing_system.time_to_l(self.time_offsets)
            ]
        )
        return lensing.uniform_luminosity(A, super().flux_array)


class LightcurveGenerator(FlatLightcurveGenerator):
    """
    As defined in Nicholas M. Sorabella et al., ‘Modeling Long-Term Variability in
    Stellar-Compact Object Binary Systems for Mass Determinations’, The Astrophysical
    Journal 936, no. 1 (August 2022): 63, https://doi.org/10.3847/1538-4357/ac82b7.
    """

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
    def flux_array(self):
        return (
            self.sl.flux_array * (1 + (self.ev.flux_array + self.db.flux_array))
        ) * super().flux_array


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
            * numpy.sin(phases)
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
