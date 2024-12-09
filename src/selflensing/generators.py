import mpmath
import numpy
import random

from astropy.time import TimeDelta, Time
from astropy.timeseries import TimeSeries
from astropy import units, constants

from scipy import interpolate

# from PyNonUniformCircularSource import PyNonUniformCircularSource


class FlatLightcurveGenerator(object):
    def __init__(
        self,
        lensing_system=None,
        times=None,
        time_start=None,
        time_end=None,
        peak_time=None,
        bins=100,
        data_path=None,
        timeseries=None,
        baseline=1.0,
    ):
        if data_path is not None or timeseries is not None:
            return super().__init__(data_path=data_path, timeseries=timeseries)

        self.__set_times(times, time_start, time_end, bins)
        if peak_time is None:
            peak_time = random.choice(list(self.times))
        self.peak_time = peak_time
        self.lensing_system = lensing_system
        self.baseline = baseline

    def __set_times(self, times, time_start, time_end, bins):
        if times is not None:
            self.times = times
        else:
            if None in (time_start, time_end, bins):
                raise TypeError(
                    "Must initialise with either times or (time_start, time_end, bins)"
                )
            self.time_start = time_start
            self.time_end = time_end
            self.bins = bins
            self.times = numpy.linspace(self.time_start, self.time_end, num=self.bins)

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
        return numpy.full(len(self.times), self.baseline)

    @property
    def flux_err_array(self):
        return numpy.full(len(self.times), numpy.nan)

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
        phases = self.lensing_system.time_to_phase(self.time_offsets)
        rv = self.lensing_system.rv(phases)
        return (
            super().flux_array
            * ((3 - self.alpha) * (rv / constants.c) * self.lensing_system.sin_i).value
        )


class EllipsoidalVariationsLightcurveGenerator(FlatLightcurveGenerator):
    """
    As defined in Nicholas M. Sorabella et al., ‘Modeling Long-Term Variability in
    Stellar-Compact Object Binary Systems for Mass Determinations’, The Astrophysical
    Journal 936, no. 1 (August 2022): 63, https://doi.org/10.3847/1538-4357/ac82b7.
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
                    * (self.lensing_system.Rcomp / self.lensing_system.a).decompose()
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
    xlp = numpy.linspace(-10, 10, 200)
    LD_COEFF = 0.6

    def __str__(self):
        return "Generated Self Lensing Light Curve"

    @property
    def flux_array(self):
        return super().flux_array * self.flux_offset

    @property
    def flux_offset(self):
        porb = self.lensing_system.porb
        rE = self.lensing_system.rE

        projectedOffset = numpy.hypot(self.xlp, self.lensing_system.b)
        mags = numpy.array(list(self.magnifications(projectedOffset.value.tolist())))
        interpolant = interpolate.interp1d(
            self.xlp, mags, bounds_error=False, fill_value=1
        )
        repeating_offsets = self.time_offsets % porb
        repeating_offsets = numpy.where(
            repeating_offsets < porb / 2,
            repeating_offsets,
            repeating_offsets - porb,
        )
        binnedLensPlaneX = (repeating_offsets * self.lensing_system.v_avg) / rE
        return interpolant(binnedLensPlaneX.decompose().value)

    # The following implementation is based on the C++ version at:
    # https://github.com/ou-astrophysics/self-lensing-simulator

    def magnificationAtRadius(self):
        radius_unitless2 = self.radius_unitless**2
        kernel = (2 / self.lradius_unitless) + (
            (1 + radius_unitless2) / radius_unitless2
        ) * (
            (numpy.pi / 2)
            + numpy.arcsin((radius_unitless2 - 1) / (radius_unitless2 + 1))
        )

        total = kernel / numpy.pi

        return total

    def uniform_magnification(self, sourcePlaneCoordinate):
        if abs(sourcePlaneCoordinate - self.radius_unitless) < 1e-5:
            return self.magnificationAtRadius(self.radius_unitless)

        kernel1 = (sourcePlaneCoordinate - self.radius_unitless) ** 2
        kernel2 = numpy.sqrt(4.0 + kernel1)

        ellipticN = (
            4.0
            * self.radius_unitless
            * sourcePlaneCoordinate
            / ((sourcePlaneCoordinate + self.radius_unitless) ** 2.0)
        )

        ellipticK = numpy.sqrt(4.0 * ellipticN) / kernel2

        firstTerm = ((sourcePlaneCoordinate + self.radius_unitless) * kernel2) / (
            2.0 * (self.radius_unitless**2)
        )
        secondTerm = (
            (sourcePlaneCoordinate - self.radius_unitless)
            * (4.0 + (0.5 * (sourcePlaneCoordinate**2) - self.radius_unitless))
            / (kernel2 * self.radius_unitless)
        )
        thirdTerm = (
            2.0
            * kernel1
            * (1.0 + self.radius_unitless)
            / (
                (self.radius_unitless**2)
                * (sourcePlaneCoordinate + self.radius_unitless)
                * kernel2
            )
        )

        kernel3 = (
            mpmath.ellipe(ellipticK) * firstTerm
            - mpmath.ellipk(ellipticK) * secondTerm
            + mpmath.ellippi(ellipticK, ellipticN) * thirdTerm
        )

        positiveSolutionReal = (kernel3 + numpy.pi) / (2.0 * numpy.pi)
        negativeSolutionReal = (kernel3 - numpy.pi) / (2.0 * numpy.pi)
        return positiveSolutionReal + negativeSolutionReal

    def radial_profile(self, r):
        radiusRatio = r / self.radius_unitless
        mu = numpy.sqrt(
            1.0 - (radiusRatio**2)
        )  # mu=cos(theta), theta=arcsin(radiusRatio)
        return 1.0 - self.LD_COEFF * (1.0 - mu)

    @property
    def radius_unitless(self):
        return (self.lensing_system.Rcomp / self.lensing_system.rE).value

    def profile_integral(self):
        return (
            2
            * numpy.pi
            * mpmath.quad(
                lambda r: r * self.radial_profile(r),
                [0, self.radius_unitless],
                method="gauss-legendre",
            )
        )

    def magnification_derivative(
        self, radialCoordinate, sourcePlaneCoordinate, umag=None
    ):
        if radialCoordinate < 0:
            return 0

        def f(r):
            if r < 0:
                return 0
            if umag is not None:
                return umag
            return self.uniform_magnification(sourcePlaneCoordinate)

        return mpmath.diff(f, radialCoordinate)

    def magnification_profile_integral(self, sourcePlaneCoordinate):
        umag = self.uniform_magnification(sourcePlaneCoordinate)
        return (
            2
            * numpy.pi
            * mpmath.quad(
                lambda r: (
                    r * self.radial_profile(r) * umag
                    + 0.5
                    * r
                    * self.magnification_derivative(r, sourcePlaneCoordinate, umag=umag)
                ),
                [0, self.radius_unitless],
                method="gauss-legendre",
            )
        )

    def magnification(self, sourcePlaneCoordinate):
        return (
            self.magnification_profile_integral(sourcePlaneCoordinate)
            / self.profile_integral()
        )

    def magnifications(self, sourcePlaneCoordinates):
        for sourcePlaneCoordinate in sourcePlaneCoordinates:
            yield self.magnification(sourcePlaneCoordinate)


class CombinedLightcurveGenerator(FlatLightcurveGenerator):
    """
    As defined in Nicholas M. Sorabella et al., ‘Modeling Long-Term Variability in
    Stellar-Compact Object Binary Systems for Mass Determinations’, The Astrophysical
    Journal 936, no. 1 (August 2022): 63, https://doi.org/10.3847/1538-4357/ac82b7.
    """

    def __init__(self, *args, sl=True, ev=True, db=True, **kwargs):
        super().__init__(*args, **kwargs)
        # Baseline is deliberately not passed on to these because we don't want to apply
        # it more than once.
        if sl:
            self.sl = SelfLensingLightcurveGenerator(
                self.lensing_system,
                peak_time=self.peak_time,
                times=self.times,
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
