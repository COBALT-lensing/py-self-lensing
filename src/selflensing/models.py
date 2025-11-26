import numpy

from astropy import units, constants


class SelfLensingSystem(object):
    param_units = {
        "a": units.au,
        "i": units.rad,
        "Macc": units.M_sun,
        "Rcomp": units.R_sun,
        "Mcomp": units.M_sun,
        "Teff": units.K,
        "e": units.dimensionless_unscaled,
        "periapsis_phase": units.rad,
        "d": units.kpc,
    }
    DEFAULT_TEFF = 5000 * units.K
    DEFAULT_PERIAPSIS = 0 * units.rad
    DEFAULT_E = 0 * units.dimensionless_unscaled
    DEFAULT_I = 0 * units.rad
    DEFAULT_D = 1 * units.kpc

    @units.quantity_input
    def __init__(
        self,
        Macc: units.kg,
        Mcomp: units.kg,
        Rcomp: units.m,
        a: units.m = None,
        # b: units.dimensionless_unscaled = DEFAULT_B,  # type: ignore
        Teff: units.K = DEFAULT_TEFF,
        e: units.dimensionless_unscaled = DEFAULT_E,  # type: ignore
        periapsis_phase: units.rad = DEFAULT_PERIAPSIS,
        i: units.rad = DEFAULT_I,
        Porb: units.second = None,
        d: units.kpc = DEFAULT_D,
    ):
        if a is None and Porb is None:
            raise ValueError("You must provide one of a or Porb")
        if a is None:
            a = (constants.G * (Macc + Mcomp) * (Porb / (2 * numpy.pi)) ** 2) ** (
                1.0 / 3.0
            )
        self.a = a.to(self.param_units["a"])

        self.Macc = Macc.to(self.param_units["Macc"])
        self.Rcomp = Rcomp.to(self.param_units["Rcomp"])
        self.Mcomp = Mcomp.to(self.param_units["Mcomp"])
        self.Teff = Teff.to(self.param_units["Teff"])
        self.e = e
        self.periapsis_phase = periapsis_phase.to(self.param_units["periapsis_phase"])
        self.i = i.to(self.param_units["i"])
        self.d = d.to(self.param_units["d"])

    @property
    def b(self):
        return self._b

    @b.setter
    def b(self, b):
        raise NotImplementedError

    @property
    def i(self):
        return self._i

    @i.setter
    def i(self, i):
        self._i = i
        self.sin_i = numpy.sin(i)
        # Calculate as per Wiktorowicz+2021
        self._b = (self.a * numpy.cos(i) / self.rE()).decompose()

    @property
    def b_max(self):
        """
        Maximum b where SL is possible as defined in Wiktorowicz, Grzegorz, Matthew Middleton,
        Norman Khan, Adam Ingram, Poshak Gandhi, and Hugh Dickinson. ‘Predicting the
        Self-Lensing Population in Optical Surveys’. Monthly Notices of the Royal
        Astronomical Society 507, no. 1 (11 October 2021): 374–84.
        https://doi.org/10.1093/mnras/stab2135.
        """
        return (1 + self.Rcomp / self.rE()).decompose()

    @property
    def b_source(self):
        """
        The impact parameter in units of Rcomp (source radii). b_source=1 when the lens is at
        the edge of the stellar disk.
        """
        return self.b * (self.rE() / self.Rcomp)

    @property
    def Mtot(self):
        return self.Macc + self.Mcomp

    def phase_to_l(self, phase):
        """
        Given a phase angle, compute the distance of the lens from the centre of the disk.

        When the lens is behind the star, assumes l is at its maximum.
        """
        phase_limit = 0.5 * numpy.pi * units.rad
        phase_upper_limit = 2 * numpy.pi * units.rad - phase_limit
        if isinstance(phase, numpy.ndarray):
            phase[(phase > phase_limit) & (phase < phase_upper_limit)] = phase_limit
        elif phase > phase_limit and phase < phase_upper_limit:
            phase = 0.5 * numpy.pi * units.rad
        return numpy.hypot(
            numpy.abs(
                self.r_sep(phase=phase).to(units.m)
                * numpy.sin(phase)
                / self.Rcomp.to(units.m)
            ).value,
            self.b_source.value,
        )

    def phase_to_b(self, phase):
        return self.phase_to_l(phase) / (self.rE() / self.Rcomp)

    @property
    @units.quantity_input
    def porb(self) -> units.day:
        return (
            2
            * numpy.pi
            * numpy.sqrt(self.a**3 / (constants.G * self.Mtot)).to(units.day)
        )

    def r(self, m, phase):
        """
        The distance from the centre of mass of the object with mass 'm' at 'phase'
        """
        r_max = self.a * m / self.Mtot
        return (r_max * (1 - self.e**2)) / (
            1 + self.e * numpy.cos(phase - self.periapsis_phase)
        )

    def rE(self, phase=0):
        return numpy.sqrt(
            (4 * constants.G * self.Macc * self.r_sep(phase) * self.sin_i)
            / constants.c**2
        ).to(units.R_sun)

    @property
    def r_roche(self):
        """
        The Roche lobe radius at periapsis.
        https://ui.adsabs.harvard.edu/abs/1983ApJ...268..368E/abstract
        """
        q = self.Mcomp / self.Macc
        q23 = q ** (2.0 / 3.0)
        A = self.r_sep(self.periapsis_phase)
        return A * ((0.49 * q23) / (0.6 * q23 + numpy.log(1 + q ** (1.0 / 3.0))))

    def r_sep(self, phase=0):
        """
        The separation in space between acc and comp at phase
        """
        return self.r(self.Macc, phase) + self.r(self.Mcomp, phase)

    @property
    def tau_eff(self):
        """
        Effective crossing time as defined in Wiktorowicz, Grzegorz, Matthew Middleton,
        Norman Khan, Adam Ingram, Poshak Gandhi, and Hugh Dickinson. ‘Predicting the
        Self-Lensing Population in Optical Surveys’. Monthly Notices of the Royal
        Astronomical Society 507, no. 1 (11 October 2021): 374–84.
        https://doi.org/10.1093/mnras/stab2135.
        """
        return (
            ((self.porb * (self.rE() + self.Rcomp)) / (numpy.pi * self.a * self.sin_i))
            * numpy.sqrt(1 - (self.b / self.b_max) ** 2)
        ).to(units.day)

    def time_to_l(self, time):
        return self.phase_to_l(self.time_to_phase(time))

    def time_to_b(self, time):
        return self.phase_to_b(self.time_to_phase(time))

    @units.quantity_input
    def time_to_phase(self, time: units.day) -> units.rad:
        """
        Convert time offsets to angular orbital phases.
        """
        time = time % self.porb
        return (((time / self.porb).decompose() * 2 * numpy.pi)) * units.rad

    def v(self, m, phase):
        """
        Orbital velocity.
        """
        v = numpy.sqrt(
            (constants.G * self.Mtot)
            * (2 / self.r(m, phase) - 1 / (self.a * m / self.Mtot))
        )
        return v.to(units.m / units.s)

    def v_tot(self, phase):
        return self.v(self.Mtot, phase)

    def v_acc(self, phase):
        return self.v(self.Macc, phase)

    @property
    def v_avg(self):
        return (2 * numpy.pi * self.a / self.porb).to(units.m / units.s)

    def v_comp(self, phase):
        return self.v(self.Mcomp, phase)
