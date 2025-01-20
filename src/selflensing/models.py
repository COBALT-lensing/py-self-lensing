import numpy

from functools import cached_property

from astropy import units, constants


CONST_C2 = (constants.c**2).decompose().value
CONST_4G = (4 * constants.G).decompose().value
CONST_G = constants.G.decompose().value
CONST_2PI = 2 * numpy.pi
CONST_HALF_PI = 0.5 * numpy.pi
CONST_R_SUN = (1 * units.R_sun).decompose().value


class SelfLensingSystem(object):
    param_units = {
        "a": units.au,
        "b": units.dimensionless_unscaled,
        "Macc": units.M_sun,
        "Rcomp": units.R_sun,
        "Mcomp": units.M_sun,
        "Teff": units.K,
        "e": units.dimensionless_unscaled,
        "periapsis_phase": units.rad,
    }
    DEFAULT_TEFF = 5000 * units.K
    DEFAULT_PERIAPSIS = 0 * units.rad
    DEFAULT_E = 0 * units.dimensionless_unscaled

    @units.quantity_input
    def __init__(
        self,
        a: units.m,
        b: units.dimensionless_unscaled,
        Macc: units.kg,
        Mcomp: units.kg,
        Rcomp: units.m,
        Teff: units.K = DEFAULT_TEFF,
        e: units.dimensionless_unscaled = DEFAULT_E,
        periapsis_phase: units.rad = DEFAULT_PERIAPSIS,
    ):
        self.b = b.decompose().value
        self.a = a.decompose().value
        self.Macc = Macc.decompose().value
        self.Rcomp = Rcomp.decompose().value
        self.Mcomp = Mcomp.decompose().value
        self.Teff = Teff.decompose().value
        self.e = e.decompose().value
        self.periapsis_phase = periapsis_phase.decompose().value

    @cached_property
    def a_acc(self):
        return self.a * self.Mcomp / self.Mtot

    @cached_property
    def a_comp(self):
        return self.a * self.Macc / self.Mtot

    @cached_property
    def b_max(self):
        """
        Maximum b where SL is possible as defined in Wiktorowicz, Grzegorz, Matthew Middleton,
        Norman Khan, Adam Ingram, Poshak Gandhi, and Hugh Dickinson. ‘Predicting the
        Self-Lensing Population in Optical Surveys’. Monthly Notices of the Royal
        Astronomical Society 507, no. 1 (11 October 2021): 374–84.
        https://doi.org/10.1093/mnras/stab2135.
        """
        return 1 + self.Rcomp / self.rE

    @cached_property
    def Mtot(self):
        return self.Macc + self.Mcomp

    @cached_property
    def porb(self):
        return CONST_2PI * numpy.sqrt(self.a**3 / (CONST_G * self.Mtot))

    @cached_property
    def rE(self):
        return numpy.sqrt((CONST_4G * self.Macc * self.a) / CONST_C2) / CONST_R_SUN

    @cached_property
    def sin_i(self):
        """
        sin of orbital inclination

        As defined in Nicholas M. Sorabella et al., ‘Modeling Long-Term Variability in
        Stellar-Compact Object Binary Systems for Mass Determinations’, The Astrophysical
        Journal 936, no. 1 (August 2022): 63, https://doi.org/10.3847/1538-4357/ac82b7.
        """
        psi = numpy.arcsin(
            (self.b * self.rE) / self.a
        )  # The angular inclination of the source
        return numpy.sin(CONST_HALF_PI - psi)

    @cached_property
    def tau_eff(self):
        """
        Effective crossing time as defined in Wiktorowicz, Grzegorz, Matthew Middleton,
        Norman Khan, Adam Ingram, Poshak Gandhi, and Hugh Dickinson. ‘Predicting the
        Self-Lensing Population in Optical Surveys’. Monthly Notices of the Royal
        Astronomical Society 507, no. 1 (11 October 2021): 374–84.
        https://doi.org/10.1093/mnras/stab2135.
        """
        return (
            (self.porb * (self.rE + self.Rcomp)) / (numpy.pi * self.a * self.sin_i)
        ) * numpy.sqrt(1 - (self.b / self.b_max) ** 2)

    @units.quantity_input
    def time_to_phase(self, time: units.day):
        """
        Convert time offsets to angular orbital phases.
        """
        time = time.decompose().value % self.porb
        return (time / self.porb) * CONST_2PI

    def v(self, a, phase):
        """
        Orbital velocity.
        """
        r = (a * (1 - self.e**2)) / (
            1 + self.e * numpy.cos(phase - self.periapsis_phase)
        )
        return numpy.sqrt((CONST_G * self.Mtot) * (2 / r - 1 / a))

    def v_tot(self, phase):
        return self.v(self.a, phase)

    def v_acc(self, phase):
        return self.v(self.a_acc, phase)

    @cached_property
    def v_avg(self):
        return CONST_2PI * self.a / self.porb

    def v_comp(self, phase):
        return self.v(self.a_comp, phase)
