from quaxed import numpy
from jax import numpy as jnp

from astropy import units, constants
from unxt import Quantity

import jax
import random

from .gp.PhysicalRelations import (
    stellar_radius_Rsun,
    effective_temperature_K,
)
from .gp.FluxFinderSensitivity import conv_mass_to_mag

jax.config.update("jax_enable_x64", True)

DEFAULT_LIMITS = {
    "a": (0.1, 200),
    "i": (0.5 * jnp.pi - 0.0001, 0.5 * jnp.pi),
    "Macc": (0.5, 50),
    "Mcomp": (0.5, 50),
    "e": (0, 0.999999),
    "periapsis_phase": (0, 2 * jnp.pi),
    "d": (0.1, 100),
}


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
    DEFAULT_PERIAPSIS = 0 * units.rad
    DEFAULT_E = 0 * units.dimensionless_unscaled
    DEFAULT_I = 0 * units.rad
    DEFAULT_D = 1 * units.kpc

    @classmethod
    def factory(
        cls,
        total=None,
        rlof=False,
        sl_possible=True,
        **kwargs,
    ):
        default_log_dist = ("a", "Macc", "Mcomp")
        for p in default_log_dist:
            k = f"{p}_dist"
            if k not in kwargs:
                kwargs[k] = "log"

        while True:
            params = {}
            for p, (p_min, p_max) in DEFAULT_LIMITS.items():
                if kwargs.get(f"{p}_dist", "uniform") == "log":
                    params[p] = (
                        10 ** random.uniform(numpy.log10(p_min), numpy.log10(p_max))
                        * cls.param_units[p]
                    )
                else:
                    params[p] = random.uniform(p_min, p_max) * cls.param_units[p]

            sls = cls(**params)

            if (rlof or not sls.rlof) and (not sl_possible or sls.sl_possible):
                yield sls

                if total is not None:
                    if total == 0:
                        break
                    total -= 1

    @units.quantity_input
    def __init__(
        self,
        Macc: units.kg,
        Mcomp: units.kg,
        Rcomp: units.m = None,
        a: units.m = None,
        # b: units.dimensionless_unscaled = DEFAULT_B,  # type: ignore
        Teff: units.K = None,
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
        self.a = Quantity.from_(a.to(self.param_units["a"]))

        if Rcomp is None:
            Rcomp = stellar_radius_Rsun(Mcomp.to(units.M_sun).value) * units.R_sun

        if Teff is None:
            Teff = Quantity.from_(
                effective_temperature_K(Mcomp.to(units.M_sun).value), units.K
            )
        self.Teff = Quantity.from_(Teff.to(self.param_units["Teff"]))

        self.Macc = Quantity.from_(Macc.to(self.param_units["Macc"]))
        self.Rcomp = Quantity.from_(Rcomp.to(self.param_units["Rcomp"]))
        self.Mcomp = Quantity.from_(Mcomp.to(self.param_units["Mcomp"]))
        self.e = Quantity.from_(e)
        self.periapsis_phase = Quantity.from_(
            periapsis_phase.to(self.param_units["periapsis_phase"])
        )
        self.i = Quantity.from_(i.to(self.param_units["i"]))
        self.d = Quantity.from_(d.to(self.param_units["d"]))

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
        # Floating point errors can make this a small negative number
        self._b = numpy.abs((self.a * numpy.cos(i) / self.rE()).decompose([]))

    @property
    def b_max(self):
        """
        Maximum b where SL is possible as defined in Wiktorowicz, Grzegorz, Matthew Middleton,
        Norman Khan, Adam Ingram, Poshak Gandhi, and Hugh Dickinson. ‘Predicting the
        Self-Lensing Population in Optical Surveys’. Monthly Notices of the Royal
        Astronomical Society 507, no. 1 (11 October 2021): 374–84.
        https://doi.org/10.1093/mnras/stab2135.
        """
        return (1 + self.Rcomp / self.rE()).decompose([])

    @property
    def sl_possible(self):
        return self.b <= self.b_max

    @property
    def b_source(self):
        """
        The impact parameter in units of Rcomp (source radii). b_source=1 when the lens is at
        the edge of the stellar disk.
        """
        return (self.b * (self.rE() / self.Rcomp)).decompose([])

    @property
    def Mtot(self):
        return self.Macc + self.Mcomp

    @property
    def Tmag(self):
        return conv_mass_to_mag(self.Mcomp.value, self.d.value)[0]

    @units.quantity_input
    def phase_to_l(self, phase: units.rad):
        """
        Given a phase angle, compute the distance of the lens from the centre of the disk.

        When the lens is behind the star, assumes l is at its maximum.
        """
        phase_limit = 0.5 * numpy.pi * units.rad
        phase_upper_limit = 2 * numpy.pi * units.rad - phase_limit
        if isinstance(phase, numpy.ndarray) or phase.shape != tuple():
            phase = jnp.array(phase.value)
            phase = jnp.where(
                (phase > phase_limit.value) & (phase < phase_upper_limit.value),
                jnp.full_like(phase, phase_limit.value),
                phase,
            )
            phase = Quantity(phase, units.rad)
        elif phase > phase_limit and phase < phase_upper_limit:
            phase = 0.5 * numpy.pi * units.rad
        return numpy.hypot(
            numpy.abs(
                self.r_sep(phase=phase).to(units.m).value
                * numpy.sin(phase.value)
                / self.Rcomp.to(units.m).value
            ),
            self.b_source.value,
        )

    @units.quantity_input
    def phase_to_b(self, phase: units.rad):
        return (self.phase_to_l(phase) / (self.rE() / self.Rcomp)).decompose([]).value

    @property
    def porb(self) -> units.day:
        a = self.a.to(units.m).value.astype(numpy.float64)
        m = self.Mtot.to(units.kg).value.astype(float)
        G = constants.G.value.astype(float)
        return Quantity((2 * numpy.pi * ((a**3 / (G * m)) ** 0.5)), "s").to(units.day)

    # @units.quantity_input
    def r(self, m: units.kg, phase: units.rad) -> units.m:
        """
        The distance from the centre of mass of the object with mass 'm' at 'phase'
        """
        r_max = self.a * m / self.Mtot
        return (r_max * (1 - self.e**2)) / (
            1
            + self.e
            * numpy.cos(
                phase.to(units.rad).value - self.periapsis_phase.to(units.rad).value
            )
        )

    @units.quantity_input
    def rE(self, phase: units.rad = 0 * units.rad) -> Quantity:
        m = self.Macc.to(units.kg).value
        G = constants.G.value
        r = self.r_sep(phase).to(units.m).value
        c2 = (constants.c**2).value
        return Quantity(((4 * G * m * r * self.sin_i.value) / c2) ** 0.5, "m")

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

    @property
    def rlof(self):
        return self.r_roche <= self.Rcomp

    # @units.quantity_input
    def r_sep(self, phase: units.rad = 0 * units.rad) -> units.m:
        """
        The separation in space between acc and comp at phase
        """
        return (self.r(self.Macc, phase) + self.r(self.Mcomp, phase)).to(units.m)

    @property
    def tau_eff(self):
        """
        Effective crossing time as defined in Wiktorowicz, Grzegorz, Matthew Middleton,
        Norman Khan, Adam Ingram, Poshak Gandhi, and Hugh Dickinson. ‘Predicting the
        Self-Lensing Population in Optical Surveys’. Monthly Notices of the Royal
        Astronomical Society 507, no. 1 (11 October 2021): 374–84.
        https://doi.org/10.1093/mnras/stab2135.
        """
        porb = self.porb.decompose([]).value
        rE = self.rE().decompose([]).value
        Rcomp = self.Rcomp.decompose([]).value
        a = self.a.decompose([]).value
        b = self.b.decompose([]).value
        b_max = self.b_max.decompose([]).value
        sin_i = self.sin_i.decompose([]).value
        return Quantity(
            ((porb * (rE + Rcomp)) / (numpy.pi * a * sin_i))
            * (1 - (b / b_max) ** 2) ** 0.5,
            "s",
        ).to(units.day)

    @property
    def tau_sl(self):
        """
        Self-lensing duration as defined in Masuda, Kento, and Kenta Hotokezaka. ‘Prospects
        of Finding Detached Black Hole–Star Binaries with TESS’. The Astrophysical Journal
        883, no. 2 (2019): 169. https://doi.org/10.3847/1538-4357/ab3a4f.
        """
        # TODO: This can't be the correct way to handle the units...
        Rcomp = self.Rcomp.to(units.R_sun).value
        porb = self.porb.to(units.day).value
        Mtot = self.Mtot.to(units.M_sun).value
        return (
            1.8 * units.hour * numpy.pi / 4 * porb ** (1 / 3) * Mtot ** (-1 / 3) * Rcomp
        ).to(units.day)

    def time_to_l(self, time):
        return self.phase_to_l(self.time_to_phase(time))

    def time_to_b(self, time):
        p = self.time_to_phase(time)
        r = self.phase_to_b(p)
        return r

    @units.quantity_input
    def time_to_phase(self, time: units.day):
        """
        Convert time offsets to angular orbital phases.
        """
        time = time.value % self.porb.value
        return Quantity((((time / self.porb.value) * 2 * numpy.pi)), units.rad)

    @units.quantity_input
    def v(self, m: units.kg, phase: units.rad) -> units.m / units.s:
        """
        Orbital velocity.
        """
        v = (
            (constants.G * self.Mtot)
            * (2 / self.r(m, phase) - 1 / (self.a * m / self.Mtot))
        ) ** 0.5
        return v.to(units.m / units.s)

    @units.quantity_input
    def v_tot(self, phase: units.rad) -> units.m / units.s:
        return self.v(self.Mtot, phase)

    @units.quantity_input
    def v_acc(self, phase: units.rad) -> units.m / units.s:
        return self.v(self.Macc, phase)

    @property
    def v_avg(self):
        return (2 * numpy.pi * self.a / self.porb).to(units.m / units.s)

    @units.quantity_input
    def v_comp(self, phase: units.rad):
        # return self.v(self.Mcomp, phase)
        return Quantity(
            (
                (
                    constants.G.value
                    * self.Macc.to(units.kg).value
                    / self.r_sep(phase).to(units.m).value
                )
                ** 0.5
            ),
            units.m / units.s,
        )
