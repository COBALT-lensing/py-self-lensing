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
from .orbits import (
    crossing_time_days,
    mean_relative_speed_m_per_s,
    relative_polar_velocity_m_per_s,
    einstein_radius_m,
    impact_parameter_source_radii,
    orbital_period_days,
    orbital_phase,
    orbital_separation_m,
    source_offset_einstein_radii,
    source_offset_source_radii,
)

jax.config.update("jax_enable_x64", True)

DEFAULT_LIMITS = {
    "a": (0.1, 200),
    # Allow both lensing and non-lensing binary orientations.
    "i": (0.0, 0.5 * jnp.pi),
    "Macc": (0.1, 50),
    "Mcomp": (0.5, 50),
    "e": (0, 0.999999),
    "periapsis_phase": (0, 2 * jnp.pi),
    "d": (0.1, 100),
}


def _quantity_value(quantity, unit=None):
    if unit is not None and hasattr(quantity, "to"):
        quantity = quantity.to(unit)
    return quantity.value if hasattr(quantity, "value") else quantity


def _as_quantity(quantity, unit=None):
    if unit is not None and hasattr(quantity, "to"):
        quantity = quantity.to(unit)
    if hasattr(quantity, "value") and hasattr(quantity, "unit"):
        return Quantity(jnp.asarray(quantity.value), quantity.unit)
    return Quantity(jnp.asarray(quantity), unit)


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
        total,
        rlof=False,
        sl_possible=True,
        param_samples=None,
        **kwargs,
    ):
        default_log_dist = ("a", "e")

        for p in default_log_dist:
            k = f"{p}_dist"
            if k not in kwargs:
                kwargs[k] = "log"

        while total > 0:
            params = {}
            if param_samples is not None:
                params = param_samples.sample(1).to_dict("records")[0]

            for p, (p_min, p_max) in DEFAULT_LIMITS.items():

                if p in params:
                    continue
                if kwargs.get(f"{p}_dist", "uniform") == "log":
                    offset_limits = p_min == 0
                    if offset_limits:
                        p_min += 1
                        p_max += 1
                    params[p] = 10 ** random.uniform(
                        numpy.log10(p_min), numpy.log10(p_max)
                    )
                    if offset_limits:
                        params[p] -= 1
                else:
                    params[p] = random.uniform(p_min, p_max)

            for p, v in params.items():
                params[p] = v * cls.param_units[p]

            sls = cls(**params)

            if (rlof or not sls.rlof) and (not sl_possible or sls.sl_possible):
                yield sls
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
        self.a = _as_quantity(a, self.param_units["a"])

        if Rcomp is None:
            Rcomp = Quantity(
                stellar_radius_Rsun(Mcomp.to(units.M_sun).value), units.R_sun
            )

        if Teff is None:
            Teff = Quantity(
                effective_temperature_K(Mcomp.to(units.M_sun).value), units.K
            )
        self.Teff = _as_quantity(Teff, self.param_units["Teff"])

        self.Macc = _as_quantity(Macc, self.param_units["Macc"])
        self.Rcomp = _as_quantity(Rcomp, self.param_units["Rcomp"])
        self.Mcomp = _as_quantity(Mcomp, self.param_units["Mcomp"])
        self.e = _as_quantity(e, self.param_units["e"])
        self.periapsis_phase = _as_quantity(
            periapsis_phase, self.param_units["periapsis_phase"]
        )
        self.i = _as_quantity(i, self.param_units["i"])
        self.d = _as_quantity(d, self.param_units["d"])

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

    @property
    def _b(self):
        # Derived on access: changing e, masses, a, or peri must not leave a cache.
        radius = _quantity_value(self.rE(), units.m)
        value = _quantity_value(self.b_source)*_quantity_value(self.Rcomp, units.m)
        return Quantity(jnp.where(radius > 0, value/jnp.where(radius > 0, radius, 1.), jnp.inf),
                        units.dimensionless_unscaled)

    @property
    def b_max(self):
        """
        Maximum b where SL is possible as defined in Wiktorowicz, Grzegorz, Matthew Middleton,
        Norman Khan, Adam Ingram, Poshak Gandhi, and Hugh Dickinson. ‘Predicting the
        Self-Lensing Population in Optical Surveys’. Monthly Notices of the Royal
        Astronomical Society 507, no. 1 (11 October 2021): 374–84.
        https://doi.org/10.1093/mnras/stab2135.
        """
        value = 1 + _quantity_value(self.Rcomp, units.m) / _quantity_value(
            self.rE(), units.m
        )
        return Quantity(value, units.dimensionless_unscaled)

    @property
    def sl_possible(self):
        # For RE > 0, b <= b_max is equivalent to d <= Rcomp + RE,
        # since b=d/RE and b_max=1+Rcomp/RE. Use physical distances to
        # avoid division by zero, and reject RE=0 explicitly: otherwise
        # both b and b_max can be infinite and inf <= inf would pass.
        radius = _quantity_value(self.rE(), units.m)
        return (radius > 0) & (_quantity_value(self.b_source)*_quantity_value(self.Rcomp, units.m)
                               <= _quantity_value(self.Rcomp, units.m)+radius)

    @property
    def b_source(self):
        """
        The impact parameter in units of Rcomp (source radii). b_source=1 when the lens is at
        the edge of the stellar disk.
        """
        value = impact_parameter_source_radii(
            _quantity_value(self.a, units.m),
            _quantity_value(self.i, units.rad),
            _quantity_value(self.Rcomp, units.m),
            _quantity_value(self.e),
            _quantity_value(self.periapsis_phase, units.rad),
        )
        return Quantity(value, units.dimensionless_unscaled)

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

        Returns the full projected separation on either half-orbit; rE is
        zero behind the source, and magnification is gated separately.
        """
        value = source_offset_source_radii(
            _quantity_value(phase, units.rad),
            _quantity_value(self.a, units.m),
            _quantity_value(self.Rcomp, units.m),
            _quantity_value(self.e),
            _quantity_value(self.periapsis_phase, units.rad),
            _quantity_value(self.b_source),
        )
        return Quantity(value, units.dimensionless_unscaled)

    @units.quantity_input
    def phase_to_b(self, phase: units.rad):
        return source_offset_einstein_radii(
            _quantity_value(phase, units.rad),
            _quantity_value(self.a, units.m),
            _quantity_value(self.Rcomp, units.m),
            _quantity_value(self.rE(phase), units.m),
            _quantity_value(self.e),
            _quantity_value(self.periapsis_phase, units.rad),
            _quantity_value(self.b_source),
        )

    @property
    def porb(self) -> units.day:
        value = orbital_period_days(
            _quantity_value(self.a, units.m),
            _quantity_value(self.Mtot, units.kg),
        )
        return Quantity(value, units.day)

    # @units.quantity_input
    def r(self, m: units.kg, phase: units.rad) -> units.m:
        """
        Barycentric distance of the body opposite mass m at phase.
        Passing Mtot gives the relative binary separation.
        """
        value = orbital_separation_m(
            _quantity_value(phase, units.rad),
            _quantity_value(self.a, units.m),
            _quantity_value(self.e),
            _quantity_value(self.periapsis_phase, units.rad),
        ) * (
            _quantity_value(m, units.kg)
            / _quantity_value(self.Mtot, units.kg)
        )
        return Quantity(value, units.m)

    @units.quantity_input
    def rE(self, phase: units.rad = 0 * units.rad) -> Quantity:
        value = einstein_radius_m(
            _quantity_value(self.a, units.m),
            _quantity_value(self.sin_i),
            _quantity_value(self.Macc, units.kg),
            _quantity_value(self.e),
            _quantity_value(self.periapsis_phase, units.rad),
            _quantity_value(phase, units.rad),
        )
        return Quantity(value, units.m)

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
        value = orbital_separation_m(
            _quantity_value(phase, units.rad),
            _quantity_value(self.a, units.m),
            _quantity_value(self.e),
            _quantity_value(self.periapsis_phase, units.rad),
        )
        return Quantity(value, units.m)

    def _crossing_time(self, include_einstein):
        value = crossing_time_days(
            _quantity_value(self.a, units.m), _quantity_value(self.i, units.rad),
            _quantity_value(self.Macc, units.kg), _quantity_value(self.Mtot, units.kg),
            _quantity_value(self.Rcomp, units.m), _quantity_value(self.e),
            _quantity_value(self.periapsis_phase, units.rad), include_einstein=include_einstein)
        return Quantity(value, units.day)

    @property
    def tau_eff(self):
        """Local straight-track crossing of Rstar+RE; not a measured FWHM.

        Uses the full eccentric sky velocity and local closest approach at the
        longitude-zero epoch. Radius and velocity are frozen for this estimate.
        """
        return self._crossing_time(True)

    @property
    def tau_sl(self):
        """Local stellar-disc crossing time for this orbit and inclination.

        Replaces the former circular, orientation-averaged scaling. Non-crossing
        tracks return NaN. No Einstein radius is added to the stellar boundary.
        """
        return self._crossing_time(False)

    def time_to_l(self, time):
        return self.phase_to_l(self.time_to_phase(time))

    def time_to_b(self, time):
        p = self.time_to_phase(time)
        r = self.phase_to_b(p)
        return r

    @units.quantity_input
    def time_to_phase(self, time: units.day):
        """
        Convert time offsets to true orbital longitudes relative to peak lensing.
        """
        phase = orbital_phase(
            _quantity_value(time, units.day),
            _quantity_value(self.porb, units.day),
            _quantity_value(self.e),
            _quantity_value(self.periapsis_phase, units.rad),
        )
        return Quantity(phase, units.rad)

    @units.quantity_input
    def v(self, m: units.kg, phase: units.rad) -> units.m / units.s:
        """Barycentric speed for the body opposite mass m.

        As with r(m), m is the OTHER body's mass. Passing Mtot gives the
        relative two-body speed. Use v_acc/v_comp for named components.
        """
        vr, vt = relative_polar_velocity_m_per_s(
            _quantity_value(phase, units.rad), _quantity_value(self.a, units.m),
            _quantity_value(self.Mtot, units.kg), _quantity_value(self.e),
            _quantity_value(self.periapsis_phase, units.rad))
        speed = jnp.hypot(vr, vt)*_quantity_value(m, units.kg)/_quantity_value(self.Mtot, units.kg)
        return Quantity(speed, units.m/units.s)

    @units.quantity_input
    def v_tot(self, phase: units.rad) -> units.m / units.s:
        return self.v(self.Mtot, phase)

    @units.quantity_input
    def v_acc(self, phase: units.rad) -> units.m / units.s:
        return self.v(self.Mcomp, phase)

    @property
    def v_avg(self):
        """Time-average relative speed along the eccentric ellipse."""
        return Quantity(mean_relative_speed_m_per_s(
            _quantity_value(self.a, units.m), _quantity_value(self.Mtot, units.kg),
            _quantity_value(self.e)), units.m/units.s)

    @units.quantity_input
    def v_comp(self, phase: units.rad):
        return self.v(self.Macc, phase)
