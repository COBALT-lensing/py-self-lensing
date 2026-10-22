import unittest

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
from astropy import units
from numpy.testing import assert_allclose

from selflensing.batch import (
    batched_gp_light_curve,
    batched_light_curve,
    batched_light_curve_with_gp_noise,
    make_batched_light_curve_generator,
    make_batched_gp_light_curve_generator,
)
from selflensing.lightcurves import (
    PARAMETER_ORDER,
    as_light_curve_parameter_array,
)
from selflensing.orbits import (
    einstein_radius_m,
    impact_parameter_source_radii,
    orbital_period_days,
    orbital_phase,
    orbital_separation_m,
    source_offset_einstein_radii,
    source_offset_source_radii,
)
from selflensing.gp.LensingGP import build_model_additive, make_kernel_fn
from selflensing.generators import LightcurveGenerator
from selflensing.models import SelfLensingSystem


def _array(value):
    return np.asarray(value.value if hasattr(value, "value") else value)


def _system_from_params(params):
    return SelfLensingSystem(
        a=params["a"] * units.au,
        i=params["i"] * units.rad,
        Macc=params["Macc"] * units.M_sun,
        Mcomp=params["Mcomp"] * units.M_sun,
        Rcomp=params["Rcomp"] * units.R_sun,
        Teff=params["Teff"] * units.K,
        e=params["e"] * units.dimensionless_unscaled,
        periapsis_phase=params["periapsis_phase"] * units.rad,
    )


def _generator_flux(
    params,
    times,
    peak_time,
    baseline=1.0,
    sl=True,
    ev=True,
    db=True,
):
    return _array(
        LightcurveGenerator(
            _system_from_params(params),
            times=times,
            peak_time=peak_time,
            baseline=baseline,
            sl=sl,
            ev=ev,
            db=db,
            lensing_method="IntegratedWittMaoJax",
        ).flux_array
    )


class BatchLightCurveTest(unittest.TestCase):
    def test_batched_light_curve_matches_generator(self):
        params = {
            "a": 1.0,
            "i": 0.5 * np.pi - 0.002,
            "Macc": 9.0,
            "Mcomp": 1.1,
            "Rcomp": 1.2,
            "Teff": 5800.0,
            "e": 0.2,
            "periapsis_phase": 0.4,
        }
        times = np.linspace(-0.5, 0.5, 41)

        generator_flux = _generator_flux(
            params,
            times,
            peak_time=0.0,
        )
        batch_flux = np.asarray(batched_light_curve(times, params))[0]

        assert_allclose(batch_flux, generator_flux, rtol=2e-9, atol=2e-9)

    def test_batched_light_curve_matches_generator_with_peak_and_baseline(self):
        params = {
            "a": 1.3,
            "i": 0.5 * np.pi - 0.003,
            "Macc": 7.0,
            "Mcomp": 1.4,
            "Rcomp": 1.5,
            "Teff": 6200.0,
            "e": 0.1,
            "periapsis_phase": 0.8,
        }
        times = np.linspace(10.0, 11.0, 37)
        peak_time = 10.35
        baseline = 1.7

        generator_flux = _generator_flux(
            params,
            times,
            peak_time=peak_time,
            baseline=baseline,
        )
        batch_flux = np.asarray(
            batched_light_curve(
                times,
                params,
                peak_times=peak_time,
                baseline=baseline,
            )
        )[0]

        assert_allclose(batch_flux, generator_flux, rtol=2e-9, atol=2e-9)

    def test_batched_light_curve_matches_generator_for_multiple_rows(self):
        params = [
            {
                "a": 1.0,
                "i": 0.5 * np.pi - 0.002,
                "Macc": 9.0,
                "Mcomp": 1.1,
                "Rcomp": 1.2,
                "Teff": 5800.0,
                "e": 0.2,
                "periapsis_phase": 0.4,
            },
            {
                "a": 1.3,
                "i": 0.5 * np.pi - 0.003,
                "Macc": 7.0,
                "Mcomp": 1.4,
                "Rcomp": 1.5,
                "Teff": 6200.0,
                "e": 0.1,
                "periapsis_phase": 0.8,
            },
        ]
        times = np.linspace(-0.25, 0.25, 31)
        peak_times = np.array([0.0, 0.05])
        baseline = np.array([1.0, 1.3])

        generator_flux = np.stack(
            [
                _generator_flux(
                    row,
                    times,
                    peak_time=peak_times[index],
                    baseline=baseline[index],
                )
                for index, row in enumerate(params)
            ]
        )
        batch_flux = np.asarray(
            batched_light_curve(
                times,
                params,
                peak_times=peak_times,
                baseline=baseline,
            )
        )

        assert_allclose(batch_flux, generator_flux, rtol=2e-9, atol=2e-9)

    def test_batched_light_curve_accepts_rows_and_compiled_generator(self):
        rows = [
            (1.0, 1.5, 9.0, 1.1, 1.2, 5800.0, 0.2, 0.4),
            (1.2, 1.4, 7.0, 1.4, 1.5, 6200.0, 0.1, 0.8),
        ]
        times = np.linspace(-0.25, 0.25, 17)

        eager_flux = batched_light_curve(times, rows)
        generate = make_batched_light_curve_generator(times)
        compiled_flux = generate(rows)

        self.assertEqual(eager_flux.shape, (2, times.size))
        assert_allclose(np.asarray(compiled_flux), np.asarray(eager_flux))

    def test_as_light_curve_parameter_array_accepts_list_of_dicts(self):
        params = [
            dict(zip(PARAMETER_ORDER, row))
            for row in [
                (1.0, 1.5, 9.0, 1.1, 1.2, 5800.0, 0.2, 0.4),
                (1.2, 1.4, 7.0, 1.4, 1.5, 6200.0, 0.1, 0.8),
            ]
        ]

        array = as_light_curve_parameter_array(params)

        self.assertEqual(array.shape, (2, len(PARAMETER_ORDER)))

    def test_batched_gp_light_curve_matches_single_gp_sample(self):
        key = jax.random.key(123)
        times = np.linspace(-0.25, 0.25, 17)
        mean_flux = np.stack(
            [
                np.linspace(0.99, 1.01, times.size),
                np.linspace(1.02, 0.98, times.size),
            ]
        )
        gp_params = np.array(
            [
                [-4.6, -3.0, -4.2, 5.0, -3.2],
                [-4.4, -3.1, -4.1, 6.0, -3.3],
            ]
        )

        batch_flux = np.asarray(
            batched_gp_light_curve(
                key,
                times,
                mean_flux,
                gp_params,
                kernel_type="GS",
                noise_measure=np.array([1e-6, 2e-6]),
            )
        )

        keys = jax.random.split(key, 2)
        kernel_fn = make_kernel_fn("GS")
        expected = []
        for index in range(2):
            theta = np.concatenate([gp_params[index], [0.0]])
            gp = build_model_additive(
                theta=theta,
                nhyp=5,
                kernel_fn=kernel_fn,
                T=times * 86400.0,
                include_lens_mean=0,
                noise_measure=np.array([1e-6, 2e-6])[index],
                mean_vals=mean_flux[index],
            )
            expected.append(np.asarray(gp.sample(keys[index])))

        assert_allclose(batch_flux, np.stack(expected), rtol=1e-12, atol=1e-12)

    def test_batched_light_curve_with_gp_noise_shape_and_reproducibility(self):
        key = jax.random.key(456)
        times = np.linspace(-0.2, 0.2, 13)
        params = [
            (1.0, 0.5 * np.pi - 0.002, 9.0, 1.1, 1.2, 5800.0, 0.2, 0.4),
            (1.2, 0.5 * np.pi - 0.003, 7.0, 1.4, 1.5, 6200.0, 0.1, 0.8),
        ]
        gp_params = np.array(
            [
                [-4.6, -3.0],
                [-4.4, -3.1],
            ]
        )

        first = batched_light_curve_with_gp_noise(
            key,
            times,
            params,
            gp_params,
            kernel_type="G",
        )
        second = batched_light_curve_with_gp_noise(
            key,
            times,
            params,
            gp_params,
            kernel_type="G",
        )

        self.assertEqual(first.shape, (2, times.size))
        assert_allclose(np.asarray(first), np.asarray(second))

    def test_compiled_batched_gp_light_curve_generator_matches_eager(self):
        key = jax.random.key(789)
        times = np.linspace(-0.2, 0.2, 13)
        mean_flux = np.ones((2, times.size))
        gp_params = np.array(
            [
                [-4.6, -3.0],
                [-4.4, -3.1],
            ]
        )

        eager = batched_gp_light_curve(
            key,
            times,
            mean_flux,
            gp_params,
            kernel_type="G",
        )
        generate = make_batched_gp_light_curve_generator(
            times,
            gp_params,
            kernel_type="G",
        )
        compiled = generate(key, mean_flux)

        assert_allclose(np.asarray(compiled), np.asarray(eager))

    def test_self_lensing_system_uses_shared_orbital_helpers(self):
        params = {
            "a": 1.0,
            "i": 0.5 * np.pi - 0.002,
            "Macc": 9.0,
            "Mcomp": 1.1,
            "Rcomp": 1.2,
            "Teff": 5800.0,
            "e": 0.2,
            "periapsis_phase": 0.4,
        }
        sls = _system_from_params(params)
        phase = jnp.linspace(0.0, 2.0 * jnp.pi, 9)
        time = jnp.linspace(-0.2, 0.2, 9)

        a_m = sls.a.to(units.m).value
        total_mass_kg = sls.Mtot.to(units.kg).value
        macc_kg = sls.Macc.to(units.kg).value
        rcomp_m = sls.Rcomp.to(units.m).value
        sin_i = sls.sin_i.value
        eccentricity = sls.e.value
        periapsis_phase = sls.periapsis_phase.to(units.rad).value
        b_source = impact_parameter_source_radii(a_m, sls.i.value, rcomp_m, eccentricity, periapsis_phase)

        assert_allclose(
            sls.porb.value,
            np.asarray(orbital_period_days(a_m, total_mass_kg)),
        )
        assert_allclose(
            sls.r_sep(phase * units.rad).value,
            np.asarray(orbital_separation_m(phase, a_m, eccentricity, periapsis_phase)),
        )
        assert_allclose(
            sls.rE(phase * units.rad).value,
            np.asarray(
                einstein_radius_m(
                    a_m,
                    sin_i,
                    macc_kg,
                    eccentricity,
                    periapsis_phase,
                    phase,
                )
            ),
        )
        expected_phase = orbital_phase(
            time,
            sls.porb.value,
            eccentricity,
            periapsis_phase,
        )
        assert_allclose(sls.time_to_phase(time * units.day).value, expected_phase)
        assert_allclose(
            sls.phase_to_l(phase * units.rad).value,
            source_offset_source_radii(
                phase,
                a_m,
                rcomp_m,
                eccentricity,
                periapsis_phase,
                b_source,
            ),
        )
        assert_allclose(
            sls.phase_to_b(phase * units.rad),
            source_offset_einstein_radii(
                phase,
                a_m,
                rcomp_m,
                sls.rE(phase * units.rad).to(units.m).value,
                eccentricity,
                periapsis_phase,
                b_source,
            ),
        )


if __name__ == "__main__":
    unittest.main()
