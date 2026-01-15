from functools import cached_property

import numpy as np
import jax.numpy as jnp
import jax
import emcee
import jax.numpy as jnp

import os
import random
import time

from . import (
    FluxFinderSensitivity as ff,
    LensingGP as lgp,
    MCMCfitting as mcf,
    PhysicalRelations as physr,
    PriorsSamplers as ps,
)

PWD = os.path.dirname(os.path.abspath(__file__))


class BaseGP(object):
    pass


class TESSGP(BaseGP):
    csv_paths = [
        f"{PWD}/data/Disk_f40_TESS_output.csv.gz",
        f"{PWD}/data/Bulge_f40_TESS_output.csv.gz",
    ]
    default_kernel_type = "GS"
    mu_bounds = [(0.0, 7.0)]  # log10 μ
    cadence_s = 600.0
    use_kde_flag = True
    Q_s = 5.0  # S: quality factor (broad)
    log_tau_f_bounds = (2, 6.5)

    def __init__(
        self,
        sls,
        lc_times=None,
        observed_lc=None,
        mean_vals=None,
        nwalkers=None,
        kernel_type=default_kernel_type,
    ):
        self.kernel_type = kernel_type
        self.sls = sls

        if lc_times is not None:
            self.lc_times = lc_times
        if observed_lc is not None:
            self.observed_lc = observed_lc
        self.mean_vals = mean_vals
        self.nwalkers = nwalkers

    @cached_property
    def model(self):
        nhyp = len(self.theta) - 1

        kernel_fn = lgp.make_kernel_fn(self.kernel_type)
        gp_model = lgp.build_model_additive(
            theta=self.theta,
            nhyp=nhyp,
            kernel_fn=kernel_fn,
            T=self.T,
            include_lens_mean=0,
            noise_measure=self.noise_measure.astype(float),
            mean_vals=self.mean_vals,
        )

        return gp_model

    def condition(self, y):
        r = self.model.condition(y)
        self.model = r.gp
        return r.log_probability

    @property
    def seed(self):
        prev_seed = getattr(self, "_prev_seed", 0)
        seed = (int(time.time()) + prev_seed) % (2**32)
        self._prev_seed = seed
        return seed

    def generate(self):
        # --- assemble theta and build GP ---
        # print("theta=", noise_only_theta)
        # nhyp = len(self.gs_hypers)

        # kernel_fn = lgp.make_kernel_fn(self.kernel_type)
        # noise_gp = lgp.build_model_additive(
        #     theta=self.theta,
        #     nhyp=nhyp,
        #     kernel_fn=kernel_fn,
        #     T=self.T,
        #     include_lens_mean=0,
        #     noise_measure=self.noise_measure.astype(float),
        # )

        return self.model.sample(jax.random.key(self.seed))

    @property
    def sls(self):
        return self._sls

    @sls.setter
    def sls(self, sls):
        self._sls = sls
        test_Mag, noise_measure = ff.conv_mass_to_mag(
            M=sls.Mcomp.value, D_kpc=sls.d.value
        )

        self.mu_test = jnp.log10(
            ff.counts_per_cadence_from_mag(
                test_Mag.astype(float), cadence_s=self.cadence_s
            )
        )

        # scale white-noise measure to your convention (fractional per cadence)
        self.noise_measure = noise_measure.astype(float) * (
            10**self.mu_test
        )  # * np.sqrt(60.0/2.0)

        # --- build GS kernel hypers directly (physics + μ bands) ---
        if "G" in self.kernel_type:
            self.lnu_g = jnp.log10(
                physr.granulation_frequency_relation(self.sls.Mcomp.value).astype(float)
            )  # G: log10 ν_gran [Hz]
            self.lsig_g = ps.expected_log_sigma(jnp.asarray(self.mu_test), "G").astype(
                float
            )  # G: log10 σ_gran
            self.g_hypers = jnp.array(
                [
                    self.lnu_g,
                    self.lsig_g,
                ],
                dtype=float,
            )
        else:
            self.g_hypers = jnp.array([])

        if "S" in self.kernel_type:
            self.lnu_s = jnp.log10(
                physr.asteroseismic_frequency_relation(self.sls.Mcomp.value).astype(
                    float
                )
            )  # S: log10 ν_seis [Hz]

            self.lsig_s = ps.expected_log_sigma(jnp.asarray(self.mu_test), "S").astype(
                float
            )  # S: log10 σ_seis

            self.s_hypers = jnp.array([self.lnu_s, self.Q_s, self.lsig_s], dtype=float)
        else:
            self.s_hypers = jnp.array([])

        if "F" in self.kernel_type:
            self.f_hypers = jnp.array(
                [
                    10 ** random.uniform(*self.log_tau_f_bounds),
                    ps.expected_log_sigma(jnp.asarray(self.mu_test), "F"),
                ]
            )
        else:
            self.f_hypers = jnp.array([])

    @property
    def theta(self):
        return jnp.concatenate(
            [
                jnp.asarray(self.g_hypers, dtype=float),
                jnp.asarray(self.s_hypers, dtype=float),
                jnp.asarray(self.f_hypers, dtype=float),
                jnp.array([self.mu_test], dtype=float),
            ]
        )

    @property
    def mean_vals(self):
        return self._mean_vals

    @mean_vals.setter
    def mean_vals(self, v):
        self._mean_vals = v
        if hasattr(self, "logprob"):
            del self.logprob
        if hasattr(self, "model"):
            del self.model

    def set_mean_amp(self, v):
        self.mean_vals = self.mu_test * v

    @property
    def kernel_bounds(self):
        T_days = 27.0
        T_seconds = T_days * 86400.0
        nu_res = 1.0 / T_seconds  # ~4.3e-7 Hz
        nu_nyq = 1.0 / (2.0 * self.cadence_s)  # ~8.33e-4 Hz

        log10_nu_res = float(np.log10(nu_res))  # ~ -6.37
        log10_nu_nyq = float(np.log10(nu_nyq))  # ~ -3.08

        # ---------------------------------------------
        # Kernel hyperparameter bounds (Hz & seconds)
        # G: [log10(nu_gran_Hz), log10(sigma)]
        # S: [log10(nu_seis_Hz), Q,            log10(sigma)]
        # E: [log10(nu_ellip_Hz),              log10(sigma)]  with nu_ellip = 2/P
        # F: [log10(tau_d [s]),                log10(sigma)]
        # ---------------------------------------------
        return {
            # Granulation frequency within observable band (resolution .. Nyquist)
            # Note: MS granulation typically near/above Nyquist at 10-min cadence; this still lets data push to the cap.
            "G": [(log10_nu_res, log10_nu_nyq), (-6.0, 2.0)],
            # Seismic ν_max for MS is usually >> Nyquist at this cadence – allow same band but consider disabling 'S' for pure MS fits.
            "S": [(log10_nu_res, log10_nu_nyq), (0.5, 20.0), (-6.0, 2.0)],
            # Ellipsoidal: your kernel expects log10(ν_ellip [Hz]).
            # If you prefer to bound P_ellip instead, translate via log10(ν_ellip) = log10(2) - log10(P).
            # Here we simply constrain ν_ellip to the observable band:
            "E": [(log10_nu_res, log10_nu_nyq), (-6.0, 2.0)],
            # Flare-like colored noise: τ_d in seconds (we use log10 τ_d).
            # Give it from a few minutes to many days within a sector.
            "F": [
                (2.0, 6.5),
                (-6.0, 2.0),
            ],  # log10 τ_d ∈ [10^2, 10^6.5] s; log10 sigma flexible
        }

    @property
    def lens_bounds(self):
        return [
            # (0.1, 20.0),                 # M_s [M_sun]  (MS assumption)
            (self.sls.Mcomp.value * 0.9, self.sls.Mcomp.value * 1.1),
            (1.4, 20.0),  # M_co [M_sun]
            (5.0, 8.0),  # log10 P_orb [s]  (~1.16 d to ~31.7 yr)
            (3 * np.pi / 8, 5 * np.pi / 8),  # inclination [rad]
            (0.0, 2 * np.pi),  # phase offset [rad]
        ]

    @cached_property
    def full_bounds(self):
        full_bounds = ps.build_combined_bounds(
            kernel_type=self.kernel_type,
            kernel_bounds_dict=self.kernel_bounds,
            lens_bounds=self.lens_bounds,
            mu_bounds=self.mu_bounds,
            include_lens_mean=1,
        )
        return tuple(map(tuple, full_bounds))

    @cached_property
    def kernel_mu_bounds(self):
        full_bounds = ps.build_combined_bounds(
            kernel_type=self.kernel_type,
            kernel_bounds_dict=self.kernel_bounds,
            lens_bounds=self.lens_bounds,
            mu_bounds=self.mu_bounds,
            include_lens_mean=0,
        )
        return tuple(map(tuple, full_bounds))

    @cached_property
    def kde_output_tuple(self):
        return ps.build_kde_3d(
            csv_paths=self.csv_paths,
            bandwidth=None,
            grid_sizes=(50, 50, 50),
            expand_frac=0.05,
        )

    @cached_property
    def nhyp(self):
        return mcf.count_nhyp(self.kernel_type)

    @property
    def ndim(self):
        n_lens_params = 5
        return self.nhyp + n_lens_params + 1

    @property
    def nwalkers(self):
        return self._nwalkers

    @nwalkers.setter
    def nwalkers(self, v):
        if v is None:
            self._nwalkers = 4 * self.ndim
        else:
            self._nwalkers = v

    @cached_property
    def p0(self):
        return mcf.init_walkers(
            n_walkers=self.nwalkers,
            param_bounds=self.full_bounds,
            kernel_type=self.kernel_type,
            include_lens_mean=False,
            # --- KDE Changes ---
            use_kde=self.use_kde_flag,  # <<< Uses the flag from our check
            kde3d_pack=self.kde_output_tuple,  # <<< Passes the 6-tuple
            # --- Physical prior enabled ---
            phys_scale=True,
            T=np.asarray(self.T),
            y=np.asarray(self.y),
            cadence_s=self.cadence_s,
        )

    @cached_property
    def logprob(self):
        return mcf.make_emcee_vectorized_logprob(
            kernel_type=self.kernel_type,
            nhyp=self.nhyp,
            noise_measure=self.noise_measure.astype(float),
            param_bounds=self.full_bounds,
            # --- KDE Changes ---
            kde3d_args=self.kde_output_tuple,  # <<< Pass the 6-tuple
            # --- Physical prior enabled ---
            phys_scale=True,
            include_lens_mean=False,
            mean_vals=self.mean_vals,
        )

    @cached_property
    def logprob_single(self):
        single = mcf.make_logposterior_fn(
            kernel_type=self.kernel_type,
            nhyp=self.nhyp,
            noise_measure=self.noise_measure,
            param_bounds=self.kernel_mu_bounds,
            kde3d_args=self.kde_output_tuple,
            phys_scale=True,
            include_lens_mean=False,
            mean_vals=self.mean_vals,
        )

        @jax.jit
        def wrapped():
            return single(theta=self.theta, T=self.T, y=self.y)

        return wrapped
        # print(ps.log_prior_bounds(self.theta, self.kernel_mu_bounds))
        # p = mcf._log_posterior_single(
        #     theta=self.theta,
        #     T=self.T,
        #     y=self.y,
        #     kernel_fn=lgp.make_kernel_fn(self.kernel_type),
        #     log_prior_phys_fn=None,  # ps.make_log_prior_physical_jit(
        #     # self.kernel_type
        #     # ),  # from ps.make_log_prior_physical_jit (or None)
        #     kde_lp_fn=None,  # from ps.make_kde3d_logprior_jit (or None)
        #     nhyp=self.nhyp,
        #     noise_measure=self.noise_measure,
        #     combined_bounds=self.kernel_mu_bounds,
        #     include_lens_mean=0,
        #     mean_vals=self.mean_vals,
        # )

        # # print("logprob:", p, self.theta, self.y)
        # return p

    @property
    def lc_times(self):
        return self._lc_times

    @lc_times.setter
    def lc_times(self, v):
        self._lc_times = v
        self.T = jnp.asarray(v, dtype=float)

    @property
    def observed_lc(self):
        return self._observed_lc

    @observed_lc.setter
    def observed_lc(self, v):
        self._observed_lc = v
        self.y = v

    def fit(self):
        sampler_lens = emcee.EnsembleSampler(
            nwalkers=self.nwalkers,
            ndim=self.ndim,
            log_prob_fn=self.logprob,
            vectorize=True,
            args=(self.T, self.y),
        )

        nburn, nprod = 500, 2000
        # print(f"Running burn-in ({nburn} steps)...")
        p0, _, _ = sampler_lens.run_mcmc(self.p0, nburn, progress=True)
        sampler_lens.reset()
        # print(f"Running production ({nprod} steps)...")
        sampler_lens.run_mcmc(p0, nprod, progress=True)

        # ---- 6) Summaries ----
        # *** Use sampler_lens to get chains ***
        flat_chain_lens = sampler_lens.get_chain(flat=True)
        logp_lens = sampler_lens.get_log_prob(flat=True)

        # print("\n--- MCMC Run Complete ---")
        # print(
        #     f"Final acceptance fraction: {np.mean(sampler_lens.acceptance_fraction):.3f}"
        # )
        # print("Chain shape:", flat_chain_lens.shape)
        # print(
        #     f"logp mean/median: {np.mean(logp_lens):.2f} / {np.median(logp_lens):.2f}"
        # )

        # Unpack a quick MAP-ish estimate
        best_idx = np.argmax(logp_lens)
        theta_hat = flat_chain_lens[best_idx]
        return theta_hat, sampler_lens


# class GPMethods(object):
#     TESS = 0

#     methods = {
#         TESS: TESSGP,
#     }

#     def get(self, k):
#         return self.methods.get(k)
