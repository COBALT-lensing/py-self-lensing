import numpy as np
import jax.numpy as jnp
import jax
import emcee
import jax.numpy as jnp

import os
import time

from . import (
    FluxFinderSensitivity as ff,
    LensingGP as lgp,
    MCMCfitting as mcf,
    PhysicalRelations as physr,
    PriorsSamplers as ps,
)

PWD = os.path.dirname(os.path.abspath(__file__))


class TESSGP(object):
    csv_paths = [
        f"{PWD}/data/Disk_f40_TESS_output.csv.gz",
        f"{PWD}/data/Bulge_f40_TESS_output.csv.gz",
    ]

    def __init__(self, sls):
        self.sls = sls

        test_Mag, noise_measure = ff.conv_mass_to_mag(
            M=self.sls.Mcomp.value, D_kpc=self.sls.d.value
        )
        # print("TESS mag:", float(test_Mag))

        self.mu_test = np.log10(
            ff.counts_per_cadence_from_mag(float(test_Mag), cadence_s=120.0)
        )

        # scale white-noise measure to your convention (fractional per cadence)
        self.noise_measure = float(noise_measure) * (
            10**self.mu_test
        )  # * np.sqrt(60.0/2.0)

        self.seed = int(time.time()) % (
            2**32
        )  # Generate a seed based on the current time

    def generate_noise(self, times):
        # --- build GS kernel hypers directly (physics + μ bands) ---
        lnu_g = np.log10(
            float(physr.granulation_frequency_relation(self.sls.Mcomp.value))
        )  # G: log10 ν_gran [Hz]
        lsig_g = float(
            ps.expected_log_sigma(jnp.asarray(self.mu_test), "G")
        )  # G: log10 σ_gran

        lnu_s = np.log10(
            float(physr.asteroseismic_frequency_relation(self.sls.Mcomp.value))
        )  # S: log10 ν_seis [Hz]
        Q_s = 5.0  # S: quality factor (broad)
        lsig_s = float(
            ps.expected_log_sigma(jnp.asarray(self.mu_test), "S")
        )  # S: log10 σ_seis

        gs_hypers = np.array([lnu_g, lsig_g, lnu_s, Q_s, lsig_s], dtype=float)

        # --- assemble theta and build GP ---
        noise_only_theta = jnp.concatenate(
            [
                jnp.asarray(gs_hypers, dtype=float),
                jnp.array([self.mu_test], dtype=float),
            ]
        )
        print("theta=", noise_only_theta)
        nhyp = len(gs_hypers)

        kernel_fn = lgp.make_kernel_fn("GS")
        noise_gp = lgp.build_model_additive(
            theta=noise_only_theta,
            nhyp=nhyp,
            kernel_fn=kernel_fn,
            T=jnp.asarray(times, dtype=float),
            include_lens_mean=0,
            noise_measure=float(self.noise_measure),
        )

        return noise_gp.sample(jax.random.key(self.seed))

    def fit(self, lc_times, observed_lc):
        # Segment characteristics
        T_days = 27.0
        cadence_s = 600.0
        T_seconds = T_days * 86400.0
        nu_res = 1.0 / T_seconds  # ~4.3e-7 Hz
        nu_nyq = 1.0 / (2.0 * cadence_s)  # ~8.33e-4 Hz

        log10_nu_res = float(np.log10(nu_res))  # ~ -6.37
        log10_nu_nyq = float(np.log10(nu_nyq))  # ~ -3.08
        LOG10_2 = float(np.log10(2.0))

        # ---------------------------
        # Lens + μ (unchanged units)
        # ---------------------------
        lens_bounds = [
            # (0.1, 20.0),                 # M_s [M_sun]  (MS assumption)
            (self.sls.Mcomp.value * 0.9, self.sls.Mcomp.value * 1.1),
            (1.4, 20.0),  # M_co [M_sun]
            (5.0, 8.0),  # log10 P_orb [s]  (~1.16 d to ~31.7 yr)
            (3 * np.pi / 8, 5 * np.pi / 8),  # inclination [rad]
            (0.0, 2 * np.pi),  # phase offset [rad]
        ]
        mu_bounds = [(0.0, 7.0)]  # log10 μ

        # ---------------------------------------------
        # Kernel hyperparameter bounds (Hz & seconds)
        # G: [log10(nu_gran_Hz), log10(sigma)]
        # S: [log10(nu_seis_Hz), Q,            log10(sigma)]
        # E: [log10(nu_ellip_Hz),              log10(sigma)]  with nu_ellip = 2/P
        # F: [log10(tau_d [s]),                log10(sigma)]
        # ---------------------------------------------
        kernel_bounds_dict = {
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

        # ---- 0) Data ----
        T = jnp.asarray(lc_times, dtype=float)
        y = jnp.asarray(observed_lc, dtype=float)  # Using the lensing light curve

        # --- New KDE Check: Try to build it directly ---
        # try:
        #     print("Attempting to build KDE output tuple...")
        #     # This line will raise a NameError if any variable is missing
        #     kde_output_tuple = (kde_obj_3d, inv_stds_3d, grid_Ms, grid_Mco, grid_logP, log_pdf_3d)
        #     use_kde_flag = True
        #     print("Successfully built KDE tuple. KDE is ON.")

        # except NameError as e:
        #     print(f"A required KDE variable is missing: {e}")
        #     print("   Please go back and re-run Cell [12] successfully to enable the KDE.")
        #     kde_output_tuple = None
        #     use_kde_flag = False

        kde_output_tuple = ps.build_kde_3d(
            csv_paths=self.csv_paths,
            bandwidth=None,
            grid_sizes=(50, 50, 50),
            expand_frac=0.05,
        )
        use_kde_flag = True

        # ---- 1) Bounds for FULL GS + Lensing model ----
        print("Building bounds for 'GS + Lensing' model...")
        full_bounds = ps.build_combined_bounds(
            kernel_type="GS",
            kernel_bounds_dict=kernel_bounds_dict,
            lens_bounds=lens_bounds,
            mu_bounds=mu_bounds,
            include_lens_mean=1,
        )
        full_bounds = tuple(map(tuple, full_bounds))

        # ---- 2) Derive sizes ----
        nhyp = mcf.count_nhyp("GS")
        n_lens_params = 5
        ndim = nhyp + n_lens_params + 1
        print(
            f"Model: 'GS + Lensing'. nhyp={nhyp}, n_lens_params={n_lens_params}, ndim={ndim}"
        )

        # ---- 3) Walker init ----
        nwalkers = 4 * ndim
        print(f"Initializing {nwalkers} walkers (Option B: Prior-informed)...")
        p0 = mcf.init_walkers(
            n_walkers=nwalkers,
            param_bounds=full_bounds,
            kernel_type="GS",
            include_lens_mean=True,
            # --- KDE Changes ---
            use_kde=use_kde_flag,  # <<< Uses the flag from our check
            kde3d_pack=kde_output_tuple,  # <<< Passes the 6-tuple
            # --- Physical prior enabled ---
            phys_scale=True,
            T=np.asarray(T),
            y=np.asarray(y),
            cadence_s=cadence_s,
        )

        # ---- 4) Vectorized log-prob and sampler ----
        print("Building JAX-compiled log-probability function...")

        logprob = mcf.make_emcee_vectorized_logprob(
            kernel_type="GS",
            nhyp=nhyp,
            noise_measure=float(self.noise_measure),
            param_bounds=full_bounds,
            # --- KDE Changes ---
            kde3d_args=kde_output_tuple,  # <<< Pass the 6-tuple
            # --- Physical prior enabled ---
            phys_scale=True,
            include_lens_mean=True,
        )

        # --- Sampler Creation and Run ---
        sampler_lens = emcee.EnsembleSampler(  # <<< Renamed to sampler_lens
            nwalkers=nwalkers,
            ndim=ndim,
            log_prob_fn=logprob,
            vectorize=True,
            args=(T, y),
        )

        # ---- 5) Run burn-in + production ----
        nburn, nprod = 500, 2000  # Keep these shorter for testing
        print(f"Running burn-in ({nburn} steps)...")
        p0, _, _ = sampler_lens.run_mcmc(p0, nburn, progress=True)
        sampler_lens.reset()
        print(f"Running production ({nprod} steps)...")
        sampler_lens.run_mcmc(p0, nprod, progress=True)

        # ---- 6) Summaries ----
        # *** Use sampler_lens to get chains ***
        flat_chain_lens = sampler_lens.get_chain(flat=True)
        logp_lens = sampler_lens.get_log_prob(flat=True)

        print("\n--- MCMC Run Complete ---")
        print(
            f"Final acceptance fraction: {np.mean(sampler_lens.acceptance_fraction):.3f}"
        )
        print("Chain shape:", flat_chain_lens.shape)
        print(
            f"logp mean/median: {np.mean(logp_lens):.2f} / {np.median(logp_lens):.2f}"
        )

        # Unpack a quick MAP-ish estimate
        best_idx = np.argmax(logp_lens)
        theta_hat = flat_chain_lens[best_idx]
        return theta_hat, sampler_lens
        # # --- Updated printout for lensing model ---
        # print("\nBest θ (GS + Lensing + logμ):")
        # print(f"  G: log10 ν_g, log10 σ_g  = {theta_hat[0:2]}")
        # print(f"  S: log10 ν_s, Q, log10 σ = {theta_hat[2:5]}")
        # print(f"  Lens: Ms, Mco, logP, inc, delta = {theta_hat[5:10]}")
        # print(f"  log10 μ                  = {theta_hat[10]:.4f}")

        # # (Optional) sanity check:
        # print("\nRe-computing log-likelihood at best-fit θ...")
        # kernel_fn = lgp.make_kernel_fn("GS")
        # gp_hat = lgp.build_model_additive(
        #     theta=jnp.asarray(theta_hat),
        #     nhyp=nhyp,
        #     kernel_fn=kernel_fn,
        #     T=T,
        #     include_lens_mean=1,
        #     noise_measure=float(self.noise_measure),
        # )
        # print(f"log L(θ̂): {float(gp_hat.log_probability(y)):.2f}")
