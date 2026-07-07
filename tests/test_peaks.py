import unittest

import numpy as np

from selflensing.peaks import PeakDetection, dedupe_overlapping_peaks, detect_peaks


class DetectPeaksTest(unittest.TestCase):
    def test_dedupes_overlapping_peaks_by_width(self):
        narrow = PeakDetection(
            index=10,
            time=1.0,
            width=0.1,
            amplitude=0.2,
            amplitude_err=0.01,
            amplitude_snr=20.0,
            delta_chi2=100.0,
            p_value=0.0,
            baseline_intercept=1.0,
            baseline_slope=0.0,
            window_start=0.8,
            window_end=1.2,
            n_points=20,
        )
        wide = PeakDetection(
            index=12,
            time=1.1,
            width=0.2,
            amplitude=0.18,
            amplitude_err=0.01,
            amplitude_snr=18.0,
            delta_chi2=80.0,
            p_value=0.0,
            baseline_intercept=1.0,
            baseline_slope=0.0,
            window_start=0.7,
            window_end=1.5,
            n_points=40,
        )
        separate = PeakDetection(
            index=50,
            time=5.0,
            width=0.15,
            amplitude=0.2,
            amplitude_err=0.01,
            amplitude_snr=20.0,
            delta_chi2=90.0,
            p_value=0.0,
            baseline_intercept=1.0,
            baseline_slope=0.0,
            window_start=4.8,
            window_end=5.2,
            n_points=20,
        )

        detections = dedupe_overlapping_peaks([narrow, separate, wide])

        self.assertEqual(detections, [wide, separate])

    def test_detects_peak_on_slanted_baseline(self):
        rng = np.random.default_rng(42)
        time = np.linspace(0.0, 10.0, 501)
        sigma = np.full_like(time, 0.02)
        center = 5.2
        width = 0.18
        baseline = 1.0 + 0.04 * (time - center)
        peak = 0.25 * np.exp(-0.5 * ((time - center) / width) ** 2)
        flux = baseline + peak + rng.normal(0.0, sigma)

        detections = detect_peaks(
            time,
            flux,
            sigma,
            widths=np.array([0.12, 0.18, 0.24]),
            candidate_prominence=0.08,
            min_delta_chi2=25.0,
            min_amplitude_snr=5.0,
            max_peaks=1,
        )

        self.assertEqual(len(detections), 1)
        self.assertAlmostEqual(detections[0].time, center, delta=0.08)
        self.assertAlmostEqual(detections[0].width, width, delta=0.07)
        self.assertGreater(detections[0].delta_chi2, 25.0)
        self.assertGreater(detections[0].amplitude_snr, 5.0)

    def test_rejects_slanted_baseline_without_peak(self):
        rng = np.random.default_rng(43)
        time = np.linspace(0.0, 10.0, 501)
        sigma = np.full_like(time, 0.02)
        flux = 1.0 + 0.04 * (time - 5.0) + rng.normal(0.0, sigma)

        detections = detect_peaks(
            time,
            flux,
            sigma,
            widths=np.array([0.12, 0.18, 0.24]),
            candidate_prominence=0.08,
            min_delta_chi2=25.0,
            min_amplitude_snr=5.0,
        )

        self.assertEqual(detections, [])


if __name__ == "__main__":
    unittest.main()
