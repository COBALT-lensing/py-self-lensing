import unittest

import numpy as np

from selflensing.matched_filtering import (
    MatchedFilterTemplate,
    detect_matched_filter_peaks,
)


class MatchedFilteringTest(unittest.TestCase):
    def test_detects_template_and_returns_light_curve_window(self):
        time = np.arange(200.0)
        template_time = np.arange(-20.0, 20.5, 0.5)
        template_flux = np.exp(-0.5 * (template_time / 4.0) ** 2)
        resampled_template_time = np.arange(-20.0, 21.0)
        resampled_template_flux = np.interp(
            resampled_template_time,
            template_time,
            template_flux,
        )
        flux = np.ones_like(time)
        start_index = 80
        flux[
            start_index : start_index + len(resampled_template_flux)
        ] += 0.2 * resampled_template_flux

        matches = detect_matched_filter_peaks(
            time,
            flux,
            [
                MatchedFilterTemplate(
                    template_flux,
                    template_time,
                )
            ],
            prominence=0.1,
        )

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].template_index, 0)
        self.assertEqual(matches[0].start_index, start_index)
        self.assertEqual(matches[0].peak_index, 100)
        self.assertEqual(matches[0].end_index, 120)
        self.assertEqual(matches[0].start_time, time[start_index])
        self.assertEqual(matches[0].peak_time, time[100])
        self.assertEqual(matches[0].end_time, time[120])

    def test_limits_matches_by_score_across_template_bank(self):
        time = np.arange(200.0)
        template_time = np.arange(-10.0, 11.0)
        narrow = np.exp(-0.5 * (template_time / 2.0) ** 2)
        wide = np.exp(-0.5 * (template_time / 5.0) ** 2)
        flux = np.ones_like(time)
        flux[90 : 90 + len(narrow)] += 0.2 * narrow

        matches = detect_matched_filter_peaks(
            time,
            flux,
            [
                MatchedFilterTemplate(wide, template_time),
                MatchedFilterTemplate(narrow, template_time),
            ],
            prominence=0.1,
            min_score_sigma=None,
            max_matches=1,
        )

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].template_index, 1)

    def test_dedupes_overlapping_template_matches_by_default(self):
        time = np.arange(200.0)
        template_time = np.arange(-10.0, 11.0)
        narrow = np.exp(-0.5 * (template_time / 2.0) ** 2)
        wide = np.exp(-0.5 * (template_time / 5.0) ** 2)
        flux = np.ones_like(time)
        flux[90 : 90 + len(narrow)] += 0.2 * narrow

        matches = detect_matched_filter_peaks(
            time,
            flux,
            [
                MatchedFilterTemplate(wide, template_time),
                MatchedFilterTemplate(narrow, template_time),
            ],
            prominence=0.1,
            min_score_sigma=None,
        )

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].template_index, 1)

    def test_can_return_overlapping_template_matches(self):
        time = np.arange(200.0)
        template_time = np.arange(-10.0, 11.0)
        narrow = np.exp(-0.5 * (template_time / 2.0) ** 2)
        wide = np.exp(-0.5 * (template_time / 5.0) ** 2)
        flux = np.ones_like(time)
        flux[90 : 90 + len(narrow)] += 0.2 * narrow

        matches = detect_matched_filter_peaks(
            time,
            flux,
            [
                MatchedFilterTemplate(wide, template_time),
                MatchedFilterTemplate(narrow, template_time),
            ],
            prominence=0.1,
            min_score_sigma=None,
            dedupe_overlaps=False,
        )

        self.assertEqual(len(matches), 2)

    def test_rejects_template_longer_than_light_curve(self):
        with self.assertRaisesRegex(ValueError, "longer than the light curve"):
            detect_matched_filter_peaks(
                np.arange(5.0),
                np.ones(5),
                [MatchedFilterTemplate(np.arange(6.0), np.arange(6.0))],
            )

    def test_default_score_threshold_rejects_low_absolute_score_peak(self):
        time = np.arange(200.0)
        template_time = np.arange(-10.0, 11.0)
        template_flux = np.exp(-0.5 * (template_time / 2.0) ** 2)
        flux = np.ones_like(time)
        flux[90 : 90 + len(template_flux)] += 0.01 * template_flux

        matches = detect_matched_filter_peaks(
            time,
            flux,
            [MatchedFilterTemplate(template_flux, template_time)],
            prominence=0.001,
        )

        self.assertEqual(matches, [])

    def test_can_disable_default_score_threshold(self):
        time = np.arange(200.0)
        template_time = np.arange(-10.0, 11.0)
        template_flux = np.exp(-0.5 * (template_time / 2.0) ** 2)
        flux = np.ones_like(time)
        flux[90 : 90 + len(template_flux)] += 0.01 * template_flux

        matches = detect_matched_filter_peaks(
            time,
            flux,
            [MatchedFilterTemplate(template_flux, template_time)],
            prominence=0.001,
            min_score_sigma=None,
        )

        self.assertEqual(len(matches), 1)


if __name__ == "__main__":
    unittest.main()
