"""Tests for evolving_median weighting method."""

import pandas as pd

from mga4all.spores import calculate_weights_evolving


def test_calculate_weights_evolving_median_basic_scenario(asset_indices):
    """Tests a basic case with absolute capacities using the median."""
    latest_deployment = pd.Series([600, 300, 200], index=asset_indices)
    median_deployment = pd.Series([700, 300, 100], index=asset_indices)

    expected = pd.Series(
        [
            7.0,  # change = abs(600-700)/700 = 0.1428 -> weight = 1/0.1428 = 7.0
            1000.0,  # change = 0 -> hits clip_min -> weight = 1000
            1.0,  # change = abs(200-100)/100 = 1.0 -> weight = 1.0
        ],
        index=asset_indices,
    )

    actual = calculate_weights_evolving(latest_deployment, median_deployment)
    pd.testing.assert_series_equal(actual, expected)


def test_calculate_weights_evolving_median_robust_zero_median(asset_indices):
    """Tests the robust logic where median deployment is zero."""
    latest_deployment = pd.Series([800, 200, 500], index=asset_indices)
    median_deployment = pd.Series([0, 0, 0], index=asset_indices)

    expected = pd.Series(
        [
            0.0,  # ROBUST logic: median=0 -> weight=0
            0.0,  # ROBUST logic: median=0 -> weight=0
            0.0,  # ROBUST logic: median=0 -> weight=0
        ],
        index=asset_indices,
    )

    actual = calculate_weights_evolving(latest_deployment, median_deployment)
    pd.testing.assert_series_equal(actual, expected)


def test_calculate_weights_evolving_median_latest_is_zero(asset_indices):
    """Tests the case where a previously used tech is not in the latest deployment."""
    latest_deployment = pd.Series([0, 500, 1000], index=asset_indices)
    median_deployment = pd.Series([600, 500, 500], index=asset_indices)

    expected = pd.Series(
        [
            1.0,  # change = abs(0-600)/600 = 1.0 -> weight = 1.0
            1000.0,  # change = 0 -> weight = 1000
            1.0,  # change = abs(1000-500)/500 = 1.0 -> weight = 1.0
        ],
        index=asset_indices,
    )

    actual = calculate_weights_evolving(latest_deployment, median_deployment)
    pd.testing.assert_series_equal(actual, expected)
