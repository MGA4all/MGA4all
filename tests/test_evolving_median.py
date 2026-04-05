"""Tests for evolving_median weighting method."""

import pandas as pd
from .conftest import MockPypsaNetwork

from mga4all.spores import (
    median_deployment,
    calculate_weights_evolving,
)


def test_calculate_median_deployment_odd_history(spore_tech_indices):
    """Tests that calculate_median_deployment correctly finds the median of an odd-sized history."""
    history = [
        pd.Series([100, 400, 50], index=spore_tech_indices),
        pd.Series([200, 200, 50], index=spore_tech_indices),
        pd.Series([0, 300, 0], index=spore_tech_indices),
        # {"Generator": {"p_nom": {"solar": 100, "wind": 400, "gas": 50}}},
        # {"Generator": {"p_nom": {"solar": 200, "wind": 200, "gas": 50}}},
        # {"Generator": {"p_nom": {"solar": 0, "wind": 300}}},  # Missing gas
    ]
    expected = pd.Series(
        [
            100.0,  # Median of [100, 200, 0] -> median of sorted [0, 100, 200] is 100.
            300.0,  # Median of [400, 200, 300] is 300
            50.0,  # Median of [50, 50, 0] is 50
        ],
        index=spore_tech_indices,
    )

    actual = median_deployment(history)
    pd.testing.assert_series_equal(actual, expected)


def test_calculate_median_deployment_even_history(spore_tech_indices):
    """Tests that calculate_median_deployment correctly averages the middle two values for an even-sized history."""
    history = [
        pd.Series([100, 200, 0], index=spore_tech_indices),
        pd.Series([300, 500, 0], index=spore_tech_indices),
        # {"Generator": {"p_nom": {"solar": 100, "wind": 200}}},
        # {"Generator": {"p_nom": {"solar": 300, "wind": 500}}},
    ]
    expected = pd.Series(
        [
            200.0,  # Median of [100, 300] is (100+300)/2 = 200
            350.0,  # Median of [200, 500] is (200+500)/2 = 350
            0.0,  # Median of [0, 0] is 0
        ],
        index=spore_tech_indices,
    )

    actual = median_deployment(history)
    pd.testing.assert_series_equal(actual, expected)


def test_calculate_weights_evolving_median_basic_scenario(spore_tech_indices):
    """Tests a basic case with absolute capacities using the median."""
    history = [
        pd.Series([900, 200, 50], index=spore_tech_indices),
        pd.Series([200, 800, 150], index=spore_tech_indices),
        pd.Series([700, 300, 100], index=spore_tech_indices),
    ]
    # Evolving Median: solar=700, wind=300, gas=100

    latest_deployment_data = {"solar": 600, "wind": 300, "gas": 200}
    latest_spore_mock = MockPypsaNetwork(latest_deployment_data)

    expected = pd.Series(
        [
            7.0,  # change = abs(600-700)/700 = 0.1428 -> weight = 1/0.1428 = 7.0
            1000.0,  # change = 0 -> hits clip_min -> weight = 1000
            1.0,  # change = abs(200-100)/100 = 1.0 -> weight = 1.0
        ],
        index=spore_tech_indices,
    )

    actual = calculate_weights_evolving(
        latest_spore_mock, history, spore_tech_indices, median_deployment
    )
    pd.testing.assert_series_equal(actual, expected)


def test_calculate_weights_evolving_median_robust_zero_median(spore_tech_indices):
    """Tests the robust logic where median deployment is zero."""
    history = [
        pd.Series([800, 200, 0], index=spore_tech_indices),
        pd.Series([0, 0, 0], index=spore_tech_indices),
        pd.Series([0, 0, 0], index=spore_tech_indices),
        # {"Generator": {"p_nom": {"solar": 800, "wind": 200}}},
        # {"Generator": {"p_nom": {"solar": 0, "wind": 0}}},
        # {"Generator": {"p_nom": {"solar": 0, "wind": 0}}},
    ]
    # Evolving Median: solar=0, wind=0, gas=0

    latest_deployment_data = {"solar": 800, "wind": 200, "gas": 500}  # 'gas' is new
    latest_spore_mock = MockPypsaNetwork(latest_deployment_data)

    expected = pd.Series(
        [
            0.0,  # ROBUST logic: median=0 -> weight=0
            0.0,  # ROBUST logic: median=0 -> weight=0
            0.0,  # ROBUST logic: median=0 -> weight=0
        ],
        index=spore_tech_indices,
    )

    actual = calculate_weights_evolving(
        latest_spore_mock, history, spore_tech_indices, median_deployment
    )
    pd.testing.assert_series_equal(actual, expected)


def test_calculate_weights_evolving_median_latest_is_zero(spore_tech_indices):
    """Tests the case where a previously used tech is not in the latest deployment."""
    history = [
        pd.Series([1000, 500, 800], index=spore_tech_indices),
        pd.Series([200, 500, 200], index=spore_tech_indices),
    ]
    # Evolving Median: solar=600, wind=500, gas=500

    latest_deployment_data = {"solar": 0, "wind": 500, "gas": 1000}  # solar is now 0
    latest_spore_mock = MockPypsaNetwork(latest_deployment_data)

    expected = pd.Series(
        [
            1.0,  # change = abs(0-600)/600 = 1.0 -> weight = 1.0
            1000.0,  # change = 0 -> weight = 1000
            1.0,  # change = abs(1000-500)/500 = 1.0 -> weight = 1.0
        ],
        index=spore_tech_indices,
    )

    actual = calculate_weights_evolving(
        latest_spore_mock, history, spore_tech_indices, median_deployment
    )
    pd.testing.assert_series_equal(actual, expected)
