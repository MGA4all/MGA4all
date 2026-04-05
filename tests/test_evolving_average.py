"""Tests for evolving_average weighting method."""

import pandas as pd

from .conftest import MockPypsaNetwork

from mga4all.spores import (
    average_deployment,
    calculate_weights_evolving,
    get_tech_deployment,
)


def test_get_tech_deployment(spore_tech_indices):
    """Tests that get_tech_deployment correctly extracts absolute p_nom_opt values."""
    deployment_data = {"solar": 150.5, "wind": 300.0, "gas": 50.2}
    network_mock = MockPypsaNetwork(deployment_data)

    expected = pd.Series(
        deployment_data.values(), index=spore_tech_indices, name="deployment"
    )

    actual = get_tech_deployment(network_mock, spore_tech_indices)
    pd.testing.assert_series_equal(actual, expected)


def test_calculate_average_deployment(spore_tech_indices):
    """Tests that calculate_average_deployment correctly averages absolute capacities."""
    history = [
        pd.Series([100, 200, 50], index=spore_tech_indices),
        pd.Series([100, 400, 50], index=spore_tech_indices),
        pd.Series([100, 300, 0], index=spore_tech_indices),
        # {"Generator": {"p_nom": {"solar": 100, "wind": 200, "gas": 50}}},
        # {"Generator": {"p_nom": {"solar": 200, "wind": 400, "gas": 50}}},
        # {"Generator": {"p_nom": {"solar": 0, "wind": 300}}},  # Missing gas
    ]
    expected = pd.Series(
        [
            (100 + 200 + 0) / 3,
            (200 + 400 + 300) / 3,
            (50 + 50 + 0) / 3,
        ],  # Handles missing key
        index=spore_tech_indices,
    )

    actual = average_deployment(history)
    pd.testing.assert_series_equal(actual, expected)


def test_evolving_average_basic_scenario_absolute(spore_tech_indices):
    """Tests a basic case with absolute capacities."""
    history = [
        pd.Series([800, 200, 100], index=spore_tech_indices),
        pd.Series([200, 800, 100], index=spore_tech_indices),
    ]
    # Evolving Average: solar=500, wind=500, gas=100

    latest_deployment_data = {"solar": 400, "wind": 500, "gas": 200}
    latest_spore_mock = MockPypsaNetwork(latest_deployment_data)

    expected = pd.Series(
        [
            5.0,  # 1 / abs(400 - 500) / 500 = 1 / 0.2 = 5.0
            1000.0,  # 1 / abs(500 - 500) / 500 = 1 / max(0, 0.001) = 1000.0
            1.0,  # 1 / abs(200 - 100) / 100 = 1 / 1.0 -> weight = 1.0
        ],
        index=spore_tech_indices,
    )

    actual = calculate_weights_evolving(
        latest_spore_mock, history, spore_tech_indices, average_deployment
    )
    pd.testing.assert_series_equal(actual, expected)


def test_evolving_average_robust_zero_average_absolute(spore_tech_indices):
    """Tests the robust logic where average deployment is zero."""
    history = [
        pd.Series([800, 200, 0], index=spore_tech_indices),
    ]
    # Evolving Average: solar=800, wind=200, gas=0

    latest_deployment_data = {"solar": 800, "wind": 200, "gas": 500}  # 'gas' is new
    latest_spore_mock = MockPypsaNetwork(latest_deployment_data)

    expected = pd.Series(
        [
            1000.0,  # 1 / abs(800 - 800) / 800 = 1 / max(0, 0.001) = 1000.0
            1000.0,  # 1 / abs(200 - 200) / 200 = 1 / max(0, 0.001) = 1000.0
            0.0,  # ROBUST logic: if average_deployed_cap = 0, set weight = 0 to encourage deployment.
        ],
        index=spore_tech_indices,
    )

    actual = calculate_weights_evolving(
        latest_spore_mock, history, spore_tech_indices, average_deployment
    )
    pd.testing.assert_series_equal(actual, expected)


def test_evolving_average_latest_is_zero_absolute(spore_tech_indices):
    """Tests the case where a previously used tech is not in the latest deployment."""
    history = [
        pd.Series([1000, 500, 200], index=spore_tech_indices),
        pd.Series([0, 500, 800], index=spore_tech_indices),
    ]
    # Evolving Average: solar=500, wind=500, gas=500

    latest_deployment_data = {"solar": 0, "wind": 500, "gas": 1000}  # solar is now 0
    latest_spore_mock = MockPypsaNetwork(latest_deployment_data)

    expected = pd.Series(
        [
            1.0,  # 1 / abs(0 - 500) / 500 = 1.0
            1000.0,  # 1 / abs(500 - 500) / 500 = 1 / max(0, 0.001) = 1000.0
            1.0,  # 1 / abs(1000 - 500) / 500 = 1.0
        ],
        index=spore_tech_indices,
    )

    actual = calculate_weights_evolving(
        latest_spore_mock, history, spore_tech_indices, average_deployment
    )
    pd.testing.assert_series_equal(actual, expected)
