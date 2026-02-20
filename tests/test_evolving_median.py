"""Tests for evolving_median weighting method."""

import pytest
from .conftest import MockPypsaNetwork, assert_nested_dict_approx

from mga4all.spores import (
    calculate_median_deployment,
    calculate_weights_evolving,
)


def test_calculate_median_deployment_odd_history(spore_techs_dict):
    """Tests that calculate_median_deployment correctly finds the median of an odd-sized history."""
    history = [
        {"Generator": {"p_nom": {"solar": 100, "wind": 400, "gas": 50}}},
        {"Generator": {"p_nom": {"solar": 200, "wind": 200, "gas": 50}}},
        {"Generator": {"p_nom": {"solar": 0, "wind": 300}}},  # Missing gas
    ]
    expected = {
        "Generator": {
            "p_nom": {
                "solar": 100,  # Median of [100, 200, 0] -> median of sorted [0, 100, 200] is 100.
                "wind": 300,  # Median of [400, 200, 300] is 300
                "gas": 50,  # Median of [50, 50, 0] is 50
            }
        }
    }

    actual = calculate_median_deployment(history, spore_techs_dict)
    assert_nested_dict_approx(actual, expected)


def test_calculate_median_deployment_even_history(spore_techs_dict):
    """Tests that calculate_median_deployment correctly averages the middle two values for an even-sized history."""
    history = [
        {"Generator": {"p_nom": {"solar": 100, "wind": 200}}},
        {"Generator": {"p_nom": {"solar": 300, "wind": 500}}},
    ]
    expected = {
        "Generator": {
            "p_nom": {
                "solar": 200,  # Median of [100, 300] is (100+300)/2 = 200
                "wind": 350,  # Median of [200, 500] is (200+500)/2 = 350
                "gas": 0,  # Median of [0, 0] is 0
            }
        }
    }

    actual = calculate_median_deployment(history, spore_techs_dict)
    assert_nested_dict_approx(actual, expected)


def test_calculate_weights_evolving_median_basic_scenario(spore_techs_dict):
    """Tests a basic case with absolute capacities using the median."""
    history = [
        {"Generator": {"p_nom": {"solar": 900, "wind": 200, "gas": 50}}},
        {"Generator": {"p_nom": {"solar": 200, "wind": 800, "gas": 150}}},
        {"Generator": {"p_nom": {"solar": 700, "wind": 300, "gas": 100}}},
    ]
    # Evolving Median: solar=700, wind=300, gas=100

    latest_deployment_data = {"solar": 600, "wind": 300, "gas": 200}
    latest_spore_mock = MockPypsaNetwork(latest_deployment_data)

    expected = {
        "Generator": {
            "p_nom": {
                "solar": pytest.approx(
                    7.0
                ),  # change = abs(600-700)/700 = 0.1428 -> weight = 1/0.1428 = 7.0
                "wind": 1000.0,  # change = 0 -> hits clip_min -> weight = 1000
                "gas": 1.0,  # change = abs(200-100)/100 = 1.0 -> weight = 1.0
            }
        }
    }

    actual = calculate_weights_evolving(
        latest_spore_mock, history, spore_techs_dict, calculate_median_deployment
    )
    assert_nested_dict_approx(actual, expected)


def test_calculate_weights_evolving_median_robust_zero_median(spore_techs_dict):
    """Tests the robust logic where median deployment is zero."""
    history = [
        {"Generator": {"p_nom": {"solar": 800, "wind": 200}}},
        {"Generator": {"p_nom": {"solar": 0, "wind": 0}}},
        {"Generator": {"p_nom": {"solar": 0, "wind": 0}}},
    ]
    # Evolving Median: solar=0, wind=0, gas=0

    latest_deployment_data = {"solar": 800, "wind": 200, "gas": 500}  # 'gas' is new
    latest_spore_mock = MockPypsaNetwork(latest_deployment_data)

    expected = {
        "Generator": {
            "p_nom": {
                "solar": 0.0,  # ROBUST logic: median=0 -> weight=0
                "wind": 0.0,  # ROBUST logic: median=0 -> weight=0
                "gas": 0.0,  # ROBUST logic: median=0 -> weight=0
            }
        }
    }

    actual = calculate_weights_evolving(
        latest_spore_mock, history, spore_techs_dict, calculate_median_deployment
    )
    assert_nested_dict_approx(actual, expected)


def test_calculate_weights_evolving_median_latest_is_zero(spore_techs_dict):
    """Tests the case where a previously used tech is not in the latest deployment."""
    history = [
        {"Generator": {"p_nom": {"solar": 1000, "wind": 500, "gas": 800}}},
        {"Generator": {"p_nom": {"solar": 200, "wind": 500, "gas": 200}}},
    ]
    # Evolving Median: solar=600, wind=500, gas=500

    latest_deployment_data = {"solar": 0, "wind": 500, "gas": 1000}  # solar is now 0
    latest_spore_mock = MockPypsaNetwork(latest_deployment_data)

    expected = {
        "Generator": {
            "p_nom": {
                "solar": 1.0,  # change = abs(0-600)/600 = 1.0 -> weight = 1.0
                "wind": 1000.0,  # change = 0 -> weight = 1000
                "gas": 1.0,  # change = abs(1000-500)/500 = 1.0 -> weight = 1.0
            }
        }
    }

    actual = calculate_weights_evolving(
        latest_spore_mock, history, spore_techs_dict, calculate_median_deployment
    )
    assert_nested_dict_approx(actual, expected)
