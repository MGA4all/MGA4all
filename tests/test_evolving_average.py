"""Tests for evolving_average weighting method."""

from .conftest import MockPypsaNetwork, assert_nested_dict_approx

from mga4all.spores import (
    calculate_average_deployment,
    calculate_weights_evolving_average,
    get_tech_deployment,
)


def test_get_tech_deployment(spore_techs_dict):
    """Tests that get_tech_deployment correctly extracts absolute p_nom_opt values."""
    deployment_data = {"solar": 150.5, "wind": 300.0, "gas": 50.2}
    network_mock = MockPypsaNetwork(deployment_data)

    expected = {"Generator": {"p_nom": deployment_data}}

    actual = get_tech_deployment(network_mock, spore_techs_dict)
    assert_nested_dict_approx(actual, expected)


def test_calculate_average_deployment(spore_techs_dict):
    """Tests that calculate_average_deployment correctly averages absolute capacities."""
    history = [
        {"Generator": {"p_nom": {"solar": 100, "wind": 200, "gas": 50}}},
        {"Generator": {"p_nom": {"solar": 200, "wind": 400, "gas": 50}}},
        {"Generator": {"p_nom": {"solar": 0, "wind": 300}}},  # Missing gas
    ]
    expected = {
        "Generator": {
            "p_nom": {
                "solar": (100 + 200 + 0) / 3,
                "wind": (200 + 400 + 300) / 3,
                "gas": (50 + 50 + 0) / 3,  # Handles missing key
            }
        }
    }

    actual = calculate_average_deployment(history, spore_techs_dict)
    assert_nested_dict_approx(actual, expected)


def test_evolving_average_basic_scenario_absolute(spore_techs_dict):
    """Tests a basic case with absolute capacities."""
    history = [
        {"Generator": {"p_nom": {"solar": 800, "wind": 200, "gas": 100}}},
        {"Generator": {"p_nom": {"solar": 200, "wind": 800, "gas": 100}}},
    ]
    # Evolving Average: solar=500, wind=500, gas=100

    latest_deployment_data = {"solar": 400, "wind": 500, "gas": 200}
    latest_spore_mock = MockPypsaNetwork(latest_deployment_data)

    expected = {
        "Generator": {
            "p_nom": {
                "solar": 5.0,  # 1 / abs(400 - 500) / 500 = 1 / 0.2 = 5.0
                "wind": 1000.0,  # 1 / abs(500 - 500) / 500 = 1 / max(0, 0.001) = 1000.0
                "gas": 1.0,  # 1 / abs(200 - 100) / 100 = 1 / 1.0 -> weight = 1.0
            }
        }
    }
    # Correcting solar calculation
    expected["Generator"]["p_nom"]["solar"] = 5.0

    actual = calculate_weights_evolving_average(
        latest_spore_mock, history, spore_techs_dict
    )
    assert_nested_dict_approx(actual, expected)


def test_evolving_average_robust_zero_average_absolute(spore_techs_dict):
    """Tests the robust logic where average deployment is zero."""
    history = [{"Generator": {"p_nom": {"solar": 800, "wind": 200}}}]
    # Evolving Average: solar=800, wind=200, gas=0

    latest_deployment_data = {"solar": 800, "wind": 200, "gas": 500}  # 'gas' is new
    latest_spore_mock = MockPypsaNetwork(latest_deployment_data)

    expected = {
        "Generator": {
            "p_nom": {
                "solar": 1000.0,  # 1 / abs(800 - 800) / 800 = 1 / max(0, 0.001) = 1000.0
                "wind": 1000.0,  # 1 / abs(200 - 200) / 200 = 1 / max(0, 0.001) = 1000.0
                "gas": 0.0,  # ROBUST logic: if average_deployed_cap = 0, set weight = 0 to encourage deployment.
            }
        }
    }

    actual = calculate_weights_evolving_average(
        latest_spore_mock, history, spore_techs_dict
    )
    assert_nested_dict_approx(actual, expected)


def test_evolving_average_latest_is_zero_absolute(spore_techs_dict):
    """Tests the case where a previously used tech is not in the latest deployment."""
    history = [
        {"Generator": {"p_nom": {"solar": 1000, "wind": 500, "gas": 200}}},
        {"Generator": {"p_nom": {"solar": 0, "wind": 500, "gas": 800}}},
    ]
    # Evolving Average: solar=500, wind=500, gas=500

    latest_deployment_data = {"solar": 0, "wind": 500, "gas": 1000}  # solar is now 0
    latest_spore_mock = MockPypsaNetwork(latest_deployment_data)

    expected = {
        "Generator": {
            "p_nom": {
                "solar": 1.0,  # 1 / abs(0 - 500) / 500 = 1.0
                "wind": 1000.0,  # 1 / abs(500 - 500) / 500 = 1 / max(0, 0.001) = 1000.0
                "gas": 1.0,  # 1 / abs(1000 - 500) / 500 = 1.0
            }
        }
    }

    actual = calculate_weights_evolving_average(
        latest_spore_mock, history, spore_techs_dict
    )
    assert_nested_dict_approx(actual, expected)
