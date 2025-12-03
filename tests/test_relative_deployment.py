"""Tests for calculate_weights_relative_deployment and calculate_weights_relative_deployment_normalized."""

from .conftest import MockPypsaNetwork, assert_nested_dict_approx

from mga4all.spores import (
    calculate_weights_relative_deployment,
    calculate_weights_relative_deployment_normalized,
)


def test_relative_deployment_first_iteration(spore_techs_dict):
    """Tests the cumulative method on the first iteration where prev_weights are zero."""
    # On the first iteration, prev_weights are all zero.
    prev_weights = spore_techs_dict

    # The least-cost solution is represented by this mock network where p_nom_max defaults to 1.0
    latest_spore_mock = MockPypsaNetwork(
        p_nom_opt_data={"solar": 0.8, "wind": 0.2, "gas": 0.0}
    )

    # Expected output is simply the relative deployment of the least-cost solution.
    expected = {
        "Generator": {
            "p_nom": {
                "solar": 0.8,
                "wind": 0.2,
                "gas": 0.0,
            }
        }
    }

    actual = calculate_weights_relative_deployment(latest_spore_mock, prev_weights)
    assert_nested_dict_approx(actual, expected)


def test_relative_deployment_subsequent_iteration(spore_techs_dict):
    """Tests the cumulative sum on a subsequent iteration with non-zero previous weights."""
    # Previous weights from a prior step.
    prev_weights = {"Generator": {"p_nom": {"solar": 0.8, "wind": 0.2, "gas": 0.0}}}

    # The new spore found has a different deployment.
    latest_spore_mock = MockPypsaNetwork(
        p_nom_opt_data={"solar": 0.1, "wind": 0.7, "gas": 0.5}
    )

    # Expected output is the element-wise sum of prev_weights + new deployment.
    expected = {
        "Generator": {
            "p_nom": {
                "solar": 0.8 + 0.1,  # 0.9
                "wind": 0.2 + 0.7,  # 0.9
                "gas": 0.0 + 0.5,  # 0.5
            }
        }
    }

    actual = calculate_weights_relative_deployment(latest_spore_mock, prev_weights)
    assert_nested_dict_approx(actual, expected)


def test_relative_deployment_respects_p_nom_max(spore_techs_dict):
    """Tests that the calculation correctly uses relative deployment (opt/max)."""
    prev_weights = {"Generator": {"p_nom": {"solar": 0.1, "wind": 0.9, "gas": 0.3}}}

    # Define both opt and max to calculate relative deployment.
    latest_spore_mock = MockPypsaNetwork(
        # FIX: Added 'gas': 0.0 to make the data consistent.
        p_nom_opt_data={"solar": 1.0, "wind": 0.5, "gas": 0.0},
        p_nom_max_data={"solar": 2.0, "wind": 4.0, "gas": 1.0},
    )
    # Relative deployments are now: solar = 1.0/2.0=0.5, wind = 0.5/4.0=0.125, gas=0.0/1.0=0.0

    expected = {
        "Generator": {
            "p_nom": {
                "solar": 0.1 + 0.5,  # 0.6
                "wind": 0.9 + 0.125,  # 1.025
                "gas": 0.3 + 0.0,  # 0.3
            }
        }
    }

    actual = calculate_weights_relative_deployment(latest_spore_mock, prev_weights)
    assert_nested_dict_approx(actual, expected)


def test_calculate_weights_relative_deployment_normalized_basic_normalization():
    """Tests that the cumulative weights are correctly normalized."""
    prev_weights = {"Generator": {"p_nom": {"solar": 0.5, "wind": 0.1, "gas": 0.0}}}
    latest_spore_mock = MockPypsaNetwork(
        p_nom_opt_data={"solar": 0.3, "wind": 0.7, "gas": 0.0}
    )
    # Cumulative sum = {'solar': 0.8, 'wind': 0.8, 'gas': 0.0}
    # Max weight = 0.8

    expected = {
        "Generator": {
            "p_nom": {
                "solar": 0.8 / 0.8,  # 1.0
                "wind": 0.8 / 0.8,  # 1.0
                "gas": 0.0 / 0.8,  # 0.0
            }
        }
    }

    actual = calculate_weights_relative_deployment_normalized(
        latest_spore_mock, prev_weights
    )
    assert_nested_dict_approx(actual, expected)


def test_calculate_weights_relative_deployment_normalized_max_value_changes():
    """Tests normalization when a new technology deployment creates a new maximum weight."""
    prev_weights = {
        "Generator": {"p_nom": {"solar": 1.0, "wind": 0.8, "gas": 0.2}}  # Max is 1.0
    }
    latest_spore_mock = MockPypsaNetwork(
        p_nom_opt_data={"solar": 0, "wind": 0, "gas": 1.0}
    )
    # Cumulative sum = {'solar': 1.0, 'wind': 0.8, 'gas': 1.2}, and new max weight = 1.2

    expected = {
        "Generator": {
            "p_nom": {
                "solar": 1.0 / 1.2,  # ~0.8333
                "wind": 0.8 / 1.2,  # ~0.6667
                "gas": 1.2 / 1.2,  # 1.0
            }
        }
    }

    actual = calculate_weights_relative_deployment_normalized(
        latest_spore_mock, prev_weights
    )
    assert_nested_dict_approx(actual, expected)


def test_calculate_weights_relative_deployment_normalized_all_zero_case(
    spore_techs_dict,
):
    """Tests that the function handles the case of all-zero weights without a ZeroDivisionError."""
    prev_weights = spore_techs_dict  # All zeros
    latest_spore_mock = MockPypsaNetwork(
        p_nom_opt_data={"solar": 0.0, "wind": 0.0, "gas": 0.0}
    )
    # Cumulative sum is all zeros, max weight is 0.

    # The function should not normalize and just return zeros.
    expected = {"Generator": {"p_nom": {"solar": 0.0, "wind": 0.0, "gas": 0.0}}}

    actual = calculate_weights_relative_deployment_normalized(
        latest_spore_mock, prev_weights
    )
    assert_nested_dict_approx(actual, expected)
