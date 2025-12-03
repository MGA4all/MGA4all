"""Tests for calculate_weights_random."""

from mga4all.spores import calculate_weights_random


def assert_random_weights_properties(result_dict, template_dict, upper_bound):
    """Asserts that a dictionary of random weights has the correct properties.

    1.  Checks that the structure (keys) matches the template.
    2.  Checks that the final values are floats.
    3.  Checks that the final values are within the range [0, upper_bound].
    """
    # 1. Check that the keys at the current level match
    assert result_dict.keys() == template_dict.keys(), "Dictionary keys do not match"

    for key, value in result_dict.items():
        if isinstance(value, dict):
            # If the value is a dictionary, recurse to check the next level
            assert_random_weights_properties(value, template_dict[key], upper_bound)
        else:
            # 2. Check that the final value is a float
            assert isinstance(value, float), f"Value for key '{key}' is not a float"
            # 3. Check that the value is within the specified range
            assert (
                0 <= value <= upper_bound
            ), f"Value {value} is out of range [0, {upper_bound}]"


def test_calculate_weights_random_basic_case(spore_techs_dict):
    """Tests the properties of the output from calculate_weights_random with a standard upper bound."""
    upper_bound = 100

    # Generate the random weights
    actual_weights = calculate_weights_random(spore_techs_dict, upper_bound)

    # Assert that the output has the correct structure, types, and value ranges
    assert_random_weights_properties(actual_weights, spore_techs_dict, upper_bound)


def test_calculate_weights_random_upper_bound_one(spore_techs_dict):
    """Tests the properties of the output when the upper bound is 1."""
    upper_bound = 1
    actual_weights = calculate_weights_random(spore_techs_dict, upper_bound)

    assert_random_weights_properties(actual_weights, spore_techs_dict, upper_bound)


def test_calculate_weights_random_is_not_deterministic(spore_techs_dict):
    """Tests that two consecutive calls produce different results, proving it's random."""
    upper_bound = 100

    # Call the function twice
    result1 = calculate_weights_random(spore_techs_dict, upper_bound)
    result2 = calculate_weights_random(spore_techs_dict, upper_bound)

    # It's astronomically unlikely they will be identical.
    # This confirms the function is not returning the same numbers every time.
    assert result1 != result2


def test_calculate_weights_random_with_more_complex_structure():
    """Tests that the property-checking works with a more complex input dictionary."""
    upper_bound = 50
    complex_techs_dict = {
        "Generator": {"p_nom": {"solar": 0, "wind": 0, "gas": 0}},
        "Store": {"e_nom": {"battery": 0}},
    }

    actual_weights = calculate_weights_random(complex_techs_dict, upper_bound)
    assert_random_weights_properties(actual_weights, complex_techs_dict, upper_bound)
