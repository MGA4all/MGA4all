"""Tests for calculate_weights_random."""

import pandas as pd

from mga4all.spores import set_weights_random


def assert_random_weights_properties(result, upper_bound):
    """Asserts that a dictionary of random weights has the correct properties.

    2.  Checks that the final values are floats.
    3.  Checks that the final values are within the range [0, upper_bound].
    """
    assert result.dtype == "float64"
    assert (0 < result).all()
    assert (result < upper_bound).all()


def test_calculate_weights_random_basic_case(spore_tech_indices):
    """Tests the properties of the output from calculate_weights_random with a standard upper bound."""
    upper_bound = 100

    # Generate the random weights
    actual_weights = set_weights_random(spore_tech_indices, upper_bound)

    # Assert that the output has the correct structure, types, and value ranges
    assert_random_weights_properties(actual_weights, upper_bound)


def test_calculate_weights_random_upper_bound_one(spore_tech_indices):
    """Tests the properties of the output when the upper bound is 1."""
    upper_bound = 1
    actual_weights = set_weights_random(spore_tech_indices, upper_bound)

    assert_random_weights_properties(actual_weights, upper_bound)


def test_calculate_weights_random_is_not_deterministic(spore_tech_indices):
    """Tests that two consecutive calls produce different results, proving it's random."""
    upper_bound = 100

    # Call the function twice
    result1 = set_weights_random(spore_tech_indices, upper_bound)
    result2 = set_weights_random(spore_tech_indices, upper_bound)

    # It's astronomically unlikely they will be identical.
    # This confirms the function is not returning the same numbers every time.
    assert not result1.equals(result2)


def test_calculate_weights_random_with_more_complex_structure():
    """Tests that the property-checking works with a more complex input dictionary."""
    upper_bound = 50
    complex_techs = pd.MultiIndex.from_tuples(
        [
            ("Generator", "p_nom", "solar"),
            ("Generator", "p_nom", "wind"),
            ("Generator", "p_nom", "gas"),
            ("Store", "e_nom", "battery"),
        ],
        names=["component", "attribute", "asset"],
    )

    actual_weights = set_weights_random(complex_techs, upper_bound)
    assert_random_weights_properties(actual_weights, upper_bound)
