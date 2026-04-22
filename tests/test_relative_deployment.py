"""Tests for calculate_weights_relative_deployment and calculate_weights_relative_deployment_normalized."""

import pandas as pd

from mga4all.spores import (
    initialize_weights,
    calculate_weights_relative_deployment,
)


def test_relative_deployment_first_iteration(asset_indices):
    """Tests the cumulative method on the first iteration where prev_weights are zero."""
    # On the first iteration, prev_weights are all zero.
    relative_deployment = pd.Series([0.8, 0.2, 0.0], index=asset_indices)
    prev_weights = initialize_weights(asset_indices)

    # Expected output is simply the relative deployment of the least-cost solution.
    expected = relative_deployment

    actual = calculate_weights_relative_deployment(relative_deployment, prev_weights)
    pd.testing.assert_series_equal(actual, expected)


def test_relative_deployment_subsequent_iteration(asset_indices):
    """Tests the cumulative sum on a subsequent iteration with non-zero previous weights."""
    relative_deployment = pd.Series([0.1, 0.7, 0.5], index=asset_indices)
    # Previous weights from a prior step.
    prev_weights = pd.Series([0.8, 0.2, 0.0], index=asset_indices)

    # Expected output is the element-wise sum of prev_weights + new deployment.
    expected = pd.Series(
        [
            0.8 + 0.1,  # 0.9
            0.2 + 0.7,  # 0.9
            0.0 + 0.5,  # 0.5
        ],
        index=asset_indices,
    )

    actual = calculate_weights_relative_deployment(relative_deployment, prev_weights)
    pd.testing.assert_series_equal(actual, expected)


def test_relative_deployment_respects_p_nom_max(asset_indices):
    """Tests that the calculation correctly uses relative deployment (opt/max)."""
    relative_deployment = pd.Series([0.5, 0.125, 0.0], index=asset_indices)
    prev_weights = pd.Series([0.1, 0.9, 0.3], index=asset_indices)

    expected = pd.Series(
        [
            0.1 + 0.5,  # 0.6
            0.9 + 0.125,  # 1.025
            0.3 + 0.0,  # 0.3
        ],
        index=asset_indices,
    )

    actual = calculate_weights_relative_deployment(relative_deployment, prev_weights)
    pd.testing.assert_series_equal(actual, expected)


def test_calculate_weights_relative_deployment_normalized_basic_normalization(
    asset_indices,
):
    """Tests that the cumulative weights are correctly normalized."""
    relative_deployment = pd.Series([0.3, 0.7, 0.0], index=asset_indices)
    prev_weights = pd.Series([0.5, 0.1, 0.0], index=asset_indices)
    # Cumulative sum = {'solar': 0.8, 'wind': 0.8, 'gas': 0.0}
    # Max weight = 0.8

    expected = pd.Series(
        [
            0.8 / 0.8,  # 1.0
            0.8 / 0.8,  # 1.0
            0.0 / 0.8,  # 0.0
        ],
        index=asset_indices,
    )

    actual = calculate_weights_relative_deployment(
        relative_deployment, prev_weights, normalize=True
    )
    pd.testing.assert_series_equal(actual, expected)


def test_calculate_weights_relative_deployment_normalized_max_value_changes(
    asset_indices,
):
    """Tests normalization when a new technology deployment creates a new maximum weight."""
    relative_deployment = pd.Series([0.0, 0.0, 1.0], index=asset_indices)
    prev_weights = pd.Series([1.0, 0.8, 0.2], index=asset_indices)
    # Cumulative sum = {'solar': 1.0, 'wind': 0.8, 'gas': 1.2}, and new max weight = 1.2

    expected = pd.Series(
        [
            1.0 / 1.2,  # ~0.8333
            0.8 / 1.2,  # ~0.6667
            1.2 / 1.2,  # 1.0
        ],
        index=asset_indices,
    )

    actual = calculate_weights_relative_deployment(
        relative_deployment, prev_weights, normalize=True
    )
    pd.testing.assert_series_equal(actual, expected)


def test_calculate_weights_relative_deployment_normalized_all_zero_case(
    asset_indices,
):
    """Tests that the function handles the case of all-zero weights without a ZeroDivisionError."""
    relative_deployment = pd.Series([0.0, 0.0, 0.0], index=asset_indices)
    prev_weights = initialize_weights(asset_indices)  # All zeros
    # Cumulative sum is all zeros, max weight is 0.

    # The function should not normalize and just return zeros.
    expected = pd.Series([0.0, 0.0, 0.0], index=asset_indices)

    actual = calculate_weights_relative_deployment(
        relative_deployment, prev_weights, normalize=True
    )
    pd.testing.assert_series_equal(actual, expected)
