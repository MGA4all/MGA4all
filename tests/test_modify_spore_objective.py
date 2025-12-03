import linopy
import pandas as pd
import pytest
from .conftest import MockPypsaNetwork
from linopy.testing import assert_linequal

from mga4all.spores import modify_objective


def setup_model_and_network():
    """Helper Function for Test Setup. Creates a basic PyPSA network and a corresponding Linopy model for testing."""
    # 1. Create a mock PyPSA network with extendable generators
    n = MockPypsaNetwork(p_nom_opt_data={"solar": 0, "wind": 0, "gas": 0})

    # 2. Create a Linopy model
    m = linopy.Model()

    # 3. Create a capacity variable that mirrors what PyPSA would create
    coords = [n.generators.index]
    m.add_variables(coords=coords, name="Generator-p_nom", lower=0)

    return n, m


def test_modify_objective_diversify_only():
    """Tests the standard 'diversify' mode. The objective should only include the diversification term."""
    n, m = setup_model_and_network()
    weights = {"Generator": {"p_nom": {"solar": 0.5, "wind": 1.0}}}
    config = {
        "objective_sense": "min",
        "spores_mode": "diversify",
        "diversification_coefficient": 10,
    }

    m = modify_objective(n, m, weights, config)

    # Manually build the expected linear expression
    # sense=1, coeff=10. solar: 1*10*0.5=5. wind: 1*10*1.0=10. gas: 1*10*0=0.
    capacity_vars = m.variables["Generator-p_nom"]

    # 1. Create a pandas Series for the final expected coefficients.
    expected_coeffs = pd.Series(
        {"solar": 5.0, "wind": 10.0, "gas": 0.0},
        index=n.generators.index,  # Ensure the index matches the variable's index
    )

    # 2. Multiply the coefficients by the variable and sum the result.
    expected_expr = (expected_coeffs * capacity_vars).sum()

    assert_linequal(m.objective.expression, expected_expr)


def test_modify_objective_intensify_and_diversify():
    """Tests 'intensify and diversify' mode. The objective should include both terms."""
    n, m = setup_model_and_network()
    weights = {"Generator": {"p_nom": {"solar": 0.5, "wind": 1.0, "gas": 0.2}}}
    config = {
        "objective_sense": "min",
        "spores_mode": "intensify and diversify",
        "diversification_coefficient": 10,
        "intensification_coefficient": 100,
        "intensifiable_technologies": ["gas"],
    }
    m = modify_objective(n, m, weights, config)
    capacity_vars = m.variables["Generator-p_nom"]
    expected_coeffs = pd.Series(
        {"solar": 5.0, "wind": 10.0, "gas": 102.0}, index=n.generators.index
    )
    expected_expr = (expected_coeffs * capacity_vars).sum()
    assert_linequal(m.objective.expression, expected_expr)


def test_modify_objective_maximization_sense():
    """Tests that an 'objective_sense' of 'max' correctly negates the coefficients."""
    n, m = setup_model_and_network()
    weights = {"Generator": {"p_nom": {"solar": 0.5, "wind": 1.0}}}
    config = {
        "objective_sense": "max",
        "spores_mode": "diversify",
        "diversification_coefficient": 10,
    }
    m = modify_objective(n, m, weights, config)
    capacity_vars = m.variables["Generator-p_nom"]
    expected_coeffs = pd.Series(
        {"solar": -5.0, "wind": -10.0, "gas": 0.0}, index=n.generators.index
    )
    expected_expr = (expected_coeffs * capacity_vars).sum()
    assert_linequal(m.objective.expression, expected_expr)


def test_modify_objective_raises_error_on_bad_attribute():
    """Tests that a ValueError is raised for an incorrect capacity attribute in the weights."""
    n, m = setup_model_and_network()

    # Using 'p_nom_min' which is not the defined capacity attribute for Generator
    bad_weights = {"Generator": {"p_nom_min": {"solar": 0.5}}}
    config = {
        "objective_sense": "min",
        "spores_mode": "diversify",
        "diversification_coefficient": 10,
    }

    with pytest.raises(
        ValueError, match="Unknown capacity attribute p_nom_min for Generator"
    ):
        modify_objective(n, m, bad_weights, config)
