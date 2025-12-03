from unittest.mock import MagicMock

import linopy
import pytest
from linopy.testing import assert_linequal

from mga4all.spores import (
    create_modified_model,
    optimize_model_and_assign_solution_to_network,
)


@pytest.fixture
def mock_pypsa_and_linopy_env(mocker):
    """Sets up a comprehensive mock environment for testing the orchestration of model creation and optimization."""
    # 1. Mock the PyPSA Network and its optimize attribute
    mock_network = MagicMock(name="MockPypsaNetwork")
    mock_network.optimize = MagicMock(name="optimize_attribute")

    # 2. Mock the Linopy Model that will be "created"
    mock_model = MagicMock(name="MockLinopyModel")

    # 3. Mock the objective expression
    # We need a real LinearExpression to test the constraint math
    m_real = linopy.Model()
    x = m_real.add_variables(name="x")
    mock_objective_expr = 10 * x  # A simple, real expression

    mock_model.objective = mock_objective_expr

    # 4. Set up the return value for create_model
    mock_network.optimize.create_model.return_value = mock_model

    # 5. Mock the function we call internally
    mock_modify_objective = mocker.patch(
        "mga4all.spores.modify_objective",
        return_value=mock_model,  # It should return the modified model
    )

    return mock_network, mock_model, mock_modify_objective


def test_create_modified_model(mock_pypsa_and_linopy_env):
    """Tests that the function correctly adds the budget constraint and calls other functions in the correct order."""
    mock_network, mock_model, mock_modify_objective = mock_pypsa_and_linopy_env

    config = {"spores_slack": 0.1}
    optimal_cost = 1000.0
    weights = {"some": "weights"}

    model = create_modified_model(mock_network, config, optimal_cost, weights)

    # 1. Check that the model was created inside the function
    mock_network.optimize.create_model.assert_called_once()

    # 2. Check that the budget constraint was added correctly
    mock_model.add_constraints.assert_called_once()
    # Get the arguments the mock was called with
    call_args = mock_model.add_constraints.call_args
    actual_constraint = call_args.args[0]
    actual_name = call_args.kwargs["name"]

    assert actual_name == "budget-constraint"

    # Manually build the expected constraint expression
    expected_lhs = mock_model.objective
    expected_rhs = (1 + config["spores_slack"]) * optimal_cost

    # Linopy constraints have .lhs, .rhs, and .sign attributes
    assert_linequal(actual_constraint.lhs, expected_lhs)
    assert actual_constraint.rhs == pytest.approx(expected_rhs)
    assert actual_constraint.sign == "<="

    # 3. Check that modify_objective was called with the correct arguments
    mock_modify_objective.assert_called_once_with(
        mock_network, mock_model, weights, config
    )

    # 4. Check that the function returns the correct objects
    assert model is mock_model


def test_optimize_model_and_assign_solution_to_network(mock_pypsa_and_linopy_env):
    """Tests that the function correctly calls the solve method and assigns the solution back to the network."""
    mock_network, mock_model, _ = mock_pypsa_and_linopy_env

    returned_network, returned_model = optimize_model_and_assign_solution_to_network(
        mock_network, mock_model, {"highs": {}}
    )

    # Check that the model's solve method was called exactly once with the correct solver name.
    mock_model.solve.assert_called_once_with(solver_name="highs")

    # 2. Check that the network's solution assignment methods were called.
    # We access these through the mock_network's .optimize attribute, which is also a mock.
    mock_network.optimize.assign_solution.assert_called_once()
    mock_network.optimize.assign_duals.assert_called_once()

    # 3. Check that the function returns the same objects that were passed into it.
    assert returned_network is mock_network
    assert returned_model is mock_model
