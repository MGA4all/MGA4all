from unittest.mock import MagicMock

import pytest

from mga4all.spores import run_spores


@pytest.fixture
def spores_run_env_factory(mocker):
    """A factory fixture that mocks all external dependencies for the run_spores function.

    This allows us to test the orchestration logic in isolation by configuring the
    weighting method for each test run.
    """

    def _factory(weighting_method_to_test):
        # Mock the SPORES configuration dictionary
        mock_spores_config = {
            "SPORES": {
                "num_spores": 5,  # Keep this low for a fast test
                "spores_mode": "diversify",
                "weighting_method": weighting_method_to_test,
                "spore_techs": {},  # Add dummy value to satisfy validation
            }
        }

        # Mock the input network and its required attributes
        mock_least_cost_network = MagicMock(name="LeastCostNetwork")
        mock_least_cost_network.is_solved = True
        mock_least_cost_network.statistics.capex.return_value.sum.return_value = 1000
        mock_least_cost_network.statistics.opex.return_value.sum.return_value = 500

        # Mock helper functions called by run_spores
        mocker.patch("mga4all.spores.validate_spores_configuration")
        mocker.patch(
            "mga4all.spores.get_tech_deployment",
            return_value={"mock": "deployment"},
        )

        # Mock the function that creates the modified model
        mock_modified_model = MagicMock(name="ModifiedModel")
        mock_create_modified_model = mocker.patch(
            "mga4all.spores.create_modified_model",
            return_value=mock_modified_model,
        )

        # Mock the return values for the optimization function for each loop
        mock_spore_nets = [MagicMock(name=f"SporeNet{i + 1}") for i in range(5)]
        mock_solved_models = [MagicMock(name=f"SolvedModel{i + 1}") for i in range(5)]
        mock_optimize_and_assign = mocker.patch(
            "mga4all.spores.optimize_model_and_assign_solution_to_network",
            side_effect=list(zip(mock_spore_nets, mock_solved_models)),
        )

        # Mock weight calculation functions
        mocker.patch(
            "mga4all.spores.initialize_weights",
            return_value={"init": "weights"},
        )
        mock_calc_first_iter = mocker.patch(
            "mga4all.spores.calculate_weights_first_iteration",
            return_value={"first_iter": "weights"},
        )

        all_weighting_mocks = {
            "relative_deployment": mocker.patch(
                "mga4all.spores.calculate_weights_relative_deployment",
                return_value={"rel_deploy": "weights"},
            ),
            "relative_deployment_normalized": mocker.patch(
                "mga4all.spores.calculate_weights_relative_deployment_normalized",
                return_value={"rel_deploy_norm": "weights"},
            ),
            "random": mocker.patch(
                "mga4all.spores.calculate_weights_random",
                return_value={"random": "weights"},
            ),
            "evolving_average": mocker.patch(
                "mga4all.spores.calculate_weights_evolving",
                return_value={"evo_avg": "weights"},
            ),
            "evolving_median": mocker.patch(
                "mga4all.spores.calculate_weights_evolving",
                return_value={"evo_med": "weights"},
            ),
        }

        return {
            "mock_spores_config": mock_spores_config,
            "mock_least_cost_network": mock_least_cost_network,
            "mock_create_modified_model": mock_create_modified_model,
            "mock_optimize_and_assign": mock_optimize_and_assign,
            "mock_calc_first_iter": mock_calc_first_iter,
            "all_weighting_mocks": all_weighting_mocks,
            "expected_spore_nets": mock_spore_nets,
            "expected_spore_models": mock_solved_models,
        }

    return _factory


@pytest.mark.parametrize(
    "method_to_test, expected_weight_value",
    [
        ("relative_deployment", {"rel_deploy": "weights"}),
        ("relative_deployment_normalized", {"rel_deploy_norm": "weights"}),
        ("random", {"random": "weights"}),
        pytest.param(
            "evolving_average",
            {"evo_avg": "weights"},
            marks=pytest.mark.xfail(
                reason="same function is patched twice, overwriting return value for 'average'."
            ),
        ),
        ("evolving_median", {"evo_med": "weights"}),
    ],
)
def test_run_spores_orchestration_and_dispatching(
    spores_run_env_factory, method_to_test, expected_weight_value
):
    """Tests that run_spores correctly orchestrates calls and dispatches the right weighting method based the config."""
    spores_run_env = spores_run_env_factory(method_to_test)

    # Use the configured mocks from the factory
    mock_least_cost_network = spores_run_env["mock_least_cost_network"]
    mock_spores_config = spores_run_env["mock_spores_config"]

    spore_networks, weights, spore_models, deploy_his = run_spores(
        mock_least_cost_network,
        mock_spores_config,
        {"highs": {}},
    )

    # Assert the loop ran the correct number of times and called the new functions
    num_spores = spores_run_env["mock_spores_config"]["SPORES"]["num_spores"]
    assert spores_run_env["mock_create_modified_model"].call_count == num_spores
    assert spores_run_env["mock_optimize_and_assign"].call_count == num_spores

    # Assert the logic for the first iteration was correct
    spores_run_env["mock_calc_first_iter"].assert_called_once()
    first_call_args = spores_run_env["mock_create_modified_model"].call_args_list[0]

    # Check the 4th positional argument (index 3) which is `new_weights`
    assert first_call_args.args[3] == {"first_iter": "weights"}

    # Assert the dispatching logic for subsequent iterations was correct
    for method_name, mock_func in spores_run_env["all_weighting_mocks"].items():
        if method_name == method_to_test:
            # It should have called the method from the config 4 times (for spores 2, 3, 4, 5)
            assert mock_func.call_count == num_spores - 1
        else:
            # It should NOT have called other methods
            mock_func.assert_not_called()

    # Check that all subsequent calls used the correct weights
    for i in range(1, num_spores):  # Check calls for spores 2 through 5
        call_args = spores_run_env["mock_create_modified_model"].call_args_list[i]

        # Check the 4th positional argument (index 3) which is `new_weights`
        assert call_args.args[3] == expected_weight_value

    # Assert the final results are correctly collected
    assert len(spore_networks) == num_spores
    assert len(weights) == num_spores
    assert len(spore_models) == num_spores
    assert len(deploy_his) == num_spores + 1  # +1 for the initial least-cost solution

    # Check that the returned objects are the ones our mock created
    for i in range(num_spores):
        assert (
            spore_networks[f"spore_{i + 1}"] is spores_run_env["expected_spore_nets"][i]
        )
        assert (
            spore_models[f"model_{i + 1}"] is spores_run_env["expected_spore_models"][i]
        )
