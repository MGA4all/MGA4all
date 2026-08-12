import pandas as pd
import numpy as np

from .model_interface_pypsa import (
    match_config_techs_to_model_techs,
    extract_diversified_capacity,
    extract_minimum_feasible_cost,
    create_mga_model,
    add_slack_constraint,
    assign_mga_objective,
)


def setup_mga_model(test_config, network_costopt):
    network = network_costopt
    minimum_cost = extract_minimum_feasible_cost(network)
    slack = test_config["cost_slack"]
    network_mga, model_mga = create_mga_model(network)
    add_slack_constraint(model_mga, minimum_cost, slack)
    return (network_mga, model_mga)


def create_target_variables(test_config, network_mga):
    spatial = test_config["spatially_explicit"]
    target_techs = match_config_techs_to_model_techs(test_config, network_mga)
    deployed_capacity_series = extract_diversified_capacity(
        target_techs, network_mga, spatial
    )
    return target_techs, deployed_capacity_series, spatial


def compute_hsj_weights(
    deployed_capacity_series,
    previous_weights_series,
    noise_threshold=0.001,
    weighting_method="integer",
):
    mga_weights_series = previous_weights_series
    new_weights_series = deployed_capacity_series
    if weighting_method == "integer":
        new_weights_series[:] = (deployed_capacity_series > noise_threshold).astype(int)
    mga_weights_series += new_weights_series
    return mga_weights_series


def update_mga_objective(
    network_mga, model_mga, mga_weights_series, target_techs, spatial
):
    assign_mga_objective(
        network_mga, model_mga, mga_weights_series, target_techs, spatial
    )
    return (network_mga, model_mga)


def hop_skip_jump_algorithm(test_config, network_costopt, noise_threshold=0.001):
    mga_alternatives = {}
    mga_spatial_alternatives = {}
    mga_weights = {}

    network_mga, model_mga = setup_mga_model(test_config, network_costopt)
    target_techs, deployed_capacity_series, spatially_explicit = (
        create_target_variables(test_config, network_mga)
    )

    mga_weights[0] = deployed_capacity_series.replace(
        deployed_capacity_series.values, 0
    )  # empty series
    mga_spatial_alternatives[0] = extract_diversified_capacity(
        target_techs, network_costopt, spatial=True
    )
    mga_alternatives[0] = extract_diversified_capacity(
        target_techs, network_costopt, spatial=False
    )
    for iteration in range(1, test_config["alternatives"] + 1):
        previous_weights_series = mga_weights[iteration - 1]
        if spatially_explicit == True:
            deployed_capacity_series = mga_spatial_alternatives[iteration - 1].copy()
        else:
            deployed_capacity_series = mga_alternatives[iteration - 1].copy()

        mga_weights_series = compute_hsj_weights(
            deployed_capacity_series, previous_weights_series, noise_threshold
        )
        network_mga, model_mga = update_mga_objective(
            network_mga, model_mga, mga_weights_series, target_techs, spatially_explicit
        )
        network_mga.optimize.solve_model(log_to_console=False)

        # Storing capacity results for further inspection
        ## TODO: make the result saving and inspection smoother and
        ## standardised across methods
        mga_alternatives[iteration] = extract_diversified_capacity(
            target_techs, network_mga, spatial=False
        )
        mga_spatial_alternatives[iteration] = extract_diversified_capacity(
            target_techs, network_mga, spatial=True
        )
        mga_weights[iteration] = mga_weights_series.copy()

    return mga_alternatives, mga_spatial_alternatives, mga_weights
