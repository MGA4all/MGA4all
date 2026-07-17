import pandas as pd
import numpy as np

from .model_interface_pypsa import (
    match_config_techs_to_model_techs,
    extract_diversified_capacity,
    extract_minimum_feasible_cost,
    create_mga_model,
    add_slack_constraint,
    assign_mga_objective
)

#-----

def setup_mga_model(test_config,n_costopt):

    n = n_costopt
    minimum_cost = extract_minimum_feasible_cost(n)
    
    slack = test_config["cost_slack"]
    
    n_mga, m_mga = create_mga_model(n)

    add_slack_constraint(m_mga, minimum_cost, slack)
    
    return(n_mga, m_mga)

def create_target_variables(test_config, n_mga):

    spatial = test_config['spatially_explicit']
    target_techs = match_config_techs_to_model_techs(test_config,n_mga)

    deployed_capacity_series = extract_diversified_capacity(target_techs, n_mga, spatial)
    
    return target_techs, deployed_capacity_series, spatial
    
def compute_hsj_weights(deployed_capacity_series, previous_weights_series, noise_threshold=0.001, weighting_method="integer"):
    
    mga_weights_series = previous_weights_series
    new_weights_series = deployed_capacity_series 
    if weighting_method == "integer":
        new_weights_series[:] = (deployed_capacity_series > noise_threshold).astype(int)
    else:
        pass
    mga_weights_series += new_weights_series

    return mga_weights_series

def update_mga_objective(n_mga, m_mga, mga_weights_series, target_techs, spatial):

    assign_mga_objective(n_mga, m_mga, mga_weights_series, target_techs, spatial)

    return (n_mga, m_mga)

def hop_skip_jump_algorithm(test_config, n_costopt, noise_threshold=0.001):

    mga_alternatives = {}
    mga_spatial_alternatives = {}
    mga_weights = {}
    
    n_mga, m_mga = setup_mga_model(test_config,n_costopt)
    target_techs, deployed_capacity_series, spatially_explicit = create_target_variables(test_config, n_mga)
    
    mga_weights[0] = deployed_capacity_series.replace(deployed_capacity_series.values,0) # empty series
    mga_spatial_alternatives[0] = extract_diversified_capacity(target_techs, n_costopt, spatial=True)
    mga_alternatives[0] = extract_diversified_capacity(target_techs, n_costopt, spatial=False)

    for j in range(1,test_config["alternatives"]+1):
        
        previous_weights_series = mga_weights[j-1]
        if spatially_explicit == True:
            deployed_capacity_series = mga_spatial_alternatives[j-1].copy()
        else:
            deployed_capacity_series = mga_alternatives[j-1].copy()        

        mga_weights_series = compute_hsj_weights(deployed_capacity_series, previous_weights_series, noise_threshold)
        n_mga, m_mga = update_mga_objective(n_mga, m_mga, mga_weights_series, target_techs, spatially_explicit)
        n_mga.optimize.solve_model(log_to_console=False)

        # Storing capacity results for further inspection
        ## TODO: make the result saving and inspection smoother and
        ## standardised across methods
        
        mga_alternatives[j] = extract_diversified_capacity(target_techs, n_mga, spatial=False)
        mga_spatial_alternatives[j] = extract_diversified_capacity(target_techs, n_mga, spatial=True)
        mga_weights[j] = mga_weights_series.copy()

    return mga_alternatives, mga_spatial_alternatives, mga_weights
