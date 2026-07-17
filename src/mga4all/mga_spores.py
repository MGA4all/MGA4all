import pandas as pd
import numpy as np
from pandas.api.types import is_number

from .model_interface_pypsa import (
    match_config_techs_to_model_techs,
    extract_diversified_capacity,
    extract_intensified_capacity,
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

    diversified_technologies_series = extract_diversified_capacity(target_techs, n_mga, spatial)

    return target_techs, diversified_technologies_series, spatial

def create_intensification_variables(n_mga, spatial, target_techs, test_config):

    intensified_technologies_series = extract_intensified_capacity(target_techs, test_config, n_mga, spatial)

    return intensified_technologies_series

def normalise_l2(weights_series):

    weights_series = weights_series / np.sqrt((weights_series**2).sum())

    return weights_series

def compute_diversification_weights(deployed_capacity_series, noise_threshold=0.001, weighting_method="integer"):

    new_weights_series = deployed_capacity_series 
    if weighting_method == "integer":
        new_weights_series[:] = (deployed_capacity_series > noise_threshold).astype(int)
    else:
        pass

    # Perturb if all values are the same

    diversification_weights_series = normalise_l2(new_weights_series)

    return diversification_weights_series

def compute_intensification_weights(intensified_technologies_series):

    if is_number(intensified_technologies_series) == False:
        intensification_weights_series = intensified_technologies_series
    else:
        intensification_weights_series = 0

    return intensification_weights_series


def compute_coefficients(test_config):

    if isinstance(test_config["intensification_coefficient"], int):    
        intensify_coeff = abs(test_config["intensification_coefficient"])
    else:
        intensify_coeff = 1

    if (test_config["diversification_coefficient"] == "auto") and (test_config["intensification_coefficient"] != 0):
        diversify_coeff = abs(intensify_coeff) 
    elif (test_config["diversification_coefficient"] == "auto") and (test_config["intensification_coefficient"] == 0):
        diversify_coeff = 1
    else:
        diversify_coeff = test_config["diversification_coefficient"]

    return intensify_coeff, diversify_coeff

def compute_combined_weights(intensification_weights_series, diversification_weights_series, intensify_coeff, diversify_coeff):

    if isinstance(intensification_weights_series,int):
        combined_weights_series = normalise_l2(
                (diversify_coeff*diversification_weights_series) + 
                (intensify_coeff*intensification_weights_series)
        )
    else:    
    
        combined_weights_series = normalise_l2(
            pd.concat(
                (
                    (diversify_coeff*diversification_weights_series),(intensify_coeff*intensification_weights_series)
                ), axis=1
            ).sum(axis=1)
        )
    # Check if direction is good, else perturb

    return combined_weights_series

def update_mga_objective(n_mga, m_mga, mga_weights_series, target_techs, spatial):

    assign_mga_objective(n_mga, m_mga, mga_weights_series, target_techs, spatial)

    return (n_mga, m_mga)

def spores_algorithm(test_config, n_costopt, noise_threshold=0.001):

    mga_alternatives = {}
    mga_spatial_alternatives = {}
    mga_weights = {}
    
    n_mga, m_mga = setup_mga_model(test_config,n_costopt)
    target_techs, diversified_technologies_series, spatially_explicit = create_target_variables(test_config, n_mga)
    intensified_technologies_series = create_intensification_variables(n_mga, spatially_explicit, target_techs, test_config)
    intensify_coeff, diversify_coeff = compute_coefficients(test_config)
    
    mga_weights[0] = diversified_technologies_series.replace(diversified_technologies_series.values,0) # empty series
    mga_spatial_alternatives[0] = extract_diversified_capacity(target_techs, n_costopt, spatial=True)
    mga_alternatives[0] = extract_diversified_capacity(target_techs, n_costopt, spatial=False)

    intensification_weights_series = compute_intensification_weights(intensified_technologies_series)

    for j in range(1,test_config["alternatives"]+1):
        
        if j==1 and test_config["intensification_coefficient"] != 0 and isinstance(intensification_weights_series, pd.Series):

            mga_weights_series = compute_combined_weights(intensification_weights_series, mga_weights[0], intensify_coeff, diversify_coeff)
        
        elif j==1 and (test_config["intensification_coefficient"] == 0 or intensification_weights_series == 0):

            previous_weights_series = mga_weights[j-1]
            if spatially_explicit == True:
                diversified_technologies_series = mga_spatial_alternatives[j-1].copy()
            else:
                diversified_technologies_series = mga_alternatives[j-1].copy()        

            mga_weights_series = compute_diversification_weights(diversified_technologies_series, previous_weights_series, noise_threshold)
        
        else:
            previous_weights_series = mga_weights[j-1]
            if spatially_explicit == True:
                diversified_technologies_series = mga_spatial_alternatives[j-1].copy()
            else:
                diversified_technologies_series = mga_alternatives[j-1].copy()

            diversification_weights_series = compute_diversification_weights(diversified_technologies_series, previous_weights_series, noise_threshold)
            
            mga_weights_series = compute_combined_weights(intensification_weights_series, diversification_weights_series, intensify_coeff, diversify_coeff)
        
        n_mga, m_mga = update_mga_objective(n_mga, m_mga, mga_weights_series, target_techs, spatially_explicit)
        n_mga.optimize.solve_model(log_to_console=False)

        # Storing capacity results for further inspection
        ## TODO: make the result saving and inspection smoother and
        ## standardised across methods
        
        mga_alternatives[j] = extract_diversified_capacity(target_techs, n_mga, spatial=False)
        mga_spatial_alternatives[j] = extract_diversified_capacity(target_techs, n_mga, spatial=True)
        mga_weights[j] = mga_weights_series.copy()

    return mga_alternatives, mga_spatial_alternatives, mga_weights


#----- NEW: Intelligent SPORES ----- #

from scipy.stats import spearmanr

def ranking_similarity(a, b):
    rank_a = a.argsort().argsort()
    rank_b = b.argsort().argsort()

    corr, _ = spearmanr(rank_a, rank_b)

    return corr

def angular_similarity(a, b):
    return float((a * b).sum())

def perturb_rank_flip(s, flip_strength=0.25, min_positive=1e-6, normalise=True):

    values = s.to_numpy().copy()

    eligible = np.where(values >= 0)[0]

    if len(eligible) < 2:
        return s.copy()

    eligible_values = values[eligible]
    sorted_local = np.argsort(eligible_values)

    k = np.random.randint(len(sorted_local) - 1)

    i = eligible[sorted_local[k]]
    j = eligible[sorted_local[k + 1]]

    gap = values[j] - values[i]
    delta = gap / 2 + flip_strength * np.random.rand()

    new_i = values[i] + delta
    new_j = values[j] - delta

    if new_j <= min_positive:
        delta = values[j] - min_positive
        new_i = values[i] + delta
        new_j = min_positive

    values[i] = new_i
    values[j] = new_j

    perturbed = pd.Series(values, index=s.index, name=s.name)

    if normalise:
        norm = np.linalg.norm(perturbed)
        if norm > 0:
            perturbed /= norm

    return perturbed

def perturb_noise(s, noise_scale=0.05, normalise=True):

    values = s.to_numpy().copy()

    values += np.random.normal(0, noise_scale, len(values))

    perturbed = pd.Series(values, index=s.index, name=s.name)

    if normalise:
        norm = np.linalg.norm(perturbed)
        if norm > 0:
            perturbed /= norm

    return perturbed

def is_different_enough(new_weights, stored_weights, cosine_threshold=0.97, ranking_threshold=0.9):

    stored_weights = list(stored_weights.values())
    
    if len(stored_weights) == 0:
        return True

    for old in stored_weights:

        # angular similarity
        cosine_similarity = angular_similarity(new_weights, old)
        
        # ranking similarity
        corr_ranking = ranking_similarity(new_weights, old)

        if cosine_similarity > cosine_threshold:
            return False
        
        if corr_ranking > ranking_threshold:
            return False

    return True


def spores_algorithm_adaptive(test_config, n_costopt, noise_threshold=0.001):

    MAX_NOISE_ATTEMPTS = 50

    mga_alternatives = {}
    mga_spatial_alternatives = {}
    mga_weights = {}
    
    n_mga, m_mga = setup_mga_model(test_config,n_costopt)
    target_techs, diversified_technologies_series, spatially_explicit = create_target_variables(test_config, n_mga)
    intensified_technologies_series = create_intensification_variables(n_mga, spatially_explicit, target_techs, test_config)
    intensify_coeff, diversify_coeff = compute_coefficients(test_config)
    
    mga_weights[0] = diversified_technologies_series.replace(diversified_technologies_series.values,0) # empty series
    mga_spatial_alternatives[0] = extract_diversified_capacity(target_techs, n_costopt, spatial=True)
    mga_alternatives[0] = extract_diversified_capacity(target_techs, n_costopt, spatial=False)

    intensification_weights_series = compute_intensification_weights(intensified_technologies_series)

    for j in range(1,test_config["alternatives"]+1):
        
        if j==1 and test_config["intensification_coefficient"] != 0 and isinstance(intensification_weights_series, pd.Series):

            mga_weights_series = compute_combined_weights(intensification_weights_series, mga_weights[0], intensify_coeff, diversify_coeff)
        
        elif j==1 and (test_config["intensification_coefficient"] == 0 or intensification_weights_series == 0):

            previous_weights_series = mga_weights[j-1]
            if spatially_explicit == True:
                diversified_technologies_series = mga_spatial_alternatives[j-1].copy()
            else:
                diversified_technologies_series = mga_alternatives[j-1].copy()        

            mga_weights_series = compute_diversification_weights(diversified_technologies_series, previous_weights_series, noise_threshold)
        
        else:
            previous_weights_series = mga_weights[j-1]
            if spatially_explicit == True:
                diversified_technologies_series = mga_spatial_alternatives[j-1].copy()
            else:
                diversified_technologies_series = mga_alternatives[j-1].copy()

            diversification_weights_series = compute_diversification_weights(diversified_technologies_series, previous_weights_series, noise_threshold)
            
            mga_weights_series = compute_combined_weights(intensification_weights_series, diversification_weights_series, intensify_coeff, diversify_coeff)
        

            #---SMART part----#

            # Check if different enough and perturb if not

            if is_different_enough(
                    mga_weights_series,
                    mga_weights
                ):
                pass
            else:
            
                # 1. Ranking perturbation first            
                candidate = perturb_rank_flip(mga_weights_series)
                
                # 2. Noise perturbation if needed
                attempt = 0
                
                while (
                    not is_different_enough(
                        candidate,
                        mga_weights
                    )
                    and attempt < MAX_NOISE_ATTEMPTS
                ):
                
                    candidate = perturb_noise(
                        candidate,
                        noise_scale=0.03 * (attempt + 1)
                    )
                
                    attempt += 1
                
                # Final accepted candidate
                mga_weights_series = candidate

            #---End of SMART part---#

        n_mga, m_mga = update_mga_objective(n_mga, m_mga, mga_weights_series, target_techs, spatially_explicit)
        n_mga.optimize.solve_model(log_to_console=False)

        # Storing capacity results for further inspection
        ## TODO: make the result saving and inspection smoother and
        ## standardised across methods
        
        mga_alternatives[j] = extract_diversified_capacity(target_techs, n_mga, spatial=False)
        mga_spatial_alternatives[j] = extract_diversified_capacity(target_techs, n_mga, spatial=True)
        mga_weights[j] = mga_weights_series.copy()

    return mga_alternatives, mga_spatial_alternatives, mga_weights

