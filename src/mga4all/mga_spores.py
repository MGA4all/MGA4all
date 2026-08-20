import pandas as pd
import numpy as np
from pandas.api.types import is_number
from scipy.stats import spearmanr

from .model_interface_pypsa import (
    match_config_techs_to_model_techs,
    extract_diversified_capacity,
    extract_intensified_capacity,
    extract_minimum_feasible_cost,
    create_mga_model,
    add_slack_constraint,
    assign_mga_objective,
)


def setup_mga_model(config, network):
    minimum_cost = extract_minimum_feasible_cost(network)
    slack = config["cost_slack"]
    network_mga, model_mga = create_mga_model(network)
    add_slack_constraint(model_mga, minimum_cost, slack)
    return network_mga, model_mga


def create_target_variables(config, network_mga):
    spatial = config["spatially_explicit"]
    target_techs = match_config_techs_to_model_techs(config, network_mga)
    diversified_technologies_series = extract_diversified_capacity(
        target_techs, network_mga, spatial
    )
    return target_techs, diversified_technologies_series, spatial


def create_intensification_variables(network_mga, spatial, target_techs, config):
    intensified_technologies_series = extract_intensified_capacity(
        target_techs, config, network_mga, spatial
    )
    return intensified_technologies_series


def normalise_l2(weights_series):
    weights_series = weights_series / np.sqrt((weights_series**2).sum())
    return weights_series


def compute_diversification_weights(
    deployed_capacity_series,
    previous_weights_series,
    noise_threshold=0.001,
    weighting_method="integer",
):
    new_weights_series = deployed_capacity_series
    if weighting_method == "integer":
        new_weights_series[:] = (deployed_capacity_series > noise_threshold).astype(int)
    updated_weights_series = previous_weights_series + new_weights_series
    # Perturb if all values are the same
    diversification_weights_series = normalise_l2(updated_weights_series)
    return diversification_weights_series


def compute_intensification_weights(intensified_technologies_series):
    if not is_number(intensified_technologies_series):
        intensification_weights_series = intensified_technologies_series
    else:
        intensification_weights_series = 0
    return intensification_weights_series


def compute_coefficients(config):
    if isinstance(config["intensification_coefficient"], int):
        intensify_coeff = abs(config["intensification_coefficient"])
    else:
        intensify_coeff = 1
    if (config["diversification_coefficient"] == "auto") and (
        config["intensification_coefficient"] != 0
    ):
        diversify_coeff = abs(intensify_coeff)
    elif (config["diversification_coefficient"] == "auto") and (
        config["intensification_coefficient"] == 0
    ):
        diversify_coeff = 1
    else:
        diversify_coeff = config["diversification_coefficient"]
    return intensify_coeff, diversify_coeff


def compute_combined_weights(
    intensification_weights_series,
    diversification_weights_series,
    intensify_coeff,
    diversify_coeff,
):
    if isinstance(intensification_weights_series, int):
        combined_weights_series = normalise_l2(
            (diversify_coeff * diversification_weights_series)
            + (intensify_coeff * intensification_weights_series)
        )
    else:
        combined_weights_series = normalise_l2(
            pd.concat(
                (
                    (diversify_coeff * diversification_weights_series),
                    (intensify_coeff * intensification_weights_series),
                ),
                axis=1,
            ).sum(axis=1)
        )
    # Check if direction is good, else perturb
    return combined_weights_series


def update_mga_objective(
    network_mga, model_mga, mga_weights_series, target_techs, spatial
):
    assign_mga_objective(
        network_mga, model_mga, mga_weights_series, target_techs, spatial
    )
    return (network_mga, model_mga)


def spores_algorithm(config, network_costopt, noise_threshold=0.001):
    mga_alternatives = {}
    mga_spatial_alternatives = {}
    mga_weights = {}

    network_mga, model_mga = setup_mga_model(config, network_costopt)
    target_techs, diversified_technologies_series, spatially_explicit = (
        create_target_variables(config, network_mga)
    )
    intensified_technologies_series = create_intensification_variables(
        network_mga, spatially_explicit, target_techs, config
    )
    intensify_coeff, diversify_coeff = compute_coefficients(config)

    mga_weights[0] = pd.Series(0, index=diversified_technologies_series.index)
    mga_diversification_weights = pd.Series(
        0, index=diversified_technologies_series.index
    )
    mga_spatial_alternatives[0] = extract_diversified_capacity(
        target_techs, network_costopt, spatial=True
    )
    mga_alternatives[0] = extract_diversified_capacity(
        target_techs, network_costopt, spatial=False
    )

    intensification_weights_series = compute_intensification_weights(
        intensified_technologies_series
    )

    for iteration in range(1, config["alternatives"] + 1):

        # If intensification is required, the first iteration
        # implements only pure intensification to find the extreme
        if (
            iteration == 1
            and config["intensification_coefficient"] != 0
            and isinstance(intensification_weights_series, pd.Series)
        ):
            diversification_weights_series = pd.Series(
                0, index=diversified_technologies_series.index
            )
            mga_weights_series = compute_combined_weights(
                intensification_weights_series,
                diversification_weights_series,
                intensify_coeff,
                diversify_coeff,
            )

        # If intensification is not required, simple diversification applies
        elif iteration == 1 and (
            config["intensification_coefficient"] == 0
            or intensification_weights_series == 0
        ):
            previous_weights_series = mga_diversification_weights
            if spatially_explicit:
                diversified_technologies_series = mga_spatial_alternatives[
                    iteration - 1
                ].copy()
            else:
                diversified_technologies_series = mga_alternatives[iteration - 1].copy()

            diversification_weights_series = compute_diversification_weights(
                diversified_technologies_series,
                previous_weights_series,
                noise_threshold,
            )
            mga_weights_series = diversification_weights_series

        # In all other cases, both intensification and diversification are
        # accounted for, and the coefficients determine their importance
        else:
            previous_weights_series = mga_diversification_weights
            if spatially_explicit:
                diversified_technologies_series = mga_spatial_alternatives[
                    iteration - 1
                ].copy()
            else:
                diversified_technologies_series = mga_alternatives[iteration - 1].copy()

            diversification_weights_series = compute_diversification_weights(
                diversified_technologies_series,
                previous_weights_series,
                noise_threshold,
            )
            mga_weights_series = compute_combined_weights(
                intensification_weights_series,
                diversification_weights_series,
                intensify_coeff,
                diversify_coeff,
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
        mga_diversification_weights = diversification_weights_series

    return mga_alternatives, mga_spatial_alternatives, mga_weights


# ----- NEW: Intelligent SPORES ----- #
def ranking_similarity(weights_a, weights_b):
    rank_a = weights_a.argsort().argsort()
    rank_b = weights_b.argsort().argsort()
    corr, pvalue = spearmanr(rank_a, rank_b)
    return corr


def angular_similarity(weights_a, weights_b):
    return float((weights_a * weights_b).sum())


def perturb_rank_flip(
    weights_series, flip_strength=0.25, min_positive=1e-6, normalise=True
):
    values = weights_series.to_numpy().copy()

    eligible = np.where(values >= 0)[0]

    if len(eligible) < 2:
        return weights_series.copy()

    eligible_values = values[eligible]
    sorted_local = np.argsort(eligible_values)

    rank_index = np.random.randint(len(sorted_local) - 1)

    index_i = eligible[sorted_local[rank_index]]
    index_j = eligible[sorted_local[rank_index + 1]]

    gap = values[index_j] - values[index_i]
    delta = gap / 2 + flip_strength * np.random.rand()

    new_i = values[index_i] + delta
    new_j = values[index_j] - delta

    if new_j <= min_positive:
        delta = values[index_j] - min_positive
        new_i = values[index_i] + delta
        new_j = min_positive

    values[index_i] = new_i
    values[index_j] = new_j

    perturbed = pd.Series(values, index=weights_series.index, name=weights_series.name)

    if normalise:
        norm = np.linalg.norm(perturbed)
        if norm > 0:
            perturbed /= norm

    return perturbed


def perturb_noise(weights_series, noise_scale=0.05, normalise=True):
    values = weights_series.to_numpy().copy()
    values += np.random.normal(0, noise_scale, len(values))
    perturbed = pd.Series(values, index=weights_series.index, name=weights_series.name)

    if normalise:
        norm = np.linalg.norm(perturbed)
        if norm > 0:
            perturbed /= norm

    return perturbed


def is_different_enough(
    new_weights, stored_weights, cosine_threshold=0.97, ranking_threshold=0.9
):
    stored_weights = list(stored_weights.values())

    if len(stored_weights) == 0:
        return True

    for old in stored_weights:
        cosine_similarity = angular_similarity(new_weights, old)
        corr_ranking = ranking_similarity(new_weights, old)

        if cosine_similarity > cosine_threshold:
            return False

        # FIXME: mismatched types, corr_ranking is possibly a tuple
        if corr_ranking > ranking_threshold:
            return False

    return True


def spores_algorithm_adaptive(config, network_costopt, noise_threshold=0.001):
    MAX_NOISE_ATTEMPTS = 50

    mga_alternatives = {}
    mga_spatial_alternatives = {}
    mga_weights = {}

    network_mga, model_mga = setup_mga_model(config, network_costopt)
    target_techs, diversified_technologies_series, spatially_explicit = (
        create_target_variables(config, network_mga)
    )
    intensified_technologies_series = create_intensification_variables(
        network_mga, spatially_explicit, target_techs, config
    )
    intensify_coeff, diversify_coeff = compute_coefficients(config)

    mga_weights[0] = pd.Series(0, index=diversified_technologies_series.index)
    mga_diversification_weights = pd.Series(
        0, index=diversified_technologies_series.index
    )
    mga_spatial_alternatives[0] = extract_diversified_capacity(
        target_techs, network_costopt, spatial=True
    )
    mga_alternatives[0] = extract_diversified_capacity(
        target_techs, network_costopt, spatial=False
    )

    intensification_weights_series = compute_intensification_weights(
        intensified_technologies_series
    )

    for iteration in range(1, config["alternatives"] + 1):
        if (
            iteration == 1
            and config["intensification_coefficient"] != 0
            and isinstance(intensification_weights_series, pd.Series)
        ):
            diversification_weights_series = pd.Series(
                0, index=diversified_technologies_series.index
            )
            mga_weights_series = compute_combined_weights(
                intensification_weights_series,
                diversification_weights_series,
                intensify_coeff,
                diversify_coeff,
            )
        elif iteration == 1 and (
            config["intensification_coefficient"] == 0
            or intensification_weights_series == 0
        ):
            previous_weights_series = mga_diversification_weights
            if spatially_explicit:
                diversified_technologies_series = mga_spatial_alternatives[
                    iteration - 1
                ].copy()
            else:
                diversified_technologies_series = mga_alternatives[iteration - 1].copy()

            diversification_weights_series = compute_diversification_weights(
                diversified_technologies_series,
                previous_weights_series,
                noise_threshold,
            )
            mga_weights_series = diversification_weights_series
        else:
            previous_weights_series = mga_diversification_weights
            if spatially_explicit:
                diversified_technologies_series = mga_spatial_alternatives[
                    iteration - 1
                ].copy()
            else:
                diversified_technologies_series = mga_alternatives[iteration - 1].copy()

            diversification_weights_series = compute_diversification_weights(
                diversified_technologies_series,
                previous_weights_series,
                noise_threshold,
            )
            mga_weights_series = compute_combined_weights(
                intensification_weights_series,
                diversification_weights_series,
                intensify_coeff,
                diversify_coeff,
            )

            # ---SMART part----#
            # Check if different enough and perturb if not
            if not is_different_enough(mga_weights_series, mga_weights):
                # 1. Ranking perturbation first
                candidate = perturb_rank_flip(mga_weights_series)
                # 2. Noise perturbation if needed
                attempt = 0
                while (
                    not is_different_enough(candidate, mga_weights)
                    and attempt < MAX_NOISE_ATTEMPTS
                ):
                    candidate = perturb_noise(
                        candidate, noise_scale=0.03 * (attempt + 1)
                    )
                    attempt += 1
                # Final accepted candidate
                mga_weights_series = candidate
            # ---End of SMART part---#

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
        mga_diversification_weights = diversification_weights_series

    return mga_alternatives, mga_spatial_alternatives, mga_weights
