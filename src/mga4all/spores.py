import logging
import numbers
import random
import statistics

import gurobipy as gp
import linopy
import numpy as np
import pandas as pd
import pypsa
import xarray as xr

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


NOMINAL_ATTRS = {
    "Generator": {"dataframe_name": "generators", "capacity_attribute": "p_nom"},
    "Line": {"dataframe_name": "lines", "capacity_attribute": "s_nom"},
    "Transformer": {"dataframe_name": "transformers", "capacity_attribute": "s_nom"},
    "Link": {"dataframe_name": "links", "capacity_attribute": "p_nom"},
    "Store": {"dataframe_name": "stores", "capacity_attribute": "e_nom"},
    "StorageUnit": {"dataframe_name": "storage_units", "capacity_attribute": "p_nom"},
}

WEIGHTING_METHODS = [
    "random",
    "evolving_median",
    "evolving_average",
    "relative_deployment",
    "relative_deployment_normalized",
]


def run_spores(
    least_cost_network: pypsa.Network,
    spores_config: dict,
    solver_options: dict,
    weighting_method: str | None = None,
    upper_bound: int = 100,
) -> tuple[
    dict[str, pypsa.Network], dict[str, dict], dict[str, linopy.Model], list[dict]
]:
    """Run the SPORES optimization to generate multiple near-optimal solutions."""
    # Validate the SPORES configuration.
    validate_spores_configuration(spores_config)

    config_data = spores_config["SPORES"]

    # Build nested dict containing spore_techs once to avoid rebuilding again in downstream functions.
    spore_techs_dict = initialize_weights(config_data)

    # If no method is passed to the function, get it from the config file.
    if weighting_method is None:
        weighting_method = config_data.get("weighting_method")

    if weighting_method not in WEIGHTING_METHODS:
        raise ValueError(
            f"Unsupported {weighting_method=}, must be one of {WEIGHTING_METHODS}."
        )

    # Get the least-cost optimal solution from the solved network.
    # Check if the network is already optimized, else raise an error.
    if not least_cost_network.is_solved:
        raise ValueError("The input network must be optimized before running SPORES.")
    optimal_cost = (
        least_cost_network.statistics.capex().sum()
        + least_cost_network.statistics.opex().sum()
    )

    # Initialize collectors to store results/history
    spore_networks = {}
    weights = {}
    spore_models = {}

    # Deployment history is needed for `evolving_average` weighting methods. Initialize the history with the least-cost
    # solution's deployment so that it has a memory of the original least-cost solution.
    deploy_his = [get_tech_deployment(least_cost_network, spore_techs_dict)]

    # Clean up model state so we can make a copy and avoid rebuilding inside the spores loop. PyPSA does not allow
    # copying networks with a solver_model attached, so we need to remove it first.
    if (
        hasattr(least_cost_network.model, "solver_model")
        and least_cost_network.model is not None
    ):
        least_cost_network.model.solver_model = None

    # Run SPORES
    for i in range(1, config_data["num_spores"] + 1):
        network = least_cost_network.copy()

        if i == 1:
            # Previous weights are needed for the relative_deployment weighting methods.
            prev_weights = initialize_weights(config_data)

            # Calculation of new weights in the 1st iteration depends on the `spores_mode`.
            new_weights = calculate_weights_first_iteration(
                network, config_data["spores_mode"], prev_weights
            )

        else:
            prev_weights = weights[
                f"weights_{i - 1}"
            ]  # Needed for relative_deployment weighting methods.

            prev_spore = spore_networks[f"spore_{i - 1}"]

            # Dispatch to the correct weighting method
            if weighting_method == "random":
                new_weights = calculate_weights_random(spore_techs_dict, upper_bound)

            elif weighting_method == "relative_deployment":
                new_weights = calculate_weights_relative_deployment(
                    prev_spore, prev_weights
                )

            elif weighting_method == "relative_deployment_normalized":
                new_weights = calculate_weights_relative_deployment_normalized(
                    prev_spore, prev_weights
                )

            elif weighting_method == "evolving_median":
                new_weights = calculate_weights_evolving_median(
                    prev_spore, deploy_his, spore_techs_dict
                )

            elif weighting_method == "evolving_average":
                new_weights = calculate_weights_evolving_average(
                    prev_spore, deploy_his, spore_techs_dict
                )

        # Create & optimize the modified model (has the new objective (tech capacities * weights) & budget constraints)
        modified_model = create_modified_model(
            network, config_data, optimal_cost, new_weights
        )
        new_spore, solved_model = optimize_model_and_assign_solution_to_network(
            network, modified_model, solver_options
        )

        weights[f"weights_{i}"] = new_weights
        spore_networks[f"spore_{i}"] = new_spore
        spore_models[f"model_{i}"] = solved_model

        # Needed for evolving_median and evolving_average weighting methods
        deploy_his.append(get_tech_deployment(new_spore, spore_techs_dict))

    return spore_networks, weights, spore_models, deploy_his


def calculate_weights_random(spore_techs_dict: dict, upper_bound: int) -> dict:
    """Generates new weights using random numbers from a uniform distribution between 0 and upper_bound."""
    new_weights = {}

    for component, attrs in spore_techs_dict.items():
        new_weights.setdefault(component, {})

        for attr, techs_weights_map in attrs.items():
            new_weights[component].setdefault(attr, {})

            for tech in techs_weights_map.keys():
                new_weights[component][attr][tech] = random.uniform(0, upper_bound)

    return new_weights


def calculate_weights_relative_deployment(n: pypsa.Network, prev_weights: dict) -> dict:
    """Calculate new weights by adding the latest relative deployment to the previous weights."""
    relative_deployment = calculate_relative_deployment(n, prev_weights)

    new_weights = {}
    for component, attrs in prev_weights.items():
        new_weights[component] = {}
        for attr, techs_weights_map in attrs.items():
            updated = {}
            for tech, weight in techs_weights_map.items():
                rel_deploy = relative_deployment[component][attr][tech]
                updated[tech] = weight + rel_deploy
            new_weights[component][attr] = updated

    return new_weights


def calculate_weights_relative_deployment_normalized(
    n: pypsa.Network, prev_weights: dict
) -> dict:
    """Calculate weights as in `calculate_weights_relative_deployment` then normalizes w.r.t. the max_weight."""
    new_weights = calculate_weights_relative_deployment(n, prev_weights)

    # Find the maximum weight from all technologies subject to SPORES.
    max_weight = 0.0
    for component, attrs in new_weights.items():
        for attr, techs_weights_map in attrs.items():
            max_val = max(techs_weights_map.values())
            if max_val > max_weight:
                max_weight = max_val

    # Normalize w.r.t. the max_weight if the max_weight is greater than 0
    if max_weight > 0:
        for component, attrs in new_weights.items():
            for attr, techs_weights_map in attrs.items():
                for tech, weight in techs_weights_map.items():
                    techs_weights_map[tech] = weight / max_weight

    return new_weights


def calculate_weights_evolving_median(
    latest_spore: pypsa.Network,
    deployment_history: list[dict],
    spore_techs_dict: dict,
    clip_min: float = 0.001,
) -> dict:
    """Calculates weights based on the reciprocal of the relative distance from the evolving median capacity.

    This weighting method uses the median instead of the average, so that the weights are not skewed by a single outlier
    spore that might have had an unusually large deployment of a specific technology. For example, if the deploy_his for
    a tech is [0, 0, 0, 0, 1000], the average would be 200. A new solution with 0 deployment would be penalized. While
    the median would be 0. A new solution with 0 deployment would get a weight of 0, identifying it as an underexplored.
    """
    median_deployment = calculate_median_deployment(
        deployment_history, spore_techs_dict
    )
    latest_deployment = get_tech_deployment(latest_spore, spore_techs_dict)

    new_weights = {}

    for component, attrs in median_deployment.items():
        new_weights.setdefault(component, {})

        for attr, techs_median_deployment_map in attrs.items():
            new_weights[component].setdefault(attr, {})

            for tech, median_deployed_cap in techs_median_deployment_map.items():
                latest_deployed_cap = (
                    latest_deployment.get(component, {}).get(attr, {}).get(tech, 0)
                )

                if median_deployed_cap > 0:
                    relative_change = (
                        abs(latest_deployed_cap - median_deployed_cap)
                        / median_deployed_cap
                    )

                    # If the relative_change is 0 (latest_deployed_cap == median), we give the relative_change a small
                    # value which will give it a large penalty (weight) since we take the reciprocal of the change.
                    weight = 1 / max(relative_change, clip_min)

                elif median_deployed_cap == 0:
                    # If the median_deployed_cap is 0, we want to encourage the deployment of this technology.
                    weight = 0.0

                new_weights[component][attr][tech] = weight

    return new_weights


def calculate_weights_evolving_average(
    latest_spore: pypsa.Network,
    deployment_history: list[dict],
    spore_techs_dict: dict,
    clip_min: float = 0.001,
) -> dict:
    """Calculates weights based on the reciprocal of the relative distance from the evolving average capacity."""
    average_deployment = calculate_average_deployment(
        deployment_history, spore_techs_dict
    )
    latest_deployment = get_tech_deployment(latest_spore, spore_techs_dict)

    new_weights = {}

    for component, attrs in average_deployment.items():
        new_weights.setdefault(component, {})

        for attr, techs_average_deployment_map in attrs.items():
            new_weights[component].setdefault(attr, {})

            for tech, average_deployed_cap in techs_average_deployment_map.items():
                latest_deployed_cap = (
                    latest_deployment.get(component, {}).get(attr, {}).get(tech, 0)
                )

                if average_deployed_cap > 0:
                    relative_change = (
                        abs(latest_deployed_cap - average_deployed_cap)
                        / average_deployed_cap
                    )

                    # If the relative_change is 0 (latest_deployed_cap == average), we give the relative_change a small
                    # value which will give it a large penalty (weight) since we take the reciprocal of the change.
                    weight = 1 / max(relative_change, clip_min)

                elif average_deployed_cap == 0:
                    # If the average_deployed_cap is 0, we want to encourage the deployment of this technology.
                    weight = 0.0

                new_weights[component][attr][tech] = weight

    return new_weights


def calculate_relative_deployment(
    n: pypsa.Network, spore_techs_dict: dict, bigM: float = 1e10
) -> dict:
    """Calculate the relative deployment (p_nom_opt/p_nom_max) of techs in the optimized network."""
    relative_deployment = {}

    for component, attrs in spore_techs_dict.items():
        extendable_techs = n.get_extendable_i(component)

        if extendable_techs.empty:
            continue

        df_name = NOMINAL_ATTRS[component]["dataframe_name"]
        df = getattr(n, df_name)
        capacity_attr = NOMINAL_ATTRS[component]["capacity_attribute"]
        max_caps = df[f"{capacity_attr}_max"][extendable_techs]
        max_caps = max_caps.replace(np.inf, bigM)
        opt_caps = df[f"{capacity_attr}_opt"][extendable_techs]
        rel_caps = (opt_caps / max_caps).to_dict()  # pd.Series to dict conversion

        relative_deployment[component] = {capacity_attr: rel_caps}

    return relative_deployment


def calculate_average_deployment(
    deployment_history: list[dict], spore_techs_dict: dict
) -> dict:
    """Calculates the average capacity deployment of spore technologies."""
    average_deployment = {}

    for component, attrs in spore_techs_dict.items():
        average_deployment.setdefault(component, {})
        for attr, techs_map in attrs.items():
            average_deployment[component].setdefault(attr, {})
            for tech in techs_map:
                total_deployed_capacity = sum(
                    deployed_capacity.get(component, {}).get(attr, {}).get(tech, 0)
                    for deployed_capacity in deployment_history
                )

                average_deployment[component][attr][tech] = (
                    total_deployed_capacity / len(deployment_history)
                )

    return average_deployment


def calculate_median_deployment(
    deployment_history: list[dict], spore_techs_dict: dict
) -> dict:
    """Calculates the median capacity deployment of spore technologies."""
    median_deployment = {}

    for component, attrs in spore_techs_dict.items():
        median_deployment.setdefault(component, {})
        for attr, techs_map in attrs.items():
            median_deployment[component].setdefault(attr, {})
            for tech in techs_map:
                # Step 1: Collect all historical deployment values for the specific tech into a list.
                deployed_capacities = [
                    deployed_capacity.get(component, {}).get(attr, {}).get(tech, 0)
                    for deployed_capacity in deployment_history
                ]

                # Step 2: Calculate the median of that list.
                # Handle the edge case where the history is empty to avoid an error.
                if not deployed_capacities:
                    median_value = 0.0
                else:
                    median_value = statistics.median(deployed_capacities)

                median_deployment[component][attr][tech] = median_value

    return median_deployment


def initialize_weights(configuration: dict) -> dict:
    """Initialize the weights of all extendable technologies in the network to zero."""
    weights = {}

    for tech in configuration["spore_technologies"]:
        for comp, comp_info in tech.items():
            attr = comp_info["attribute"]
            comp_weights = {k: 0 for k in comp_info["index"]}
            weights.setdefault(comp, {})[attr] = comp_weights

    return weights


def calculate_weights_first_iteration(
    n: pypsa.Network, spores_mode: str, prev_weights: dict
):
    """Calculate weights for the first iteration of SPORES based on spores_mode.

    This function ensures that we either start with zero weights (intensify) or
    start with weights based on the least-cost solution (diversify).

    This function assumes that `network` is the least-cost optimized network and
    `prev_weights` is 0 for all techs. There are 2 methods to compute new weights:

    If `spores_mode` is "intensify and diversify", it sets the `new_weights` to
    be thesame as the `prev_weights`. This is done so that the start of the
    exploration for subsequent SPORES is focused around the previously found
    intensified solution.

    If `spores_mode` is "diversify", it simply calls the `calculate_weights`
    function that compute `new_weights` as the sum of the `prev_weight`(which is
    0 in the first iteration) and the relative deployment of techs in the
    previously found diversified solution (in which case, is the least-cost
    solution since it is the first iteration). This is done so that the
    exploration of the solution space starts from the least-cost solution.
    """
    if spores_mode == "intensify and diversify":
        return prev_weights

    return calculate_weights_relative_deployment(n, prev_weights)


def get_tech_deployment(n: pypsa.Network, spore_techs_dict: dict) -> dict:
    """Get the deployed capacity (p_nom_opt) of spore techs in the optimized network."""
    deployment = {}

    for component, attrs in spore_techs_dict.items():
        extendable_techs = n.get_extendable_i(component)

        if extendable_techs.empty:
            continue

        df_name = NOMINAL_ATTRS[component]["dataframe_name"]
        df = getattr(n, df_name)
        capacity_attr = NOMINAL_ATTRS[component]["capacity_attribute"]
        opt_caps = df[f"{capacity_attr}_opt"][
            extendable_techs
        ].to_dict()  # pd.Series to dict conversion

        deployment[component] = {capacity_attr: opt_caps}

    return deployment


def get_tech_deployment_xarray(n: pypsa.Network, spore_techs_dict: dict) -> xr.DataArray:
    """Get the deployed capacity (p_nom_opt) of spore techs in the optimized network."""
    series_to_concat, keys_for_concat = [], []

    for component in spore_techs_dict.keys():
        extendable_techs = n.get_extendable_i(component)
        if extendable_techs.empty:
            continue

        df_name = NOMINAL_ATTRS[component]["dataframe_name"]
        capacity_attr = NOMINAL_ATTRS[component]["capacity_attribute"]

        df = getattr(n, df_name)
        capacities_series = df[f"{capacity_attr}_opt"][extendable_techs]

        series_to_concat.append(capacities_series)
        # 'extendable_techs' pd.Series already has 'asset' coordinate value, add component and attribute
        keys_for_concat.append((component, capacity_attr))

    # TODO: confirm level/coordinate name "asset"
    full_multiindex_series = pd.concat(series_to_concat, keys=keys_for_concat, names=["component", "attribute", "asset"])
    return full_multiindex_series.to_xarray()


def validate_spores_configuration(config: dict):
    """Validate a SPORES YAML config against the specified requirements."""
    if "SPORES" not in config:
        raise ValueError("Missing top-level key: 'SPORES'.")
    spores_config = config["SPORES"]

    # Must have config_name which will be used in the output folder name to save results.
    if (
        "config_name" not in spores_config
        or not isinstance(spores_config["config_name"], str)
        or not spores_config["config_name"].strip()
    ):
        raise ValueError("'config_name' must be provided as a non-empty string.")

    # Required keys
    required_keys_in_spores_config = [
        "objective_sense",
        "spores_slack",
        "num_spores",
        "weighting_method",
        "spores_mode",
        "diversification_coefficient",
        "spore_technologies",
    ]

    for key in required_keys_in_spores_config:
        if key not in spores_config:
            raise ValueError(f"Missing required key: '{key}'.")

    # objective_sense must be min for consistency.
    if spores_config["objective_sense"] != "min":
        raise ValueError(
            "'objective_sense' must be 'min'. To maximize, please set the 'diversification_coefficient' "
            "and/or 'intensification_coefficient' to negative."
        )

    # spores_slack must be between 0 and 1
    if not isinstance(spores_config["spores_slack"], numbers.Number) or not (
        0 <= spores_config["spores_slack"] <= 1
    ):
        raise ValueError("'spores_slack' must be a number between 0 and 1.")

    # num_spores must be integer >= 1
    if (
        not isinstance(spores_config["num_spores"], int)
        or spores_config["num_spores"] < 1
    ):
        raise ValueError("'num_spores' must be an integer >= 1.")

    # weighting_method must be valid
    if spores_config["weighting_method"] not in WEIGHTING_METHODS:
        raise ValueError(
            f"Unsupported {spores_config['weighting_method']=}, must be one of {WEIGHTING_METHODS}."
        )

    # spores_mode must be valid
    if spores_config["spores_mode"] not in ["diversify", "intensify and diversify"]:
        raise ValueError(
            "'spores_mode' must be either 'diversify' or 'intensify and diversify'."
        )

    # diversification_coefficient must be positive number
    diversification_coeff = spores_config["diversification_coefficient"]
    try:
        diversification_coeff = float(diversification_coeff)
    except (TypeError, ValueError):
        raise ValueError("'diversification_coefficient' must be a number.")

    if diversification_coeff <= 0:
        raise ValueError("'diversification_coefficient' must be a positive number.")

    # spore_technologies cannot be empty
    spore_technologies = spores_config["spore_technologies"]
    if not isinstance(spore_technologies, list) or not spore_technologies:
        raise ValueError("'spore_technologies' must be a non-empty list.")

    # Keys of spore_technologies must be in valid_tech_type
    valid_tech_type = NOMINAL_ATTRS.keys()
    for tech_top_key in spore_technologies:
        if not isinstance(tech_top_key, dict) or len(tech_top_key) != 1:
            raise ValueError(
                "Each element in 'spore_technologies' must be a dict with a single top-level pypsa-component key."
            )
        component = next(iter(tech_top_key))
        if component not in valid_tech_type:
            raise ValueError(
                f"Invalid pypsa-component '{component}' in 'spore_technologies'. Must be one of {valid_tech_type}."
            )

        # Extra sanity check: each must have attribute and index keys
        tech_data = tech_top_key[component]
        if "attribute" not in tech_data or not isinstance(tech_data["attribute"], str):
            raise ValueError(
                f"Component '{component}' must define an 'attribute' key with a string value."
            )
        if (
            "index" not in tech_data
            or not isinstance(tech_data["index"], list)
            or not tech_data["index"]
        ):
            raise ValueError(
                f"Component '{component}' must define a non-empty 'index' list."
            )

    # If spores_mode is "intensify and diversify", extra checks
    if spores_config["spores_mode"] == "intensify and diversify":
        try:
            spores_config["intensification_coefficient"] = float(
                spores_config["intensification_coefficient"]
            )
        except (KeyError, TypeError, ValueError):
            raise ValueError(
                "'intensification_coefficient' must be provided as a number for spores_mode 'intensify and diversify'."
            )
        if (
            "intensifiable_technologies" not in spores_config
            or not isinstance(spores_config["intensifiable_technologies"], list)
            or not spores_config["intensifiable_technologies"]
        ):
            raise ValueError(
                "'intensifiable_technologies' must be a non-empty list when 'spores_mode' is 'intensify and diversify'."
            )

    # Coupling rule: intensification_coefficient and intensifiable_technologies must be both present or both absent
    has_coeff = "intensification_coefficient" in spores_config
    has_intensifiable = "intensifiable_technologies" in spores_config
    if has_coeff != has_intensifiable:  # XOR
        raise ValueError(
            "'intensification_coefficient' and 'intensifiable_technologies' must be provided or omitted together."
        )

    # Extra check: No duplicate component-index pairs in spore_technologies
    seen_pairs = set()
    for tech in spore_technologies:
        comp = next(iter(tech))
        for idx in tech[comp]["index"]:
            pair = (comp, idx)
            if pair in seen_pairs:
                raise ValueError(f"Duplicate technology entry found: {pair}")
            seen_pairs.add(pair)

    return True


# ======================== Pypsa/linopy related code implementation section ========================
def optimize_model_and_assign_solution_to_network(
    n: pypsa.Network,
    m: linopy.Model,
    solver_options: dict,
    env: gp.Env = None,
) -> tuple[pypsa.Network, linopy.Model]:
    """Optimize a model and assign the solution back to the pypsa network for analysis."""
    solver_name = list(solver_options.keys())[0]
    kwargs = solver_options[solver_name]

    if solver_name == "gurobi" and env is not None:
        logger.info("Solving model with Gurobi using a managed environment.")
        kwargs["env"] = env
    else:
        logger.info(f"Solving model with {solver_name} without a managed environment.")

    m.solve(solver_name=solver_name, **kwargs)

    n.optimize.assign_solution()
    n.optimize.assign_duals()

    return n, m


def create_modified_model(
    n: pypsa.Network, configuration: dict, optimal_cost: float, weights: dict
) -> linopy.Model:
    """Create the modified model (with the new objective and budget constraint) from the least-cost network."""
    # 1. Access the underlying linopy model of the least-cost pypsa network
    m = n.optimize.create_model()

    # 2. Add the budget constraint to the model
    slack = configuration["spores_slack"]
    least_cost_objective = m.objective
    if not isinstance(least_cost_objective, linopy.LinearExpression):
        least_cost_objective = least_cost_objective.expression
    m.add_constraints(
        least_cost_objective <= (1 + slack) * optimal_cost, name="budget-constraint"
    )

    # 3. Modify the objective function
    m = modify_objective(n, m, weights, configuration)

    return m


def modified_model_for_spores_run(
    n: pypsa.Network,
    m: linopy.Model,
    configuration: dict,
    optimal_cost: float,
    weights: dict,
) -> linopy.Model:
    """Modify the model given model to add the new objective function and budget constraint."""
    # 1. Add the budget constraint to the model
    slack = configuration["spores_slack"]
    least_cost_objective = m.objective
    if not isinstance(least_cost_objective, linopy.LinearExpression):
        least_cost_objective = least_cost_objective.expression
    m.add_constraints(
        least_cost_objective <= (1 + slack) * optimal_cost, name="budget-constraint"
    )

    # 2. Modify the objective function
    m = modify_objective(n, m, weights, configuration)

    return m


def modify_objective(
    n: pypsa.Network, m: linopy.Model, weights: dict, configuration: dict
) -> linopy.Model:
    """Modify the objective function to optimize technology capacities instead of costs."""
    sense = parse_objective_sense(configuration["objective_sense"])
    spores_mode = configuration["spores_mode"]
    diversification_coeff = float(configuration.get("diversification_coefficient"))
    intensification_coeff = configuration.get("intensification_coefficient")
    if intensification_coeff is not None:
        intensification_coeff = float(intensification_coeff)
    intensifiable_technologies = configuration.get("intensifiable_technologies")

    objective_expressions = []
    for component, comp_info in weights.items():
        for component_attr, tech_weight_dict in comp_info.items():
            if component_attr != NOMINAL_ATTRS[component]["capacity_attribute"]:
                raise ValueError(
                    f"Unknown capacity attribute {component_attr} for {component}"
                )

            capacity_variable = m[f"{component}-{component_attr}"]

            # Build diversification terms
            tech_weight_table = (
                pd.Series(tech_weight_dict)
                .reindex(n.get_extendable_i(component))
                .fillna(0)
            )
            diversification_final_coeffs = (
                diversification_coeff * tech_weight_table * sense
            )

            # Build intensification terms
            intensification_final_coeffs = pd.Series(
                0.0, index=tech_weight_table.index
            )  # Start with zeros

            if spores_mode == "intensify and diversify" and intensification_coeff != 0:
                intensify_mask = tech_weight_table.index.isin(
                    intensifiable_technologies
                )
                intensification_value = intensification_coeff * sense
                # Apply the value only to the selected technologies
                intensification_final_coeffs[intensify_mask] = intensification_value

            # Add the coefficient Series together first
            combined_final_coeffs = (
                diversification_final_coeffs + intensification_final_coeffs
            )

            # 4. Create a single, clean LinearExpression
            objective_expressions.append(
                (combined_final_coeffs * capacity_variable).sum()
            )

    m.remove_objective()
    m.objective = sum(objective_expressions)

    return m


def parse_objective_sense(sense: str) -> int:
    """Parse the sense of the objective function."""
    if sense == "min":
        return 1
    elif sense == "max":
        return -1
    else:
        raise ValueError(f"Unknown sense: {sense}. Use 'min' or 'max'.")
