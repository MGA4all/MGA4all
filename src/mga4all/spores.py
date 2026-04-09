import logging
import numbers
from typing import Callable, Iterable

import gurobipy as gp
import linopy
import numpy as np
import pandas as pd
import pypsa

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


PYPSA_DATAFRAME_NAMES = {
    "Generator": "generators",
    "Line": "lines",
    "Transformer": "transformers",
    "Link": "links",
    "Store": "stores",
    "StorageUnit": "storage_units",
}

WEIGHTING_METHODS = frozenset(
    [
        "random",
        "evolving_median",
        "evolving_average",
        "relative_deployment",
        "relative_deployment_normalized",
    ]
)


def run_spores(
    least_cost_network: pypsa.Network,
    spores_config: dict,
    solver_options: dict,
    weighting_method: str | None = None,
    upper_bound: int = 100,
) -> tuple[
    dict[str, pypsa.Network],
    dict[str, pd.Series],
    dict[str, linopy.Model],
    list[pd.Series],
]:
    """Run the SPORES optimization to generate multiple near-optimal solutions."""
    # Validate the SPORES configuration.
    validate_spores_configuration(spores_config)

    config_data = spores_config["SPORES"]

    # Build nested dict containing spore_techs once to avoid rebuilding again in downstream functions.
    asset_indices = get_asset_multi_index(config_data)

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
    deploy_his = [get_deployment(least_cost_network, asset_indices)]

    # Clean up model state so we can make a copy and avoid rebuilding inside the spores loop. PyPSA does not allow
    # copying networks with a solver_model attached, so we need to remove it first.
    if least_cost_network and hasattr(least_cost_network.model, "solver_model"):
        least_cost_network.model.solver_model = None

    # Run SPORES
    for i in range(1, config_data["num_spores"] + 1):
        network = least_cost_network.copy()

        if i == 1:
            # Previous weights are needed for the relative_deployment weighting methods.
            prev_weights = initialize_weights(asset_indices)

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
                new_weights = set_weights_random(asset_indices, upper_bound)

            elif weighting_method == "relative_deployment":
                new_weights = calculate_weights_relative_deployment(
                    prev_spore, prev_weights
                )

            elif weighting_method == "relative_deployment_normalized":
                new_weights = calculate_weights_relative_deployment(
                    prev_spore, prev_weights, normalize=True
                )

            elif weighting_method == "evolving_median":
                deployment = pd.concat(deploy_his, axis="columns").median(axis=1)
                new_weights = calculate_weights_evolving(
                    prev_spore, deployment
                )

            elif weighting_method == "evolving_average":
                deployment = pd.concat(deploy_his, axis="columns").mean(axis=1)
                new_weights = calculate_weights_evolving(
                    prev_spore, deployment
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
        deploy_his.append(get_deployment(new_spore, asset_indices))

    return spore_networks, weights, spore_models, deploy_his


def get_asset_multi_index(configuration: dict) -> pd.MultiIndex:
    """Unpack the spore technologies information into a flat datastructure."""
    entries = [
        (component_name, component_info["attribute"], asset)
        for technology in configuration["spore_technologies"]
        for component_name, component_info in technology.items()
        for asset in component_info["index"]
    ]
    return pd.MultiIndex.from_tuples(entries, names=["component", "attribute", "asset"])


def initialize_weights(indices: pd.MultiIndex) -> pd.Series:
    """Initialize the weights of all extendable technologies in the network."""
    return pd.Series(0.0, index=indices, name="weights")


def set_weights_random(asset_indices: pd.MultiIndex, upper_bound: int) -> pd.Series:
    """Generates new weights using random numbers from a uniform distribution between 0 and upper_bound."""
    rng = np.random.default_rng()
    weights = rng.uniform(0, upper_bound, len(asset_indices))
    return pd.Series(weights, index=asset_indices, name="weights")


def get_deployment(
    n: pypsa.Network, asset_indices: pd.MultiIndex, bigM: float = 1e10, relative: bool=False
) -> pd.Series:
    """Calculate the (relative) deployment of assets in the optimized network."""
    deployment_values = []
    for component, capacity_attr, asset in asset_indices:
        df_name = PYPSA_DATAFRAME_NAMES[component]
        df = getattr(n, df_name)
        opt_caps = df[f"{capacity_attr}_opt"][asset]

        if relative:
            # set actual value in case max is infinite
            max_caps = min(df[f"{capacity_attr}_max"][asset], bigM)
            deployment_values.append(opt_caps / max_caps)
        else:
            deployment_values.append(opt_caps)

    return pd.Series(deployment_values, index=asset_indices, name="deployment")


def calculate_weights_relative_deployment(
    n: pypsa.Network, prev_weights: pd.Series, normalize: bool=False
) -> pd.Series:
    """Calculate new weights by adding the latest relative deployment to the previous weights,
    optionally normalized w.r.t. the max_weight.
    """
    relative_deployment = get_deployment(n, prev_weights.index, relative=True)
    new_weights =  prev_weights + relative_deployment

    max_weight = new_weights.max()
    if normalize and max_weight > 0:
        new_weights /= max_weight

    return new_weights


def calculate_weights_evolving(
    latest_spore: pypsa.Network,
    deployment: pd.Series,
    clip_min: float = 0.001,
) -> pd.Series:
    """Calculates weights based on the reciprocal of the relative distance from the evolving median or average capacity.

    Weighting can be done using average or median, depending on which function is given for `calculate_deployment`.

    When the median instead of the average is used for the weighting method, the weights are not skewed by an outlier
    spore that might have had an unusually large deployment of a specific technology. For example, if the deploy_his for
    a tech is [0, 0, 0, 0, 1000], the average would be 200. A new solution with 0 deployment would be penalized. While
    the median would be 0. A new solution with 0 deployment would get a weight of 0, identifying it as an underexplored.
    """
    indices = deployment.index
    latest_deployment = get_deployment(latest_spore, indices)

    relative_change = (latest_deployment - deployment).abs() / deployment
    # If the relative_change is 0 (latest_deployed == mean or median), we give the relative_change a small
    # value which will give it a large penalty (weight) since we take the reciprocal of the change.
    relative_change[relative_change < clip_min] = clip_min

    new_weights = 1 / relative_change
    # If the deployment of an asset is 0, we want to encourage the deployment of this technology.
    new_weights[deployment == 0] = 0.0
    return new_weights


def calculate_weights_first_iteration(
    n: pypsa.Network, spores_mode: str, prev_weights: pd.Series
) -> pd.Series:
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
    valid_tech_type = PYPSA_DATAFRAME_NAMES.keys()
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
    n: pypsa.Network, configuration: dict, optimal_cost: float, weights: pd.Series
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
    weights: pd.Series,
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
    n: pypsa.Network, m: linopy.Model, weights: pd.Series, configuration: dict
) -> linopy.Model:
    """Modify the objective function to optimize technology capacities instead of costs."""
    sense = parse_objective_sense(configuration["objective_sense"])
    spores_mode = configuration["spores_mode"]
    diversification_coeff = float(configuration.get("diversification_coefficient"))
    intensification_coeff = configuration.get("intensification_coefficient")
    if intensification_coeff is not None:
        intensification_coeff = float(intensification_coeff)
    intensifiable_technologies = configuration.get("intensifiable_technologies")

    group_levels = ["component", "attribute"]
    objective_expressions = []
    for (component, attribute), tech_weights in weights.groupby(level=group_levels):
        # The group-name index levels are still present with a single index value
        tech_weights.index = tech_weights.index.droplevel(group_levels)

        # If the index of `tech_weights` has a name, it won't match an unnamed index elsewhere,
        # which would result in n^2 expression elements instead of just n
        tech_weights.index.name = None

        capacity_variable = m[f"{component}-{attribute}"]

        diversification_final_coeffs = diversification_coeff * tech_weights * sense

        # Build intensification terms, starting with zeros
        intensification_final_coeffs = pd.Series(0.0, index=tech_weights.index)

        if spores_mode == "intensify and diversify" and intensification_coeff != 0:
            intensify_mask = tech_weights.index.isin(intensifiable_technologies)
            intensification_value = intensification_coeff * sense
            # Apply the value only to the selected technologies
            intensification_final_coeffs[intensify_mask] = intensification_value

        # Add the coefficient Series together first
        combined_final_coeffs = (
            diversification_final_coeffs + intensification_final_coeffs
        )

        # 4. Create a single, clean LinearExpression
        objective_expressions.append((combined_final_coeffs * capacity_variable).sum())

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
