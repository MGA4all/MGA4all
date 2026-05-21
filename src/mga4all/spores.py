import logging

import gurobipy as gp
import linopy
import numpy as np
import pandas as pd
import pypsa

from .validate import (
    PYPSAConfig,
    PYPSA_DATAFRAME_NAMES,
    SporesConfig,
    WeightingMethod,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_spores(
    least_cost_network: pypsa.Network,
    spores_config: dict,
    solver_options: dict,
    upper_bound: int = 100,
) -> list[pypsa.Network]:
    """Run the SPORES optimization to generate multiple near-optimal solutions.

    Returns
    -------
    spore_networks: list[pypsa.Network]
    """
    config = PYPSAConfig(**spores_config).SPORES

    weighting_method = config.weighting_method
    asset_indices = get_asset_multi_index(config)
    max_capacities = get_max_capacities(least_cost_network, asset_indices)

    # Check if the network is already optimized, else raise an error.
    if not least_cost_network.is_solved:
        raise ValueError("The input network must be optimized before running SPORES.")
    # PyPSA does not allow copying networks with a solver_model attached, so remove in case it is there.
    least_cost_network.model.solver_model = None

    # First spore uses zeros or relative deployment as weights
    prev_weights = pd.Series(0.0, index=asset_indices, name="weights")
    if not config.intensify:
        relative_deployment = (
            get_deployment(least_cost_network, asset_indices) / max_capacities
        )
        prev_weights = calculate_weights_relative_deployment(
            relative_deployment, prev_weights
        )
    first_spore = evaluate_weights(
        least_cost_network, prev_weights, config, solver_options
    )
    spore_networks = [first_spore]
    deploy_his = pd.DataFrame(  # Record history for evolving_average/median methods
        {
            "least_cost": get_deployment(least_cost_network, asset_indices),
            "first_spore": get_deployment(first_spore, asset_indices),
        }
    )

    # Run SPORES
    for i in range(config.num_spores - 1):
        match weighting_method:
            case WeightingMethod.RANDOM:
                new_weights = set_weights_random(asset_indices, upper_bound)
            case (
                WeightingMethod.RELATIVE_DEPLOYMENT
                | WeightingMethod.RELATIVE_DEPLOYMENT_NORMALIZED
            ):
                rel_deployment = deploy_his.iloc[:, -1] / max_capacities
                normalize = (
                    weighting_method == WeightingMethod.RELATIVE_DEPLOYMENT_NORMALIZED
                )
                new_weights = calculate_weights_relative_deployment(
                    rel_deployment, prev_weights, normalize=normalize
                )
            case WeightingMethod.EVOLVING_AVERAGE:
                weights = deploy_his.mean(axis=1)
                new_weights = calculate_weights_evolving(
                    deploy_his.iloc[:, -1], weights
                )
            case WeightingMethod.EVOLVING_MEDIAN:
                weights = deploy_his.median(axis=1)
                new_weights = calculate_weights_evolving(
                    deploy_his.iloc[:, -1], weights
                )
            case _:
                raise RuntimeError(f"{weighting_method=} unknown")

        logger.debug(f"weights for iteration {i}: {new_weights}")
        new_spore = evaluate_weights(
            least_cost_network, new_weights, config, solver_options
        )
        spore_networks.append(new_spore)

        prev_weights = new_weights
        deploy_his[i] = get_deployment(new_spore, asset_indices)  # append column

    return spore_networks


def evaluate_weights(
    base_network: pypsa.Network,
    weights: pd.Series,
    config_data: SporesConfig,
    solver_options: dict,
) -> pypsa.Network:
    """Evaluate a PyPSA network with objective modified according to given weights

    Returns
    -------
    new_spore: PyPSA.Network
    """
    optimal_cost = (
        base_network.statistics.capex().sum() + base_network.statistics.opex().sum()
    )
    network = base_network.copy()
    # Create & optimize the modified model (has the new objective (tech capacities * weights) & budget constraints)
    modified_model = create_modified_model(network, config_data, optimal_cost, weights)
    new_spore, solved_model = optimize_model_and_assign_solution_to_network(
        network, modified_model, solver_options
    )
    return new_spore


def get_asset_multi_index(configuration: SporesConfig) -> pd.MultiIndex:
    """Unpack the spore technologies information into a flat datastructure."""
    entries = [
        (asset_group.component, asset_group.attribute, asset)
        for asset_group in configuration.spore_technologies
        for asset in asset_group.assets
    ]
    return pd.MultiIndex.from_tuples(entries, names=["component", "attribute", "asset"])


def set_weights_random(asset_indices: pd.MultiIndex, upper_bound: int) -> pd.Series:
    """Generates new weights using random numbers from a uniform distribution between 0 and upper_bound."""
    rng = np.random.default_rng()
    weights = rng.uniform(0, upper_bound, len(asset_indices))
    return pd.Series(weights, index=asset_indices, name="weights")


def get_max_capacities(
    n: pypsa.Network, asset_indices: pd.MultiIndex, bigM: float = 1e10
) -> pd.Series:
    """Retrieve the maximum capacity of assets from the network."""
    capacity_values = []
    for component, capacity_attr, asset in asset_indices:
        df_name = PYPSA_DATAFRAME_NAMES[component]
        df = getattr(n, df_name)
        # Set an actual value in case max is infinite
        max_caps = min(df[f"{capacity_attr}_max"][asset], bigM)
        capacity_values.append(max_caps)

    return pd.Series(capacity_values, index=asset_indices, name="maximum capacity")


def get_deployment(n: pypsa.Network, asset_indices: pd.MultiIndex) -> pd.Series:
    """Retrieve the deployment of assets in the optimized network."""
    deployment_values = []
    for component, capacity_attr, asset in asset_indices:
        df_name = PYPSA_DATAFRAME_NAMES[component]
        df = getattr(n, df_name)
        opt_caps = df[f"{capacity_attr}_opt"][asset]
        deployment_values.append(opt_caps)

    return pd.Series(deployment_values, index=asset_indices, name="deployment")


def calculate_weights_relative_deployment(
    relative_deployment: pd.Series, prev_weights: pd.Series, normalize: bool = False
) -> pd.Series:
    """Calculate new weights by adding the latest relative deployment to the previous weights,
    optionally normalized w.r.t. the max_weight.
    """
    new_weights = prev_weights + relative_deployment

    if normalize and (max_weight := new_weights.max()) > 0:
        new_weights /= max_weight

    return new_weights


def calculate_weights_evolving(
    latest_deployment: pd.Series,
    average_deployment: pd.Series,
    clip_min: float = 0.001,
) -> pd.Series:
    """Calculates weights based on the reciprocal of the relative distance from the evolving median or average capacity.

    Weighting can be done using average or median, depending on which function is given for `calculate_deployment`.

    When the median instead of the average is used for the weighting method, the weights are not skewed by an outlier
    spore that might have had an unusually large deployment of a specific technology. For example, if the deploy_his for
    a tech is [0, 0, 0, 0, 1000], the average would be 200. A new solution with 0 deployment would be penalized. While
    the median would be 0. A new solution with 0 deployment would get a weight of 0, identifying it as an underexplored.
    """
    relative_change = (
        latest_deployment - average_deployment
    ).abs() / average_deployment
    # If the relative_change is 0 (latest_deployed == mean or median), we give the relative_change a small
    # value which will give it a large penalty (weight) since we take the reciprocal of the change.
    relative_change[relative_change < clip_min] = clip_min

    new_weights = 1 / relative_change
    # If the deployment of an asset is 0, we want to encourage the deployment of this technology.
    new_weights[average_deployment < clip_min] = 0.0
    return new_weights


# ======================== Pypsa/linopy related code implementation section ========================
def optimize_model_and_assign_solution_to_network(
    n: pypsa.Network,
    m: linopy.Model,
    solver_options: dict,
    env: gp.Env | None = None,
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
    n: pypsa.Network,
    configuration: SporesConfig,
    optimal_cost: float,
    weights: pd.Series,
) -> linopy.Model:
    """Create the modified model (with the new objective and budget constraint) from the least-cost network."""
    # 1. Access the underlying linopy model of the least-cost pypsa network
    m = n.optimize.create_model(
        include_objective_constant=False
    )  # suppress FutureWarning about objective constant

    # 2. Add the budget constraint to the model
    slack = configuration.spores_slack
    least_cost_objective = m.objective
    if not isinstance(least_cost_objective, linopy.LinearExpression):
        least_cost_objective = least_cost_objective.expression
    m.add_constraints(
        least_cost_objective <= (1 + slack) * optimal_cost, name="budget-constraint"
    )

    # 3. Modify the objective function
    m = modify_objective(n, m, weights, configuration)

    return m


def modify_objective(
    n: pypsa.Network, m: linopy.Model, weights: pd.Series, configuration: SporesConfig
) -> linopy.Model:
    """Modify the objective function to optimize technology capacities instead of costs."""
    diversification_coeff = configuration.diversification_coefficient
    intensification_coeff = configuration.intensification_coefficient
    intensifiable_technologies = configuration.intensifiable_technologies

    group_levels = ["component", "attribute"]
    objective_expressions = []
    for (component, attribute), tech_weights in weights.groupby(level=group_levels):
        # The group-name index levels are still present with a single index value
        tech_weights.index = tech_weights.index.droplevel(group_levels)

        # If the index of `tech_weights` has a name, it won't match an unnamed index elsewhere,
        # which would result in n^2 expression elements instead of just n
        tech_weights.index.name = None

        capacity_variable = m[f"{component}-{attribute}"]

        diversification_final_coeffs = diversification_coeff * tech_weights

        # Build intensification terms, starting with zeros
        intensification_final_coeffs = pd.Series(0.0, index=tech_weights.index)

        if configuration.intensify and intensifiable_technologies:
            intensify_mask = tech_weights.index.isin(intensifiable_technologies)
            # Apply the value only to the selected technologies
            intensification_final_coeffs[intensify_mask] = intensification_coeff

        # Add the coefficient Series together first
        combined_final_coeffs = (
            diversification_final_coeffs + intensification_final_coeffs
        )

        # 4. Create a single, clean LinearExpression
        objective_expressions.append((combined_final_coeffs * capacity_variable).sum())

    m.remove_objective()
    m.objective = sum(objective_expressions)

    return m
