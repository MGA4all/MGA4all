import pandas as pd
from copy import deepcopy

PYPSA_CAPACITY_VARIABLES = {
    "Generator": "p_nom",
    "Link": "p_nom",
    "Process": "p_nom",
    "StorageUnit": "p_nom",
    "Store": "e_nom",
    "Line": "s_nom",
}


def match_config_techs_to_model_techs(config, network):
    diversified_techs = set(config["diversified_technologies"])

    if config["intensified_technologies"] and (
        len(config["intensified_technologies"]) != 0
    ):
        intensified_techs = set(config["intensified_technologies"])
        config_techs = intensified_techs | diversified_techs
    else:
        intensified_techs = None
        config_techs = diversified_techs

    diversified_model_techs = {}
    intensified_model_techs = {}

    for component in network.components:
        df = component.static

        if "carrier" not in df.columns:
            continue

        carriers_present = set(df["carrier"].dropna().astype(str))
        matched_diversified = sorted(
            t for t in diversified_techs if t in carriers_present
        )
        if matched_diversified:
            diversified_model_techs[component.name] = matched_diversified

        if intensified_techs is not None:
            matched_intensified = sorted(
                t for t in intensified_techs if t in carriers_present
            )
            if matched_intensified:
                intensified_model_techs[component.name] = matched_intensified

    try:
        flat_intensified = [
            item
            for sublist in list(intensified_model_techs.values())
            for item in sublist
        ]
        flat_diversified = [
            item
            for sublist in list(diversified_model_techs.values())
            for item in sublist
        ]
        flat = flat_diversified + flat_intensified
    except Exception:
        flat = [
            item
            for sublist in list(diversified_model_techs.values())
            for item in sublist
        ]

    print(
        "Technologies {} not found in the model".format(
            sorted(list(config_techs - set(flat)))
        )
    )

    if intensified_techs is not None:
        model_techs = {}
        model_techs["intensified"] = intensified_model_techs
        model_techs["diversified"] = diversified_model_techs
    else:
        model_techs = diversified_model_techs

    return model_techs


def extract_diversified_capacity(target_techs, network, spatial=False):
    component_tables = {
        "Generator": (network.generators, "p_nom_opt", "carrier", "bus"),
        "Link": (network.links, "p_nom_opt", "carrier", "bus0"),
        "Process": (network.processes, "p_nom_opt", "carrier", "bus0"),
        "StorageUnit": (network.storage_units, "p_nom_opt", "carrier", "bus"),
        "Store": (network.stores, "e_nom_opt", "carrier", "bus"),
        "Line": (network.lines, "s_nom_opt", "carrier", "bus0"),
    }

    if "intensified" in target_techs.keys():
        target_techs = target_techs["diversified"]  # focus on diversity here

    deployed_capacity_assets = {}
    deployed_capacity_buses = {}

    for component, carriers in target_techs.items():
        df, opt_col, carrier_col, bus_col = component_tables[component]

        filtered = df[df[carrier_col].isin(carriers)]

        deployed_capacity_assets[component] = filtered[opt_col].to_dict()

        deployed_capacity_buses[component] = (
            filtered.groupby(carrier_col)[opt_col].sum().to_dict()
        )

    if spatial:
        deployed_capacity = deployed_capacity_assets
    else:
        deployed_capacity = deployed_capacity_buses

    deployed_capacity_series = pd.Series(
        {k: v for inner in deployed_capacity.values() for k, v in inner.items()}
    )

    return deployed_capacity_series


def extract_intensified_capacity(target_techs, test_config, network, spatial=False):
    component_tables = {
        "Generator": (network.generators, "p_nom_opt", "carrier", "bus"),
        "Link": (network.links, "p_nom_opt", "carrier", "bus0"),
        "Process": (network.processes, "p_nom_opt", "carrier", "bus0"),
        "StorageUnit": (network.storage_units, "p_nom_opt", "carrier", "bus"),
        "Store": (network.stores, "e_nom_opt", "carrier", "bus"),
        "Line": (network.lines, "s_nom_opt", "carrier", "bus0"),
    }

    if "intensified" in target_techs.keys() and (len(target_techs["intensified"]) != 0):
        mapping = {
            k: v
            for k, v in zip(
                test_config["intensified_technologies"],
                test_config["intensification_coefficient"],
            )
        }

        target_techs = target_techs["intensified"]  # focus on intensity here

        deployed_capacity_assets = {}
        deployed_capacity_buses = {}

        for component, carriers in target_techs.items():
            df, opt_col, carrier_col, bus_col = component_tables[component]

            filtered = df[df[carrier_col].isin(carriers)]

            # spatial=True: asset -> coefficient
            deployed_capacity_assets[component] = {
                asset: mapping[carrier]
                for asset, carrier in filtered[carrier_col].items()
            }

            # spatial=False: carrier -> coefficient
            deployed_capacity_buses[component] = {
                carrier: mapping[carrier] for carrier in filtered[carrier_col].unique()
            }

        if spatial:
            deployed_capacity = deployed_capacity_assets
        else:
            deployed_capacity = deployed_capacity_buses

        deployed_capacity_series = pd.Series(
            {k: v for inner in deployed_capacity.values() for k, v in inner.items()}
        )

    else:
        deployed_capacity_series = 0

    return deployed_capacity_series


def extract_minimum_feasible_cost(network):
    optimal_cost = network.statistics.capex().sum() + network.statistics.opex().sum()
    fixed_cost = network.statistics.installed_capex().sum()

    true_optimal_cost = optimal_cost - fixed_cost

    return true_optimal_cost


def create_mga_model(network):
    network.model.solver_model = None
    network_mga = network.copy()  # Network object

    model_mga = network_mga.optimize.create_model(
        include_objective_constant=False
    )  # Model object

    return network_mga, model_mga


def add_slack_constraint(model_mga, true_optimal_cost, slack):
    original_objective = model_mga.objective
    cost_expr = (
        original_objective
        if not hasattr(original_objective, "expression")
        else original_objective.expression
    )

    model_mga.add_constraints(
        cost_expr <= (1 + slack) * true_optimal_cost,
        name="budget",
    )


def convert_linear_weights_into_pypsa(
    network, mga_weights_series, target_techs, spatial=False
):
    pypsa_component_tables = {
        "Generator": network.generators,
        "Link": network.links,
        "Process": network.processes,
        "StorageUnit": network.storage_units,
        "Store": network.stores,
        "Line": network.lines,
    }

    weights = mga_weights_series
    pypsa_weights = {}

    if "intensified" in target_techs.keys():
        merged_target_techs = deepcopy(target_techs["diversified"])

        for component, carriers in target_techs["intensified"].items():
            if component in merged_target_techs:
                # Add only carriers that are not already present
                merged_target_techs[component].extend(
                    carrier
                    for carrier in carriers
                    if carrier not in merged_target_techs[component]
                )
            else:
                merged_target_techs[component] = carriers.copy()
    else:
        merged_target_techs = target_techs

    for component, techs in merged_target_techs.items():
        var = PYPSA_CAPACITY_VARIABLES[component]
        df = pypsa_component_tables[component]

        if spatial:
            names = df.index[df["carrier"].isin(techs)]
            coeffs = weights.loc[names]

        else:
            coeffs = {}
            for tech in techs:
                if tech not in weights.index:
                    continue
                names = df.index[df["carrier"] == tech]
                coeffs.update({name: weights.loc[tech] for name in names})
            coeffs = pd.Series(coeffs)

        pypsa_weights[component] = {var: coeffs}

    return pypsa_weights


def assign_mga_objective(
    network_mga, model_mga, mga_weights_series, target_techs, spatial=False
):
    weight_dict = convert_linear_weights_into_pypsa(
        network_mga, mga_weights_series, target_techs, spatial
    )

    mga_obj = network_mga.optimize.build_linexpr_from_weights(
        weight_dict,
        model=model_mga,
    )

    model_mga.add_objective(mga_obj, overwrite=True)
