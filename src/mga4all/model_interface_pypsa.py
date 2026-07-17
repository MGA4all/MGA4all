import pypsa
import pandas as pd
from copy import deepcopy

#---- PyPSA model interface ----#

PYPSA_CAPACITY_VARIABLES = {
    "Generator": "p_nom",
    "Link": "p_nom",
    "Process": "p_nom",
    "StorageUnit": "p_nom",
    "Store": "e_nom",
    "Line": "s_nom"
}


# Match technologies targeted by the config to the corresponding PyPSA components
def match_config_techs_to_model_techs(config, n):
    
    diversified_techs = set(config["diversified_technologies"])

    if config["intensified_technologies"] and (len(config["intensified_technologies"]) != 0):
        intensified_techs = set(config["intensified_technologies"])
        config_techs = intensified_techs | diversified_techs
    else:
        intensified_techs = None
        config_techs = diversified_techs
        
    diversified_model_techs = {}
    intensified_model_techs = {}
    
    for comp in n.components:
        df = comp.static

        if "carrier" not in df.columns:
            continue

        carriers_present = set(df["carrier"].dropna().astype(str))
        matched_diversified = sorted(t for t in diversified_techs if t in carriers_present)
        if matched_diversified:
                diversified_model_techs[comp.name] = matched_diversified

        if intensified_techs is not None:
            matched_intensified = sorted(t for t in intensified_techs if t in carriers_present)
            if matched_intensified:
                intensified_model_techs[comp.name] = matched_intensified
    try:
        flat_intensified = [item for sublist in list(intensified_model_techs.values()) for item in sublist]
        flat_diversified = [item for sublist in list(diversified_model_techs.values()) for item in sublist]
        flat = flat_diversified + flat_intensified
    except:
        flat = [item for sublist in list(diversified_model_techs.values()) for item in sublist]

    print("Technologies {} not found in the model".format(sorted(list(config_techs - set(flat)))))

    if intensified_techs is not None:
        model_techs = {}
        model_techs["intensified"] = intensified_model_techs
        model_techs["diversified"] = diversified_model_techs
    else:
        model_techs = diversified_model_techs

    return model_techs

def extract_diversified_capacity(target_techs, n, spatial=False):

    component_tables = {
        "Generator": (n.generators, "p_nom_opt", "carrier", "bus"),
        "Link": (n.links, "p_nom_opt", "carrier", "bus0"),
        "Process": (n.processes, "p_nom_opt", "carrier", "bus0"),
        "StorageUnit": (n.storage_units, "p_nom_opt", "carrier", "bus"),
        "Store": (n.stores, "e_nom_opt", "carrier", "bus"),
        "Line": (n.lines, "s_nom_opt", "carrier", "bus0"),
    }

    if "intensified" in target_techs.keys():
        target_techs = target_techs["diversified"] # focus on diversity here
    else:
        pass
    
    deployed_capacity_assets = {}
    deployed_capacity_buses = {}
        
    for component, carriers in target_techs.items():
        df, opt_col, carrier_col, bus_col = component_tables[component]
    
        filtered = df[df[carrier_col].isin(carriers)]
    
        deployed_capacity_assets[component] = filtered[opt_col].to_dict()
    
        deployed_capacity_buses[component] = (
            filtered.groupby(carrier_col)[opt_col]
            .sum()
            .to_dict()
        )

    if spatial==True:
        deployed_capacity = deployed_capacity_assets
    else:
        deployed_capacity = deployed_capacity_buses

    deployed_capacity_series = pd.Series({
        k: v
        for inner in deployed_capacity.values()
        for k, v in inner.items()
    })
    
    return deployed_capacity_series

def extract_intensified_capacity(target_techs, test_config, n, spatial=False):

    component_tables = {
        "Generator": (n.generators, "p_nom_opt", "carrier", "bus"),
        "Link": (n.links, "p_nom_opt", "carrier", "bus0"),
        "Process": (n.processes, "p_nom_opt", "carrier", "bus0"),
        "StorageUnit": (n.storage_units, "p_nom_opt", "carrier", "bus"),
        "Store": (n.stores, "e_nom_opt", "carrier", "bus"),
        "Line": (n.lines, "s_nom_opt", "carrier", "bus0"),
    }


    if "intensified" in target_techs.keys() and (len(target_techs["intensified"]) != 0):
        
        mapping = {
            k: v
            for k, v in zip(
                test_config["intensified_technologies"],
                test_config["intensification_coefficient"],
            )
        }

        target_techs = target_techs["intensified"] # focus on intensity here

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
                carrier: mapping[carrier]
                for carrier in filtered[carrier_col].unique()
            }
            
        if spatial:
            deployed_capacity = deployed_capacity_assets
        else:
            deployed_capacity = deployed_capacity_buses

        deployed_capacity_series = pd.Series({
            k: v
            for inner in deployed_capacity.values()
            for k, v in inner.items()
        })
    
    else:
        deployed_capacity_series = 0
    
    return deployed_capacity_series



def extract_minimum_feasible_cost(n):

    optimal_cost = n.statistics.capex().sum() + n.statistics.opex().sum()
    fixed_cost = n.statistics.installed_capex().sum()

    true_optimal_cost = optimal_cost - fixed_cost

    return true_optimal_cost

def create_mga_model(n):

    n.model.solver_model = None
    n_mga = n.copy() # Network object
    
    m_mga = n_mga.optimize.create_model(include_objective_constant=False) # Model object

    return (n_mga, m_mga)

def add_slack_constraint(m_mga, true_optimal_cost, slack):
    
    original_objective = m_mga.objective
    cost_expr = (
        original_objective
        if not hasattr(original_objective, "expression")
        else original_objective.expression
    )
    
    m_mga.add_constraints(
        cost_expr <= (1 + slack) * (true_optimal_cost),
        name="budget",
    )

    return

def convert_linear_weights_into_pypsa(n, mga_weights_series, target_techs, spatial=False):

    pypsa_component_tables = {
        "Generator": n.generators,
        "Link": n.links,
        "Process": n.processes,
        "StorageUnit": n.storage_units,
        "Store": n.stores,
        "Line": n.lines,
    }
        
    s = mga_weights_series
    pypsa_weights = {}

    if "intensified" in target_techs.keys():    
        merged_target_techs = deepcopy(target_techs["diversified"])

        for component, carriers in target_techs["intensified"].items():
            if component in merged_target_techs:
                # Add only carriers that are not already present
                merged_target_techs[component].extend(
                    carrier for carrier in carriers
                    if carrier not in merged_target_techs[component]
                )
            else:
                merged_target_techs[component] = carriers.copy()
    else:
        merged_target_techs = target_techs

    for component, techs in merged_target_techs.items():
        var = PYPSA_CAPACITY_VARIABLES[component]
        df = pypsa_component_tables[component]

        if spatial==True:
            names = df.index[df["carrier"].isin(techs)]
            coeffs = s.loc[names]

        elif spatial == False:
            coeffs = {}
            for tech in techs:
                if tech not in s.index:
                    continue
                names = df.index[df["carrier"] == tech]
                coeffs.update({name: s.loc[tech] for name in names})
            coeffs = pd.Series(coeffs)

        else:
            raise ValueError("`spatial` must be a boolean.")

        pypsa_weights[component] = {var: coeffs}

    return pypsa_weights

def assign_mga_objective(n_mga, m_mga, mga_weights_series, target_techs, spatial=False):

    weight_dict = convert_linear_weights_into_pypsa(n_mga, mga_weights_series, target_techs, spatial)
    
    mga_obj = n_mga.optimize.build_linexpr_from_weights(
        weight_dict,
        model=m_mga,
    )

    m_mga.add_objective(mga_obj, overwrite=True)

    return
    
#--------------------------------#
