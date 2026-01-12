import pandas as pd
import pypsa


def create_pypsa_network(num_snapshots=24) -> pypsa.Network:
    """Create a simple PyPSA network."""
    base_network = pypsa.examples.scigrid_de()
    example_network = pypsa.Network(
        snapshots=pd.date_range("2025-01-01", periods=num_snapshots, freq="h")
    )

    carriers = ["AC", "transmission", "gas", "solar", "wind", "battery"]
    for car in carriers:
        example_network.add("Carrier", car)

    example_network.add("Bus", "bus1", carrier="AC")
    example_network.add("Bus", "bus2", carrier="AC")

    example_network.add(
        "Line",
        "line",
        carrier="transmission",
        bus0="bus1",
        bus1="bus2",
        s_nom=0,
        s_nom_extendable=True,
        s_nom_max=1000,
        capital_cost=1000,
    )

    # Add a candidate gas generator at bus1
    example_network.add(
        "Generator",
        "OCGT",
        carrier="gas",
        bus="bus1",
        p_nom=0,
        p_nom_extendable=True,
        p_nom_max=1000,
        capital_cost=2000,
        marginal_cost=0,
    )

    # Add load at one bus2
    example_network.add(
        "Load",
        "demand",
        bus="bus2",
        p_set=base_network.loads_t.p_set.loc[:, "382_220kV"].values[:num_snapshots],
    )

    # Add candidate technologies to be built at either buses
    for bus in example_network.buses.index:
        example_network.add(
            "Generator",
            f"solar_{bus}",
            carrier="solar",
            bus=bus,
            p_nom=0,
            p_nom_extendable=True,
            p_nom_max=1000,
            capital_cost=400,
            marginal_cost=0,
            p_max_pu=base_network.generators_t.p_max_pu.loc[
                :, "384_220kV Solar"
            ].values[:num_snapshots],
        )

        example_network.add(
            "Generator",
            f"wind_{bus}",
            carrier="wind",
            bus=bus,
            p_nom=0,
            p_nom_extendable=True,
            p_nom_max=1000,
            capital_cost=500,
            marginal_cost=0,
            p_max_pu=base_network.generators_t.p_max_pu.loc[
                :, "457 Wind Onshore"
            ].values[:num_snapshots],
        )

        example_network.add(
            "StorageUnit",
            f"battery_{bus}",
            carrier="battery",
            bus=bus,
            p_nom=0,
            p_nom_extendable=True,
            p_nom_max=300,
            max_hours=5,
            capital_cost=500,
            marginal_cost=0,
            efficiency_store=0.95,
            efficiency_dispatch=0.95,
            cyclic_state_of_charge=False,
        )

    return example_network
