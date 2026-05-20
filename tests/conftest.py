import yaml

import pandas as pd
import pytest

from mga4all.examples import create_pypsa_network


@pytest.fixture(scope="module")
def asset_indices():
    """Fixture for the SPORE technologies dictionary."""
    return pd.MultiIndex.from_tuples(
        [
            ("Generator", "p_nom", "solar"),
            ("Generator", "p_nom", "wind"),
            ("Generator", "p_nom", "gas"),
        ],
        names=["component", "attribute", "asset"],
    )


@pytest.fixture(scope="module")
def pypsa_network():
    yield create_pypsa_network()


class MockPypsaNetwork:
    """A mock object that mimics a pypsa.Network for testing purposes.

    It's designed to work with the `calculate_relative_deployment` function.
    """

    def __init__(self, p_nom_opt_data, p_nom_max_data=None):
        techs = list(p_nom_opt_data.keys())
        p_nom_opt = list(p_nom_opt_data.values())

        # Assume p_nom_max is 1.0 for all techs for simplicity, unless specified
        if p_nom_max_data is None:
            p_nom_max = [1.0] * len(techs)
        else:
            p_nom_max = list(p_nom_max_data.values())

        self.generators = pd.DataFrame(
            {"p_nom_opt": p_nom_opt, "p_nom_max": p_nom_max}, index=techs
        )

    def get_extendable_i(self, component):
        if component == "Generator":
            return self.generators.index
        return pd.Index([])


@pytest.fixture()
def pypsa_spores_config():
    """Fixture for a sample SPORES configuration."""
    with open("configs/spores_configs/test_spores_configs/test_diversify.yaml") as f:
        config = yaml.safe_load(f)
    return config
