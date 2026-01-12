import pandas as pd
import pytest

from mga4all.examples import create_pypsa_network


@pytest.fixture(scope="module")
def spore_techs_dict():
    """Fixture for the SPORE technologies dictionary."""
    return {
        "Generator": {
            "p_nom": {
                "solar": 0,
                "wind": 0,
                "gas": 0,
            }
        }
    }


@pytest.fixture(scope="module")
def pypsa_network():
    yield create_pypsa_network()


def assert_nested_dict_approx(d1, d2):
    """Asserts that two nested dictionaries are approximately equal.

    Recursively compares dictionary keys and uses pytest.approx for
    floating-point value comparisons.
    """
    assert d1.keys() == d2.keys(), "Dictionary keys do not match"
    for key in d1:
        v1 = d1[key]
        v2 = d2[key]
        if isinstance(v1, dict) and isinstance(v2, dict):
            # If the value is another dictionary, recurse
            assert_nested_dict_approx(v1, v2)
        else:
            # Otherwise, compare the values directly using pytest.approx
            assert v1 == pytest.approx(v2), f"Value mismatch for key '{key}'"


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
