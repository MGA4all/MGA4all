import yaml

import pytest

from mga4all.examples import create_pypsa_network
from mga4all.mga_hop_skip_jump import hop_skip_jump_algorithm
from mga4all.mga_random_directions import random_directions_algorithm
from mga4all.mga_spores import spores_algorithm, spores_algorithm_adaptive
from mga4all.validate import HopSkipJumpConfig, RandomDirectionsConfig, SPORESConfig


@pytest.fixture(scope="module")
def network():
    """Fixture for repeated use of the example pypsa network."""
    network = create_pypsa_network()
    network.optimize()
    return network


@pytest.fixture(scope="module")
def hop_skip_jump_config():
    """Fixture for basic hop-skip-jump configuration."""
    with open("configs/test_config_hop_skip_jump.yaml") as f:
        return HopSkipJumpConfig.model_validate(yaml.safe_load(f))


@pytest.fixture(scope="module")
def random_directions_config():
    """Fixture for basic random-directions configuration."""
    with open("configs/test_config_random_directions.yaml") as f:
        return RandomDirectionsConfig.model_validate(yaml.safe_load(f))


@pytest.fixture(scope="module")
def spores_diversify_config():
    """Fixture for SPORES configuration using only diversification."""
    with open("configs/test_config_spores_diversify_only.yaml") as f:
        return SPORESConfig.model_validate(yaml.safe_load(f))


@pytest.fixture(scope="module")
def spores_intensify_config():
    """Fixture for SPORES configuration using diversification and intensification."""
    with open("configs/test_config_spores_diversify_intensify.yaml") as f:
        return SPORESConfig.model_validate(yaml.safe_load(f))


@pytest.mark.parametrize(
    ["configuration", "algorithm"],
    [
        ("hop_skip_jump_config", hop_skip_jump_algorithm),
        ("random_directions_config", random_directions_algorithm),
        ("spores_diversify_config", spores_algorithm),
        ("spores_diversify_config", spores_algorithm),
        ("spores_intensify_config", spores_algorithm_adaptive),
        ("spores_intensify_config", spores_algorithm_adaptive),
    ],
)
def test_examples(network, configuration, algorithm, request):
    """Test that each algorithm runs without error with the sample configuration."""
    config = request.getfixturevalue(configuration)  # fetch config from fixture name
    algorithm(config, network)
