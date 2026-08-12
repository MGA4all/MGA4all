import pytest
from pydantic import ValidationError

from mga4all.validate import SPORESConfig


@pytest.mark.parametrize(
    "key",
    [
        "config_name",
        "model_interface",
        "alternatives",
        "cost_slack",
        "spatially_explicit",
        "diversification_coefficient",
        "intensification_coefficient",
        "diversified_technologies",
        "intensified_technologies",
    ],
)
def test_missing_keys(key, spores_diversify_config_dict):
    """Test that a missing key is correctly caught."""
    del spores_diversify_config_dict[key]
    with pytest.raises(ValidationError):
        SPORESConfig.model_validate(spores_diversify_config_dict)


@pytest.mark.parametrize(
    ["key", "value"],
    [
        ("config_name", ""),
        ("model_interface", ""),
        ("alternatives", -1),
        ("cost_slack", -1),
        ("diversification_coefficient", -1),
        ("intensification_coefficient", 1.5),
        ("diversified_technologies", 1),
        ("intensified_technologies", 1),
    ],
)
def test_invalid_values(key, value, spores_diversify_config_dict):
    """Test that invalid values are caught."""
    spores_diversify_config_dict[key] = value
    with pytest.raises(ValidationError):
        SPORESConfig.model_validate(spores_diversify_config_dict)


def test_duplicates(spores_diversify_config_dict):
    """Test that a duplicate asset in `diversified_technologies` is caught."""
    spores_diversify_config_dict["diversified_technologies"].append("OCGT")
    spores_diversify_config_dict["diversified_technologies"].append("OCGT")

    with pytest.raises(ValidationError) as exception_info:
        SPORESConfig.model_validate(spores_diversify_config_dict)
    assert "Duplicate technology entries found:" in str(exception_info.value)
