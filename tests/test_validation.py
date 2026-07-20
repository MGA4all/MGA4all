import pytest
from pydantic import ValidationError

from mga4all.validate import PYPSAConfig


@pytest.mark.parametrize(
    "key",
    [
        "config_name",
        "model_interface"
        "alternatives",
        "cost_slack",
        "spatially_explicit",
        "diversification_coefficient",
        "intensification_coefficient",
        "diversified_technologies",
        "intensified_technologies"
    ],
)
def test_missing_keys(key, pypsa_spores_config_dict):
    """Test that a missing key is correctly caught."""
    del pypsa_spores_config_dict["SPORES"][key]
    with pytest.raises(ValidationError):
        PYPSAConfig.model_validate(pypsa_spores_config_dict)


@pytest.mark.parametrize(
    ["key", "value"],
    [
        ("config_name", ""),
        ("model_interface", ""),
        ("alternatives", -1),
        ("cost_slack", -1),
        ("diversification_coefficient", -1),
        ("intensification_coefficient", "5"),
        ("diversified_technologies", 1),
        ("intensified_technologies", 1),
    ],
)
def test_invalid_values(key, value, pypsa_spores_config_dict):
    """Test that invalid values are caught."""
    pypsa_spores_config_dict["SPORES"][key] = value
    with pytest.raises(ValidationError):
        PYPSAConfig.model_validate(pypsa_spores_config_dict)


@pytest.mark.parametrize(
    "values",
    [
        {"intensified_technologies": ["OCGT"]},
        {"intensification_coefficient": 1.0},
    ],
    ids=["missing coefficient", "missing technologies"],
)
def test_duplicates(pypsa_spores_config_dict):
    """Test that a duplicate asset in `diversified_technologies` is caught."""
    pypsa_spores_config_dict["SPORES"]["diversified_technologies"][0]["assets"].append("OCGT")
    pypsa_spores_config_dict["SPORES"]["diversified_technologies"][0]["assets"].append("OCGT")

    with pytest.raises(ValidationError) as exception_info:
        PYPSAConfig.model_validate(pypsa_spores_config_dict)
    assert "Duplicate asset entry found" in str(exception_info.value)
