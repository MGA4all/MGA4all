import pytest
from pydantic import ValidationError

from mga4all.validate import PYPSAConfig


@pytest.mark.parametrize(
    "key",
    [
        "config_name",
        "num_spores",
        "spores_slack",
        "diversification_coefficient",
        "weighting_method",
        "intensify",
        "spore_technologies",
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
        ("num_spores", -1),
        ("spores_slack", -1),
        ("diversification_coefficient", -1),
        ("weighting_method", "constant"),
        ("intensify", None),
        ("spore_technologies", []),
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
        {"intensifiable_technologies": ["OCGT"]},
        {"intensification_coefficient": 1.0},
        {},
    ],
    ids=["missing coefficient", "missing technologies", "both missing"],
)
def test_missing_intensification_options(values, pypsa_spores_config_dict):
    """Test that missing intensification options are caught when mode includes `intensify`."""
    spores_config = pypsa_spores_config_dict["SPORES"]
    spores_config["intensify"] = True
    for key, value in values.items():
        spores_config[key] = value

    with pytest.raises(ValidationError) as exception_info:
        PYPSAConfig.model_validate(pypsa_spores_config_dict)
    assert "provided when `intensify` is `True`" in str(exception_info.value)


@pytest.mark.parametrize(
    "values",
    [
        {"intensifiable_technologies": ["OCGT"]},
        {"intensification_coefficient": 1.0},
    ],
    ids=["missing coefficient", "missing technologies"],
)
def test_missing_optional_intensification_options(values, pypsa_spores_config_dict):
    """Test that a lacking intensification option is caught."""
    spores_config = pypsa_spores_config_dict["SPORES"]
    for key, value in values.items():
        spores_config[key] = value

    with pytest.raises(ValidationError) as exception_info:
        PYPSAConfig.model_validate(pypsa_spores_config_dict)
    assert "both provided or omitted together" in str(exception_info.value)


def test_intensifiable_subset(pypsa_spores_config_dict):
    """Test that it is caught if an intensifiable asset is not previously defined."""
    spores_config = pypsa_spores_config_dict["SPORES"]
    spores_config["intensification_coefficient"] = 1.0
    spores_config["intensifiable_technologies"] = ["not present"]

    with pytest.raises(ValidationError) as exception_info:
        PYPSAConfig.model_validate(pypsa_spores_config_dict)
    assert "were not previously defined" in str(exception_info.value)


def test_duplicates(pypsa_spores_config_dict):
    """Test that a duplicate asset in `spore_technologies` is caught."""
    pypsa_spores_config_dict["SPORES"]["spore_technologies"][0]["assets"].append("OCGT")

    with pytest.raises(ValidationError) as exception_info:
        PYPSAConfig.model_validate(pypsa_spores_config_dict)
    assert "Duplicate asset entry found" in str(exception_info.value)
