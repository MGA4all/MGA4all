import numbers

WEIGHTING_METHODS = frozenset(
    [
        "random",
        "evolving_median",
        "evolving_average",
        "relative_deployment",
        "relative_deployment_normalized",
    ]
)

PYPSA_DATAFRAME_NAMES = {
    "Generator": "generators",
    "Line": "lines",
    "Transformer": "transformers",
    "Link": "links",
    "Store": "stores",
    "StorageUnit": "storage_units",
}

def validate_spores_configuration(config: dict):
    """Validate a SPORES YAML config against the specified requirements."""
    try:
        spores_config = config["SPORES"]
    except KeyError:
        raise ValueError("Missing top-level key: 'SPORES'.")

    validate_config_name(spores_config)
    validate_required_keys(spores_config)
    validate_objective_sense(spores_config)
    validate_spores_slack(spores_config)
    validate_num_spores(spores_config)
    validate_weighting_method(spores_config)
    validate_spores_mode(spores_config)
    validate_diversification_coefficient(spores_config)
    validate_spore_technologies(spores_config)
    validate_intensify_and_diversify(spores_config)
    validate_coupling_rule(spores_config)
    validate_no_duplicates(spores_config)

    return True


def validate_config_name(spores_config):
    """Must have config_name which will be used in the output folder name to save results."""
    if (
        "config_name" not in spores_config
        or not isinstance(spores_config["config_name"], str)
        or not spores_config["config_name"].strip()
    ):
        raise ValueError("'config_name' must be provided as a non-empty string.")

def validate_required_keys(spores_config):
    # Required keys
    required_keys_in_spores_config = [
        "objective_sense",
        "spores_slack",
        "num_spores",
        "weighting_method",
        "spores_mode",
        "diversification_coefficient",
        "spore_technologies",
    ]

    for key in required_keys_in_spores_config:
        if key not in spores_config:
            raise ValueError(f"Missing required key: '{key}'.")

def validate_objective_sense(spores_config):
    """objective_sense must be min for consistency."""
    if spores_config["objective_sense"] != "min":
        raise ValueError(
            "'objective_sense' must be 'min'. To maximize, please set the 'diversification_coefficient' "
            "and/or 'intensification_coefficient' to negative."
        )

def validate_spores_slack(spores_config):
    """# spores_slack must be between 0 and 1"""
    if not isinstance(spores_config["spores_slack"], numbers.Number) or not (
        0 <= spores_config["spores_slack"] <= 1
    ):
        raise ValueError("'spores_slack' must be a number between 0 and 1.")

def validate_num_spores(spores_config):
    """num_spores must be integer >= 1"""
    if (
        not isinstance(spores_config["num_spores"], int)
        or spores_config["num_spores"] < 1
    ):
        raise ValueError("'num_spores' must be an integer >= 1.")

def validate_weighting_method(spores_config):
    """weighting_method must be valid"""
    if spores_config["weighting_method"] not in WEIGHTING_METHODS:
        raise ValueError(
            f"Unsupported {spores_config['weighting_method']=}, must be one of {WEIGHTING_METHODS}."
        )

def validate_spores_mode(spores_config):
    """spores_mode must be valid"""
    if spores_config["spores_mode"] not in ["diversify", "intensify and diversify"]:
        raise ValueError(
            "'spores_mode' must be either 'diversify' or 'intensify and diversify'."
        )

def validate_diversification_coefficient(spores_config):
    """diversification_coefficient must be positive number"""
    diversification_coeff = spores_config["diversification_coefficient"]
    try:
        diversification_coeff = float(diversification_coeff)
    except (TypeError, ValueError):
        raise ValueError("'diversification_coefficient' must be a number.")

    if diversification_coeff <= 0:
        raise ValueError("'diversification_coefficient' must be a positive number.")

def validate_spore_technologies(spores_config):
    # spore_technologies cannot be empty
    spore_technologies = spores_config["spore_technologies"]
    if not isinstance(spore_technologies, list) or not spore_technologies:
        raise ValueError("'spore_technologies' must be a non-empty list.")

    # Keys of spore_technologies must be in valid_tech_type
    valid_tech_type = PYPSA_DATAFRAME_NAMES.keys()
    for tech_top_key in spore_technologies:
        if not isinstance(tech_top_key, dict) or len(tech_top_key) != 1:
            raise ValueError(
                "Each element in 'spore_technologies' must be a dict with a single top-level pypsa-component key."
            )
        component = next(iter(tech_top_key))
        if component not in valid_tech_type:
            raise ValueError(
                f"Invalid pypsa-component '{component}' in 'spore_technologies'. Must be one of {valid_tech_type}."
            )

        # Extra sanity check: each must have attribute and index keys
        tech_data = tech_top_key[component]
        if "attribute" not in tech_data or not isinstance(tech_data["attribute"], str):
            raise ValueError(
                f"Component '{component}' must define an 'attribute' key with a string value."
            )
        if (
            "index" not in tech_data
            or not isinstance(tech_data["index"], list)
            or not tech_data["index"]
        ):
            raise ValueError(
                f"Component '{component}' must define a non-empty 'index' list."
            )

def validate_intensify_and_diversify(spores_config):
    # If spores_mode is "intensify and diversify", extra checks
    if spores_config["spores_mode"] == "intensify and diversify":
        try:
            spores_config["intensification_coefficient"] = float(
                spores_config["intensification_coefficient"]
            )
        except (KeyError, TypeError, ValueError):
            raise ValueError(
                "'intensification_coefficient' must be provided as a number for spores_mode 'intensify and diversify'."
            )
        if (
            "intensifiable_technologies" not in spores_config
            or not isinstance(spores_config["intensifiable_technologies"], list)
            or not spores_config["intensifiable_technologies"]
        ):
            raise ValueError(
                "'intensifiable_technologies' must be a non-empty list when 'spores_mode' is 'intensify and diversify'."
            )

def validate_coupling_rule(spores_config):
    # Coupling rule: intensification_coefficient and intensifiable_technologies must be both present or both absent
    has_coeff = "intensification_coefficient" in spores_config
    has_intensifiable = "intensifiable_technologies" in spores_config
    if has_coeff != has_intensifiable:  # XOR
        raise ValueError(
            "'intensification_coefficient' and 'intensifiable_technologies' must be provided or omitted together."
        )

def validate_no_duplicates(spores_config):
    # Extra check: No duplicate component-index pairs in spore_technologies
    seen_pairs = set()
    for tech in spores_config["spore_technologies"]:
        comp = next(iter(tech))
        for idx in tech[comp]["index"]:
            pair = (comp, idx)
            if pair in seen_pairs:
                raise ValueError(f"Duplicate technology entry found: {pair}")
            seen_pairs.add(pair)

