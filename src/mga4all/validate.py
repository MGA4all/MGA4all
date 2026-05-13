from typing import Annotated, Literal, get_args, TypeAlias
from typing_extensions import Self

from pydantic import (
    BaseModel,
    Field,
    PositiveFloat,
    PositiveInt,
    StringConstraints,
    model_validator,
)

WeightingMethod: TypeAlias = Literal[
    "random",
    "evolving_median",
    "evolving_average",
    "relative_deployment",
    "relative_deployment_normalized",
]
WEIGHTING_METHODS: frozenset[WeightingMethod] = frozenset(get_args(WeightingMethod))

PyPSAComponent: TypeAlias = Literal[
    "Generator", "Line", "Transformer", "Link", "Store", "StorageUnit"
]
PYPSA_DATAFRAME_NAMES: dict[PyPSAComponent, str] = dict(
    zip(
        get_args(PyPSAComponent),
        ["generators", "lines", "transformers", "links", "stores", "storage_units"],
    )
)


class AssetGroup(BaseModel):
    component: PyPSAComponent
    """PyPSA component type of the group of assets."""
    attribute: str
    """Which PyPSA attribute to optimize. E.g., `p_nom` (dispatch) or `e_nom` (capacity)."""
    assets: list[str] = Field(min_length=1)
    """Asset names of this component type to be targeted."""


class SporesConfig(BaseModel):
    config_name: Annotated[str, StringConstraints(min_length=1)]
    """Descriptive name of this configuration, used in the output folder name to save results."""

    num_spores: PositiveInt
    """Number of SPORES to generate."""
    spores_slack: Annotated[float, Field(gt=0, lt=1)]
    """..."""
    diversification_coefficient: PositiveFloat
    """Diversification coefficient, must be positive. ..."""

    weighting_method: WeightingMethod
    """Which method should be used to update weights after each iteration."""
    intensify: bool
    """Toggle if SPORES should include an intensification step."""

    spore_technologies: list[AssetGroup] = Field(min_length=1)
    """Which technologies to target during the SPORES run."""

    # Conditionally optional values
    intensification_coefficient: float | None = None
    """Intensification coefficient. Positive for minimization, negative for maximization. ..."""
    intensifiable_technologies: list[str] | None = None
    """Subset of `spore_technologies` to intensify, i.e., push to max or min deployment within slack."""

    @model_validator(mode="after")
    def check_intensification_options(self) -> Self:
        """Check that intensification options are present when that mode is selected."""
        lacks_coefficient = self.intensification_coefficient is None
        lacks_intensifiable = (
            self.intensifiable_technologies is None
            or len(self.intensifiable_technologies) == 0
        )

        if self.intensify and (lacks_coefficient or lacks_intensifiable):
            raise ValueError(
                "`intensification_coefficient` and `intensifiable_technologies` "
                "must be both provided when `intensify` is `True`."
            )
        elif lacks_coefficient is not lacks_intensifiable:  # XOR
            raise ValueError(
                "`intensification_coefficient` and `intensifiable_technologies` "
                "must be both provided or omitted together."
            )

        return self

    @model_validator(mode="after")
    def check_intensifiable_subset(self) -> Self:
        """Check that the list of intensifiable technologies is a subset of the `spores_technologies`."""
        if self.intensifiable_technologies is None:
            return self  # skip if not defined

        spores_asset_names = {
            asset
            for asset_group in self.spore_technologies
            for asset in asset_group.assets
        }
        
        difference = set(self.intensifiable_technologies) - spores_asset_names
        if len(difference) != 0:
            raise ValueError(
                "The following assets listed under `intensifiable_technologies` were not "
                "previously defined under `spores_technologies`:\n"
                + "\n".join(difference)
            )

        return self

    @model_validator(mode="after")
    def check_for_duplicates(self) -> Self:
        """Extra check: No duplicate component-index pairs in spore_technologies."""
        seen_pairs = set()
        for asset_group in self.spore_technologies:
            component = asset_group.component
            for asset in asset_group.assets:
                pair = (component, asset)
                if pair in seen_pairs:
                    raise ValueError(f"Duplicate asset entry found: {pair}")
                seen_pairs.add(pair)

        return self  # no duplicates found


class PYPSAConfig(BaseModel):
    SPORES: SporesConfig


def validate_spores_configuration(config: dict):
    """Validate a SPORES YAML config against the specified requirements."""
    PYPSAConfig.model_validate(config)
