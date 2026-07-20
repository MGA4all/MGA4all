from enum import StrEnum, auto
from typing import Annotated
from typing_extensions import Self

from pydantic import (
    BaseModel,
    Field,
    PositiveFloat,
    PositiveInt,
    StringConstraints,
    model_validator,
)


class PyPSAComponent(StrEnum):
    GENERATOR = "Generator"
    LINE = "Line"
    TRANSFORMER = "Transformer"
    LINK = "Link"
    STORE = "Store"
    STORAGEUNIT = "StorageUnit"
    PROCESS = "Process"


PYPSA_DATAFRAME_NAMES: dict[PyPSAComponent, str] = dict(
    zip(
        PyPSAComponent,
        ["generators", "lines", "transformers", "links", "stores", "storage_units", "processes"],
    )
)


class AssetGroup(BaseModel):
    component: PyPSAComponent
    """PyPSA component type of the group of assets."""
    attribute: str
    """Which PyPSA attribute to optimize. E.g., `p_nom` (dispatch) or `e_nom` (capacity)."""
    assets: list[str] = Field(min_length=1)
    """Asset names of this component type to be targeted."""


class SPORESConfig(BaseModel):
    config_name: Annotated[str, StringConstraints(min_length=1)]
    """Descriptive name of this configuration, used in the output folder name to save results."""
    model_interface: "pypsa"
    """Which model interface to use (e.g., PyPSA's)."""

    alternatives: PositiveInt
    """Number of MGA alternatives to generate."""
    cost_slack: Annotated[float, Field(gt=0, lt=1)]
    """Percentage relaxation of the optimal cost expressed as a fraction (e.g., 10% is 0.1)"""
    spatially_explicit: bool
    """Whether to target MGA variables per node or at the system level only."""
    diversification_coefficient: "auto" | PositiveFloat
    """Diversification coefficient, must be "auto" or positive."""
    intensification_coefficient: int | list[int]
    """Intensification coefficient, must be 0, 1, -1, or a list of those."""

    diversified_technologies: list[AssetGroup] = Field(min_length=0)
    """Which technologies to diversify during the MGA run."""
    intensified_technologies: list[AssetGroup] = Field(min_length=0)
    """Which technologies to intensify during the MGA run."""

    @model_validator(mode="after")
    def check_for_duplicates(self) -> Self:
        """Extra check: No duplicate component-index pairs in diversification_technologies."""
        seen_pairs = set()
        for asset_group in self.diversification_technologies:
            component = asset_group.component
            for asset in asset_group.assets:
                pair = (component, asset)
                if pair in seen_pairs:
                    raise ValueError(f"Duplicate asset entry found: {pair}")
                seen_pairs.add(pair)

        return self  # no duplicates found


class SimpleMGAConfig(BaseModel):
    config_name: Annotated[str, StringConstraints(min_length=1)]
    """Descriptive name of this configuration, used in the output folder name to save results."""
    model_interface: str
    """Which model interface to use (e.g., PyPSA's)."""

    alternatives: PositiveInt
    """Number of MGA alternatives to generate."""
    cost_slack: Annotated[float, Field(gt=0, lt=1)]
    """Percentage relaxation of the optimal cost expressed as a fraction (e.g., 10% is 0.1)"""
    spatially_explicit: bool
    """Whether to target MGA variables per node or at the system level only."""
    diversification_coefficient: "auto" | PositiveFloat
    """Diversification coefficient, must be "auto" or positive."""

    @model_validator(mode="after")
    def check_for_duplicates(self) -> Self:
        """Extra check: No duplicate component-index pairs in diversification_technologies."""
        seen_pairs = set()
        for asset_group in self.diversification_technologies:
            component = asset_group.component
            for asset in asset_group.assets:
                pair = (component, asset)
                if pair in seen_pairs:
                    raise ValueError(f"Duplicate asset entry found: {pair}")
                seen_pairs.add(pair)

        return self  # no duplicates found


class PYPSAConfig(BaseModel):
    SPORES: SPORESConfig
    HSJ: SimpleMGAConfig
    RandomDirections: SimpleMGAConfig
