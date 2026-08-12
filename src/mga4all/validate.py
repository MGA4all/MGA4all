from copy import copy
from enum import StrEnum
from typing import Annotated, Literal
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
    model_interface: Literal["pypsa"]
    """Which model interface to use (e.g., PyPSA's)."""

    alternatives: PositiveInt
    """Number of MGA alternatives to generate."""
    cost_slack: Annotated[float, Field(gt=0, lt=1)]
    """Percentage relaxation of the optimal cost expressed as a fraction (e.g., 10% is 0.1)"""
    spatially_explicit: bool
    """Whether to target MGA variables per node or at the system level only."""
    diversification_coefficient: Literal["auto"] | PositiveFloat
    """Diversification coefficient, must be "auto" or positive."""
    intensification_coefficient: int | list[int]
    """Intensification coefficient, must be 0, 1, -1, or a list of those."""

    diversified_technologies: list[str] = Field(min_length=0)
    """Which technologies to diversify during the MGA run."""
    intensified_technologies: list[str] = Field(min_length=0)
    """Which technologies to intensify during the MGA run."""

    @model_validator(mode="after")
    def check_for_duplicates(self) -> Self:
        """Extra check: No duplicate technology in diversified_technologies."""
        unique_technologies = set(self.diversified_technologies)
        if len(unique_technologies) < len(self.diversified_technologies):
            duplicate_technologies = copy(self.diversified_technologies)
            for tech in unique_technologies:
                duplicate_technologies.remove(tech)
            raise ValueError(f"Duplicate technology entries found: {duplicate_technologies}")

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
    diversification_coefficient: Literal["auto"] | PositiveFloat
    """Diversification coefficient, must be "auto" or positive."""
