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
        [
            "generators",
            "lines",
            "transformers",
            "links",
            "stores",
            "storage_units",
            "processes",
        ],
    )
)


class AssetGroup(BaseModel):
    component: PyPSAComponent
    """PyPSA component type of the group of assets."""
    attribute: str
    """Which PyPSA attribute to optimize. E.g., `p_nom` (dispatch) or `e_nom` (capacity)."""
    assets: list[str] = Field(min_length=1)
    """Asset names of this component type to be targeted."""


class SimpleMGAConfig(BaseModel):
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
    diversification_coefficient: PositiveFloat = 1
    """Diversification coefficient, must be positive if given. Default: 1."""

    diversified_technologies: list[str] = Field(min_length=0)
    """Which technologies to diversify during the MGA run."""

    @staticmethod
    def find_duplicates(items: list) -> list:
        """Return the duplicates from the given list of items."""
        duplicates = copy(items)
        for item in set(items):
            duplicates.remove(item)
        return duplicates

    @model_validator(mode="after")
    def ensure_unique_diversified_technologies(self) -> Self:
        """Ensure there are no duplicates in diversified_technologies."""
        if len(set(self.diversified_technologies)) < len(self.diversified_technologies):
            duplicates = self.find_duplicates(self.diversified_technologies)
            raise ValueError(
                f"Duplicate `diversified_technologies` entries found: {duplicates}"
            )

        return self  # no duplicates found


class HopSkipJumpConfig(SimpleMGAConfig):
    pass


class RandomDirectionsConfig(SimpleMGAConfig):
    pass


class SPORESConfig(SimpleMGAConfig):
    intensification_coefficient: Literal[-1, 0, 1] | list[Literal[-1, 0, 1]] = 0
    """Intensification coefficient, must be 0, 1, -1, or a list of those."""
    intensified_technologies: list[str] = Field(min_length=0)
    """Which technologies to intensify during the MGA run."""

    @model_validator(mode="after")
    def validate_intensification_coefficient(self):
        """Correctly set the intensification coefficients: one per technology."""
        num_intensified = len(self.intensified_technologies)
        coefficients = self.intensification_coefficient

        if num_intensified == 0:
            coefficients = 0
        elif isinstance(coefficients, list) and len(coefficients) != num_intensified:
            raise ValueError(
                f"Number of intensification coefficients {len(coefficients)} "
                f"does not match number of intensified technologies {num_intensified}"
            )

        self.intensification_coefficient = coefficients
        return self

    @model_validator(mode="after")
    def ensure_unique_intensified_technologies(self) -> Self:
        """Ensure there are no duplicates in diversified_technologies."""
        if len(set(self.intensified_technologies)) < len(self.intensified_technologies):
            duplicates = self.find_duplicates(self.intensified_technologies)
            raise ValueError(
                f"Duplicate `intensified_technologies` entries found: {duplicates}"
            )

        return self  # no duplicates found
