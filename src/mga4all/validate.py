from typing import Annotated, Literal, get_args
from typing_extensions import Self

from pydantic import (
    BaseModel,
    Field,
    PositiveFloat,
    PositiveInt,
    StringConstraints,
    confloat,
    model_validator,
)

WeightingMethod = Literal[
    "random",
    "evolving_median",
    "evolving_average",
    "relative_deployment",
    "relative_deployment_normalized",
]
WEIGHTING_METHODS: frozenset[WeightingMethod] = frozenset(get_args(WeightingMethod))

PyPSAComponent = Literal[
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
    attribute: str
    assets: list[str] = Field(min_length=1)


class SporesConfig(BaseModel):
    config_name: Annotated[str, StringConstraints(min_length=1)]

    num_spores: PositiveInt
    spores_slack: float = confloat(gt=0, lt=1)
    diversification_coefficient: PositiveFloat

    objective_sense: Literal["min"]
    weighting_method: WeightingMethod
    spores_mode: Literal["diversify", "intensify and diversify"]

    spore_technologies: list[AssetGroup] = Field(min_length=1)

    # Conditionally optional values
    intensification_coefficient: float | None = None
    intensifiable_technologies: list[str] | None = None

    @model_validator(mode="after")
    def check_intensification_options(self) -> Self:
        """Check that intensification options are present when that mode is selected."""
        intensify_mode = self.spores_mode == "intensify and diversify"
        lacks_coefficient = self.intensification_coefficient is None
        lacks_intensifiable = (
            self.intensifiable_technologies is None
            or len(self.intensifiable_technologies) == 0
        )

        if intensify_mode and (lacks_coefficient or lacks_intensifiable):
            raise ValueError(
                "'intensification_coefficient' and 'intensifiable_technologies' "
                "must be both provided for mode 'intensify and diversify'."
            )
        elif lacks_coefficient is not lacks_coefficient:  # XOR
            raise ValueError(
                "'intensification_coefficient' and 'intensifiable_technologies' "
                "must be both provided or omitted together."
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
                    raise ValueError(f"Duplicate technology entry found: {pair}")
                seen_pairs.add(pair)

        return self  # no duplicates found


class PYPSAConfig(BaseModel):
    SPORES: SporesConfig


def validate_spores_configuration(config: dict):
    """Validate a SPORES YAML config against the specified requirements."""
    PYPSAConfig.model_validate(config)
