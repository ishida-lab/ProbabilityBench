from __future__ import annotations

from dataclasses import dataclass

# A field value, such as "technology" or 4.5.
FeatureValue = str | int | float


@dataclass(frozen=True)
class DatasetExample:
    """One generated case and its exact class probabilities."""

    case_id: str
    visible: dict[str, FeatureValue]
    hidden: dict[str, FeatureValue]
    true_outcome: str
    posterior: dict[str, float]
    rendered_text: str | None = None


@dataclass(frozen=True)
class DatasetSplit:
    """Training and test cases from one draw."""

    train: list[DatasetExample]
    test: list[DatasetExample]
