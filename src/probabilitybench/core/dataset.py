from __future__ import annotations

from dataclasses import replace

import numpy as np

from .types import DatasetExample, DatasetSplit


def render_dataset(
    dataset: list[DatasetExample],
    render_fn,
) -> list[DatasetExample]:
    """Attach rendered text to each dataset example."""

    # `replace` keeps the dataclass immutable while adding the text.
    return [replace(example, rendered_text=render_fn(example.visible)) for example in dataset]


def build_split(
    world,
    hidden_attributes: list[str],
    train_size: int = 700,
    test_size: int = 300,
    seed: int = 2024,
    render_fn=None,
) -> DatasetSplit:
    """Draw train_size + test_size cases; the first train_size are for training.

    With the same seed, every row index always holds the same case, so the test
    cases are the rows that follow the training rows.
    """

    if train_size < 1 or test_size < 1:
        raise ValueError("train_size and test_size must both be positive")
    # The seed makes every split reproducible.
    dataset = world.generate_dataset(
        n=train_size + test_size,
        hidden_attributes=hidden_attributes,
        rng=np.random.default_rng(seed),
    )
    # Without a render_fn, cases keep only their structured fields.
    if render_fn is not None:
        dataset = render_dataset(dataset, render_fn=render_fn)
    return DatasetSplit(train=dataset[:train_size], test=dataset[train_size:])
