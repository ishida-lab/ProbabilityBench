"""Settings, renderer, and labels for each world."""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Callable

from . import TASK_NAMES, create_world


@dataclass(frozen=True)
class TaskSpec:
    """A world's factory, Bayes-error settings, renderer, and labels."""

    name: str
    create_world: Callable[[], Any]
    hiding_levels: list[dict[str, Any]]
    render_fn: Callable[..., str]
    positive_label: str
    negative_label: str
    nuisance_count: int = 8


def get_task(name: str, *, nuisance_count: int | None = 8) -> TaskSpec:
    """Describe one world, with eight irrelevant fields unless told otherwise."""
    if name not in TASK_NAMES:
        raise ValueError(f"Unknown task {name!r}. Available: {list(TASK_NAMES)}")
    count = 8 if nuisance_count is None else nuisance_count
    factory = partial(create_world, name, nuisance_count=count)
    prototype = factory()
    return TaskSpec(
        name=name, create_world=factory, hiding_levels=prototype.hiding_levels,
        render_fn=prototype.render, positive_label=prototype.positive_label,
        negative_label=prototype.negative_label, nuisance_count=count,
    )
