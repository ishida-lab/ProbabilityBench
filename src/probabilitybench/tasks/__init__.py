"""The finance, peer-review, and AI-safety worlds of ProbabilityBench."""

from .world import RichFiniteWorld

TASK_NAMES = ("financial_default", "paper_review", "covert_transfer")


def create_world(task: str, nuisance_count: int = 8) -> RichFiniteWorld:
    if task == "covert_transfer":
        from .covert_transfer import CovertTransferWorld
        return CovertTransferWorld(nuisance_count)
    return RichFiniteWorld(task, nuisance_count)


__all__ = ["RichFiniteWorld", "create_world", "TASK_NAMES"]
