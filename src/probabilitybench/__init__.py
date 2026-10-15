"""ProbabilityBench: text-based simulated worlds with exact class probabilities."""

from .core.dataset import build_split
from .tasks import TASK_NAMES, create_world
from .tasks.registry import get_task

__all__ = ["TASK_NAMES", "build_split", "create_world", "get_task"]
