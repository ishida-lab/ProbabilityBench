"""Send decision requests to OpenRouter and read back positive-class probabilities.

OpenRouter's Decisions API answers each question with a "noul": the probability
that the question's true criterion holds. ProbabilityBench asks one question,
"positive", so the noul is the model's probability of the positive outcome.
Requests contain memos and training labels, never test targets.
"""
from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path
from urllib import request
from urllib.error import HTTPError, URLError

DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"
# Rate limits, timeouts, and transient server or provider errors are retried.
RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504, 524, 529})


def load_dotenv(path: Path = Path(".env")) -> None:
    """Read KEY=VALUE lines into the environment, never replacing a value already set."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.removeprefix("export ").strip()
        os.environ.setdefault(key, value.strip().strip("'\""))


def api_key() -> str:
    """The OpenRouter key from the environment or a .env file in the working directory."""
    load_dotenv()
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise RuntimeError("Set OPENROUTER_API_KEY in your environment or in a .env file "
                           "(copy .env.example to .env).")
    return key


def create_decision(payload: dict, key: str, *, retries: int = 5, timeout: float = 120) -> dict:
    """POST one request, retrying transient failures with exponential backoff."""
    data = json.dumps(payload).encode()
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json",
               "X-Title": "ProbabilityBench"}
    for attempt in range(retries + 1):
        try:
            req = request.Request(DECISIONS_URL, data=data, headers=headers, method="POST")
            with request.urlopen(req, timeout=timeout) as response:
                return json.loads(response.read())
        except HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:500]
            if exc.code not in RETRYABLE_STATUS or attempt == retries:
                raise RuntimeError(f"OpenRouter returned HTTP {exc.code}: {detail}") from exc
        except (TimeoutError, URLError, OSError):
            if attempt == retries:
                raise
        time.sleep(min(2 ** attempt, 30) * (0.5 + random.random()))
    raise AssertionError("unreachable")


def positive_probability(response: dict) -> float:
    """The model's probability of the positive outcome; malformed replies raise."""
    answers = response.get("answers") if isinstance(response, dict) else None
    answer = answers.get("positive") if isinstance(answers, dict) else None
    if not isinstance(answer, dict) or answer.get("type") != "noul":
        raise ValueError("Response has no probability answer to the positive question")
    q = answer.get("noul")
    if type(q) not in (int, float) or not math.isfinite(q) or not 0 <= q <= 1:
        raise ValueError(f"Probability must be a finite number in [0, 1], got {q!r}")
    return float(q)
