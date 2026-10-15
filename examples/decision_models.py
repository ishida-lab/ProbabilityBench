"""Evaluate decision models against exact probabilities and plot cost against quality.

Every model sees the cases of the paper's cost-quality comparison: the Medium
setting of each world, 25 labeled examples, and 300 test cases from each of
seeds 2032-2034, which is 2,700 requests per model.

  run      call decision models on OpenRouter and save their probabilities
  score    mean KL divergence per world, and cost per 1,000 predictions
  plot     KL against cost for every scored model, one panel per world

Run from the repository root, for example:
  uv run python examples/decision_models.py run --limit 5
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from probabilitybench import create_world
from probabilitybench.evaluation import openrouter
from probabilitybench.evaluation.probability import bernoulli_kl, make_decision_input

WORLDS = ("paper_review", "covert_transfer", "financial_default")
WORLD_TITLES = {"paper_review": "Peer review", "covert_transfer": "Covert transfer",
                "financial_default": "Credit risk"}
SEEDS = (2032, 2033, 2034)
LEVEL = "medium_bayes_error"
SHOTS = 25
# Each seed's draw holds 1,000 training rows before its held-out cases. The
# examples are the first 25 rows; the test cases are the first 300 held out.
FIRST_TEST_ROW = 1000
TEST_CASES = 300
EPSILON = 1e-3
DEFAULT_MODELS = ("typesafe/jev-1.13", "perplexity/pplx-decider-v1.1-27b", "openai/gpt-6-luna-decisions")
RUNS = Path("outputs/decision_models")


def protocol_cases(world_name: str, seed: int, limit: int = TEST_CASES):
    """Training examples and test cases for one world and seed, plus the world itself."""
    if not 1 <= limit <= TEST_CASES:
        raise ValueError(f"limit must be between 1 and {TEST_CASES}")
    world = create_world(world_name)
    hidden = next(level["hidden"] for level in world.hiding_levels if level["name"] == LEVEL)
    rows = world.generate_dataset(FIRST_TEST_ROW + TEST_CASES, hidden, np.random.default_rng(seed))
    train = [{"rendered_text": world.render(row.visible), "true_outcome": row.true_outcome}
             for row in rows[:SHOTS]]
    return world, train, rows[FIRST_TEST_ROW:FIRST_TEST_ROW + limit]


def protocol_requests(limit: int = TEST_CASES):
    """Yield (world, seed, case_id, decision input) for every test case."""
    for world_name in WORLDS:
        for seed in SEEDS:
            world, train, test = protocol_cases(world_name, seed, limit)
            for row in test:
                yield (world_name, seed, row.case_id,
                       make_decision_input(world_name, train, {"rendered_text": world.render(row.visible)}, SHOTS))


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def run_file(runs: Path, model: str) -> Path:
    return runs / (model.replace("/", "__") + ".jsonl")


def run(models: list[str], runs: Path, limit: int, concurrency: int) -> None:
    """Query each model, appending one line per answered case; reruns resume."""
    key = openrouter.api_key()
    runs.mkdir(parents=True, exist_ok=True)
    for model in models:
        path = run_file(runs, model)
        done = {(r["world"], r["seed"], r["case_id"]) for r in read_jsonl(path)}
        todo = [(w, s, c, {"model": model, **inputs}) for w, s, c, inputs in protocol_requests(limit)
                if (w, s, c) not in done]
        print(f"{model}: {len(todo)} requests to send, {len(done)} already saved", flush=True)
        if not todo:
            continue
        failures, started = 0, time.monotonic()
        with path.open("a", encoding="utf-8") as out:
            # One request first, so a bad key, missing credit, or unknown model stops early.
            first, rest = todo[0], todo[1:]
            out.write(json.dumps(answer_record(model, first, openrouter.create_decision(first[3], key))) + "\n")
            with ThreadPoolExecutor(concurrency) as pool:
                futures = {pool.submit(openrouter.create_decision, job[3], key): job for job in rest}
                for count, future in enumerate(as_completed(futures), 2):
                    job = futures[future]
                    try:
                        record = answer_record(model, job, future.result())
                    except (RuntimeError, ValueError, OSError) as exc:
                        failures += 1
                        print(f"  failed {job[0]} seed {job[1]} {job[2]}: {exc}", file=sys.stderr)
                        continue
                    out.write(json.dumps(record) + "\n")
                    out.flush()
                    if count % 100 == 0:
                        print(f"  {count}/{len(todo)}", flush=True)
        elapsed = time.monotonic() - started
        print(f"{model}: {len(todo)} requests in {elapsed:.0f} s ({len(todo) / max(elapsed, 1e-9):.1f} per second "
              f"at concurrency {concurrency}); saved to {path}"
              + (f"; {failures} failed, rerun to retry them" if failures else ""))


def answer_record(model: str, job: tuple, response: dict) -> dict:
    world, seed, case_id, _ = job
    usage = response.get("usage") or {}
    return {"model": model, "response_model": response.get("model"), "world": world, "seed": seed,
            "case_id": case_id, "probability_positive": openrouter.positive_probability(response),
            "cost_usd": usage.get("cost"), "input_tokens": usage.get("input_tokens")}


def exact_probabilities(limit: int) -> dict[tuple, float]:
    """Evaluator-side targets, regenerated rather than stored next to the inputs."""
    truth = {}
    for world_name in WORLDS:
        for seed in SEEDS:
            world, _, test = protocol_cases(world_name, seed, limit)
            for row in test:
                truth[world_name, seed, row.case_id] = row.posterior[world.positive_label]
    return truth


def score(runs: Path, limit: int) -> list[dict]:
    """Mean KL(exact || model) per world, averaging each seed's mean over the seeds."""
    truth = exact_probabilities(limit)
    results = []
    for path in sorted(runs.glob("*.jsonl")):
        records = read_jsonl(path)
        keys = [(r["world"], r["seed"], r["case_id"]) for r in records]
        if len(set(keys)) != len(keys):
            raise ValueError(f"{path} answers some case more than once")
        unknown = set(keys) - truth.keys()
        if unknown:
            raise ValueError(f"{path} has {len(unknown)} answers for cases outside this comparison")
        name = records[0].get("model", path.stem) if records else path.stem
        for world in WORLDS:
            rows = [r for r in records if r["world"] == world]
            if not rows:
                continue
            per_seed = []
            for seed in SEEDS:
                seed_rows = [r for r in rows if r["seed"] == seed]
                if seed_rows:
                    per_seed.append(math.fsum(
                        bernoulli_kl(truth[world, seed, r["case_id"]], r["probability_positive"], EPSILON)
                        for r in seed_rows) / len(seed_rows))
            costs = [r.get("cost_usd") for r in rows]
            results.append({
                "model": name, "world": world, "kl_nats": sum(per_seed) / len(per_seed),
                "cost_per_1000": 1000 * sum(costs) / len(costs) if None not in costs else None,
                "n_cases": len(rows), "expected_cases": len(SEEDS) * limit,
            })
    return results


def print_scores(results: list[dict]) -> None:
    print(f"{'model':40s} {'world':18s} {'KL (nats)':>9s} {'USD/1,000':>10s} {'cases':>9s}")
    for r in results:
        cost = f"{r['cost_per_1000']:.4f}" if r["cost_per_1000"] is not None else "unknown"
        cases = f"{r['n_cases']}/{r['expected_cases']}"
        print(f"{r['model']:40s} {r['world']:18s} {r['kl_nats']:9.4f} {cost:>10s} {cases:>9s}")


# Colors and markers follow the paper's figures, assigned in this fixed order.
SLOTS = (("#1F4E79", "o"), ("#7A2A2A", "s"), ("#4C7A5A", "^"), ("#C07A2B", "D"), ("#8A6B9E", "v"),
         ("#4E9CAC", "P"), ("#C97087", "X"), ("#7B5600", "*"), ("#6145A3", "h"))
WORLD_COLORS = {"paper_review": "#7A2A2A", "covert_transfer": "#4C7A5A", "financial_default": "#1F4E79"}
FRONTIER = "#B03A2E"
INK = "#2B2B2B"


def frontier(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Points that no cheaper point beats on KL, in order of cost."""
    best, kept = math.inf, []
    for cost, kl in sorted(points):
        if kl < best:
            kept.append((cost, kl))
            best = kl
    return kept


def plot(results: list[dict], output: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    except ImportError:
        raise SystemExit("Plotting needs matplotlib: uv sync --extra plot") from None
    for r in results:
        if r["n_cases"] < r["expected_cases"]:
            print(f"warning: {r['model']} answered {r['n_cases']} of {r['expected_cases']} "
                  f"{r['world']} cases", file=sys.stderr)
    plotted = [r for r in results if r["cost_per_1000"] is not None and r["cost_per_1000"] > 0]
    for r in results:
        if r not in plotted:
            print(f"warning: OpenRouter reported no cost for {r['model']} on {r['world']}", file=sys.stderr)
    if not plotted:
        raise SystemExit("Nothing to plot: score some runs first")
    models = sorted({r["model"] for r in plotted})
    if len(models) > len(SLOTS):
        raise SystemExit(f"At most {len(SLOTS)} models fit one figure")
    style = dict(zip(models, SLOTS))
    label = {m: m.rsplit("/", 1)[-1] for m in models}

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK, "axes.labelcolor": INK,
                         "xtick.color": INK, "ytick.color": INK})
    fig, axes = plt.subplots(1, len(WORLDS), figsize=(10, 3.6))
    for index, (ax, world) in enumerate(zip(axes, WORLDS)):
        rows = [r for r in plotted if r["world"] == world]
        best = frontier([(r["cost_per_1000"], r["kl_nats"]) for r in rows])
        if len(best) > 1:
            ax.plot(*zip(*best), color=FRONTIER, linewidth=1.2, zorder=2)
        for r in rows:
            color, marker = style[r["model"]]
            x, y = r["cost_per_1000"], r["kl_nats"]
            ax.scatter(x, y, s=60, color=color, marker=marker, edgecolors="white", linewidths=0.8, zorder=3)
            ax.annotate(label[r["model"]], (x, y), xytext=(5, 5), textcoords="offset points",
                        fontsize=7.5, color=INK)
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.invert_yaxis()  # Lower KL is better, so better models sit higher.
        ax.margins(x=0.3, y=0.25)
        ax.grid(True, color="#E3E3E3", linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_title(f"({'abc'[index]}) {WORLD_TITLES[world]}", color=WORLD_COLORS[world])
        if index == 0:
            ax.set_ylabel("KL divergence (nats)")
        if index == 1:
            ax.set_xlabel("Cost reported by OpenRouter, USD per 1,000 predictions")
    handles = [Line2D([], [], linestyle="none", marker=style[m][1], color=style[m][0], markersize=7, label=label[m])
               for m in models]
    handles.append(Line2D([], [], color=FRONTIER, linewidth=1.2, label="Pareto frontier"))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 5), frameon=False, fontsize=8)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (("run", "query decision models on OpenRouter"),
                            ("score", "print KL and cost per world"),
                            ("plot", "draw KL against cost")):
        sub = commands.add_parser(name, help=help_text)
        sub.add_argument("--limit", type=int, default=TEST_CASES,
                         help=f"test cases per world and seed (default {TEST_CASES}); use a few for a trial")
        if name == "run":
            sub.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
            sub.add_argument("--concurrency", type=int, default=8)
        sub.add_argument("--runs", type=Path, default=RUNS)
        if name == "plot":
            sub.add_argument("--output", type=Path, default=Path("outputs/cost_quality.png"))
    args = parser.parse_args()
    if args.command == "run":
        try:
            run(args.models, args.runs, args.limit, args.concurrency)
        except RuntimeError as exc:  # A missing key, or an API error on the first request.
            raise SystemExit(f"error: {exc}") from None
    elif args.command == "score":
        print_scores(score(args.runs, args.limit))
    else:
        plot(score(args.runs, args.limit), args.output)


if __name__ == "__main__":
    main()
