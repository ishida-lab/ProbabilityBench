<div align="center">
  <a href="images/probabilitybench_logo.jpg"><img src="images/probabilitybench_logo.jpg" width="220" alt="ProbabilityBench logo: a cat with a halo holding a die on a cloud" /></a>
  <h1>ProbabilityBench</h1>
  <a href="https://arxiv.org/abs/2610.17016"><img src="https://img.shields.io/badge/arXiv-2610.17016-b31b1b.svg" alt="arXiv"></a>
  <a href="https://huggingface.co/datasets/ishidalab/probabilitybench"><img src="https://img.shields.io/badge/Hugging%20Face-Dataset-yellow.svg" alt="Hugging Face Dataset"></a>
  <a href="https://ishida-lab.github.io/blog_probabilitybench.html"><img src="https://img.shields.io/badge/Blog-Post-green.svg" alt="Blog"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT"></a>
</div>

Real-world datasets usually provide only one observed outcome per input, leaving
the true conditional probability unknown and making model uncertainty difficult
to evaluate. ProbabilityBench provides text-based simulated worlds with exact
class probabilities, so a model's reported probabilities can be compared
directly with the ground truth.

Each generated case includes a readable memo, a sampled outcome, and the exact
probability of the positive outcome given what the memo shows.

| World | Task name | Positive outcome |
|---|---|---|
| Peer review | `paper_review` | `accept` |
| Covert transfer | `covert_transfer` | `encoding` |
| Credit risk | `financial_default` | `default` |

To look at cases without installing anything, see the fixed test set on
[Hugging Face](https://huggingface.co/datasets/ishidalab/probabilitybench).

## Install

Use [uv](https://docs.astral.sh/uv/getting-started/installation/) to manage Python
and dependencies. Python 3.10 or later is supported. Clone the repository:

```bash
git clone https://github.com/ishida-lab/ProbabilityBench.git
cd ProbabilityBench
```

Install and test:

```bash
uv sync --extra dev
uv run pytest -q
```

## Generate your first cases

Save this snippet as `generate_cases.py` and run it with `uv run python generate_cases.py`:

```python
from probabilitybench.core.dataset import build_split
from probabilitybench.tasks.registry import get_task

task = get_task("paper_review")
world = task.create_world()
level = next(s for s in task.hiding_levels if s["name"] == "medium_bayes_error")
split = build_split(
    world, level["hidden"], train_size=50, test_size=20,
    seed=2032, render_fn=task.render_fn,
)

case = split.test[0]
print(case.rendered_text)                 # Model input: the complete memo
print(case.true_outcome)                  # Sampled label; evaluator only at test time
print(case.posterior[task.positive_label]) # Exact P(positive outcome | memo)
```

Each world has `low_bayes_error`, `medium_bayes_error`, and `high_bayes_error`
settings. These hide different relevant information. With the same task, seed,
and sample sizes, the three settings share sampled outcomes and case IDs,
allowing matched comparisons.

These worlds are synthetic mechanisms, not empirical models of real peer
review, credit risk, or AI behavior. Their probabilities are exact under those
mechanisms. Factor worlds sum over hidden states; covert transfer uses its
analytic likelihood. The implementation lives in
[`tasks/specifications.py`](src/probabilitybench/tasks/specifications.py),
[`tasks/world.py`](src/probabilitybench/tasks/world.py), and
[`tasks/covert_transfer.py`](src/probabilitybench/tasks/covert_transfer.py).
Numerical measurements and irrelevant-field distributions are defined in
[`tasks/numerical.py`](src/probabilitybench/tasks/numerical.py); memo rendering
and decoding are in [`tasks/prose.py`](src/probabilitybench/tasks/prose.py).

## Evaluate decision models

[`examples/decision_models.py`](examples/decision_models.py) compares decision
models with the exact probabilities and plots quality against cost, as in the
paper's cost-quality figure. Every model answers the same 2,700 requests, which
are identical to the paper's: the Medium setting of each world, 25 labeled
examples per request, and 300 test cases from each of seeds 2032–2034.

**1. Add your OpenRouter key.** Copy the template, then paste your key from
[openrouter.ai/settings/keys](https://openrouter.ai/settings/keys) into `.env`:

```bash
cp .env.example .env
```

Git ignores `.env`. A key already set as `OPENROUTER_API_KEY` in your
environment takes precedence.

**2. Run the models.** Start with five cases per world and seed to check your
key and credit:

```bash
uv run python examples/decision_models.py run --limit 5
```

Then run the full comparison. It sends only the cases still missing:

```bash
uv run python examples/decision_models.py run
```

The default models are Jev 1.13, Perplexity Decider V1.1 27B, and GPT-6 Luna
Decisions. At current OpenRouter prices, a full run costs about US$2 or less per
model. To evaluate your own decision model on OpenRouter, list it with the
others:

```bash
uv run python examples/decision_models.py run --models typesafe/jev-1.13 your-org/your-model
```

Answers are saved in `outputs/decision_models/`, one file per model.

**3. Score and plot.** `score` prints each model's mean KL divergence from the
exact probabilities per world, in nats (lower is better), and the cost per
1,000 predictions reported by OpenRouter:

```bash
uv run python examples/decision_models.py score
```

Plotting needs matplotlib, an optional extra that this command installs:

```bash
uv run --extra plot python examples/decision_models.py plot
```

This writes `outputs/cost_quality.png`: one panel per world, KL against cost,
with the Pareto frontier in red. If you ran with `--limit`, pass the same
`--limit` to `score` and `plot`.

## Citation

```bibtex
@misc{dai2026probabilitybench,
  title         = {ProbabilityBench: Model Evaluation and Calibration with Exact Ground-Truth Uncertainty},
  author        = {Dai, Yuyang and Tang, Yuting and Futami, Futoshi and Ishida, Takashi},
  year          = {2026},
  eprint        = {2610.17016},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
}
```

## License

MIT licensed; see [LICENSE](LICENSE).
