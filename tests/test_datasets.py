"""Generated cases have exact probabilities and reproducible draws."""
import numpy as np
import pytest

from probabilitybench import build_split, get_task
from probabilitybench.tasks import create_world
from probabilitybench.tasks.covert_transfer import outgoing_identifier, tag_observations


@pytest.mark.parametrize("task", ["paper_review", "covert_transfer", "financial_default"])
@pytest.mark.parametrize("level_index", [0, 1, 2])
def test_rendered_cases_have_exact_posteriors_and_stable_draws(task, level_index):
    world = create_world(task, nuisance_count=8)
    hidden = world.hiding_levels[level_index]["hidden"]
    rows = world.generate_dataset(12, hidden, np.random.default_rng(2032))
    repeat = world.generate_dataset(12, hidden, np.random.default_rng(2032))
    assert rows == repeat
    for row in rows:
        assert world.decode(world.render(row.visible)) == row.visible
        assert world.compute_posterior(row.visible) == pytest.approx(row.posterior)
        assert sum(row.posterior.values()) == pytest.approx(1)
        assert row.true_outcome in row.posterior
        assert all(0 <= p <= 1 for p in row.posterior.values())



@pytest.mark.parametrize("task", ["paper_review", "covert_transfer", "financial_default"])
def test_settings_and_irrelevant_fields_keep_case_ids_and_outcomes(task):
    reference = create_world(task, nuisance_count=8).generate_dataset(12, [], np.random.default_rng(2032))
    for count in (0, 4, 8):
        world = create_world(task, nuisance_count=count)
        for level in world.hiding_levels:
            rows = world.generate_dataset(12, level["hidden"], np.random.default_rng(2032))
            assert [(r.case_id, r.true_outcome) for r in rows] == [
                (r.case_id, r.true_outcome) for r in reference]
            if not level["hidden"]:  # Irrelevant fields never change an exact probability.
                assert [r.posterior for r in rows] == [r.posterior for r in reference]



def test_split_sizes_and_rows_follow_one_draw():
    task = get_task("financial_default")
    world = task.create_world()
    split = build_split(world, [], train_size=5, test_size=3, seed=2032, render_fn=task.render_fn)
    rows = world.generate_dataset(8, [], np.random.default_rng(2032))
    assert [len(split.train), len(split.test)] == [5, 3]
    assert [r.case_id for r in split.train + split.test] == [r.case_id for r in rows]
    assert split.test[0].rendered_text == world.render(rows[5].visible)
    with pytest.raises(ValueError, match="positive"):
        build_split(world, [], train_size=5, test_size=0)


def test_covert_transfer_posterior_matches_likelihood_ratio():
    world = create_world("covert_transfer", nuisance_count=0)
    visible = {"report_type": "progress update", **tag_observations("a"),
               "run_identifier": outgoing_identifier("progress update", "b" * 16, "a")}
    assert world.compute_posterior(visible)["encoding"] == pytest.approx(26 / 27)
    visible["run_identifier"] = "b" * 16
    assert world.compute_posterior(visible)["encoding"] == 0


@pytest.mark.parametrize("task, expected", [
    ("paper_review", (0.034, 0.091, 0.109)),
    ("covert_transfer", (0.019, 0.074, 0.145)),
    ("financial_default", (0.026, 0.108, 0.167)),
])
def test_exact_bayes_error_matches_the_paper(task, expected):
    world = create_world(task)
    errors = [world.exact_bayes_error(level["hidden"]) for level in world.hiding_levels]
    assert [round(e, 3) for e in errors] == list(expected)


