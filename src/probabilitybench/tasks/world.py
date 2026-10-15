"""Exact inference for the peer-review and credit-risk worlds.

Substantive states and the residual are finite, so every exact probability is
a sum over them. Irrelevant fields are independent of everything else and
cancel from P(outcome | visible fields). Numerical measurements identify their
original category exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from probabilitybench.core.types import DatasetExample, FeatureValue
from probabilitybench.core.identifiers import stable_case_id
from .numerical import numeric_features, nuisance_values, sample_nuisances
from probabilitybench.tasks.regimes import REGIME_NAMES
from .specifications import (
    CONTEXTS, FINANCE_FACTORS, FINANCE_NUISANCES, HIDDEN_FACTORS,
    RESIDUAL_PROBABILITIES, RESIDUAL_VALUES, REVIEW_FACTORS, REVIEW_NUISANCES,
    CURRENT_REVIEW, joint_probabilities, outcome_matrix,
)


@dataclass(frozen=True)
class CoreTable:
    keys: tuple[str, ...]
    codes: np.ndarray
    masses: np.ndarray
    posteriors: np.ndarray
    state_to_pattern: np.ndarray



def factors_for(task: str):
    if task not in CONTEXTS:
        raise ValueError(f"Unknown world: {task}")
    return REVIEW_FACTORS if task == "paper_review" else FINANCE_FACTORS


def hiding_levels(task: str) -> list[dict]:
    hidden = HIDDEN_FACTORS[task]
    return [dict(name=name, hidden=list(h)) for name, h in zip(REGIME_NAMES, ((), hidden["medium"], hidden["high"]))]


class RichFiniteWorld:
    """Peer-review or credit-risk world. Nuisance fields are independently sampled.

    ``nuisance_count`` selects a prefix of the fixed feature list, so matched
    draws at 0, 4, and 8 fields share substantive states, outcomes, IDs, and the
    common nuisance fields. Case IDs are never rendered.
    """

    def __init__(self, task: str, nuisance_count: int = 8) -> None:
        self.factors = factors_for(task)
        self.review_spec = CURRENT_REVIEW if task == "paper_review" else None
        if self.review_spec:
            self.factors = self.review_spec.factors
        if type(nuisance_count) is not int or not 0 <= nuisance_count <= 8:
            raise ValueError("nuisance_count must be an integer from 0 to 8")
        self.task = task
        self.name = task
        self.nuisance_count = nuisance_count
        self.positive_label = "accept" if task == "paper_review" else "default"
        self.negative_label = "reject" if task == "paper_review" else "no-default"
        self.outcome_name = "decision" if task == "paper_review" else "default_status"
        self.outcome_values = [self.positive_label, self.negative_label]
        self.contexts = CONTEXTS[task]
        all_nuisances = REVIEW_NUISANCES if task == "paper_review" else FINANCE_NUISANCES
        self.nuisance_values = nuisance_values(task, dict(list(all_nuisances.items())[:nuisance_count]))
        self.numeric_features = numeric_features(task, self.nuisance_values)
        self.hiding_levels = hiding_levels(task)

        sizes = [len(v) for v in self.contexts.values()] + [3] * len(self.factors)
        self.base_states = np.indices(sizes, dtype=np.int8).reshape(len(sizes), -1).T.copy()
        options = {"review": self.review_spec} if self.review_spec else {}
        self.base_masses = joint_probabilities(task, self.base_states, **options)
        self.residual_values = (self.review_spec.residual_values if self.review_spec else RESIDUAL_VALUES[task]).copy()
        self.residual_probabilities = (self.review_spec.residual_probabilities if self.review_spec else RESIDUAL_PROBABILITIES).copy()
        self.outcomes = outcome_matrix(task, self.base_states, **options)
        self.base_posteriors = self.outcomes @ self.residual_probabilities

        self.field_values: dict[str, tuple[FeatureValue, ...]] = dict(self.contexts)
        columns = [self.base_states[:, i] for i in range(len(self.contexts))]
        for j, factor in enumerate(self.factors, len(self.contexts)):
            self.field_values[factor.key] = factor.values
            columns.append(self.base_states[:, j])
        self.core_keys = tuple(self.field_values)
        self.core_field_values = dict(self.field_values)
        self.core_codes = np.column_stack(columns)
        self.field_values.update(self.nuisance_values)
        self.field_values.update({k: spec.values for k, spec in self.numeric_features.items()})
        self.attribute_names = list(self.field_values)
        self._tables: dict[tuple[str, ...], CoreTable] = {}

    def _hidden(self, hidden_attributes: Sequence[str]) -> set[str]:
        hidden = set(hidden_attributes)
        unknown = hidden - set(self.attribute_names)
        if unknown:
            raise ValueError(f"Unknown hidden attributes: {sorted(unknown)}")
        return hidden

    def core_pattern_table(self, hidden_attributes: Sequence[str]) -> CoreTable:
        """Mass and exact probability of every pattern of visible relevant fields.

        Irrelevant fields are left out because they never change a probability.
        Generated cases and the exact Bayes error both read their values from here.
        """
        hidden = self._hidden(hidden_attributes)
        keys = tuple(key for key in self.core_keys if key not in hidden)
        if keys not in self._tables:
            indices = [i for i, key in enumerate(self.core_keys) if key not in hidden]
            if indices:
                codes, inverse = np.unique(self.core_codes[:, indices], axis=0, return_inverse=True)
            else:
                codes = np.empty((1, 0), dtype=np.int8)
                inverse = np.zeros(len(self.base_states), dtype=np.int64)
            masses = np.bincount(inverse, weights=self.base_masses)
            positive = np.bincount(inverse, weights=self.base_masses * self.base_posteriors)
            self._tables[keys] = CoreTable(keys, codes, masses, positive / masses, inverse)
        return self._tables[keys]

    def exact_bayes_error(self, hidden_attributes: Sequence[str]) -> float:
        table = self.core_pattern_table(hidden_attributes)
        return float(table.masses @ np.minimum(table.posteriors, 1 - table.posteriors))

    def _validate_visible(self, visible: dict[str, FeatureValue]) -> None:
        for key, value in visible.items():
            if key in self.numeric_features:
                self.numeric_features[key].validate(value)
            elif key not in self.field_values or value not in self.field_values[key]:
                raise ValueError(f"Unknown field or unsupported value: {key}={value}")

    def core_code(self, key, value):
        if key in self.numeric_features:
            return self.numeric_features[key].code(value)
        return self.core_field_values[key].index(value)

    def compute_posterior(self, visible: dict[str, FeatureValue]) -> dict[str, float]:
        """Exact summation for any valid partial observation, including nuisances."""
        self._validate_visible(visible)
        matches = np.ones(len(self.base_states), dtype=bool)
        for j, key in enumerate(self.core_keys):
            if key in visible:
                matches &= self.core_codes[:, j] == self.core_code(key, visible[key])
        mass = self.base_masses[matches].sum()
        if mass <= 0:
            raise ValueError("Visible assignment has zero probability under the world")
        positive = float(self.base_masses[matches] @ self.base_posteriors[matches] / mass)
        return {self.positive_label: positive, self.negative_label: 1 - positive}

    def generate_dataset(
        self, n: int, hidden_attributes: Sequence[str], rng: np.random.Generator | None = None,
    ) -> list[DatasetExample]:
        if type(n) is not int or n < 0:
            raise ValueError("n must be a nonnegative integer")
        hidden = self._hidden(hidden_attributes)
        rng = np.random.default_rng() if rng is None else rng
        # Fixed stream allocation makes nuisance count and hiding independent
        # of the substantive draws. IDs have their own unrelated stream.
        seeds = rng.integers(0, 2**63, size=4, dtype=np.uint64)
        core_rng, residual_rng = (np.random.default_rng(int(s)) for s in seeds[:2])
        states = core_rng.choice(len(self.base_states), size=n, p=self.base_masses)
        residuals = residual_rng.choice(len(self.residual_values), size=n, p=self.residual_probabilities)
        nuisances = sample_nuisances(self.task, self.nuisance_values, seeds[2], n)
        # Reserve stream 8 after the eight nuisance streams. This preserves
        # substantive/residual draws and existing nominal details for a seed.
        measurements = {}
        for j, key in enumerate(self.core_keys):
            if key in self.numeric_features:
                measurement_rng = np.random.default_rng(np.random.SeedSequence(int(seeds[2]), spawn_key=(8, j)))
                measurements[key] = self.numeric_features[key].sample(
                    measurement_rng, n, self.core_codes[states, j])
        table = self.core_pattern_table(hidden_attributes)
        records = []
        for i, state in enumerate(states):
            full = {key: self.core_field_values[key][self.core_codes[state, j]] for j, key in enumerate(self.core_keys)}
            full.update({key: values[i] for key, values in {**nuisances, **measurements}.items()})
            positive = float(table.posteriors[table.state_to_pattern[state]])
            outcome = self.positive_label if self.outcomes[state, residuals[i]] else self.negative_label
            records.append(DatasetExample(
                case_id=stable_case_id(self.task, int(seeds[3]), i),
                visible={k: v for k, v in full.items() if k not in hidden},
                hidden={k: v for k, v in full.items() if k in hidden},
                true_outcome=outcome,
                posterior={self.positive_label: positive, self.negative_label: 1 - positive},
            ))
        return records

    def render(self, visible: dict[str, FeatureValue]) -> str:
        """Name-free memo. IDs and field relevance are never rendered.

        Each field has a unique phrase, making the observation exactly
        recoverable. The fixed order is schema order, independent of labels.
        """
        self._validate_visible(visible)
        from .prose import render_factor_world
        return render_factor_world(self, visible)

    def decode(self, text: str) -> dict[str, FeatureValue]:
        """Recover attributes from this renderer; reject ambiguous field values."""
        from .prose import decode_by_phrases, factor_phrases, decode_numeric, REVIEW_CONTEXT
        context = (decode_by_phrases({"contribution_type": REVIEW_CONTEXT}, text)
                   if self.task == "paper_review" else {})
        visible = decode_by_phrases(factor_phrases(self, context.get("contribution_type")), text)
        visible.update(decode_numeric(self, text))
        self._validate_visible(visible)
        return visible

