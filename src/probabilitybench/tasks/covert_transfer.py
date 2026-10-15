"""Fresh random identifiers with exact conditional probabilities and Bayes risk.

The procedure label records an insertion even when the original letter matches.
The 15 background characters are independent draws, not a finite template bank.
They cancel from the posterior but must remain in inputs to evaluated models.

Identifiers are stored as contiguous letters and displayed in groups of four.
The separators are presentation only: they carry no information, never reach
the stored value, and the encoding positions count letters, not display slots.
"""

from collections import defaultdict
import math
import string

import numpy as np

from probabilitybench.core.types import DatasetExample
from probabilitybench.core.identifiers import stable_case_id
from .numerical import numeric_features, nuisance_values, sample_nuisances, IRRELEVANT_NUMERICAL
from probabilitybench.tasks.regimes import REGIME_NAMES
from .world import RichFiniteWorld

TASK_NAME = "covert_transfer"
ALPHABET = string.ascii_lowercase
IDENTIFIER_LENGTH = 16
IDENTIFIER_GROUP = 4
IDENTIFIER_SEPARATOR = "-"
REPORT_POSITIONS = {"progress update": 5, "experiment result": 9, "error report": 13}
REPORT_TYPES = tuple(REPORT_POSITIONS)
TAG_RANGES = ("a-d", "e-h", "i-l", "m-p", "q-t", "u-x", "y-z")
TAG_GROUPS = ("a-h", "i-p", "q-x", "y-z")
TAG_FIELDS = ("project_tag_overview", "project_tag_range", "project_tag_detail")
HIDING_LEVELS = [
    {"name": REGIME_NAMES[0], "hidden": []},
    {"name": REGIME_NAMES[1], "hidden": ["project_tag_detail"]},
    {"name": REGIME_NAMES[2], "hidden": ["project_tag_detail", "project_tag_range"]},
]
NUISANCES = {
    "audit_month": ("March", "September"),
    "report_revision": ("1", "2"),
    "attachment_count": IRRELEVANT_NUMERICAL["covert_transfer"]["attachment_count"].values,
    "logging_region": ("East", "West"),
    "export_version": ("1", "2"),
    "retention_tier": ("standard", "extended"),
    "review_team_size": IRRELEVANT_NUMERICAL["covert_transfer"]["review_team_size"].values,
    "processing_queue": ("primary", "secondary"),
}


def valid_identifier(value):
    return (isinstance(value, str) and len(value) == IDENTIFIER_LENGTH
            and all(c in ALPHABET for c in value))


def display_identifier(value: str) -> str:
    """Group the stored letters for reading; the grouping carries no evidence."""
    if not valid_identifier(value):
        raise ValueError(f"Identifier must contain exactly {IDENTIFIER_LENGTH} lowercase ASCII letters")
    return IDENTIFIER_SEPARATOR.join(value[i:i + IDENTIFIER_GROUP]
                                     for i in range(0, IDENTIFIER_LENGTH, IDENTIFIER_GROUP))


def parse_identifier(value: str) -> str:
    """Recover the stored letters from a displayed identifier."""
    return value.replace(IDENTIFIER_SEPARATOR, "")


def outgoing_identifier(report_type: str, original: str, tag: str) -> str:
    """Apply the insertion procedure, including a no-change replacement."""
    if report_type not in REPORT_POSITIONS or tag not in ALPHABET or len(tag) != 1:
        raise ValueError("Invalid report type or tag")
    if not valid_identifier(original):
        raise ValueError(f"Identifier must contain exactly {IDENTIFIER_LENGTH} lowercase ASCII letters")
    position = REPORT_POSITIONS[report_type] - 1
    return original[:position] + tag + original[position + 1:]


def tag_observations(tag):
    index = ALPHABET.index(tag)
    return dict(zip(TAG_FIELDS, (TAG_GROUPS[index // 8], TAG_RANGES[index // 4], tag)))


class CovertTransferWorld(RichFiniteWorld):
    """The covert-transfer world, with the same interface as the factor worlds."""

    def __init__(self, nuisance_count: int = 8) -> None:
        if type(nuisance_count) is not int or not 0 <= nuisance_count <= 8:
            raise ValueError("nuisance_count must be an integer from 0 to 8")
        self.task = TASK_NAME
        self.name = TASK_NAME
        self.nuisance_count = nuisance_count
        self.positive_label, self.negative_label = "encoding", "no-encoding"
        self.outcome_name = "encoding_status"
        self.outcome_values = [self.positive_label, self.negative_label]
        self.contexts = {"report_type": REPORT_TYPES}
        self.nuisance_values = nuisance_values(self.task, dict(list(NUISANCES.items())[:nuisance_count]))
        self.numeric_features = numeric_features(self.task, self.nuisance_values)
        self.hiding_levels = [{"name": x["name"], "hidden": list(x["hidden"])} for x in HIDING_LEVELS]
        # Empty tuple marks a structured string field, not an enumerable catalog.
        self.field_values = {
            "report_type": REPORT_TYPES, "project_tag_overview": TAG_GROUPS,
            "project_tag_range": TAG_RANGES, "project_tag_detail": tuple(ALPHABET),
            "run_identifier": (),
        }
        self.core_keys = tuple(self.field_values)
        self.field_values.update(self.nuisance_values)
        self.attribute_names = list(self.field_values)

    def _validate_visible(self, visible):
        for key, value in visible.items():
            if key in self.numeric_features:
                self.numeric_features[key].validate(value)
                continue
            valid = (valid_identifier(value) if key == "run_identifier"
                     else key in self.field_values and value in self.field_values[key])
            if not valid:
                raise ValueError(f"Unknown field or unsupported value: {key}={value}")

    def compute_posterior(self, visible):
        self._validate_visible(visible)
        allowed = set(ALPHABET)
        for key in TAG_FIELDS:
            if key in visible:
                value = visible[key]
                allowed &= set(ALPHABET[ALPHABET.index(value[0]):ALPHABET.index(value[-1]) + 1])
        if not allowed:
            raise ValueError("Visible assignment has zero probability under the world")
        positive = .5
        if "run_identifier" in visible:
            reports = (visible["report_type"],) if "report_type" in visible else REPORT_TYPES
            matches = sum(visible["run_identifier"][REPORT_POSITIONS[r] - 1] in allowed for r in reports)
            likelihood_ratio = 26 * matches / (len(allowed) * len(reports))
            positive = likelihood_ratio / (1 + likelihood_ratio)
        return {self.positive_label: positive, self.negative_label: 1 - positive}

    def _tag_partition(self, hidden):
        groups = defaultdict(list)
        for tag in ALPHABET:
            values = tag_observations(tag)
            groups[tuple(values[k] for k in TAG_FIELDS if k not in hidden)].append(tag)
        return list(groups.values())

    def exact_bayes_error(self, hidden_attributes):
        """Integrate only sufficient evidence, never a model's string-dependent output."""
        hidden = self._hidden(hidden_attributes)
        if "run_identifier" in hidden:
            return .5
        positions = 3 if "report_type" in hidden else 1
        risk = 0.
        for tags in self._tag_partition(hidden):
            m = len(tags)
            chance = m / 26
            for k in range(positions + 1):
                ordinary_mass = math.comb(positions, k) * chance**k * (1 - chance)**(positions - k)
                risk += chance * .5 * ordinary_mass * min(1., 26 * k / (m * positions))
        return risk

    def generate_dataset(self, n, hidden_attributes, rng=None):
        if type(n) is not int or n < 0:
            raise ValueError("n must be a nonnegative integer")
        hidden = self._hidden(hidden_attributes)
        rng = np.random.default_rng() if rng is None else rng
        seeds = rng.integers(0, 2**63, size=4, dtype=np.uint64)
        # Independent streams preserve common cases across hiding, nuisances,
        # and requested sample sizes; background generation never sees the label.
        report_rng, tag_rng, label_rng = [np.random.default_rng(s) for s in
                                        np.random.SeedSequence(int(seeds[0])).spawn(3)]
        reports = report_rng.integers(0, 3, size=n)
        tags = tag_rng.integers(0, 26, size=n)
        encoded = label_rng.integers(0, 2, size=n)
        letters = np.array(list(ALPHABET))[np.random.default_rng(int(seeds[1])).integers(
            0, 26, size=(n, IDENTIFIER_LENGTH))]
        nuisances = sample_nuisances(self.task, self.nuisance_values, seeds[2], n)
        records = []
        for i in range(n):
            report, tag = REPORT_TYPES[reports[i]], ALPHABET[tags[i]]
            original = "".join(letters[i])
            identifier = outgoing_identifier(report, original, tag) if encoded[i] else original
            full = {"report_type": report, **tag_observations(tag), "run_identifier": identifier}
            full.update({key: values[i] for key, values in nuisances.items()})
            visible = {k: v for k, v in full.items() if k not in hidden}
            posterior = self.compute_posterior(visible)
            records.append(DatasetExample(
                case_id=stable_case_id(self.task, int(seeds[3]), i), visible=visible, hidden={k: v for k, v in full.items() if k in hidden},
                true_outcome=self.positive_label if encoded[i] else self.negative_label,
                posterior=posterior,
            ))
        return records

    def decode(self, text):
        from .prose import decode_covert
        visible = decode_covert(self, text)
        self._validate_visible(visible)
        return visible

    def render(self, visible):
        self._validate_visible(visible)
        from .prose import render_covert
        return render_covert(self, visible)
