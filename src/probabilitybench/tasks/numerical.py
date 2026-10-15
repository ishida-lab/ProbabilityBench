"""Author-chosen numerical supports, units, and probability distributions.

These are synthetic design parameters, not fits to empirical populations.
Ordinal measurements are uniform on a finite grid conditional on the original
category. Count distributions are truncated and renormalized, never clipped.
"""
from dataclasses import dataclass
from functools import cached_property
import math

import numpy as np
from scipy.optimize import brentq
from scipy.stats import nbinom, norm


@dataclass(frozen=True)
class NumericFeature:
    lower: int
    upper: int
    unit: str
    template: str
    median: int | None = None
    dispersion: float = 3
    distribution: str = "negative_binomial"
    zero_mass: float = 0
    decimals: int = 0
    thresholds: tuple[float, ...] = ()

    @cached_property
    def values(self):
        scale = 10 ** self.decimals
        return tuple(i if scale == 1 else i / scale
                     for i in range(self.lower * scale, self.upper * scale + 1))

    def validate(self, value):
        # Numeric-looking labels and booleans must not silently become numbers.
        if isinstance(value, (bool, str)) or not isinstance(value, (int, float)):
            raise ValueError(f"Expected a numeric value in {self.unit}: {value!r}")
        scaled = value * 10 ** self.decimals
        if (not math.isfinite(value) or not self.lower <= value <= self.upper
                or value != round(scaled) / 10 ** self.decimals):
            raise ValueError(f"Unsupported numerical value in {self.unit}: {value!r}")

    def code(self, value):
        self.validate(value)
        if not self.thresholds:
            raise ValueError("Only relevant numerical features have category thresholds")
        return int(np.searchsorted(self.thresholds, value, side="right"))

    @cached_property
    def category_values(self):
        return tuple(tuple(v for v in self.values if self.code(v) == c) for c in range(3))

    @cached_property
    def probabilities(self):
        if self.thresholds:
            raise ValueError("Relevant numerical features require a conditional category")
        values = np.asarray(self.values)

        def weights(location):
            if self.distribution == "lognormal":
                # Integer N is a rounded lognormal, conditional on the support.
                high = norm.cdf((np.log(values + .5) - location) / self.dispersion)
                low = norm.cdf((np.log(values - .5) - location) / self.dispersion)
                mass = high - low
            else:
                mean = np.exp(location)
                mass = nbinom.pmf(values, self.dispersion, self.dispersion / (self.dispersion + mean))
            if self.zero_mass:
                mass[values == 0] = 0
            mass = mass / mass.sum() * (1 - self.zero_mass)
            if self.zero_mass:
                mass[values == 0] = self.zero_mass
            return mass

        # Mid-CDF fixes a stable integer median after truncation/zero inflation.
        def residual(location):
            mass = weights(location)
            return mass[values < self.median].sum() + .5 * mass[values == self.median].sum() - .5

        location = brentq(residual, -5, 15)
        probabilities = weights(location)
        probabilities.setflags(write=False)
        return probabilities

    def sample(self, rng, n, codes=None):
        if self.thresholds:
            if codes is None:
                raise ValueError("Category codes are required for a relevant numerical feature")
            # One independent draw per case; changing visibility does not change it.
            uniforms = rng.random(n)
            return [self.category_values[int(c)][min(int(u * len(self.category_values[int(c)])),
                                                     len(self.category_values[int(c)]) - 1)]
                    for c, u in zip(codes, uniforms)]
        return [self.values[int(i)] for i in rng.choice(len(self.values), n, p=self.probabilities)]

    def render(self, value):
        self.validate(value)
        number = f"{value:.{self.decimals}f}"
        return self.template.format(value=number, s="" if value == 1 else "s")

    def scale(self, value):
        self.validate(value)
        return (value - self.lower) / (self.upper - self.lower)


RELEVANT_NUMERICAL = {
    "paper_review": {
        "evidence": NumericFeature(1, 5, "rating out of 5",
            "The strength of supporting evidence is rated {value} out of 5.",
            distribution="conditional_uniform", decimals=1, thresholds=(2.5, 3.5)),
        "clarity": NumericFeature(1, 5, "rating out of 5",
            "The writing-clarity rating is {value} out of 5.",
            distribution="conditional_uniform", decimals=1, thresholds=(2.5, 3.5)),
    },
    "financial_default": {
        "concentration": NumericFeature(0, 100, "percent of annual revenue",
            "The three largest customers account for {value}% of annual revenue.",
            distribution="conditional_uniform", decimals=1, thresholds=(20, 50)),
        "rate_exposure": NumericFeature(0, 100, "percent of outstanding debt",
            "Interest rates reset within the next 12 months on {value}% of outstanding debt.",
            distribution="conditional_uniform", decimals=1, thresholds=(25, 60)),
    },
    "covert_transfer": {},
}

IRRELEVANT_NUMERICAL = {
    "financial_default": {
        "employee_count": NumericFeature(20, 5000, "people", "covers {value} employees",
            200, 1, "lognormal"),
        "operating_sites": NumericFeature(1, 30, "locations", "{value} operating site{s}", 3, 2),
        "administrative_departments": NumericFeature(1, 20, "departments",
            "{value} administrative department{s}", 5, 6),
        "report_pages": NumericFeature(4, 80, "pages", "runs {value} pages", 20, 8),
        "chart_count": NumericFeature(0, 20, "charts", "{value} chart{s}", 5, 3),
    },
    "paper_review": {
        "author_count": NumericFeature(1, 20, "people", "by {value} author{s}", 4, 6),
        "manuscript_pages": NumericFeature(4, 30, "pages", "runs {value} pages", 10, 20),
        "figure_count": NumericFeature(0, 20, "figures", "{value} figure{s}", 5, 4),
        "table_count": NumericFeature(0, 15, "tables", "{value} table{s}", 3, 3),
        "reference_count": NumericFeature(5, 150, "references", "{value} references", 40, 5),
        "appendix_pages": NumericFeature(0, 60, "pages", "{value} page{s} of appendix", 10, 3,
            zero_mass=.2),
    },
    "covert_transfer": {
        "attachment_count": NumericFeature(0, 20, "attachments", "with {value} attachment{s}", 3, 2),
        "review_team_size": NumericFeature(1, 12, "people", "was reviewed by a team of {value}", 4, 6),
    },
}


def numeric_features(task, nuisance_keys=()):
    return {**RELEVANT_NUMERICAL[task],
            **{k: v for k, v in IRRELEVANT_NUMERICAL[task].items() if k in nuisance_keys}}


def nuisance_values(task, catalog):
    return {k: IRRELEVANT_NUMERICAL[task][k].values if k in IRRELEVANT_NUMERICAL[task] else v
            for k, v in catalog.items()}


def sample_nuisances(task, catalog, seed, n):
    children = np.random.SeedSequence(int(seed)).spawn(8)
    result = {}
    for j, (key, values) in enumerate(catalog.items()):
        rng = np.random.default_rng(children[j])
        spec = IRRELEVANT_NUMERICAL[task].get(key)
        result[key] = (spec.sample(rng, n) if spec else
                       [values[int(i)] for i in rng.integers(0, len(values), size=n)])
    return result
