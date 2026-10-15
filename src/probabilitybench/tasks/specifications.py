"""Finite, author-defined mechanisms. These are not fitted real-world models."""

from dataclasses import dataclass

import numpy as np
from .numerical import IRRELEVANT_NUMERICAL


@dataclass(frozen=True)
class Factor:
    key: str
    label: str
    values: tuple[str, str, str]


REVIEW_FACTORS = (
    Factor("novelty", "Novelty", ("incremental", "substantial", "major")),
    Factor("significance", "Problem significance", ("narrow", "moderate", "broad")),
    Factor("validity", "Technical validity", ("serious flaw", "limited concern", "sound")),
    Factor("evidence", "Evidence strength", ("weak", "moderate", "strong")),
    Factor("comparisons", "Comparison quality", ("inadequate", "partly controlled", "well controlled")),
    Factor("robustness", "Robustness", ("fragile", "mixed", "stable")),
    Factor("reproducibility", "Reproducibility", ("insufficient", "partial", "complete")),
    Factor("clarity", "Clarity", ("difficult", "adequate", "clear")),
)

FINANCE_FACTORS = (
    Factor("cash_generation", "Operating cash generation", ("weak", "moderate", "strong")),
    Factor("cash_buffer", "Cash buffer", ("limited", "moderate", "substantial")),
    Factor("debt_service", "Recurring debt service", ("light", "moderate", "heavy")),
    Factor("maturities", "Near-term principal maturities", ("dispersed", "moderate", "concentrated")),
    Factor("refinancing", "Refinancing access", ("limited", "partial", "broad")),
    Factor("rate_exposure", "Funding-rate exposure", ("low", "moderate", "high")),
    Factor("concentration", "Customer concentration", ("low", "moderate", "high")),
    Factor("flexibility", "Operating flexibility", ("limited", "moderate", "high")),
)

# Eight mutually independent details, independent of substantive/residual
# variables. Numerical supports and masses are specified in numerical.py;
# these catalogs fix field order and the remaining categorical values.
REVIEW_NUISANCES = {
    "author_count": IRRELEVANT_NUMERICAL["paper_review"]["author_count"].values,
    "manuscript_pages": IRRELEVANT_NUMERICAL["paper_review"]["manuscript_pages"].values,
    "figure_count": IRRELEVANT_NUMERICAL["paper_review"]["figure_count"].values,
    "table_count": IRRELEVANT_NUMERICAL["paper_review"]["table_count"].values,
    "reference_count": IRRELEVANT_NUMERICAL["paper_review"]["reference_count"].values,
    "appendix_pages": IRRELEVANT_NUMERICAL["paper_review"]["appendix_pages"].values,
    "submission_month": ("January", "February", "March", "April", "May", "June",
                         "July", "August", "September", "October", "November", "December"),
    "document_revision": ("1", "3"),
}
FINANCE_NUISANCES = {
    "employee_count": IRRELEVANT_NUMERICAL["financial_default"]["employee_count"].values,
    "operating_sites": IRRELEVANT_NUMERICAL["financial_default"]["operating_sites"].values,
    "administrative_departments": IRRELEVANT_NUMERICAL["financial_default"]["administrative_departments"].values,
    "reporting_month": ("March", "September"),
    "fiscal_year_end": ("June", "December"),
    "report_pages": IRRELEVANT_NUMERICAL["financial_default"]["report_pages"].values,
    "chart_count": IRRELEVANT_NUMERICAL["financial_default"]["chart_count"].values,
    "document_revision": ("1", "3"),
}

CONTEXTS = {
    "paper_review": {"contribution_type": ("theoretical", "empirical", "applied")},
    "financial_default": {
        "sector": ("technology", "manufacturing", "consumer", "healthcare"),
        "business_cycle": ("expansion", "steady", "contraction"),
    },
}
# Factors hidden at Medium and High (nested), chosen so the exact Bayes error
# rises strictly under each nested projection. Exact values come from each
# world's exact_bayes_error, not from the hiding names.
HIDDEN_FACTORS = {
    "paper_review": {"medium": ("novelty",), "high": ("novelty", "validity")},
    "financial_default": {"medium": ("cash_buffer",), "high": ("cash_buffer", "maturities")},
}
RESIDUAL_PROBABILITIES = np.array([0.03, 0.09, 0.76, 0.09, 0.03])
RESIDUAL_VALUES = {
    "paper_review": np.arange(-5., 6.),
    "financial_default": np.array([-3.0, -1.0, 0.0, 1.0, 3.0]),
}

# Columns follow REVIEW_FACTORS. Integer tenths avoid floating-point decisions
# at the acceptance boundary. Each row sums to 100 (10 in score units).
# No model results set these weights.
REVIEW_WEIGHTS_TENTHS = np.array([
    [17, 12, 20, 18, 7, 7, 8, 11],
    [10, 10, 15, 25, 13, 11, 9, 7],
    [9, 18, 13, 25, 9, 10, 8, 8],
])
REVIEW_THRESHOLD_TENTHS = 115
REVIEW_RESIDUAL_PROBABILITIES = np.array([
    .001, .004, .015, .03, .09, .72, .09, .03, .015, .004, .001,
])
REVIEW_EVIDENCE_PROBABILITIES = np.array([
    [.85, .13, .02], [.50, .30, .20], [.33, .29, .38],
])
REVIEW_ROBUSTNESS_PROBABILITIES = np.array([
    [.515, .27, .215], [.48, .27, .25], [.445, .27, .285],
])


@dataclass(frozen=True)
class ReviewSpecification:
    factors: tuple[Factor, ...]
    weights: np.ndarray
    threshold: int
    evidence_probabilities: np.ndarray
    robustness_probabilities: np.ndarray
    residual_values: np.ndarray
    residual_probabilities: np.ndarray


CURRENT_REVIEW = ReviewSpecification(
    REVIEW_FACTORS, REVIEW_WEIGHTS_TENTHS, REVIEW_THRESHOLD_TENTHS,
    REVIEW_EVIDENCE_PROBABILITIES, REVIEW_ROBUSTNESS_PROBABILITIES, RESIDUAL_VALUES["paper_review"],
    REVIEW_RESIDUAL_PROBABILITIES,
)
def joint_probabilities(task: str, states: np.ndarray, review: ReviewSpecification = CURRENT_REVIEW) -> np.ndarray:
    """A small Bayesian network with strictly positive conditional tables."""
    n_context = len(CONTEXTS[task])
    q = states[:, n_context:]
    mass = np.ones(len(states))
    if task == "paper_review":
        mass *= 1 / 3
    else:
        mass *= 1 / 4
        mass *= np.array([0.30, 0.40, 0.30])[states[:, 1]]
    for j in range(q.shape[1]):
        probabilities = np.tile([0.48, 0.27, 0.25], (len(states), 1))
        if task == "paper_review":
            if j == 2:
                probabilities[:] = [0.15, 0.35, 0.50]
            key = review.factors[j].key
            if key == "evidence":
                probabilities = review.evidence_probabilities[q[:, 2]]
            elif key == "robustness":
                probabilities = review.robustness_probabilities[q[:, 2]]
        else:
            # The credit cycle affects refinancing; weak cash generation
            # modestly increases the probability of a limited cash buffer.
            if j == 4:
                shift = 0.04 * (1.0 - states[:, 1])
                probabilities[:, 0] -= shift
                probabilities[:, 2] += shift
            if j == 1:
                shift = 0.03 * (q[:, 0] - 1.0)
                probabilities[:, 0] -= shift
                probabilities[:, 2] += shift
        mass *= probabilities[np.arange(len(states)), q[:, j]]
    if not np.isclose(mass.sum(), 1.0, atol=1e-12):
        raise ValueError("The specified conditional tables do not normalize")
    return mass / mass.sum()


def outcome_matrix(task: str, states: np.ndarray, review: ReviewSpecification = CURRENT_REVIEW) -> np.ndarray:
    """Deterministic outcomes for every substantive state and residual value."""
    q = states[:, len(CONTEXTS[task]):].astype(np.int64)
    shock = RESIDUAL_VALUES[task].astype(np.int64)
    if task == "paper_review":
        shock = review.residual_values.astype(np.int64)
        weights = review.weights[states[:, 0]]
        score = np.sum(weights * q, axis=1)
        # Strong evidence carries extra weight when comparisons are controlled;
        # stable findings reinforce sound technical work.
        score += 6 * ((q[:, 3] == 2) & (q[:, 4] == 2))
        robustness = next(j for j, factor in enumerate(review.factors) if factor.key == "robustness")
        score += 6 * ((q[:, 2] == 2) & (q[:, robustness] == 2))
        return ((score[:, None] + 10 * shock) >= review.threshold) & (q[:, 2, None] != 0)
    # Finance uses integer twentieths of the normalized monetary unit.
    cycle = states[:, 1]
    contraction = (cycle == 2).astype(np.int64)
    cash = np.array([60, 100, 140])[q[:, 0]]
    cash -= contraction * (20 + 15 * q[:, 6])
    cash += contraction * 10 * q[:, 7]
    # Sector changes the impact of concentrated customers during contraction.
    cash -= contraction * np.array([5, 10, 15, 0])[states[:, 0]] * q[:, 6]
    buffer = np.array([20, 60, 100])[q[:, 1]]
    refinancing = np.array([0, 2, 4])[q[:, 4]] * np.array([20, 15, 8])[cycle]
    service = np.array([20, 40, 60])[q[:, 2]]
    service += np.array([0, 5, 15])[cycle] * q[:, 5]
    principal = np.array([20, 60, 120])[q[:, 3]]
    # All amounts use a fixed normalized monetary unit, unrelated to headcount.
    resources = cash + buffer + refinancing
    obligations = service + principal
    return resources[:, None] + 20 * shock < obligations[:, None]
