"""Stable case identifiers."""
import hashlib

# Each world's fixed prefix for its case IDs. Changing a prefix would change
# every case ID in that world, including those in the published dataset.
CASE_ID_NAMESPACES = {
    "financial_default": "financial_default_v2_numerical_features_v1",
    "paper_review": "paper_review_v2_numerical_features_v1_review_eight_factors_v1",
    "covert_transfer": "covert_transfer_v2_iid_short_identifiers_v1_numerical_features_v1",
}


def stable_case_id(task: str, seed: int, index: int) -> str:
    """Identify a case by world, ID seed and row index; it never encodes the outcome."""
    return hashlib.blake2s(f"{CASE_ID_NAMESPACES[task]}:{seed}:{index}".encode(), digest_size=12).hexdigest()
