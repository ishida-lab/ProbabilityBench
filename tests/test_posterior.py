from probabilitybench.tasks import create_world


def test_exact_posterior_matches_manual_result() -> None:
    """Posterior should normalize and assign more mass to default in this risky case."""

    world = create_world("financial_default", nuisance_count=0)
    posterior = world.compute_posterior(
        {
            "sector": "technology",
            "cash_buffer": "limited",
            "debt_service": "heavy",
            "cash_generation": "weak",
            "business_cycle": "contraction",
        }
    )
    assert round(sum(posterior.values()), 6) == 1.0
    assert posterior["default"] > posterior["no-default"]
