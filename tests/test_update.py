import pytest
from src.beta_binomial import BetaBinomialModel

def test_invalid_alpha_beta():
    with pytest.raises(ValueError):
        BetaBinomialModel(alpha=0, beta=1)
    with pytest.raises(ValueError):
        BetaBinomialModel(alpha=1, beta=-0.5)

def test_update_edge_cases():
    model = BetaBinomialModel(alpha=1.0, beta=1.0)
    # No data observed
    model.update(k=0, n=0)
    assert model.alpha == 1.0 and model.beta == 1.0
    # All successes
    model.update(k=5, n=5)
    assert model.alpha == 6.0 and model.beta == 1.0
    # All failures
    model.update(k=0, n=3)
    assert model.alpha == 6.0 and model.beta == 4.0
