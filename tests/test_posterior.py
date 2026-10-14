import math
import numpy as np
from src.beta_binomial import BetaBinomialModel

def test_posterior_mean_uniform():
    model = BetaBinomialModel(alpha=1.0, beta=1.0)
    model.update(k=5, n=10)
    # Posterior α=6, β=6 -> mean 0.5
    assert math.isclose(model.posterior_mean(), 0.5, rel_tol=1e-9)

def test_predictive_distribution_normalization():
    model = BetaBinomialModel(alpha=2.0, beta=3.0)
    pmf = model.predictive_distribution(m=0, N=5)
    # pmf should sum to 1
    assert math.isclose(pmf.sum(), 1.0, rel_tol=1e-9)
    # Edge case: N=0 should return array([1.])
    pmf_zero = model.predictive_distribution(m=0, N=0)
    assert np.array_equal(pmf_zero, np.array([1.0]))
