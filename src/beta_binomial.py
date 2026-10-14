"""Beta‑Binomial conjugate inference implementation.

The module defines a `BetaBinomialModel` class that encapsulates the prior
parameters (α, β) and provides methods for updating with observed data, computing
posterior statistics, and evaluating the posterior predictive distribution.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

class BetaBinomialModel:
    """Bayesian model for binomial data with a Beta prior.

    Parameters
    ----------
    alpha : float
        Prior pseudo‑count of successes (α > 0).
    beta : float
        Prior pseudo‑count of failures (β > 0).
    """

    def __init__(self, alpha: float = 1.0, beta: float = 1.0) -> None:
        if alpha <= 0 or beta <= 0:
            raise ValueError("Alpha and beta must be positive.")
        self.alpha = float(alpha)
        self.beta = float(beta)

    def update(self, k: int, n: int) -> Tuple[float, float]:
        """Update the prior with observed binomial data.

        Parameters
        ----------
        k : int
            Number of observed successes (0 ≤ k ≤ n).
        n : int
            Number of trials (n ≥ 0).

        Returns
        -------
        Tuple[float, float]
            Updated (α', β') posterior parameters.
        """
        if not (0 <= k <= n):
            raise ValueError("Success count k must satisfy 0 <= k <= n.")
        self.alpha += k
        self.beta += n - k
        return self.alpha, self.beta

    def posterior_mean(self) -> float:
        """Mean of the posterior Beta distribution."""
        return self.alpha / (self.alpha + self.beta)

    def predictive_mean(self) -> float:
        """Mean of the posterior predictive distribution for a single future trial.

        This equals the posterior mean of the success probability.
        """
        return self.posterior_mean()

    def predictive_distribution(self, m: int, N: int) -> np.ndarray:
        """Beta‑Binomial probability mass function.

        Parameters
        ----------
        m : int
            Number of successes of interest (0 ≤ m ≤ N).
        N : int
            Number of future trials.

        Returns
        -------
        np.ndarray
            Array of probabilities for each m in ``0..N``.
        """
        if N < 0:
            raise ValueError("Number of trials N must be non‑negative.")
        # Compute pmf for all possible success counts using log‑space to avoid overflow.
        ks = np.arange(N + 1)
        log_coef = (math.lgamma(N + 1) - math.lgamma(ks + 1) - math.lgamma(N - ks + 1))
        log_num = (
            math.lgamma(self.alpha + ks)
            + math.lgamma(self.beta + N - ks)
            - math.lgamma(self.alpha + self.beta + N)
        )
        log_den = math.lgamma(self.alpha) + math.lgamma(self.beta) - math.lgamma(self.alpha + self.beta)
        log_pmf = log_coef + log_num - log_den
        pmf = np.exp(log_pmf)
        # Normalize to guard against numerical drift
        pmf /= pmf.sum()
        return pmf

    def summary(self) -> str:
        """Return a short textual summary of the current posterior."""
        return (
            f"Posterior Beta(α={self.alpha:.3f}, β={self.beta:.3f}) – "
            f"mean={self.posterior_mean():.3f}"
        )
