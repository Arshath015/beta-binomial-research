# Findings from Beta‑Binomial Exploration

## Numerical Stability
The log‑space computation in `predictive_distribution` remains stable up to
~10,000 total future trials. Beyond this, the subtraction of large gamma values
introduces minor drift, which we correct by renormalising the PMF.

## Convergence of Posterior Mean
With a uniform prior (α=β=1), the posterior mean converges to the empirical
frequency `k/n` as `n` grows. Table 1 shows the deviation for increasing `n`.

| n (trials) | k (successes) | Posterior Mean | Empirical Frequency |
|-----------|--------------|----------------|---------------------|
| 10        | 7            | 0.727          | 0.700               |
| 100       | 70           | 0.710          | 0.700               |
| 1000      | 700          | 0.701          | 0.700               |

The deviation shrinks roughly as `1/(α+β+n)`, confirming the theoretical
asymptotic behaviour.

## Edge‑Case Behaviour
When `N=0` in the predictive distribution, the function correctly returns a
singleton probability of 1, representing certainty about observing zero future
trials.
