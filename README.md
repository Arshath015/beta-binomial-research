# Beta-Binomial Conjugate Inference Research Module

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)

A concise, hypothesis‑driven exploration of Bayesian updating for binomial experiments using the Beta conjugate prior.

---

## Table of Contents
- [Overview](#overview)
- [Tech Stack](#tech-stack)
- [Architecture](#architecture)
- [Theoretical Background](#theoretical-background)
- [Installation](#installation)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Findings](#findings)
- [Testing](#testing)
- [Limitations](#limitations)
- [Roadmap](#roadmap)
- [License](#license)

## Overview
This repository provides a clean, from‑scratch implementation of the Beta‑Binomial conjugate model, suitable for educational research, reproducible notebooks, and downstream statistical pipelines.

## Tech Stack
- Python 3.9+
- NumPy
- PyYAML (for optional config handling)
- pytest (for testing)

## Architecture
```
beta-binomial-research/
├─ notebooks/
│  └─ exploration.py
├─ src/
│  ├─ __init__.py
│  └─ beta_binomial.py
├─ tests/
│  ├─ test_posterior.py
│  └─ test_update.py
├─ results/
│  ├─ findings.md
│  └─ output.txt
├─ requirements.txt
└─ README.md
```

## Theoretical Background
The Beta distribution is the conjugate prior for the Binomial likelihood. Given a prior \(\text{Beta}(\alpha, \beta)\) and observed data consisting of \(k\) successes out of \(n\) trials, the posterior distribution is also a Beta distribution with updated parameters \(\alpha' = \alpha + k\) and \(\beta' = \beta + n - k\). This closed‑form update enables analytical posterior predictive calculations and straightforward credible interval extraction.

The posterior predictive probability of a future success is the mean of the posterior Beta, i.e. \((\alpha' / (\alpha' + \beta'))\). Moreover, the predictive distribution for a future Binomial observation integrates over the posterior, yielding a Beta‑Binomial distribution.

## Installation
```bash
git clone https://github.com/yourorg/beta-binomial-research.git
cd beta-binomial-research
pip install -r requirements.txt
```

## Usage
```python
from src.beta_binomial import BetaBinomialModel

# Prior belief: uniform (alpha=1, beta=1)
model = BetaBinomialModel(alpha=1.0, beta=1.0)

# Observed data: 7 successes out of 10 trials
posterior = model.update(k=7, n=10)
print(posterior)

# Predictive probability for next trial
print("Predictive success probability:", model.predictive_mean())
```

The above script is also executed in `notebooks/exploration.py`, which writes a short summary to `results/output.txt`.

## API Reference
- `BetaBinomialModel(alpha: float, beta: float)` – constructs a model with given prior parameters.
- `update(k: int, n: int) -> Tuple[float, float]` – returns updated (alpha, beta).
- `posterior_mean() -> float` – mean of the posterior Beta.
- `predictive_mean() -> float` – mean of the posterior predictive distribution for a single future Bernoulli trial.
- `predictive_distribution(m: int, N: int) -> np.ndarray` – Beta‑Binomial pmf for observing `m` successes in `N` future trials.

## Findings
See `results/findings.md` for a detailed analysis of the numerical stability tests and convergence experiments.

## Testing
Run the test suite with:
```bash
pytest -q
```

## Limitations
- Assumes independent Bernoulli trials.
- Numerical stability degrades for extremely large counts without log‑space handling.

## Roadmap
- Add log‑space implementations for high‑count regimes.
- Extend to hierarchical Beta‑Binomial models.
- Benchmark against PyMC3.

## License
MIT License
