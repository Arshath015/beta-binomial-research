# %%
"""Exploratory notebook‑style script for Beta‑Binomial inference.

Runs a simple scenario, prints the posterior summary, and writes a short
result file to ``results/output.txt``.
"""

from src.beta_binomial import BetaBinomialModel

# Prior: uniform belief
model = BetaBinomialModel(alpha=1.0, beta=1.0)

# Simulated observation: 12 successes out of 20 trials
posterior_params = model.update(k=12, n=20)
print("Updated parameters:", posterior_params)
print(model.summary())

# Predictive probability for a single future trial
pred_mean = model.predictive_mean()
print("Predictive success probability:", pred_mean)

# Write a concise report
output_path = "results/output.txt"
with open(output_path, "w") as f:
    f.write("Beta‑Binomial Exploration Result\n")
    f.write(f"Posterior α={model.alpha:.3f}, β={model.beta:.3f}\n")
    f.write(f"Posterior mean={model.posterior_mean():.3f}\n")
    f.write(f"Predictive mean={pred_mean:.3f}\n")
print(f"Report written to {output_path}")
# %%
