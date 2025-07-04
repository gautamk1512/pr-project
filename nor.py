import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Parameters for five different normal distributions
params = [
    {"mu": 0, "sigma": 1, "label": "μ=0, σ=1"},
    {"mu": 0, "sigma": 2, "label": "μ=0, σ=2"},
    {"mu": 0, "sigma": 0.5, "label": "μ=0, σ=0.5"},
    {"mu": 2, "sigma": 1, "label": "μ=2, σ=1"},
    {"mu": -2, "sigma": 1, "label": "μ=-2, σ=1"},
]

x = np.linspace(-10, 10, 1000)

# Plot each distribution
plt.figure(figsize=(10, 6))
for p in params:
    y = norm.pdf(x, p["mu"], p["sigma"])
    plt.plot(x, y, label=p["label"])

plt.title("Gaussian (Normal) Distributions with Different μ and σ")
plt.xlabel("x")
plt.ylabel("Probability Density")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
