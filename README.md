# Arrow-Debreu General Equilibrium Engine

Compute **Walrasian equilibria** in Arrow-Debreu exchange economies using tatonnement dynamics, the Eisenberg-Gale convex programme, and Edgeworth box analysis.

[![CI](https://github.com/abhi-wadhwa/arrow-debreu-equilibrium/actions/workflows/ci.yml/badge.svg)](https://github.com/abhi-wadhwa/arrow-debreu-equilibrium/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

An **Arrow-Debreu exchange economy** consists of $n$ agents and $m$ goods. Each agent $i$ is endowed with a bundle $\omega_i \in \mathbb{R}^m_+$ and has a utility function $u_i : \mathbb{R}^m_+ \to \mathbb{R}$ representing her preferences. A **competitive (Walrasian) equilibrium** is a price vector $p^* \in \mathbb{R}^m_+$ and an allocation $(x_1^*, \ldots, x_n^*)$ such that:

1. **Utility maximisation**: each agent maximises utility on her budget set:

$$x_i^* = \arg\max_{x \geq 0} \; u_i(x) \quad \text{s.t.} \quad p^* \cdot x \leq p^* \cdot \omega_i$$

2. **Market clearing**: aggregate demand equals aggregate supply:

$$\sum_{i=1}^n x_i^* = \sum_{i=1}^n \omega_i$$

The **Arrow-Debreu theorem** (1954) guarantees that under standard assumptions (continuity, monotonicity, convexity of preferences), a competitive equilibrium exists.

---

## Theory

### Excess Demand and Walras' Law

The **excess demand function** $z : \mathbb{R}^m_{++} \to \mathbb{R}^m$ is defined as:

$$z(p) = \sum_{i=1}^n \left[ x_i^*(p) - \omega_i \right]$$

**Walras' Law** states that for all $p > 0$:

$$p \cdot z(p) = 0$$

This follows from budget exhaustion: each agent spends her entire income. At equilibrium, $z(p^*) = 0$.

### Tatonnement

The **tatonnement process** models a Walrasian auctioneer who adjusts prices in the direction of excess demand:

$$\frac{dp_j}{dt} = z_j(p)$$

The discrete version updates:

$$p_j^{t+1} = p_j^t + \eta \cdot z_j(p^t)$$

followed by projection onto the unit simplex $\Delta^{m-1} = \{p \geq 0 : \sum_j p_j = 1\}$.

For economies with **gross substitutes** (where an increase in the price of one good raises the demand for all other goods), tatonnement is globally stable and converges to the unique equilibrium.

### Utility Functions

| Type | Formula | Demand $x_j^*$ |
|------|---------|-----------------|
| **Cobb-Douglas** | $u(x) = \prod_j x_j^{\alpha_j}$ | $x_j = \frac{\alpha_j}{\sum_k \alpha_k} \cdot \frac{I}{p_j}$ |
| **CES** | $u(x) = \left(\sum_j \alpha_j x_j^\rho\right)^{1/\rho}$ | $x_j = \frac{\alpha_j^\sigma p_j^{-\sigma}}{\sum_k \alpha_k^\sigma p_k^{-\sigma}} \cdot \frac{I}{p_j}$ |
| **Leontief** | $u(x) = \min_j \frac{x_j}{\alpha_j}$ | $x_j = \frac{\alpha_j}{\sum_k \alpha_k p_k} \cdot I$ |
| **Quasi-linear** | $u(x) = x_0 + \sum_{j \geq 1} \alpha_j \ln x_j$ | $x_j = \frac{\alpha_j p_0}{p_j}, \quad x_0 = \frac{I - \sum_{j \geq 1} \alpha_j p_0}{p_0}$ |

where $I = p \cdot \omega_i$ is income and $\sigma = 1/(1-\rho)$ is the elasticity of substitution.

### Eisenberg-Gale Convex Programme

For homogeneous utility functions, the competitive equilibrium can be computed by solving:

$$\max_{x_{ij} \geq 0} \sum_{i=1}^n \log \, u_i(x_i) \quad \text{s.t.} \quad \sum_{i=1}^n x_{ij} \leq \sum_{i=1}^n \omega_{ij}, \; \forall j$$

The dual variables (shadow prices) of the feasibility constraints give the equilibrium prices.

### Edgeworth Box

For the 2-agent, 2-good case, the **Edgeworth box** provides a geometric representation:

- **Indifference curves**: level sets $\{x : u_i(x) = c\}$
- **Contract curve**: the locus of Pareto-optimal allocations where $\text{MRS}_1 = \text{MRS}_2$:

$$\frac{\partial u_1 / \partial x_1}{\partial u_1 / \partial x_2} = \frac{\partial u_2 / \partial x_1}{\partial u_2 / \partial x_2}$$

- **Core**: allocations on the contract curve that both agents prefer to the endowment (individually rational + Pareto-optimal)
- **Competitive equilibrium**: the unique point on the contract curve where the price line through the endowment is tangent to both agents' indifference curves

### Welfare Theorems

**First Welfare Theorem**: Every competitive equilibrium allocation is Pareto-optimal. No feasible reallocation can make one agent better off without making another worse off.

**Second Welfare Theorem**: Every Pareto-optimal allocation can be supported as a competitive equilibrium with appropriate lump-sum transfers. If $x^*$ is Pareto-optimal, there exist prices $p$ and transfers $T_i$ such that $x^*$ is a CE for the economy where each agent's income is $p \cdot \omega_i + T_i$.

---

## Installation

```bash
git clone https://github.com/abhi-wadhwa/arrow-debreu-equilibrium.git
cd arrow-debreu-equilibrium
pip install -e ".[dev]"
```

## Quick Start

```python
import numpy as np
from src.core.economy import Agent, ExchangeEconomy
from src.core.utilities import CobbDouglas
from src.core.tatonnement import tatonnement

# Define agents
alice = Agent(
    endowment=np.array([8.0, 2.0]),
    utility=CobbDouglas([0.4, 0.6]),
    name="Alice",
)
bob = Agent(
    endowment=np.array([2.0, 8.0]),
    utility=CobbDouglas([0.6, 0.4]),
    name="Bob",
)

# Create economy and find equilibrium
economy = ExchangeEconomy(agents=[alice, bob])
result = tatonnement(economy, tol=1e-10)

print(f"Prices: {result.prices}")
print(f"Converged: {result.converged}")
print(f"Allocations: {economy.allocations(result.prices)}")
```

## CLI

```bash
# Run the demo
python -m src.cli demo

# Run tatonnement on a random economy
python -m src.cli tatonnement

# Edgeworth box analysis
python -m src.cli edgeworth

# Launch the Streamlit dashboard
python -m src.cli app
```

## Interactive Dashboard

```bash
streamlit run src/viz/app.py
```

The dashboard includes four tabs:

1. **Edgeworth Box Explorer** -- drag sliders to adjust endowments and preferences; see indifference curves, the contract curve, core, and competitive equilibrium in real time.
2. **Tatonnement Dynamics** -- watch prices converge on the simplex with excess demand convergence plots.
3. **Multi-Good Economy** -- excess demand functions plotted against price, with Walras' Law verification.
4. **Welfare Analysis** -- utility comparison (endowment vs. CE) and welfare theorem verification.

## Tests

```bash
pytest tests/ -v
```

Key test cases:
- 2-agent Cobb-Douglas analytic CE matches numerical solution to 6 decimal places
- Walras' Law $p \cdot z(p) = 0$ verified for 1000 random price vectors
- Market clearing: all $|z_j(p^*)| < \varepsilon$
- First and Second Welfare Theorems verified

## Docker

```bash
docker build -t arrow-debreu .
docker run -p 8501:8501 arrow-debreu
```

## Project Structure

```
arrow-debreu-equilibrium/
├── src/
│   ├── core/
│   │   ├── economy.py          # Agent and ExchangeEconomy classes
│   │   ├── utilities.py        # Cobb-Douglas, CES, Leontief, quasi-linear
│   │   ├── demand.py           # Demand and excess demand computation
│   │   ├── tatonnement.py      # Discrete and ODE-based tatonnement
│   │   ├── eisenberg_gale.py   # Convex programme for CE computation
│   │   ├── edgeworth.py        # Edgeworth box: contract curve, core, CE
│   │   └── welfare.py          # First and Second welfare theorem checks
│   ├── viz/
│   │   └── app.py              # Streamlit interactive dashboard
│   └── cli.py                  # Command-line interface
├── tests/                      # Comprehensive test suite
├── examples/
│   └── demo.py                 # Full walkthrough of all features
├── pyproject.toml
├── Makefile
├── Dockerfile
└── .github/workflows/ci.yml
```

## References

- Arrow, K. J., & Debreu, G. (1954). *Existence of an equilibrium for a competitive economy*. Econometrica, 22(3), 265-290.
- Mas-Colell, A., Whinston, M. D., & Green, J. R. (1995). *Microeconomic Theory*. Oxford University Press.
- Eisenberg, E., & Gale, D. (1959). *Consensus of subjective probabilities: The pari-mutuel method*. Annals of Mathematical Statistics, 30(1), 165-168.
- Scarf, H. E. (1967). *The approximation of fixed points of a continuous mapping*. SIAM Journal on Applied Mathematics, 15(5), 1328-1343.

## License

MIT
