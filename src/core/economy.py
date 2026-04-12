"""Exchange economy model: agents, endowments, and utilities.

An Arrow-Debreu exchange economy consists of *n* agents and *m* goods.
Each agent *i* is endowed with a non-negative bundle omega_i in R^m and
has a utility function u_i : R^m_+ -> R that she seeks to maximise
subject to her budget constraint  p . x <= p . omega_i.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
from numpy.typing import NDArray

from src.core.utilities import UtilityFunction


@dataclass
class Agent:
    """An agent in an exchange economy.

    Parameters
    ----------
    endowment : NDArray[np.float64]
        Initial endowment vector omega_i (length m).
    utility : UtilityFunction
        Utility function object (must implement __call__ and demand).
    name : str
        Optional human-readable label.
    """

    endowment: NDArray[np.float64]
    utility: UtilityFunction
    name: str = ""

    def __post_init__(self) -> None:
        self.endowment = np.asarray(self.endowment, dtype=np.float64)
        if np.any(self.endowment < 0):
            raise ValueError(f"Endowment must be non-negative, got {self.endowment}")

    @property
    def num_goods(self) -> int:
        return len(self.endowment)

    def income(self, prices: NDArray[np.float64]) -> float:
        """Compute agent's income at given prices: p . omega."""
        return float(np.dot(prices, self.endowment))

    def demand(self, prices: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute Marshallian demand at given prices.

        Uses the analytic demand method on the utility function if available,
        otherwise falls back to numerical optimisation.
        """
        prices = np.asarray(prices, dtype=np.float64)
        inc = self.income(prices)
        if hasattr(self.utility, "demand"):
            return self.utility.demand(prices, inc)
        # Fallback: numerical optimisation
        return _numerical_demand(self.utility, prices, inc)

    def utility_at(self, x: NDArray[np.float64]) -> float:
        """Evaluate utility at bundle x."""
        return self.utility(x)


def _numerical_demand(
    utility: UtilityFunction,
    prices: NDArray[np.float64],
    income: float,
) -> NDArray[np.float64]:
    """Solve max u(x) s.t. p.x <= I, x >= 0 via scipy."""
    from scipy.optimize import minimize

    m = len(prices)
    x0 = np.full(m, income / (m * np.mean(prices)))

    def neg_u(x: NDArray[np.float64]) -> float:
        return -utility(np.maximum(x, 1e-12))

    constraints = [
        {"type": "ineq", "fun": lambda x: income - np.dot(prices, x)},
    ]
    bounds = [(1e-12, None)] * m
    result = minimize(
        neg_u,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 2000, "ftol": 1e-14},
    )
    return np.maximum(result.x, 0.0)


@dataclass
class ExchangeEconomy:
    """An Arrow-Debreu pure exchange economy.

    Parameters
    ----------
    agents : list of Agent
        The agents in the economy.
    """

    agents: List[Agent] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.agents:
            m = self.agents[0].num_goods
            for i, a in enumerate(self.agents):
                if a.num_goods != m:
                    raise ValueError(
                        f"Agent {i} has {a.num_goods} goods, expected {m}."
                    )

    @property
    def num_agents(self) -> int:
        return len(self.agents)

    @property
    def num_goods(self) -> int:
        if not self.agents:
            return 0
        return self.agents[0].num_goods

    @property
    def aggregate_endowment(self) -> NDArray[np.float64]:
        """Total endowment across all agents."""
        return np.sum([a.endowment for a in self.agents], axis=0)

    def excess_demand(self, prices: NDArray[np.float64]) -> NDArray[np.float64]:
        """Aggregate excess demand z(p) = sum_i [x_i(p) - omega_i].

        By Walras' Law, p . z(p) = 0 for all p.
        """
        prices = np.asarray(prices, dtype=np.float64)
        total_demand = np.zeros(self.num_goods, dtype=np.float64)
        total_endowment = np.zeros(self.num_goods, dtype=np.float64)
        for agent in self.agents:
            total_demand += agent.demand(prices)
            total_endowment += agent.endowment
        return total_demand - total_endowment

    def walras_law_check(self, prices: NDArray[np.float64]) -> float:
        """Return p . z(p) -- should be zero by Walras' Law."""
        prices = np.asarray(prices, dtype=np.float64)
        z = self.excess_demand(prices)
        return float(np.dot(prices, z))

    def allocations(
        self, prices: NDArray[np.float64]
    ) -> list[NDArray[np.float64]]:
        """Return the list of demand bundles for each agent at given prices."""
        prices = np.asarray(prices, dtype=np.float64)
        return [agent.demand(prices) for agent in self.agents]

    def utility_profile(self, prices: NDArray[np.float64]) -> NDArray[np.float64]:
        """Utility of each agent at the competitive equilibrium allocation."""
        allocs = self.allocations(prices)
        return np.array(
            [a.utility_at(x) for a, x in zip(self.agents, allocs)]
        )
