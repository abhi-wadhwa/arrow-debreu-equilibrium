r"""Edgeworth box analysis for 2-agent, 2-good exchange economies.

The Edgeworth box is the fundamental geometric tool for visualising
exchange in a two-person, two-good economy.  Key objects:

- **Indifference curves**: level sets of u_i(x) = c.
- **Contract curve**: the set of Pareto-optimal allocations
  (tangencies of indifference curves, i.e., MRS_1 = MRS_2).
- **Core**: allocations that both agents prefer to their endowments
  and are Pareto-optimal.
- **Competitive equilibrium (CE)**: the unique allocation on the
  contract curve where the price line through the endowment point
  is tangent to both agents' indifference curves.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq

from src.core.economy import Agent, ExchangeEconomy
from src.core.utilities import CobbDouglas


@dataclass
class EdgeworthResult:
    """Result of Edgeworth box analysis.

    Attributes
    ----------
    contract_curve : NDArray
        Array of shape (K, 2) giving allocations to agent 1 on the
        contract curve.
    core : NDArray
        Subset of the contract curve that is in the core.
    ce_allocation : NDArray
        Competitive equilibrium allocation for agent 1 (shape (2,)).
    ce_prices : NDArray
        Equilibrium price vector (shape (2,), normalised so p1=1).
    total_endowment : NDArray
        Aggregate endowment omega_1 + omega_2.
    """

    contract_curve: NDArray[np.float64]
    core: NDArray[np.float64]
    ce_allocation: NDArray[np.float64]
    ce_prices: NDArray[np.float64]
    total_endowment: NDArray[np.float64]


class EdgeworthBox:
    """Edgeworth box for a 2-agent, 2-good economy.

    Parameters
    ----------
    agent1 : Agent
        First agent.
    agent2 : Agent
        Second agent.
    """

    def __init__(self, agent1: Agent, agent2: Agent) -> None:
        if agent1.num_goods != 2 or agent2.num_goods != 2:
            raise ValueError("Edgeworth box requires exactly 2 goods.")
        self.agent1 = agent1
        self.agent2 = agent2
        self.total = agent1.endowment + agent2.endowment
        self.economy = ExchangeEconomy(agents=[agent1, agent2])

    def mrs(self, agent: Agent, x: NDArray[np.float64]) -> float:
        """Marginal rate of substitution: (du/dx1) / (du/dx2)."""
        g = agent.utility.gradient(x)
        if abs(g[1]) < 1e-300:
            return 1e15
        return float(g[0] / g[1])

    def indifference_curve(
        self,
        agent_idx: int,
        utility_level: float,
        n_points: int = 200,
    ) -> NDArray[np.float64]:
        """Compute indifference curve for an agent at a given utility level.

        For Cobb-Douglas u(x1,x2) = x1^a * x2^b, the curve is
        x2 = (c / x1^a)^{1/b}.

        Parameters
        ----------
        agent_idx : int
            0 for agent 1, 1 for agent 2.
        utility_level : float
            Target utility level.
        n_points : int
            Number of points to sample.

        Returns
        -------
        NDArray of shape (K, 2)
            Points (x1, x2) on the indifference curve within the box.
        """
        agent = [self.agent1, self.agent2][agent_idx]

        if agent_idx == 0:
            x1_max = self.total[0]
            x2_max = self.total[1]
        else:
            x1_max = self.total[0]
            x2_max = self.total[1]

        points = []
        x1_vals = np.linspace(1e-6, x1_max - 1e-6, n_points)

        for x1 in x1_vals:
            # Solve u(x1, x2) = utility_level for x2
            if isinstance(agent.utility, CobbDouglas):
                a, b = agent.utility.alphas
                # x1^a * x2^b = c  =>  x2 = (c / x1^a)^{1/b}
                val = utility_level / (x1 ** a)
                if val > 0:
                    x2 = val ** (1.0 / b)
                else:
                    continue
            else:
                # Numerical root-finding
                def _residual(x2_try: float) -> float:
                    bundle = np.array([x1, x2_try])
                    if agent_idx == 1:
                        bundle = bundle  # already in agent 2's coordinates
                    return agent.utility(bundle) - utility_level

                try:
                    x2 = brentq(_residual, 1e-10, x2_max * 2, xtol=1e-12)
                except ValueError:
                    continue

            if agent_idx == 0:
                if 0 < x2 < x2_max:
                    points.append([x1, x2])
            else:
                # Agent 2's allocation in agent 1's coordinates
                x1_a1 = self.total[0] - x1
                x2_a1 = self.total[1] - x2
                if 0 < x1_a1 < self.total[0] and 0 < x2_a1 < self.total[1]:
                    points.append([x1_a1, x2_a1])

        if not points:
            return np.empty((0, 2))
        return np.array(points)

    def contract_curve(self, n_points: int = 300) -> NDArray[np.float64]:
        """Compute the contract curve (set of Pareto-optimal allocations).

        At Pareto optima, MRS_1(x) = MRS_2(omega - x).

        Returns
        -------
        NDArray of shape (K, 2)
            Allocations to agent 1 on the contract curve.
        """
        points = []
        x1_vals = np.linspace(1e-4, self.total[0] - 1e-4, n_points)

        for x1 in x1_vals:
            # For each x1 for agent 1, find x2 such that MRS_1 = MRS_2
            def _mrs_diff(x2: float) -> float:
                bundle1 = np.array([x1, x2])
                bundle2 = self.total - bundle1
                if np.any(bundle2 < 1e-10):
                    return 1e10
                mrs1 = self.mrs(self.agent1, bundle1)
                mrs2 = self.mrs(self.agent2, bundle2)
                return mrs1 - mrs2

            try:
                x2_star = brentq(
                    _mrs_diff, 1e-6, self.total[1] - 1e-6, xtol=1e-10
                )
                points.append([x1, x2_star])
            except ValueError:
                continue

        if not points:
            return np.empty((0, 2))
        return np.array(points)

    def core(
        self, contract_curve: NDArray[np.float64] | None = None
    ) -> NDArray[np.float64]:
        """Compute the core: Pareto-optimal allocations preferred by both.

        The core is the subset of the contract curve where both agents
        achieve at least their endowment utility.

        Parameters
        ----------
        contract_curve : NDArray, optional
            Pre-computed contract curve.  If None, computed internally.

        Returns
        -------
        NDArray of shape (K, 2)
        """
        if contract_curve is None:
            contract_curve = self.contract_curve()

        u1_endow = self.agent1.utility(self.agent1.endowment)
        u2_endow = self.agent2.utility(self.agent2.endowment)

        core_points = []
        for pt in contract_curve:
            bundle1 = pt
            bundle2 = self.total - pt
            if (
                self.agent1.utility(bundle1) >= u1_endow - 1e-10
                and self.agent2.utility(bundle2) >= u2_endow - 1e-10
            ):
                core_points.append(pt)

        if not core_points:
            return np.empty((0, 2))
        return np.array(core_points)

    def competitive_equilibrium(self) -> tuple[NDArray, NDArray]:
        """Find the competitive equilibrium (CE).

        At the CE, the price ratio equals the MRS and the allocation
        lies on the budget line through the endowment.

        Returns
        -------
        (allocation, prices) : tuple
            allocation is agent 1's bundle (shape (2,)).
            prices is normalised price vector with p[0] = 1.
        """
        # Find price ratio p1/p2 such that markets clear
        def _excess_good1(price_ratio: float) -> float:
            """Excess demand for good 1 at price (1, price_ratio)."""
            p = np.array([1.0, price_ratio])
            z = self.economy.excess_demand(p)
            return z[0]

        try:
            p_ratio = brentq(_excess_good1, 1e-4, 1e4, xtol=1e-12)
        except ValueError:
            # Fallback: grid search then refine
            ratios = np.logspace(-3, 3, 1000)
            z_arr = np.array([_excess_good1(r) for r in ratios])
            # Find sign change
            sign_changes = np.where(np.diff(np.sign(z_arr)))[0]
            if len(sign_changes) == 0:
                raise RuntimeError("Could not find competitive equilibrium.")
            idx = sign_changes[0]
            p_ratio = brentq(
                _excess_good1, ratios[idx], ratios[idx + 1], xtol=1e-12
            )

        prices = np.array([1.0, p_ratio])
        prices = prices / prices.sum()
        allocation = self.agent1.demand(prices)
        return allocation, prices

    def analyse(self, n_points: int = 300) -> EdgeworthResult:
        """Run full Edgeworth box analysis.

        Returns
        -------
        EdgeworthResult
        """
        cc = self.contract_curve(n_points)
        core_pts = self.core(cc)
        ce_alloc, ce_prices = self.competitive_equilibrium()

        return EdgeworthResult(
            contract_curve=cc,
            core=core_pts,
            ce_allocation=ce_alloc,
            ce_prices=ce_prices,
            total_endowment=self.total,
        )

    def budget_line(
        self,
        prices: NDArray[np.float64],
        n_points: int = 200,
    ) -> NDArray[np.float64]:
        """Compute the budget line through the endowment point.

        Returns points (x1, x2) satisfying p . x = p . omega_1
        within the box boundaries.
        """
        income = np.dot(prices, self.agent1.endowment)
        x1_vals = np.linspace(0, self.total[0], n_points)
        points = []
        for x1 in x1_vals:
            if abs(prices[1]) < 1e-15:
                continue
            x2 = (income - prices[0] * x1) / prices[1]
            if 0 <= x2 <= self.total[1]:
                points.append([x1, x2])
        if not points:
            return np.empty((0, 2))
        return np.array(points)
