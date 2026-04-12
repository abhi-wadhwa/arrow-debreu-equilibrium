r"""Welfare theorem verification for Arrow-Debreu exchange economies.

First Welfare Theorem (FWT)
---------------------------
Every competitive equilibrium allocation is Pareto-optimal.
Verification: check that no feasible reallocation can improve one agent's
utility without reducing another's.

Second Welfare Theorem (SWT)
----------------------------
Every Pareto-optimal allocation can be supported as a competitive
equilibrium (with appropriate lump-sum transfers).
Verification: given a Pareto-optimal allocation, find supporting prices
from the common MRS and verify market clearing with transfers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from src.core.economy import Agent, ExchangeEconomy


@dataclass
class WelfareResult:
    """Result of welfare theorem verification.

    Attributes
    ----------
    theorem : str
        ``"first"`` or ``"second"``.
    holds : bool
        Whether the theorem property is verified.
    details : str
        Human-readable explanation.
    utilities_ce : NDArray
        Utility profile at the tested allocation.
    utilities_deviation : NDArray | None
        Utility profile of the best deviation found (FWT only).
    supporting_prices : NDArray | None
        Supporting prices (SWT only).
    """

    theorem: str
    holds: bool
    details: str
    utilities_ce: NDArray[np.float64]
    utilities_deviation: NDArray[np.float64] | None = None
    supporting_prices: NDArray[np.float64] | None = None


def verify_first_welfare_theorem(
    economy: ExchangeEconomy,
    prices: NDArray[np.float64],
    n_trials: int = 500,
) -> WelfareResult:
    r"""Verify the First Welfare Theorem.

    Check that the competitive equilibrium allocation at *prices* is
    Pareto-optimal by trying to find a feasible Pareto-improving deviation.

    Parameters
    ----------
    economy : ExchangeEconomy
        The exchange economy.
    prices : NDArray
        Equilibrium price vector.
    n_trials : int
        Number of random deviations to test.

    Returns
    -------
    WelfareResult
    """
    prices = np.asarray(prices, dtype=np.float64)
    n = economy.num_agents
    m = economy.num_goods
    total = economy.aggregate_endowment
    allocs = economy.allocations(prices)
    u_ce = np.array([economy.agents[i].utility(allocs[i]) for i in range(n)])

    # Try to find a Pareto-dominating feasible allocation
    found_deviation = False
    best_deviation_utilities = None

    for _ in range(n_trials):
        # Generate a random feasible allocation
        # Split the total endowment among agents randomly
        splits = np.random.dirichlet(np.ones(n), size=m)  # shape (m, n)
        candidate = []
        for i in range(n):
            x_i = total * splits[:, i]
            candidate.append(x_i)

        u_cand = np.array(
            [economy.agents[i].utility(candidate[i]) for i in range(n)]
        )

        # Pareto improvement: all >= and at least one >
        if np.all(u_cand >= u_ce - 1e-10) and np.any(u_cand > u_ce + 1e-10):
            found_deviation = True
            best_deviation_utilities = u_cand
            break

    # Also try a more directed search: for each agent, try to improve
    # their utility while keeping others at least as well off
    if not found_deviation:
        for target in range(n):
            result = _try_pareto_improvement(economy, allocs, u_ce, target)
            if result is not None:
                found_deviation = True
                best_deviation_utilities = result
                break

    holds = not found_deviation
    if holds:
        details = (
            "First Welfare Theorem VERIFIED: no Pareto-improving deviation "
            f"found in {n_trials} random trials + directed search. "
            "The competitive equilibrium allocation is Pareto-optimal."
        )
    else:
        details = (
            "First Welfare Theorem VIOLATED: a Pareto-improving deviation was "
            f"found. CE utilities: {u_ce}, deviation utilities: "
            f"{best_deviation_utilities}."
        )

    return WelfareResult(
        theorem="first",
        holds=holds,
        details=details,
        utilities_ce=u_ce,
        utilities_deviation=best_deviation_utilities,
    )


def _try_pareto_improvement(
    economy: ExchangeEconomy,
    allocs: list[NDArray[np.float64]],
    u_ce: NDArray[np.float64],
    target: int,
) -> NDArray[np.float64] | None:
    """Try to find a Pareto improvement by maximising agent *target*'s
    utility subject to all others being at least as well off and feasibility.
    """
    n = economy.num_agents
    m = economy.num_goods
    total = economy.aggregate_endowment
    dim = n * m

    def neg_target_utility(x_flat: NDArray) -> float:
        x_target = x_flat[target * m : (target + 1) * m]
        return -economy.agents[target].utility(np.maximum(x_target, 1e-12))

    constraints = []
    # Feasibility: sum_i x_{ij} = total_j for all j
    for j in range(m):
        def _feas(x_flat: NDArray, j_=j) -> float:
            s = sum(x_flat[i * m + j_] for i in range(n))
            return total[j_] - s  # >= 0 means sum <= total
        def _feas2(x_flat: NDArray, j_=j) -> float:
            s = sum(x_flat[i * m + j_] for i in range(n))
            return s - total[j_]  # >= 0 means sum >= total
        constraints.append({"type": "eq", "fun": lambda x, j_=j: sum(x[i * m + j_] for i in range(n)) - total[j_]})

    # Each non-target agent must be at least as well off
    for i in range(n):
        if i == target:
            continue
        def _u_constraint(x_flat: NDArray, i_=i) -> float:
            x_i = x_flat[i_ * m : (i_ + 1) * m]
            return economy.agents[i_].utility(np.maximum(x_i, 1e-12)) - u_ce[i_]
        constraints.append({"type": "ineq", "fun": _u_constraint})

    x0 = np.concatenate(allocs)
    bounds = [(1e-12, None)] * dim

    result = minimize(
        neg_target_utility,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 2000, "ftol": 1e-14},
    )

    if result.success:
        u_new = np.array([
            economy.agents[i].utility(
                np.maximum(result.x[i * m : (i + 1) * m], 1e-12)
            )
            for i in range(n)
        ])
        if (
            np.all(u_new >= u_ce - 1e-10)
            and u_new[target] > u_ce[target] + 1e-8
        ):
            return u_new
    return None


def verify_second_welfare_theorem(
    economy: ExchangeEconomy,
    target_allocation: list[NDArray[np.float64]],
) -> WelfareResult:
    r"""Verify the Second Welfare Theorem.

    Given a Pareto-optimal allocation, find supporting prices p such that
    each agent's allocation maximises their utility on the budget set
    defined by p (possibly with lump-sum transfers).

    Parameters
    ----------
    economy : ExchangeEconomy
        The exchange economy.
    target_allocation : list of NDArray
        The Pareto-optimal allocation to support.

    Returns
    -------
    WelfareResult
    """
    n = economy.num_agents
    m = economy.num_goods

    u_target = np.array([
        economy.agents[i].utility(target_allocation[i]) for i in range(n)
    ])

    # At a Pareto optimum, the gradients of utility are proportional:
    #   grad u_i(x_i) = lambda_i * p
    # So p is proportional to grad u_i(x_i) for any i with x_i >> 0.

    # Use agent 0's gradient to find prices
    g0 = economy.agents[0].utility.gradient(target_allocation[0])
    if np.max(np.abs(g0)) < 1e-15:
        return WelfareResult(
            theorem="second",
            holds=False,
            details="Could not find supporting prices: zero gradient.",
            utilities_ce=u_target,
        )

    prices = g0 / g0.sum()  # normalise to simplex

    # Verify that each agent's allocation maximises utility on the
    # budget set { x : p.x <= p.x_i* }
    # i.e., check that MRS = price ratio for all agents
    holds = True
    details_parts = []

    for i in range(n):
        g_i = economy.agents[i].utility.gradient(target_allocation[i])
        # Check proportionality: g_i = lambda_i * prices
        if np.max(np.abs(g_i)) < 1e-15:
            continue
        g_i_normalised = g_i / g_i.sum()
        error = np.max(np.abs(g_i_normalised - prices))
        if error > 1e-4:
            holds = False
            details_parts.append(
                f"Agent {i}: gradient not proportional to prices "
                f"(max error = {error:.6e})."
            )
        else:
            details_parts.append(
                f"Agent {i}: supporting prices verified (error = {error:.6e})."
            )

    # Compute transfers
    transfers = np.array([
        np.dot(prices, target_allocation[i]) - np.dot(prices, economy.agents[i].endowment)
        for i in range(n)
    ])
    details_parts.append(f"Transfers: {transfers}")
    details_parts.append(f"Transfer balance: {transfers.sum():.6e}")

    if holds:
        details = (
            "Second Welfare Theorem VERIFIED: supporting prices found. "
            + " ".join(details_parts)
        )
    else:
        details = (
            "Second Welfare Theorem check: issues found. "
            + " ".join(details_parts)
        )

    return WelfareResult(
        theorem="second",
        holds=holds,
        details=details,
        utilities_ce=u_target,
        supporting_prices=prices,
    )


def is_pareto_optimal(
    economy: ExchangeEconomy,
    allocation: list[NDArray[np.float64]],
    n_trials: int = 200,
) -> bool:
    """Check whether an allocation is Pareto-optimal by random search."""
    n = economy.num_agents
    m = economy.num_goods
    total = economy.aggregate_endowment
    u_alloc = np.array([
        economy.agents[i].utility(allocation[i]) for i in range(n)
    ])

    for _ in range(n_trials):
        splits = np.random.dirichlet(np.ones(n), size=m)
        candidate = [total * splits[:, i] for i in range(n)]
        u_cand = np.array([
            economy.agents[i].utility(candidate[i]) for i in range(n)
        ])
        if np.all(u_cand >= u_alloc - 1e-10) and np.any(u_cand > u_alloc + 1e-10):
            return False
    return True
