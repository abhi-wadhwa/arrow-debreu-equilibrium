"""Demand computation for Arrow-Debreu exchange economies.

Given prices, solve each agent's constrained optimisation problem
    max  u_i(x)
    s.t. p . x <= p . omega_i,  x >= 0
and compute aggregate (excess) demand.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from src.core.economy import Agent, ExchangeEconomy


def compute_demand(
    agent: Agent,
    prices: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute Marshallian demand for a single agent.

    Parameters
    ----------
    agent : Agent
        The agent whose demand to compute.
    prices : NDArray
        Price vector (length m).

    Returns
    -------
    NDArray
        Optimal consumption bundle x* (length m).
    """
    return agent.demand(prices)


def compute_excess_demand(
    economy: ExchangeEconomy,
    prices: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute aggregate excess demand z(p).

    Parameters
    ----------
    economy : ExchangeEconomy
        The exchange economy.
    prices : NDArray
        Price vector (length m).

    Returns
    -------
    NDArray
        Excess demand z(p) = sum_i [x_i*(p) - omega_i].

    Notes
    -----
    By Walras' Law: p . z(p) = 0  for all p > 0.
    """
    return economy.excess_demand(prices)


def compute_individual_excess_demand(
    agent: Agent,
    prices: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Excess demand for a single agent: x_i*(p) - omega_i."""
    return agent.demand(prices) - agent.endowment


def verify_walras_law(
    economy: ExchangeEconomy,
    prices: NDArray[np.float64],
    tol: float = 1e-8,
) -> bool:
    """Check that Walras' Law holds: |p . z(p)| < tol."""
    return abs(economy.walras_law_check(prices)) < tol


def market_clearing_check(
    economy: ExchangeEconomy,
    prices: NDArray[np.float64],
    tol: float = 1e-6,
) -> tuple[bool, NDArray[np.float64]]:
    """Check whether markets clear at given prices.

    Returns
    -------
    (clears, excess_demand) : tuple
        clears is True if all |z_j(p)| < tol.
    """
    z = economy.excess_demand(prices)
    clears = bool(np.all(np.abs(z) < tol))
    return clears, z
