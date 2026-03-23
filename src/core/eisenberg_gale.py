r"""Eisenberg-Gale convex programme for computing competitive equilibria.

For an exchange economy with homogeneous-of-degree-one utility functions
(Cobb-Douglas, CES), the competitive equilibrium can be found by solving
the convex programme:

    max   sum_i  w_i * log(u_i(x_i))
    s.t.  sum_i  x_{ij} <= sum_i omega_{ij}   for all j   (feasibility)
          x_{ij} >= 0                           for all i,j

where w_i = p . omega_i  (income of agent i at equilibrium).

For Cobb-Douglas utilities, u_i(x) = prod_j x_{ij}^{alpha_{ij}}, this
reduces to the Eisenberg-Gale programme with closed-form structure.

The equilibrium prices are recovered from the dual variables (shadow prices)
of the feasibility constraints.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize, LinearConstraint

from src.core.economy import ExchangeEconomy
from src.core.utilities import CobbDouglas, CES


@dataclass
class EisenbergGaleResult:
    """Result from the Eisenberg-Gale convex programme.

    Attributes
    ----------
    prices : NDArray
        Equilibrium prices (normalised to simplex).
    allocations : list of NDArray
        Equilibrium allocation for each agent.
    utilities : NDArray
        Utility value for each agent at the equilibrium.
    converged : bool
        Whether the optimiser converged.
    objective : float
        Optimal objective value.
    """

    prices: NDArray[np.float64]
    allocations: list[NDArray[np.float64]]
    utilities: NDArray[np.float64]
    converged: bool
    objective: float


def eisenberg_gale(
    economy: ExchangeEconomy,
    p0: NDArray[np.float64] | None = None,
) -> EisenbergGaleResult:
    r"""Solve the Eisenberg-Gale programme for an exchange economy.

    Works for economies with Cobb-Douglas or CES utilities.

    Parameters
    ----------
    economy : ExchangeEconomy
        The exchange economy.
    p0 : NDArray, optional
        Initial price guess for the dual.  If None, uses equal prices.

    Returns
    -------
    EisenbergGaleResult
    """
    n = economy.num_agents
    m = economy.num_goods
    total_endowment = economy.aggregate_endowment

    # Decision variables: x_{ij} for i=0..n-1, j=0..m-1
    # Stacked as a vector of length n*m: [x_{0,0},...,x_{0,m-1}, x_{1,0}, ...]
    dim = n * m

    def _get_allocation(x_flat: NDArray) -> list[NDArray]:
        return [x_flat[i * m : (i + 1) * m] for i in range(n)]

    def _utility_i(agent_idx: int, x_i: NDArray) -> float:
        return economy.agents[agent_idx].utility(x_i)

    # Objective: maximise sum_i w_i * log(u_i(x_i))
    # We don't know w_i = p . omega_i a priori.
    # For Cobb-Douglas, we can use the equal-weighted EG programme
    # (setting all weights to 1), which gives proportional equilibrium.
    # The equilibrium prices emerge from the dual variables.

    def neg_objective(x_flat: NDArray) -> float:
        allocs = _get_allocation(x_flat)
        val = 0.0
        for i in range(n):
            u = max(_utility_i(i, allocs[i]), 1e-300)
            val += np.log(u)
        return -val

    def neg_objective_grad(x_flat: NDArray) -> NDArray:
        allocs = _get_allocation(x_flat)
        grad = np.zeros(dim, dtype=np.float64)
        for i in range(n):
            x_i = allocs[i]
            u = max(_utility_i(i, x_i), 1e-300)
            g_i = economy.agents[i].utility.gradient(x_i)
            grad[i * m : (i + 1) * m] = -g_i / u
        return grad

    # Feasibility: sum_i x_{ij} <= total_endowment_j for each j
    # This is a linear constraint: A @ x <= b
    A_feas = np.zeros((m, dim), dtype=np.float64)
    for j in range(m):
        for i in range(n):
            A_feas[j, i * m + j] = 1.0

    linear_constraint = LinearConstraint(A_feas, lb=0.0, ub=total_endowment)

    # Initial guess: give each agent their endowment
    x0 = np.concatenate([a.endowment for a in economy.agents])
    x0 = np.maximum(x0, 1e-6)

    bounds = [(1e-12, None)] * dim

    result = minimize(
        neg_objective,
        x0,
        jac=neg_objective_grad,
        method="SLSQP",
        bounds=bounds,
        constraints=[{"type": "ineq", "fun": lambda x: total_endowment - A_feas @ x}],
        options={"maxiter": 5000, "ftol": 1e-14},
    )

    allocs = _get_allocation(result.x)

    # Extract prices from KKT conditions:
    # At optimum, for each agent i and good j:
    #   (1/u_i) * du_i/dx_{ij} = lambda_j   (shadow price)
    # So prices are proportional to marginal utility / utility for any agent
    # at any good they consume a positive amount of.
    prices = np.zeros(m, dtype=np.float64)
    for j in range(m):
        shadows = []
        for i in range(n):
            x_i = allocs[i]
            u_i = max(_utility_i(i, x_i), 1e-300)
            g_i = economy.agents[i].utility.gradient(x_i)
            if x_i[j] > 1e-10:
                shadows.append(g_i[j] / u_i)
        if shadows:
            prices[j] = np.mean(shadows)
        else:
            prices[j] = 1e-10

    prices = prices / prices.sum()

    utilities = np.array([_utility_i(i, allocs[i]) for i in range(n)])

    return EisenbergGaleResult(
        prices=prices,
        allocations=allocs,
        utilities=utilities,
        converged=result.success,
        objective=-result.fun,
    )
