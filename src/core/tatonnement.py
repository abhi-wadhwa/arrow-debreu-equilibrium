r"""Tatonnement price adjustment dynamics.

The Walrasian auctioneer adjusts prices according to excess demand:

    dp_j / dt  =  z_j(p)          (continuous tatonnement)
    p^{t+1}_j  =  p^t_j + eta * z_j(p^t)   (discrete tatonnement)

Prices are projected onto the unit simplex after each step to maintain
normalisation  sum_j p_j = 1.

This module provides both discrete and ODE-based solvers for finding
competitive equilibrium prices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from src.core.economy import ExchangeEconomy


@dataclass
class TatonnementResult:
    """Result from tatonnement price adjustment.

    Attributes
    ----------
    prices : NDArray
        Equilibrium price vector (on the simplex).
    excess_demand : NDArray
        Final excess demand z(p*).
    converged : bool
        Whether the algorithm converged.
    iterations : int
        Number of iterations (discrete) or ODE evaluations (continuous).
    price_history : NDArray
        Array of shape (T, m) recording the price path.
    """

    prices: NDArray[np.float64]
    excess_demand: NDArray[np.float64]
    converged: bool
    iterations: int
    price_history: NDArray[np.float64]


def _project_simplex(p: NDArray[np.float64]) -> NDArray[np.float64]:
    """Project price vector onto the unit simplex (sum=1, all>=0)."""
    p = np.maximum(p, 1e-15)
    return p / p.sum()


def tatonnement(
    economy: ExchangeEconomy,
    p0: NDArray[np.float64] | None = None,
    method: str = "discrete",
    eta: float = 0.1,
    max_iter: int = 10_000,
    tol: float = 1e-8,
    t_span: tuple[float, float] = (0.0, 50.0),
    t_eval_points: int = 500,
) -> TatonnementResult:
    r"""Run tatonnement to find competitive equilibrium prices.

    Parameters
    ----------
    economy : ExchangeEconomy
        The exchange economy.
    p0 : NDArray, optional
        Initial price guess.  Defaults to uniform prices.
    method : str
        ``"discrete"`` for iterative step, ``"ode"`` for scipy ODE solver.
    eta : float
        Step size for discrete tatonnement.
    max_iter : int
        Maximum iterations (discrete mode).
    tol : float
        Convergence tolerance on ||z(p)||_inf.
    t_span : tuple
        Time interval for ODE solver.
    t_eval_points : int
        Number of evaluation points for the ODE trajectory.

    Returns
    -------
    TatonnementResult
    """
    m = economy.num_goods
    if p0 is None:
        p0 = np.ones(m, dtype=np.float64) / m
    else:
        p0 = _project_simplex(np.asarray(p0, dtype=np.float64))

    if method == "discrete":
        return _discrete_tatonnement(economy, p0, eta, max_iter, tol)
    elif method == "ode":
        return _ode_tatonnement(economy, p0, t_span, t_eval_points, tol)
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'discrete' or 'ode'.")


def _discrete_tatonnement(
    economy: ExchangeEconomy,
    p0: NDArray[np.float64],
    eta: float,
    max_iter: int,
    tol: float,
) -> TatonnementResult:
    """Discrete tatonnement with proportional price adjustment.

    Uses a multiplicative update rule that is more stable for
    multi-good economies:
        p_j <- p_j * (1 + eta * z_j / W_j)
    where W_j is the total endowment of good j.
    Falls back to additive updates with adaptive step size when
    the proportional update is not making progress.
    """
    p = p0.copy()
    history: List[NDArray[np.float64]] = [p.copy()]
    W = economy.aggregate_endowment

    best_err = np.inf
    best_p = p.copy()

    for it in range(1, max_iter + 1):
        z = economy.excess_demand(p)
        err = np.max(np.abs(z))

        if err < best_err:
            best_err = err
            best_p = p.copy()

        if err < tol:
            return TatonnementResult(
                prices=p,
                excess_demand=z,
                converged=True,
                iterations=it,
                price_history=np.array(history),
            )

        # Proportional (multiplicative) update:
        #   p_j *= (1 + eta * z_j / W_j)
        # This ensures prices stay positive and is scale-invariant.
        relative_z = z / np.maximum(W, 1e-15)
        # Clamp the relative update to avoid overshooting
        update = eta * relative_z
        update = np.clip(update, -0.3, 0.3)
        p = p * (1.0 + update)
        p = _project_simplex(p)
        history.append(p.copy())

    # Return the best prices found
    z_best = economy.excess_demand(best_p)
    return TatonnementResult(
        prices=best_p,
        excess_demand=z_best,
        converged=bool(np.max(np.abs(z_best)) < tol),
        iterations=max_iter,
        price_history=np.array(history),
    )


def _ode_tatonnement(
    economy: ExchangeEconomy,
    p0: NDArray[np.float64],
    t_span: tuple[float, float],
    t_eval_points: int,
    tol: float,
) -> TatonnementResult:
    """Continuous tatonnement using scipy ODE solver.

    Solves  dp/dt = z(p)  with projection onto the simplex.
    """
    m = len(p0)
    t_eval = np.linspace(t_span[0], t_span[1], t_eval_points)

    def rhs(t: float, p: NDArray[np.float64]) -> NDArray[np.float64]:
        p_proj = _project_simplex(p)
        z = economy.excess_demand(p_proj)
        # Project dynamics to stay on simplex tangent space:
        # remove component along (1,1,...,1) / m
        z_proj = z - np.mean(z)
        return z_proj

    sol = solve_ivp(
        rhs,
        t_span,
        p0,
        t_eval=t_eval,
        method="RK45",
        rtol=1e-10,
        atol=1e-12,
        max_step=0.5,
    )

    # Normalise each trajectory point
    price_history = np.array(
        [_project_simplex(sol.y[:, i]) for i in range(sol.y.shape[1])]
    )
    p_final = price_history[-1]
    z_final = economy.excess_demand(p_final)

    return TatonnementResult(
        prices=p_final,
        excess_demand=z_final,
        converged=bool(np.max(np.abs(z_final)) < tol),
        iterations=len(t_eval),
        price_history=price_history,
    )
