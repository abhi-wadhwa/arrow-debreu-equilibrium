"""Utility function library for Arrow-Debreu exchange economies.

Provides callable utility classes with analytic gradient support:
  - Cobb-Douglas:  u(x) = prod_j x_j^{alpha_j}
  - CES:           u(x) = (sum_j alpha_j x_j^rho)^{1/rho}
  - Leontief:      u(x) = min_j (x_j / alpha_j)
  - Quasi-linear:  u(x) = x_0 + sum_{j>=1} alpha_j ln(x_j)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class UtilityFunction(ABC):
    """Abstract base class for utility functions."""

    @abstractmethod
    def __call__(self, x: NDArray[np.float64]) -> float:
        """Evaluate utility at consumption bundle *x*."""

    @abstractmethod
    def gradient(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return the gradient of the utility function at *x*."""

    @property
    @abstractmethod
    def num_goods(self) -> int:
        """Number of goods in the economy."""


class CobbDouglas(UtilityFunction):
    r"""Cobb-Douglas utility.

    .. math::
        u(x) = \prod_{j=1}^{m} x_j^{\alpha_j}

    Parameters
    ----------
    alphas : array-like
        Exponents for each good.  Need not sum to 1 (they are stored as-is).
    """

    def __init__(self, alphas: NDArray[np.float64] | list[float]) -> None:
        self.alphas = np.asarray(alphas, dtype=np.float64)
        if np.any(self.alphas <= 0):
            raise ValueError("All Cobb-Douglas exponents must be positive.")

    @property
    def num_goods(self) -> int:
        return len(self.alphas)

    def __call__(self, x: NDArray[np.float64]) -> float:
        x = np.asarray(x, dtype=np.float64)
        return float(np.prod(np.power(np.maximum(x, 1e-300), self.alphas)))

    def gradient(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        x = np.asarray(x, dtype=np.float64)
        x_safe = np.maximum(x, 1e-300)
        u_val = np.prod(np.power(x_safe, self.alphas))
        return self.alphas * u_val / x_safe

    # ------------- Analytic demand (budget-share rule) ----------------
    def demand(
        self,
        prices: NDArray[np.float64],
        income: float,
    ) -> NDArray[np.float64]:
        r"""Marshallian demand for Cobb-Douglas utility.

        .. math::
            x_j^* = \frac{\alpha_j}{\sum_k \alpha_k} \cdot \frac{I}{p_j}
        """
        prices = np.asarray(prices, dtype=np.float64)
        shares = self.alphas / self.alphas.sum()
        return shares * income / prices


class CES(UtilityFunction):
    r"""Constant Elasticity of Substitution utility.

    .. math::
        u(x) = \left(\sum_{j=1}^{m} \alpha_j\, x_j^{\rho}\right)^{1/\rho}

    where :math:`\sigma = 1/(1-\rho)` is the elasticity of substitution.

    Parameters
    ----------
    alphas : array-like
        Share (weight) parameters.
    rho : float
        Substitution parameter.  :math:`\rho < 1, \rho \neq 0`.
        - :math:`\rho \to 0` recovers Cobb-Douglas.
        - :math:`\rho \to -\infty` recovers Leontief.
    """

    def __init__(
        self,
        alphas: NDArray[np.float64] | list[float],
        rho: float,
    ) -> None:
        self.alphas = np.asarray(alphas, dtype=np.float64)
        if abs(rho) < 1e-12:
            raise ValueError("rho must be non-zero (use CobbDouglas for rho->0).")
        self.rho = float(rho)

    @property
    def num_goods(self) -> int:
        return len(self.alphas)

    def __call__(self, x: NDArray[np.float64]) -> float:
        x = np.asarray(x, dtype=np.float64)
        x_safe = np.maximum(x, 1e-300)
        inner = np.sum(self.alphas * np.power(x_safe, self.rho))
        return float(np.power(np.maximum(inner, 1e-300), 1.0 / self.rho))

    def gradient(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        x = np.asarray(x, dtype=np.float64)
        x_safe = np.maximum(x, 1e-300)
        inner = np.sum(self.alphas * np.power(x_safe, self.rho))
        outer = np.power(np.maximum(inner, 1e-300), (1.0 / self.rho) - 1.0)
        return outer * self.alphas * np.power(x_safe, self.rho - 1.0)

    def demand(
        self,
        prices: NDArray[np.float64],
        income: float,
    ) -> NDArray[np.float64]:
        r"""Marshallian demand for CES utility.

        .. math::
            x_j^* = \frac{\alpha_j^{\sigma}\, p_j^{-\sigma}}
                         {\sum_k \alpha_k^{\sigma}\, p_k^{-\sigma}} \cdot
                    \frac{I}{p_j}

        where :math:`\sigma = 1 / (1 - \rho)`.
        """
        prices = np.asarray(prices, dtype=np.float64)
        sigma = 1.0 / (1.0 - self.rho)
        weights = np.power(self.alphas, sigma) * np.power(prices, -sigma)
        shares = weights / weights.sum()
        return shares * income / prices


class Leontief(UtilityFunction):
    r"""Leontief (perfect complements) utility.

    .. math::
        u(x) = \min_j \frac{x_j}{\alpha_j}

    Parameters
    ----------
    alphas : array-like
        Coefficients for each good (proportions in which goods are consumed).
    """

    def __init__(self, alphas: NDArray[np.float64] | list[float]) -> None:
        self.alphas = np.asarray(alphas, dtype=np.float64)
        if np.any(self.alphas <= 0):
            raise ValueError("All Leontief coefficients must be positive.")

    @property
    def num_goods(self) -> int:
        return len(self.alphas)

    def __call__(self, x: NDArray[np.float64]) -> float:
        x = np.asarray(x, dtype=np.float64)
        return float(np.min(x / self.alphas))

    def gradient(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Sub-gradient: 1/alpha_j at the binding good, 0 elsewhere."""
        x = np.asarray(x, dtype=np.float64)
        ratios = x / self.alphas
        j = int(np.argmin(ratios))
        g = np.zeros_like(x)
        g[j] = 1.0 / self.alphas[j]
        return g

    def demand(
        self,
        prices: NDArray[np.float64],
        income: float,
    ) -> NDArray[np.float64]:
        r"""Marshallian demand for Leontief utility.

        The consumer spends all income on a fixed-proportions basket:

        .. math::
            x_j^* = \frac{\alpha_j}{\sum_k \alpha_k p_k} \cdot I
        """
        prices = np.asarray(prices, dtype=np.float64)
        cost_per_unit = np.dot(self.alphas, prices)
        units = income / cost_per_unit
        return self.alphas * units


class QuasiLinear(UtilityFunction):
    r"""Quasi-linear utility.

    .. math::
        u(x) = x_0 + \sum_{j=1}^{m-1} \alpha_j \ln(x_j)

    Good 0 is the numeraire (linear term).

    Parameters
    ----------
    alphas : array-like
        Coefficients for the log terms (goods 1 through m-1).
    """

    def __init__(self, alphas: NDArray[np.float64] | list[float]) -> None:
        self.alphas = np.asarray(alphas, dtype=np.float64)
        if np.any(self.alphas <= 0):
            raise ValueError("All quasi-linear coefficients must be positive.")

    @property
    def num_goods(self) -> int:
        return len(self.alphas) + 1  # numeraire + log goods

    def __call__(self, x: NDArray[np.float64]) -> float:
        x = np.asarray(x, dtype=np.float64)
        x_safe = np.maximum(x[1:], 1e-300)
        return float(x[0] + np.sum(self.alphas * np.log(x_safe)))

    def gradient(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        x = np.asarray(x, dtype=np.float64)
        g = np.zeros_like(x)
        g[0] = 1.0
        x_safe = np.maximum(x[1:], 1e-300)
        g[1:] = self.alphas / x_safe
        return g

    def demand(
        self,
        prices: NDArray[np.float64],
        income: float,
    ) -> NDArray[np.float64]:
        r"""Marshallian demand for quasi-linear utility.

        Interior solution (when income is large enough):

        .. math::
            x_j = \frac{\alpha_j p_0}{p_j}, \quad j \geq 1

        and the remainder goes to the numeraire good 0.
        """
        prices = np.asarray(prices, dtype=np.float64)
        m = len(prices)
        x = np.zeros(m, dtype=np.float64)
        # Spend alpha_j * p_0 / p_j on each non-numeraire good
        p0 = prices[0]
        spending_others = self.alphas * p0  # cost for goods 1..m-1
        total_other_spending = np.sum(spending_others)
        if total_other_spending <= income:
            x[1:] = self.alphas * p0 / prices[1:]
            x[0] = (income - total_other_spending) / p0
        else:
            # Corner solution: scale spending proportionally
            scale = income / total_other_spending
            x[1:] = scale * self.alphas * p0 / prices[1:]
            x[0] = 0.0
        return x
