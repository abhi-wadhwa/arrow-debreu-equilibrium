"""Tests for demand computation and Walras' Law."""

import numpy as np
import pytest

from src.core.economy import Agent, ExchangeEconomy
from src.core.utilities import CobbDouglas, CES, Leontief, QuasiLinear
from src.core.demand import (
    compute_demand,
    compute_excess_demand,
    verify_walras_law,
    market_clearing_check,
)


class TestCobbDouglasDemand:
    """Test Cobb-Douglas demand matches the analytic budget-share formula."""

    def test_two_goods_demand(self) -> None:
        """x_j = (alpha_j / sum_alpha) * I / p_j."""
        alphas = [0.4, 0.6]
        u = CobbDouglas(alphas)
        prices = np.array([2.0, 3.0])
        income = 100.0
        x = u.demand(prices, income)
        expected = np.array([0.4 / 1.0 * 100.0 / 2.0, 0.6 / 1.0 * 100.0 / 3.0])
        np.testing.assert_allclose(x, expected, atol=1e-10)

    def test_three_goods_demand(self) -> None:
        alphas = [1.0, 2.0, 3.0]
        u = CobbDouglas(alphas)
        prices = np.array([1.0, 2.0, 3.0])
        income = 60.0
        total_alpha = 6.0
        expected = np.array([
            1.0 / total_alpha * income / 1.0,
            2.0 / total_alpha * income / 2.0,
            3.0 / total_alpha * income / 3.0,
        ])
        x = u.demand(prices, income)
        np.testing.assert_allclose(x, expected, atol=1e-10)

    def test_budget_exhaustion(self) -> None:
        """Agent spends all income."""
        u = CobbDouglas([0.3, 0.7])
        prices = np.array([5.0, 2.0])
        income = 50.0
        x = u.demand(prices, income)
        spent = np.dot(prices, x)
        assert abs(spent - income) < 1e-10


class TestCESDemand:
    """Test CES demand formula."""

    def test_ces_demand_basic(self) -> None:
        alphas = [0.5, 0.5]
        rho = 0.5
        u = CES(alphas, rho)
        prices = np.array([1.0, 1.0])
        income = 10.0
        x = u.demand(prices, income)
        # With equal weights and equal prices, demand is equal
        np.testing.assert_allclose(x[0], x[1], atol=1e-10)
        # Budget exhaustion
        assert abs(np.dot(prices, x) - income) < 1e-10

    def test_ces_budget_exhaustion(self) -> None:
        u = CES([0.3, 0.7], rho=-0.5)
        prices = np.array([3.0, 2.0])
        income = 30.0
        x = u.demand(prices, income)
        assert abs(np.dot(prices, x) - income) < 1e-10


class TestLeontiefDemand:
    """Test Leontief demand."""

    def test_leontief_proportions(self) -> None:
        alphas = [1.0, 2.0]
        u = Leontief(alphas)
        prices = np.array([1.0, 1.0])
        income = 3.0
        x = u.demand(prices, income)
        # x = alpha * I / (alpha . p) = [1, 2] * 3 / (1 + 2) = [1, 2]
        np.testing.assert_allclose(x, [1.0, 2.0], atol=1e-10)

    def test_leontief_budget_exhaustion(self) -> None:
        u = Leontief([2.0, 3.0])
        prices = np.array([5.0, 1.0])
        income = 26.0
        x = u.demand(prices, income)
        assert abs(np.dot(prices, x) - income) < 1e-10


class TestQuasiLinearDemand:
    """Test quasi-linear demand."""

    def test_interior_solution(self) -> None:
        alphas = [2.0]  # u = x0 + 2*ln(x1)
        u = QuasiLinear(alphas)
        prices = np.array([1.0, 1.0])
        income = 10.0
        x = u.demand(prices, income)
        # x1 = alpha * p0 / p1 = 2, x0 = (I - 2) / p0 = 8
        np.testing.assert_allclose(x, [8.0, 2.0], atol=1e-10)


class TestWalrasLaw:
    """Test that Walras' Law p.z(p) = 0 holds for all price vectors."""

    def _make_economy(self) -> ExchangeEconomy:
        a1 = Agent(endowment=np.array([8.0, 2.0]), utility=CobbDouglas([0.4, 0.6]))
        a2 = Agent(endowment=np.array([2.0, 8.0]), utility=CobbDouglas([0.6, 0.4]))
        return ExchangeEconomy(agents=[a1, a2])

    def test_walras_law_uniform_prices(self) -> None:
        economy = self._make_economy()
        p = np.array([0.5, 0.5])
        assert abs(economy.walras_law_check(p)) < 1e-10

    def test_walras_law_random_prices(self) -> None:
        economy = self._make_economy()
        rng = np.random.RandomState(42)
        for _ in range(100):
            p = rng.dirichlet(np.ones(2))
            pz = economy.walras_law_check(p)
            assert abs(pz) < 1e-10, f"Walras' Law violated at p={p}: p.z={pz}"

    def test_walras_law_three_goods(self) -> None:
        a1 = Agent(endowment=np.array([5.0, 3.0, 2.0]), utility=CobbDouglas([0.2, 0.5, 0.3]))
        a2 = Agent(endowment=np.array([2.0, 4.0, 6.0]), utility=CobbDouglas([0.4, 0.3, 0.3]))
        a3 = Agent(endowment=np.array([3.0, 3.0, 2.0]), utility=CobbDouglas([0.3, 0.3, 0.4]))
        economy = ExchangeEconomy(agents=[a1, a2, a3])
        rng = np.random.RandomState(99)
        for _ in range(50):
            p = rng.dirichlet(np.ones(3))
            assert abs(economy.walras_law_check(p)) < 1e-10


class TestMarketClearing:
    """Test market clearing at equilibrium prices."""

    def test_two_agent_cobb_douglas_analytic(self) -> None:
        """2-agent Cobb-Douglas: analytic CE prices and market clearing.

        For agents with u_A = x1^a * x2^(1-a) and u_B = x1^b * x2^(1-b):
        With omega_A = (w_A1, w_A2) and omega_B = (w_B1, w_B2),
        the CE price ratio p1/p2 can be computed analytically.
        """
        a = 0.4
        b = 0.6
        w_a = np.array([8.0, 2.0])
        w_b = np.array([2.0, 8.0])

        agent_a = Agent(endowment=w_a, utility=CobbDouglas([a, 1 - a]))
        agent_b = Agent(endowment=w_b, utility=CobbDouglas([b, 1 - b]))
        economy = ExchangeEconomy(agents=[agent_a, agent_b])

        # Find CE numerically
        from src.core.tatonnement import tatonnement
        result = tatonnement(economy, tol=1e-12)

        clears, z = market_clearing_check(economy, result.prices, tol=1e-6)
        assert clears, f"Markets do not clear: z = {z}"
        assert np.max(np.abs(z)) < 1e-8

    def test_verify_walras_law_function(self) -> None:
        a1 = Agent(endowment=np.array([5.0, 5.0]), utility=CobbDouglas([0.5, 0.5]))
        a2 = Agent(endowment=np.array([5.0, 5.0]), utility=CobbDouglas([0.5, 0.5]))
        economy = ExchangeEconomy(agents=[a1, a2])
        assert verify_walras_law(economy, np.array([0.5, 0.5]))
