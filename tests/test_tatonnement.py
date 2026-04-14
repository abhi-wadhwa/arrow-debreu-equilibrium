"""Tests for tatonnement price adjustment dynamics."""

import numpy as np
import pytest

from src.core.economy import Agent, ExchangeEconomy
from src.core.tatonnement import tatonnement
from src.core.utilities import CobbDouglas


class TestDiscreteTatonnement:
    """Test discrete tatonnement convergence."""

    def _two_agent_economy(self) -> ExchangeEconomy:
        a1 = Agent(endowment=np.array([8.0, 2.0]), utility=CobbDouglas([0.4, 0.6]))
        a2 = Agent(endowment=np.array([2.0, 8.0]), utility=CobbDouglas([0.6, 0.4]))
        return ExchangeEconomy(agents=[a1, a2])

    def test_convergence(self) -> None:
        economy = self._two_agent_economy()
        result = tatonnement(economy, method="discrete", tol=1e-8)
        assert result.converged
        assert np.max(np.abs(result.excess_demand)) < 1e-8

    def test_prices_on_simplex(self) -> None:
        economy = self._two_agent_economy()
        result = tatonnement(economy, method="discrete", tol=1e-8)
        assert abs(result.prices.sum() - 1.0) < 1e-12
        assert np.all(result.prices > 0)

    def test_price_history_recorded(self) -> None:
        economy = self._two_agent_economy()
        result = tatonnement(economy, method="discrete", tol=1e-8)
        # History includes the initial price + at least one iteration
        assert result.price_history.shape[0] >= 1
        assert result.price_history.shape[1] == 2
        # With asymmetric endowments/preferences, should take > 1 step
        assert result.iterations >= 1

    def test_match_analytic_ce_six_decimals(self) -> None:
        """Match analytic CE to 6 decimal places for 2-agent Cobb-Douglas.

        Analytic solution:
        For u_A = x1^a x2^(1-a), u_B = x1^b x2^(1-b)
        with omega_A = (w1_A, w2_A), omega_B = (w1_B, w2_B):

        Demand for good 1:
          x1_A = a * (p1*w1_A + p2*w2_A) / p1
          x1_B = b * (p1*w1_B + p2*w2_B) / p1

        Market clearing for good 1:
          x1_A + x1_B = w1_A + w1_B = W1

        This gives:
          a*(p1*w1_A + p2*w2_A) + b*(p1*w1_B + p2*w2_B) = p1 * W1

        Solving for p1/p2:
          p1/p2 = (a*w2_A + b*w2_B) / ((1-a)*w1_A + (1-b)*w1_B)
        """
        a, b = 0.4, 0.6
        w_a = np.array([8.0, 2.0])
        w_b = np.array([2.0, 8.0])

        # Analytic price ratio
        p_ratio_analytic = (a * w_a[1] + b * w_b[1]) / ((1 - a) * w_a[0] + (1 - b) * w_b[0])
        p_analytic = np.array([p_ratio_analytic, 1.0])
        p_analytic = p_analytic / p_analytic.sum()

        # Compute analytic allocations
        agent_a = Agent(endowment=w_a, utility=CobbDouglas([a, 1 - a]))
        agent_b = Agent(endowment=w_b, utility=CobbDouglas([b, 1 - b]))
        x_a_analytic = agent_a.demand(p_analytic)
        x_b_analytic = agent_b.demand(p_analytic)

        # Numerical solution
        economy = ExchangeEconomy(agents=[agent_a, agent_b])
        result = tatonnement(economy, method="discrete", tol=1e-12, max_iter=50000)

        # Match prices to 6 decimal places
        np.testing.assert_allclose(
            result.prices, p_analytic, atol=1e-6,
            err_msg="Tatonnement prices do not match analytic CE to 6 decimals",
        )

        # Match allocations to 6 decimal places
        allocs = economy.allocations(result.prices)
        np.testing.assert_allclose(allocs[0], x_a_analytic, atol=1e-6)
        np.testing.assert_allclose(allocs[1], x_b_analytic, atol=1e-6)

    def test_symmetric_economy(self) -> None:
        """In a symmetric economy, CE prices should be equal."""
        a1 = Agent(endowment=np.array([5.0, 5.0]), utility=CobbDouglas([0.5, 0.5]))
        a2 = Agent(endowment=np.array([5.0, 5.0]), utility=CobbDouglas([0.5, 0.5]))
        economy = ExchangeEconomy(agents=[a1, a2])
        result = tatonnement(economy, tol=1e-10)
        assert result.converged
        np.testing.assert_allclose(result.prices[0], result.prices[1], atol=1e-8)


class TestODETatonnement:
    """Test ODE-based tatonnement."""

    def test_ode_convergence(self) -> None:
        a1 = Agent(endowment=np.array([8.0, 2.0]), utility=CobbDouglas([0.4, 0.6]))
        a2 = Agent(endowment=np.array([2.0, 8.0]), utility=CobbDouglas([0.6, 0.4]))
        economy = ExchangeEconomy(agents=[a1, a2])
        result = tatonnement(economy, method="ode", tol=1e-6, t_span=(0, 100))
        assert np.max(np.abs(result.excess_demand)) < 1e-4

    def test_ode_prices_on_simplex(self) -> None:
        a1 = Agent(endowment=np.array([5.0, 5.0]), utility=CobbDouglas([0.3, 0.7]))
        a2 = Agent(endowment=np.array([5.0, 5.0]), utility=CobbDouglas([0.7, 0.3]))
        economy = ExchangeEconomy(agents=[a1, a2])
        result = tatonnement(economy, method="ode", t_span=(0, 80))
        assert abs(result.prices.sum() - 1.0) < 1e-8
        assert np.all(result.prices > 0)


class TestMultiAgentTatonnement:
    """Test tatonnement with more than 2 agents."""

    def test_three_agents_three_goods(self) -> None:
        np.random.seed(42)
        agents = []
        for _ in range(3):
            alphas = np.random.dirichlet(np.ones(3))
            endow = np.random.exponential(5.0, size=3)
            agents.append(Agent(endowment=endow, utility=CobbDouglas(alphas)))
        economy = ExchangeEconomy(agents=agents)
        result = tatonnement(economy, tol=1e-8)
        assert result.converged
        assert np.max(np.abs(result.excess_demand)) < 1e-6

    def test_five_agents_four_goods(self) -> None:
        np.random.seed(99)
        agents = []
        for _ in range(5):
            alphas = np.random.dirichlet(np.ones(4))
            endow = np.random.exponential(3.0, size=4)
            agents.append(Agent(endowment=endow, utility=CobbDouglas(alphas)))
        economy = ExchangeEconomy(agents=agents)
        result = tatonnement(economy, tol=1e-7, max_iter=20000)
        assert result.converged
        assert np.max(np.abs(result.excess_demand)) < 1e-6


class TestInvalidInputs:
    """Test error handling."""

    def test_invalid_method(self) -> None:
        a1 = Agent(endowment=np.array([5.0, 5.0]), utility=CobbDouglas([0.5, 0.5]))
        economy = ExchangeEconomy(agents=[a1])
        with pytest.raises(ValueError, match="Unknown method"):
            tatonnement(economy, method="invalid")
