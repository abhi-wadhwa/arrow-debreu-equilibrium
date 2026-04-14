"""Tests for welfare theorem verification."""

import numpy as np

from src.core.economy import Agent, ExchangeEconomy
from src.core.tatonnement import tatonnement
from src.core.utilities import CobbDouglas
from src.core.welfare import (
    is_pareto_optimal,
    verify_first_welfare_theorem,
    verify_second_welfare_theorem,
)


class TestFirstWelfareTheorem:
    """Test First Welfare Theorem: CE allocations are Pareto-optimal."""

    def _make_economy(self) -> ExchangeEconomy:
        a1 = Agent(endowment=np.array([8.0, 2.0]), utility=CobbDouglas([0.4, 0.6]))
        a2 = Agent(endowment=np.array([2.0, 8.0]), utility=CobbDouglas([0.6, 0.4]))
        return ExchangeEconomy(agents=[a1, a2])

    def test_fwt_holds(self) -> None:
        economy = self._make_economy()
        result = tatonnement(economy, tol=1e-10)
        fwt = verify_first_welfare_theorem(economy, result.prices)
        assert fwt.holds, fwt.details
        assert fwt.theorem == "first"

    def test_fwt_symmetric_economy(self) -> None:
        a1 = Agent(endowment=np.array([5.0, 5.0]), utility=CobbDouglas([0.5, 0.5]))
        a2 = Agent(endowment=np.array([5.0, 5.0]), utility=CobbDouglas([0.5, 0.5]))
        economy = ExchangeEconomy(agents=[a1, a2])
        result = tatonnement(economy, tol=1e-10)
        fwt = verify_first_welfare_theorem(economy, result.prices)
        assert fwt.holds

    def test_fwt_three_agents(self) -> None:
        np.random.seed(42)
        agents = []
        for _ in range(3):
            alphas = np.random.dirichlet(np.ones(2))
            endow = np.random.exponential(5.0, size=2)
            agents.append(Agent(endowment=endow, utility=CobbDouglas(alphas)))
        economy = ExchangeEconomy(agents=agents)
        result = tatonnement(economy, tol=1e-10)
        fwt = verify_first_welfare_theorem(economy, result.prices)
        assert fwt.holds


class TestSecondWelfareTheorem:
    """Test Second Welfare Theorem: Pareto-optimal allocations can be
    supported as CE with transfers."""

    def test_swt_at_ce(self) -> None:
        """The CE allocation itself should be supportable."""
        a1 = Agent(endowment=np.array([8.0, 2.0]), utility=CobbDouglas([0.4, 0.6]))
        a2 = Agent(endowment=np.array([2.0, 8.0]), utility=CobbDouglas([0.6, 0.4]))
        economy = ExchangeEconomy(agents=[a1, a2])
        result = tatonnement(economy, tol=1e-10)
        allocs = economy.allocations(result.prices)
        swt = verify_second_welfare_theorem(economy, allocs)
        assert swt.holds, swt.details
        assert swt.theorem == "second"
        assert swt.supporting_prices is not None

    def test_swt_equal_split(self) -> None:
        """Equal split of identical goods with identical agents is Pareto-optimal
        and should be supportable."""
        a1 = Agent(endowment=np.array([8.0, 2.0]), utility=CobbDouglas([0.5, 0.5]))
        a2 = Agent(endowment=np.array([2.0, 8.0]), utility=CobbDouglas([0.5, 0.5]))
        economy = ExchangeEconomy(agents=[a1, a2])
        # Equal split
        allocs = [np.array([5.0, 5.0]), np.array([5.0, 5.0])]
        swt = verify_second_welfare_theorem(economy, allocs)
        assert swt.holds


class TestParetoOptimality:
    """Test Pareto optimality check."""

    def test_ce_is_pareto_optimal(self) -> None:
        a1 = Agent(endowment=np.array([8.0, 2.0]), utility=CobbDouglas([0.4, 0.6]))
        a2 = Agent(endowment=np.array([2.0, 8.0]), utility=CobbDouglas([0.6, 0.4]))
        economy = ExchangeEconomy(agents=[a1, a2])
        result = tatonnement(economy, tol=1e-10)
        allocs = economy.allocations(result.prices)
        assert is_pareto_optimal(economy, allocs, n_trials=500)

    def test_endowment_may_not_be_pareto_optimal(self) -> None:
        """The endowment is generally not Pareto-optimal when agents
        have different preferences."""
        a1 = Agent(endowment=np.array([9.0, 1.0]), utility=CobbDouglas([0.2, 0.8]))
        a2 = Agent(endowment=np.array([1.0, 9.0]), utility=CobbDouglas([0.8, 0.2]))
        economy = ExchangeEconomy(agents=[a1, a2])
        allocs = [a1.endowment, a2.endowment]
        # With such extreme preferences and endowments, the endowment
        # allocation is likely not Pareto-optimal
        result = is_pareto_optimal(economy, allocs, n_trials=1000)
        # This should fail since agent 1 wants good 2 and agent 2 wants good 1
        assert not result


class TestWelfareUtilities:
    """Test utility values in welfare results."""

    def test_ce_utilities_improve_over_endowment(self) -> None:
        """At CE, both agents should be at least as well off as at endowment
        (voluntary trade is beneficial)."""
        a1 = Agent(endowment=np.array([8.0, 2.0]), utility=CobbDouglas([0.4, 0.6]))
        a2 = Agent(endowment=np.array([2.0, 8.0]), utility=CobbDouglas([0.6, 0.4]))
        economy = ExchangeEconomy(agents=[a1, a2])
        result = tatonnement(economy, tol=1e-10)
        allocs = economy.allocations(result.prices)

        u_endow = [a1.utility(a1.endowment), a2.utility(a2.endowment)]
        u_ce = [a1.utility(allocs[0]), a2.utility(allocs[1])]

        assert u_ce[0] >= u_endow[0] - 1e-8
        assert u_ce[1] >= u_endow[1] - 1e-8
