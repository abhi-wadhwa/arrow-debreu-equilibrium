"""Tests for Edgeworth box analysis."""

import numpy as np
import pytest

from src.core.economy import Agent
from src.core.edgeworth import EdgeworthBox
from src.core.utilities import CobbDouglas


class TestEdgeworthBox:
    """Test Edgeworth box construction and analysis."""

    def _make_box(self) -> EdgeworthBox:
        a = Agent(endowment=np.array([8.0, 2.0]), utility=CobbDouglas([0.4, 0.6]))
        b = Agent(endowment=np.array([2.0, 8.0]), utility=CobbDouglas([0.6, 0.4]))
        return EdgeworthBox(a, b)

    def test_total_endowment(self) -> None:
        box = self._make_box()
        np.testing.assert_allclose(box.total, [10.0, 10.0])

    def test_contract_curve_non_empty(self) -> None:
        box = self._make_box()
        cc = box.contract_curve()
        assert len(cc) > 10
        # All points inside the box
        assert np.all(cc[:, 0] >= 0) and np.all(cc[:, 0] <= 10.0)
        assert np.all(cc[:, 1] >= 0) and np.all(cc[:, 1] <= 10.0)

    def test_contract_curve_mrs_equal(self) -> None:
        """On the contract curve, MRS_1 = MRS_2."""
        box = self._make_box()
        cc = box.contract_curve(n_points=100)
        for pt in cc[10:-10]:  # skip boundary points
            bundle1 = pt
            bundle2 = box.total - pt
            mrs1 = box.mrs(box.agent1, bundle1)
            mrs2 = box.mrs(box.agent2, bundle2)
            assert abs(mrs1 - mrs2) < 1e-4, (
                f"MRS mismatch at {pt}: MRS1={mrs1:.6f}, MRS2={mrs2:.6f}"
            )

    def test_core_subset_of_contract_curve(self) -> None:
        box = self._make_box()
        cc = box.contract_curve()
        core = box.core(cc)
        assert len(core) > 0
        assert len(core) <= len(cc)

    def test_core_individual_rationality(self) -> None:
        """All core allocations are individually rational."""
        box = self._make_box()
        core = box.core()
        u1_endow = box.agent1.utility(box.agent1.endowment)
        u2_endow = box.agent2.utility(box.agent2.endowment)
        for pt in core:
            bundle1 = pt
            bundle2 = box.total - pt
            assert box.agent1.utility(bundle1) >= u1_endow - 1e-8
            assert box.agent2.utility(bundle2) >= u2_endow - 1e-8

    def test_ce_on_contract_curve(self) -> None:
        """CE allocation is close to the contract curve."""
        box = self._make_box()
        cc = box.contract_curve(n_points=500)
        ce_alloc, ce_prices = box.competitive_equilibrium()
        # Find closest point on contract curve
        dists = np.linalg.norm(cc - ce_alloc, axis=1)
        min_dist = np.min(dists)
        assert min_dist < 0.1, f"CE not near contract curve: min dist = {min_dist}"

    def test_ce_budget_constraint(self) -> None:
        """At CE, both agents satisfy their budget constraints."""
        box = self._make_box()
        ce_alloc, ce_prices = box.competitive_equilibrium()
        # Agent 1
        income1 = np.dot(ce_prices, box.agent1.endowment)
        spending1 = np.dot(ce_prices, ce_alloc)
        assert abs(spending1 - income1) < 1e-6

        # Agent 2
        alloc2 = box.total - ce_alloc
        income2 = np.dot(ce_prices, box.agent2.endowment)
        spending2 = np.dot(ce_prices, alloc2)
        assert abs(spending2 - income2) < 1e-6

    def test_ce_market_clearing(self) -> None:
        """At CE, total demand equals total supply."""
        box = self._make_box()
        ce_alloc, ce_prices = box.competitive_equilibrium()
        alloc2 = box.total - ce_alloc
        total_demand = ce_alloc + alloc2
        np.testing.assert_allclose(total_demand, box.total, atol=1e-8)

    def test_symmetric_economy_ce(self) -> None:
        """Symmetric economy should have CE at the centre of the box."""
        a = Agent(endowment=np.array([5.0, 5.0]), utility=CobbDouglas([0.5, 0.5]))
        b = Agent(endowment=np.array([5.0, 5.0]), utility=CobbDouglas([0.5, 0.5]))
        box = EdgeworthBox(a, b)
        ce_alloc, ce_prices = box.competitive_equilibrium()
        np.testing.assert_allclose(ce_alloc, [5.0, 5.0], atol=1e-4)
        np.testing.assert_allclose(ce_prices[0], ce_prices[1], atol=1e-4)


class TestEdgeworthBoxValidation:
    """Test input validation."""

    def test_requires_two_goods(self) -> None:
        a = Agent(endowment=np.array([1.0, 2.0, 3.0]), utility=CobbDouglas([0.3, 0.3, 0.4]))
        b = Agent(endowment=np.array([3.0, 2.0, 1.0]), utility=CobbDouglas([0.4, 0.3, 0.3]))
        with pytest.raises(ValueError, match="exactly 2 goods"):
            EdgeworthBox(a, b)


class TestEdgeworthAnalyse:
    """Test the full analyse() method."""

    def test_analyse_returns_all_fields(self) -> None:
        a = Agent(endowment=np.array([6.0, 4.0]), utility=CobbDouglas([0.3, 0.7]))
        b = Agent(endowment=np.array([4.0, 6.0]), utility=CobbDouglas([0.7, 0.3]))
        box = EdgeworthBox(a, b)
        result = box.analyse()
        assert result.contract_curve.shape[1] == 2
        assert result.core.shape[1] == 2
        assert len(result.ce_allocation) == 2
        assert len(result.ce_prices) == 2
        assert len(result.total_endowment) == 2

    def test_budget_line(self) -> None:
        a = Agent(endowment=np.array([8.0, 2.0]), utility=CobbDouglas([0.4, 0.6]))
        b = Agent(endowment=np.array([2.0, 8.0]), utility=CobbDouglas([0.6, 0.4]))
        box = EdgeworthBox(a, b)
        _, prices = box.competitive_equilibrium()
        bl = box.budget_line(prices)
        assert len(bl) > 0
        # All points on the budget line should satisfy p.x = p.omega
        income = np.dot(prices, a.endowment)
        for pt in bl:
            assert abs(np.dot(prices, pt) - income) < 1e-6


class TestIndifferenceCurves:
    """Test indifference curve computation."""

    def test_agent1_ic(self) -> None:
        a = Agent(endowment=np.array([5.0, 5.0]), utility=CobbDouglas([0.5, 0.5]))
        b = Agent(endowment=np.array([5.0, 5.0]), utility=CobbDouglas([0.5, 0.5]))
        box = EdgeworthBox(a, b)
        u_level = a.utility(np.array([3.0, 3.0]))
        ic = box.indifference_curve(0, u_level)
        assert len(ic) > 0
        # All points should have approximately the same utility
        for pt in ic:
            assert abs(a.utility(pt) - u_level) < 1e-4

    def test_agent2_ic(self) -> None:
        a = Agent(endowment=np.array([5.0, 5.0]), utility=CobbDouglas([0.5, 0.5]))
        b = Agent(endowment=np.array([5.0, 5.0]), utility=CobbDouglas([0.5, 0.5]))
        box = EdgeworthBox(a, b)
        u_level = b.utility(np.array([4.0, 4.0]))
        ic = box.indifference_curve(1, u_level)
        assert len(ic) > 0
