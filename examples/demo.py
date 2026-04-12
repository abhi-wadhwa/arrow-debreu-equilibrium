"""Demonstration of the Arrow-Debreu General Equilibrium Engine.

This script walks through a complete analysis of a 2-agent, 2-good
exchange economy with Cobb-Douglas utilities, illustrating:
  1. Setting up agents with endowments and preferences
  2. Computing competitive equilibrium via tatonnement
  3. Verifying Walras' Law and market clearing
  4. Edgeworth box analysis (contract curve, core, CE)
  5. Eisenberg-Gale convex programme
  6. Welfare theorem verification
"""

import numpy as np

from src.core.economy import Agent, ExchangeEconomy
from src.core.utilities import CobbDouglas, CES, Leontief, QuasiLinear
from src.core.demand import (
    compute_demand,
    compute_excess_demand,
    verify_walras_law,
    market_clearing_check,
)
from src.core.tatonnement import tatonnement
from src.core.eisenberg_gale import eisenberg_gale
from src.core.edgeworth import EdgeworthBox
from src.core.welfare import (
    verify_first_welfare_theorem,
    verify_second_welfare_theorem,
    is_pareto_optimal,
)


def main() -> None:
    print("=" * 70)
    print("  Arrow-Debreu General Equilibrium Engine -- Full Demo")
    print("=" * 70)

    # ─────────────────────────────────────────────
    # 1. Define the economy
    # ─────────────────────────────────────────────
    print("\n[1] ECONOMY SETUP")
    print("-" * 40)

    agent_a = Agent(
        endowment=np.array([8.0, 2.0]),
        utility=CobbDouglas([0.4, 0.6]),
        name="Alice",
    )
    agent_b = Agent(
        endowment=np.array([2.0, 8.0]),
        utility=CobbDouglas([0.6, 0.4]),
        name="Bob",
    )
    economy = ExchangeEconomy(agents=[agent_a, agent_b])

    print(f"  Alice: omega = {agent_a.endowment}, u = x1^0.4 * x2^0.6")
    print(f"  Bob:   omega = {agent_b.endowment}, u = x1^0.6 * x2^0.4")
    print(f"  Aggregate endowment: {economy.aggregate_endowment}")

    # ─────────────────────────────────────────────
    # 2. Analytic competitive equilibrium
    # ─────────────────────────────────────────────
    print("\n[2] ANALYTIC COMPETITIVE EQUILIBRIUM")
    print("-" * 40)

    a, b = 0.4, 0.6
    w_a, w_b = agent_a.endowment, agent_b.endowment

    # Analytic price ratio: p1/p2 = (a*w2_A + b*w2_B) / ((1-a)*w1_A + (1-b)*w1_B)
    p_ratio = (a * w_a[1] + b * w_b[1]) / ((1 - a) * w_a[0] + (1 - b) * w_b[0])
    p_analytic = np.array([p_ratio, 1.0])
    p_analytic = p_analytic / p_analytic.sum()

    x_a_analytic = agent_a.demand(p_analytic)
    x_b_analytic = agent_b.demand(p_analytic)

    print(f"  Price ratio p1/p2 = {p_ratio:.6f}")
    print(f"  Normalised prices = {p_analytic}")
    print(f"  Alice demand: {x_a_analytic}")
    print(f"  Bob demand:   {x_b_analytic}")
    print(f"  Market clearing: {x_a_analytic + x_b_analytic} = {economy.aggregate_endowment}")

    # ─────────────────────────────────────────────
    # 3. Tatonnement (discrete)
    # ─────────────────────────────────────────────
    print("\n[3] TATONNEMENT (DISCRETE)")
    print("-" * 40)

    result_disc = tatonnement(economy, method="discrete", tol=1e-10)
    print(f"  Converged: {result_disc.converged}")
    print(f"  Iterations: {result_disc.iterations}")
    print(f"  Prices: {result_disc.prices}")
    print(f"  Max |z_j|: {np.max(np.abs(result_disc.excess_demand)):.2e}")

    # Verify match with analytic
    price_error = np.max(np.abs(result_disc.prices - p_analytic))
    print(f"  Price error vs. analytic: {price_error:.2e}")

    # ─────────────────────────────────────────────
    # 4. Tatonnement (ODE)
    # ─────────────────────────────────────────────
    print("\n[4] TATONNEMENT (ODE)")
    print("-" * 40)

    result_ode = tatonnement(economy, method="ode", t_span=(0, 100))
    print(f"  Prices: {result_ode.prices}")
    print(f"  Max |z_j|: {np.max(np.abs(result_ode.excess_demand)):.2e}")

    # ─────────────────────────────────────────────
    # 5. Walras' Law verification
    # ─────────────────────────────────────────────
    print("\n[5] WALRAS' LAW VERIFICATION")
    print("-" * 40)

    rng = np.random.RandomState(0)
    max_violation = 0.0
    for _ in range(1000):
        p = rng.dirichlet(np.ones(2))
        pz = abs(economy.walras_law_check(p))
        max_violation = max(max_violation, pz)
    print(f"  Max |p.z(p)| over 1000 random prices: {max_violation:.2e}")
    print(f"  Walras' Law holds: {max_violation < 1e-10}")

    # ─────────────────────────────────────────────
    # 6. Market clearing check
    # ─────────────────────────────────────────────
    print("\n[6] MARKET CLEARING")
    print("-" * 40)

    clears, z = market_clearing_check(economy, result_disc.prices, tol=1e-8)
    print(f"  Markets clear: {clears}")
    print(f"  Excess demand: {z}")

    # ─────────────────────────────────────────────
    # 7. Edgeworth box analysis
    # ─────────────────────────────────────────────
    print("\n[7] EDGEWORTH BOX ANALYSIS")
    print("-" * 40)

    box = EdgeworthBox(agent_a, agent_b)
    ew = box.analyse(n_points=500)

    print(f"  Contract curve: {len(ew.contract_curve)} points")
    print(f"  Core: {len(ew.core)} points")
    print(f"  CE allocation (Alice): {ew.ce_allocation}")
    print(f"  CE prices: {ew.ce_prices}")

    # ─────────────────────────────────────────────
    # 8. Eisenberg-Gale convex programme
    # ─────────────────────────────────────────────
    print("\n[8] EISENBERG-GALE CONVEX PROGRAMME")
    print("-" * 40)

    eg = eisenberg_gale(economy)
    print(f"  Converged: {eg.converged}")
    print(f"  Prices: {eg.prices}")
    print(f"  Alice allocation: {eg.allocations[0].round(6)}")
    print(f"  Bob allocation: {eg.allocations[1].round(6)}")
    print(f"  Objective (sum log u): {eg.objective:.6f}")

    # ─────────────────────────────────────────────
    # 9. Welfare theorem verification
    # ─────────────────────────────────────────────
    print("\n[9] WELFARE THEOREMS")
    print("-" * 40)

    allocs = economy.allocations(result_disc.prices)

    fwt = verify_first_welfare_theorem(economy, result_disc.prices)
    print(f"  First Welfare Theorem holds: {fwt.holds}")

    swt = verify_second_welfare_theorem(economy, allocs)
    print(f"  Second Welfare Theorem holds: {swt.holds}")
    if swt.supporting_prices is not None:
        print(f"  Supporting prices: {swt.supporting_prices}")

    po = is_pareto_optimal(economy, allocs)
    print(f"  CE allocation is Pareto-optimal: {po}")

    # ─────────────────────────────────────────────
    # 10. Gains from trade
    # ─────────────────────────────────────────────
    print("\n[10] GAINS FROM TRADE")
    print("-" * 40)

    u_endow_a = agent_a.utility(agent_a.endowment)
    u_endow_b = agent_b.utility(agent_b.endowment)
    u_ce_a = agent_a.utility(allocs[0])
    u_ce_b = agent_b.utility(allocs[1])

    print(f"  Alice: u(endowment) = {u_endow_a:.6f}, u(CE) = {u_ce_a:.6f}, "
          f"gain = {(u_ce_a / u_endow_a - 1) * 100:.2f}%")
    print(f"  Bob:   u(endowment) = {u_endow_b:.6f}, u(CE) = {u_ce_b:.6f}, "
          f"gain = {(u_ce_b / u_endow_b - 1) * 100:.2f}%")

    # ─────────────────────────────────────────────
    # 11. CES economy example
    # ─────────────────────────────────────────────
    print("\n[11] CES ECONOMY (rho = -1, Leontief-like)")
    print("-" * 40)

    agent_c = Agent(
        endowment=np.array([7.0, 3.0]),
        utility=CES([0.5, 0.5], rho=-1.0),
        name="Carol",
    )
    agent_d = Agent(
        endowment=np.array([3.0, 7.0]),
        utility=CES([0.5, 0.5], rho=-1.0),
        name="Dave",
    )
    ces_economy = ExchangeEconomy(agents=[agent_c, agent_d])
    ces_result = tatonnement(ces_economy, tol=1e-8)
    print(f"  Converged: {ces_result.converged}")
    print(f"  Prices: {ces_result.prices}")
    print(f"  Max |z_j|: {np.max(np.abs(ces_result.excess_demand)):.2e}")

    print("\n" + "=" * 70)
    print("  Demo complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
