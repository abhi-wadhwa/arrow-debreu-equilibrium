"""Command-line interface for the Arrow-Debreu equilibrium engine.

Usage:
    python -m src.cli demo          Run the 2-agent Cobb-Douglas demo
    python -m src.cli tatonnement   Run tatonnement on a random economy
    python -m src.cli edgeworth     Compute Edgeworth box analysis
    python -m src.cli app           Launch the Streamlit dashboard
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from src.core.economy import Agent, ExchangeEconomy
from src.core.utilities import CobbDouglas, CES, Leontief
from src.core.demand import compute_excess_demand, verify_walras_law, market_clearing_check
from src.core.tatonnement import tatonnement
from src.core.eisenberg_gale import eisenberg_gale
from src.core.edgeworth import EdgeworthBox
from src.core.welfare import verify_first_welfare_theorem, verify_second_welfare_theorem


def demo() -> None:
    """Run the classic 2-agent, 2-good Cobb-Douglas example."""
    print("=" * 60)
    print("Arrow-Debreu Exchange Economy: 2-Agent Cobb-Douglas Demo")
    print("=" * 60)

    agent_a = Agent(
        endowment=np.array([8.0, 2.0]),
        utility=CobbDouglas([0.4, 0.6]),
        name="Agent A",
    )
    agent_b = Agent(
        endowment=np.array([2.0, 8.0]),
        utility=CobbDouglas([0.6, 0.4]),
        name="Agent B",
    )
    economy = ExchangeEconomy(agents=[agent_a, agent_b])

    print(f"\nAgent A: omega = {agent_a.endowment}, u(x) = x1^0.4 * x2^0.6")
    print(f"Agent B: omega = {agent_b.endowment}, u(x) = x1^0.6 * x2^0.4")
    print(f"Aggregate endowment: {economy.aggregate_endowment}")

    # Tatonnement
    print("\n--- Tatonnement ---")
    result = tatonnement(economy, tol=1e-10)
    print(f"Converged: {result.converged}")
    print(f"Iterations: {result.iterations}")
    print(f"Equilibrium prices: {result.prices}")
    print(f"Excess demand: {result.excess_demand}")
    print(f"Max |z_j|: {np.max(np.abs(result.excess_demand)):.2e}")

    # Allocations
    allocs = economy.allocations(result.prices)
    print(f"\nAgent A demand: {allocs[0]}")
    print(f"Agent B demand: {allocs[1]}")

    # Walras' Law
    print(f"\nWalras' Law check: p.z(p) = {economy.walras_law_check(result.prices):.2e}")

    # Welfare
    print("\n--- Welfare Theorems ---")
    fwt = verify_first_welfare_theorem(economy, result.prices)
    print(f"First Welfare Theorem: {'HOLDS' if fwt.holds else 'VIOLATED'}")
    swt = verify_second_welfare_theorem(economy, allocs)
    print(f"Second Welfare Theorem: {'HOLDS' if swt.holds else 'VIOLATED'}")

    # Eisenberg-Gale
    print("\n--- Eisenberg-Gale Convex Programme ---")
    eg = eisenberg_gale(economy)
    print(f"EG converged: {eg.converged}")
    print(f"EG prices: {eg.prices}")
    print(f"EG allocations: {[a.round(4) for a in eg.allocations]}")

    # Edgeworth box
    print("\n--- Edgeworth Box ---")
    box = EdgeworthBox(agent_a, agent_b)
    ew_result = box.analyse()
    print(f"CE allocation (Agent A): {ew_result.ce_allocation}")
    print(f"CE prices: {ew_result.ce_prices}")
    print(f"Contract curve: {len(ew_result.contract_curve)} points")
    print(f"Core: {len(ew_result.core)} points")


def run_tatonnement() -> None:
    """Run tatonnement on a random multi-agent economy."""
    print("=" * 60)
    print("Tatonnement on Random 4-Agent, 3-Good Economy")
    print("=" * 60)

    np.random.seed(123)
    agents = []
    for i in range(4):
        alphas = np.random.dirichlet(np.ones(3))
        endow = np.random.exponential(5.0, size=3)
        agents.append(Agent(
            endowment=endow,
            utility=CobbDouglas(alphas),
            name=f"Agent {i+1}",
        ))
        print(f"Agent {i+1}: alpha={alphas.round(3)}, omega={endow.round(3)}")

    economy = ExchangeEconomy(agents=agents)
    print(f"\nAggregate endowment: {economy.aggregate_endowment.round(3)}")

    for method in ["discrete", "ode"]:
        print(f"\n--- Method: {method} ---")
        result = tatonnement(economy, method=method, tol=1e-8)
        print(f"Converged: {result.converged}")
        print(f"Prices: {result.prices.round(6)}")
        print(f"Max |z_j|: {np.max(np.abs(result.excess_demand)):.2e}")

        clears, z = market_clearing_check(economy, result.prices, tol=1e-6)
        print(f"Markets clear: {clears}")
        print(f"Walras' Law: {verify_walras_law(economy, result.prices)}")


def run_edgeworth() -> None:
    """Compute and display Edgeworth box analysis."""
    print("=" * 60)
    print("Edgeworth Box Analysis")
    print("=" * 60)

    agent_a = Agent(
        endowment=np.array([6.0, 4.0]),
        utility=CobbDouglas([0.5, 0.5]),
        name="Agent A",
    )
    agent_b = Agent(
        endowment=np.array([4.0, 6.0]),
        utility=CobbDouglas([0.5, 0.5]),
        name="Agent B",
    )

    box = EdgeworthBox(agent_a, agent_b)
    result = box.analyse()

    print(f"Total endowment: {result.total_endowment}")
    print(f"CE allocation (Agent A): {result.ce_allocation}")
    print(f"CE allocation (Agent B): {result.total_endowment - result.ce_allocation}")
    print(f"CE prices: {result.ce_prices}")
    print(f"Contract curve points: {len(result.contract_curve)}")
    print(f"Core points: {len(result.core)}")


def launch_app() -> None:
    """Launch the Streamlit dashboard."""
    import subprocess
    import os

    app_path = os.path.join(os.path.dirname(__file__), "viz", "app.py")
    subprocess.run(["streamlit", "run", app_path])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Arrow-Debreu General Equilibrium Engine CLI"
    )
    parser.add_argument(
        "command",
        choices=["demo", "tatonnement", "edgeworth", "app"],
        help="Command to run",
    )
    args = parser.parse_args()

    if args.command == "demo":
        demo()
    elif args.command == "tatonnement":
        run_tatonnement()
    elif args.command == "edgeworth":
        run_edgeworth()
    elif args.command == "app":
        launch_app()


if __name__ == "__main__":
    main()
