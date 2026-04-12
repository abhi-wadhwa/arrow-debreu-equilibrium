"""Streamlit interactive dashboard for Arrow-Debreu exchange economies.

Features:
  1. Edgeworth Box Explorer - 2D interactive visualisation with
     indifference curves, contract curve, core, and CE price line.
  2. Tatonnement Animation - prices on the simplex converging to equilibrium.
  3. Multi-good economy - excess demand plots across price space.
  4. Welfare Analysis - utility allocations at CE vs. endowment.
"""

from __future__ import annotations

import sys
import os

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection

# Ensure the project root is on the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.core.economy import Agent, ExchangeEconomy
from src.core.utilities import CobbDouglas, CES, Leontief
from src.core.tatonnement import tatonnement
from src.core.eisenberg_gale import eisenberg_gale
from src.core.edgeworth import EdgeworthBox
from src.core.welfare import verify_first_welfare_theorem, verify_second_welfare_theorem


def main() -> None:
    st.set_page_config(
        page_title="Arrow-Debreu Equilibrium Engine",
        page_icon="<=>",
        layout="wide",
    )

    st.title("Arrow-Debreu General Equilibrium Engine")
    st.markdown(
        "Compute Walrasian equilibria in Arrow-Debreu exchange economies. "
        "Explore the Edgeworth box, watch tatonnement converge, and verify "
        "the fundamental welfare theorems."
    )

    tab1, tab2, tab3, tab4 = st.tabs([
        "Edgeworth Box Explorer",
        "Tatonnement Dynamics",
        "Multi-Good Economy",
        "Welfare Analysis",
    ])

    with tab1:
        _edgeworth_box_tab()
    with tab2:
        _tatonnement_tab()
    with tab3:
        _multi_good_tab()
    with tab4:
        _welfare_tab()


# ──────────────────────────────────────────────────────────────────
#  Tab 1: Edgeworth Box Explorer
# ──────────────────────────────────────────────────────────────────
def _edgeworth_box_tab() -> None:
    st.header("Edgeworth Box Explorer")
    st.markdown(
        "A 2-agent, 2-good economy. Agent A's origin is bottom-left; "
        "Agent B's origin is top-right."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Agent A (Cobb-Douglas)")
        a1_alpha = st.slider("Agent A: alpha_1", 0.1, 2.0, 0.4, 0.05, key="a1a")
        a1_beta = st.slider("Agent A: alpha_2", 0.1, 2.0, 0.6, 0.05, key="a1b")
        omega_a1 = st.slider("Agent A: endowment good 1", 0.1, 10.0, 8.0, 0.1, key="oa1")
        omega_a2 = st.slider("Agent A: endowment good 2", 0.1, 10.0, 2.0, 0.1, key="oa2")

    with col2:
        st.subheader("Agent B (Cobb-Douglas)")
        b1_alpha = st.slider("Agent B: alpha_1", 0.1, 2.0, 0.6, 0.05, key="b1a")
        b1_beta = st.slider("Agent B: alpha_2", 0.1, 2.0, 0.4, 0.05, key="b1b")
        omega_b1 = st.slider("Agent B: endowment good 1", 0.1, 10.0, 2.0, 0.1, key="ob1")
        omega_b2 = st.slider("Agent B: endowment good 2", 0.1, 10.0, 8.0, 0.1, key="ob2")

    agent_a = Agent(
        endowment=np.array([omega_a1, omega_a2]),
        utility=CobbDouglas([a1_alpha, a1_beta]),
        name="Agent A",
    )
    agent_b = Agent(
        endowment=np.array([omega_b1, omega_b2]),
        utility=CobbDouglas([b1_alpha, b1_beta]),
        name="Agent B",
    )

    box = EdgeworthBox(agent_a, agent_b)

    with st.spinner("Computing Edgeworth box analysis..."):
        result = box.analyse(n_points=400)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    total = result.total_endowment

    # Box
    ax.set_xlim(0, total[0])
    ax.set_ylim(0, total[1])
    ax.set_aspect("equal")
    ax.set_xlabel("Good 1 (Agent A)", fontsize=12)
    ax.set_ylabel("Good 2 (Agent A)", fontsize=12)

    # Secondary axes for Agent B
    ax2 = ax.secondary_xaxis("top", functions=(lambda x: total[0] - x, lambda x: total[0] - x))
    ax3 = ax.secondary_yaxis("right", functions=(lambda y: total[1] - y, lambda y: total[1] - y))
    ax2.set_xlabel("Good 1 (Agent B)", fontsize=12)
    ax3.set_ylabel("Good 2 (Agent B)", fontsize=12)

    # Contract curve
    if len(result.contract_curve) > 0:
        ax.plot(
            result.contract_curve[:, 0],
            result.contract_curve[:, 1],
            "g-",
            linewidth=2,
            label="Contract Curve",
            alpha=0.8,
        )

    # Core
    if len(result.core) > 0:
        ax.plot(
            result.core[:, 0],
            result.core[:, 1],
            "g-",
            linewidth=5,
            alpha=0.4,
            label="Core",
        )

    # Budget line
    bl = box.budget_line(result.ce_prices)
    if len(bl) > 0:
        ax.plot(bl[:, 0], bl[:, 1], "b--", linewidth=1.5, label="Budget Line", alpha=0.7)

    # Indifference curves at CE
    ce_u_a = agent_a.utility(result.ce_allocation)
    ce_u_b = agent_b.utility(total - result.ce_allocation)
    ic_a = box.indifference_curve(0, ce_u_a)
    ic_b = box.indifference_curve(1, ce_u_b)
    if len(ic_a) > 0:
        ax.plot(ic_a[:, 0], ic_a[:, 1], "r-", linewidth=1.5, label="IC Agent A (CE)", alpha=0.7)
    if len(ic_b) > 0:
        ax.plot(ic_b[:, 0], ic_b[:, 1], "m-", linewidth=1.5, label="IC Agent B (CE)", alpha=0.7)

    # Indifference curves at endowment
    endow_u_a = agent_a.utility(agent_a.endowment)
    endow_u_b = agent_b.utility(agent_b.endowment)
    ic_a_endow = box.indifference_curve(0, endow_u_a)
    ic_b_endow = box.indifference_curve(1, endow_u_b)
    if len(ic_a_endow) > 0:
        ax.plot(ic_a_endow[:, 0], ic_a_endow[:, 1], "r:", linewidth=1, alpha=0.5, label="IC Agent A (endowment)")
    if len(ic_b_endow) > 0:
        ax.plot(ic_b_endow[:, 0], ic_b_endow[:, 1], "m:", linewidth=1, alpha=0.5, label="IC Agent B (endowment)")

    # Points
    ax.plot(*agent_a.endowment, "ko", markersize=10, label="Endowment", zorder=5)
    ax.plot(*result.ce_allocation, "r*", markersize=15, label="CE Allocation", zorder=5)

    ax.legend(loc="lower left", fontsize=9)
    ax.set_title("Edgeworth Box", fontsize=14)
    ax.grid(True, alpha=0.3)

    st.pyplot(fig)

    # Results summary
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        st.metric("CE Price Ratio (p1/p2)", f"{result.ce_prices[0]/result.ce_prices[1]:.4f}")
        st.metric("Agent A Utility (CE)", f"{ce_u_a:.4f}")
        st.metric("Agent A Utility (endowment)", f"{endow_u_a:.4f}")
    with col_r2:
        st.metric("CE Allocation A", f"({result.ce_allocation[0]:.3f}, {result.ce_allocation[1]:.3f})")
        st.metric("Agent B Utility (CE)", f"{ce_u_b:.4f}")
        st.metric("Agent B Utility (endowment)", f"{endow_u_b:.4f}")


# ──────────────────────────────────────────────────────────────────
#  Tab 2: Tatonnement Dynamics
# ──────────────────────────────────────────────────────────────────
def _tatonnement_tab() -> None:
    st.header("Tatonnement Price Dynamics")
    st.markdown(
        r"Watch the Walrasian auctioneer adjust prices: $dp_j/dt = z_j(p)$."
    )

    col1, col2 = st.columns(2)
    with col1:
        n_agents = st.slider("Number of agents", 2, 5, 3, key="tat_n")
        n_goods = st.slider("Number of goods", 2, 5, 3, key="tat_m")
    with col2:
        method = st.selectbox("Method", ["discrete", "ode"], key="tat_method")
        eta = st.slider("Step size (discrete)", 0.01, 0.5, 0.1, 0.01, key="tat_eta")

    # Generate random economy
    np.random.seed(42)
    agents = []
    for i in range(n_agents):
        alphas = np.random.dirichlet(np.ones(n_goods))
        endowment = np.random.exponential(5.0, size=n_goods)
        agents.append(Agent(
            endowment=endowment,
            utility=CobbDouglas(alphas),
            name=f"Agent {i+1}",
        ))
    economy = ExchangeEconomy(agents=agents)

    if st.button("Run Tatonnement", key="run_tat"):
        with st.spinner("Running tatonnement..."):
            result = tatonnement(
                economy,
                method=method,
                eta=eta,
                max_iter=5000,
                tol=1e-8,
            )

        st.success(f"Converged: {result.converged} | Iterations: {result.iterations}")
        st.write(f"Equilibrium prices: {result.prices}")
        st.write(f"Max |excess demand|: {np.max(np.abs(result.excess_demand)):.2e}")

        # Plot price trajectories
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Price paths
        ax = axes[0]
        for j in range(n_goods):
            ax.plot(result.price_history[:, j], label=f"Good {j+1}")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Price")
        ax.set_title("Price Convergence")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Excess demand convergence
        ax = axes[1]
        ed_norms = []
        for t in range(len(result.price_history)):
            z = economy.excess_demand(result.price_history[t])
            ed_norms.append(np.max(np.abs(z)))
        ax.semilogy(ed_norms)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Max |z_j(p)|")
        ax.set_title("Excess Demand Convergence")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)

        # Simplex plot for 3 goods
        if n_goods == 3:
            _plot_simplex_trajectory(result.price_history)


def _plot_simplex_trajectory(history: np.ndarray) -> None:
    """Plot price trajectory on the 2-simplex (ternary diagram)."""
    st.subheader("Price Trajectory on the Simplex")

    fig, ax = plt.subplots(figsize=(7, 6))

    # Convert barycentric to Cartesian
    def bary_to_cart(p: np.ndarray) -> tuple[float, float]:
        x = 0.5 * (2 * p[1] + p[2])
        y = (np.sqrt(3) / 2) * p[2]
        return x, y

    # Draw simplex triangle
    vertices = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2]])
    triangle = plt.Polygon(vertices, fill=False, edgecolor="black", linewidth=2)
    ax.add_patch(triangle)

    # Plot trajectory
    coords = np.array([bary_to_cart(p) for p in history])
    ax.plot(coords[:, 0], coords[:, 1], "b-", alpha=0.5, linewidth=0.8)
    ax.plot(coords[0, 0], coords[0, 1], "go", markersize=10, label="Start", zorder=5)
    ax.plot(coords[-1, 0], coords[-1, 1], "r*", markersize=15, label="Equilibrium", zorder=5)

    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.0)
    ax.set_aspect("equal")
    ax.legend()
    ax.set_title("Price Simplex (3 goods)")
    ax.axis("off")

    st.pyplot(fig)


# ──────────────────────────────────────────────────────────────────
#  Tab 3: Multi-Good Economy
# ──────────────────────────────────────────────────────────────────
def _multi_good_tab() -> None:
    st.header("Multi-Good Economy: Excess Demand Analysis")
    st.markdown(
        "Explore how excess demand varies with prices in a 2-good economy."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Agent 1")
        mg_a1 = st.slider("alpha_1", 0.1, 2.0, 0.3, 0.05, key="mg_a1")
        mg_b1 = st.slider("alpha_2", 0.1, 2.0, 0.7, 0.05, key="mg_b1")
        mg_e1 = st.slider("endowment (good 1, good 2)", 0.1, 10.0, (6.0, 2.0), 0.1, key="mg_e1")
    with col2:
        st.subheader("Agent 2")
        mg_a2 = st.slider("alpha_1 ", 0.1, 2.0, 0.7, 0.05, key="mg_a2")
        mg_b2 = st.slider("alpha_2 ", 0.1, 2.0, 0.3, 0.05, key="mg_b2")
        mg_e2 = st.slider("endowment (good 1, good 2) ", 0.1, 10.0, (2.0, 6.0), 0.1, key="mg_e2")

    agent1 = Agent(
        endowment=np.array([mg_e1[0], mg_e1[1]]),
        utility=CobbDouglas([mg_a1, mg_b1]),
    )
    agent2 = Agent(
        endowment=np.array([mg_e2[0], mg_e2[1]]),
        utility=CobbDouglas([mg_a2, mg_b2]),
    )
    economy = ExchangeEconomy(agents=[agent1, agent2])

    # Excess demand as function of p1 (with p2 = 1 - p1)
    p1_vals = np.linspace(0.01, 0.99, 500)
    z1_vals = []
    z2_vals = []
    for p1 in p1_vals:
        p = np.array([p1, 1.0 - p1])
        z = economy.excess_demand(p)
        z1_vals.append(z[0])
        z2_vals.append(z[1])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(p1_vals, z1_vals, "r-", linewidth=2, label="z_1(p) (good 1)")
    ax.plot(p1_vals, z2_vals, "b-", linewidth=2, label="z_2(p) (good 2)")
    ax.axhline(y=0, color="k", linewidth=0.8, linestyle="--")
    ax.set_xlabel("p_1 (price of good 1)", fontsize=12)
    ax.set_ylabel("Excess Demand", fontsize=12)
    ax.set_title("Excess Demand Functions", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Mark equilibrium
    z1_arr = np.array(z1_vals)
    sign_changes = np.where(np.diff(np.sign(z1_arr)))[0]
    if len(sign_changes) > 0:
        idx = sign_changes[0]
        p1_eq = p1_vals[idx]
        ax.axvline(x=p1_eq, color="green", linewidth=1.5, linestyle=":", label=f"CE at p1={p1_eq:.3f}")
        ax.legend(fontsize=11)

    st.pyplot(fig)

    # Walras' Law verification
    st.subheader("Walras' Law Verification: p . z(p) = 0")
    walras_vals = []
    for p1 in p1_vals:
        p = np.array([p1, 1.0 - p1])
        walras_vals.append(economy.walras_law_check(p))

    fig2, ax2 = plt.subplots(figsize=(10, 3))
    ax2.plot(p1_vals, walras_vals, "purple", linewidth=1.5)
    ax2.set_xlabel("p_1")
    ax2.set_ylabel("p . z(p)")
    ax2.set_title("Walras' Law: p . z(p) should be identically zero")
    ax2.grid(True, alpha=0.3)
    max_violation = np.max(np.abs(walras_vals))
    ax2.set_ylim(-max(max_violation * 2, 1e-12), max(max_violation * 2, 1e-12))
    st.pyplot(fig2)
    st.metric("Max |p . z(p)|", f"{max_violation:.2e}")


# ──────────────────────────────────────────────────────────────────
#  Tab 4: Welfare Analysis
# ──────────────────────────────────────────────────────────────────
def _welfare_tab() -> None:
    st.header("Welfare Theorem Verification")
    st.markdown(
        "**First Welfare Theorem**: Every CE allocation is Pareto-optimal.\n\n"
        "**Second Welfare Theorem**: Every Pareto-optimal allocation can be "
        "supported as a CE with transfers."
    )

    # Set up economy
    col1, col2 = st.columns(2)
    with col1:
        w_a1 = st.slider("Agent A alpha_1", 0.1, 2.0, 0.5, 0.05, key="w_a1")
        w_a2 = st.slider("Agent A alpha_2", 0.1, 2.0, 0.5, 0.05, key="w_a2")
        w_e_a = st.slider("Agent A endowment", 0.1, 10.0, (7.0, 3.0), 0.1, key="w_ea")
    with col2:
        w_b1 = st.slider("Agent B alpha_1", 0.1, 2.0, 0.3, 0.05, key="w_b1")
        w_b2 = st.slider("Agent B alpha_2", 0.1, 2.0, 0.7, 0.05, key="w_b2")
        w_e_b = st.slider("Agent B endowment", 0.1, 10.0, (3.0, 7.0), 0.1, key="w_eb")

    agent_a = Agent(
        endowment=np.array([w_e_a[0], w_e_a[1]]),
        utility=CobbDouglas([w_a1, w_a2]),
        name="Agent A",
    )
    agent_b = Agent(
        endowment=np.array([w_e_b[0], w_e_b[1]]),
        utility=CobbDouglas([w_b1, w_b2]),
        name="Agent B",
    )
    economy = ExchangeEconomy(agents=[agent_a, agent_b])

    if st.button("Run Welfare Analysis", key="run_welfare"):
        with st.spinner("Computing equilibrium and verifying welfare theorems..."):
            tat_result = tatonnement(economy, tol=1e-10)
            prices = tat_result.prices

            allocs = economy.allocations(prices)
            u_ce = economy.utility_profile(prices)
            u_endow = np.array([
                agent_a.utility(agent_a.endowment),
                agent_b.utility(agent_b.endowment),
            ])

            fwt = verify_first_welfare_theorem(economy, prices)
            swt = verify_second_welfare_theorem(economy, allocs)

        # Display results
        st.subheader("Competitive Equilibrium")
        col_a, col_b = st.columns(2)
        with col_a:
            st.write(f"**Prices**: ({prices[0]:.4f}, {prices[1]:.4f})")
            st.write(f"**Agent A allocation**: ({allocs[0][0]:.4f}, {allocs[0][1]:.4f})")
            st.write(f"**Agent B allocation**: ({allocs[1][0]:.4f}, {allocs[1][1]:.4f})")
        with col_b:
            st.write(f"**Max |excess demand|**: {np.max(np.abs(tat_result.excess_demand)):.2e}")

        # Utility comparison
        st.subheader("Utility at CE vs. Endowment")
        fig, ax = plt.subplots(figsize=(8, 4))
        x_pos = np.arange(2)
        width = 0.35
        bars1 = ax.bar(x_pos - width/2, u_endow, width, label="Endowment", color="salmon")
        bars2 = ax.bar(x_pos + width/2, u_ce, width, label="CE Allocation", color="steelblue")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(["Agent A", "Agent B"])
        ax.set_ylabel("Utility")
        ax.set_title("Utility: Endowment vs. Competitive Equilibrium")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        for bar, val in zip(bars1, u_endow):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)
        for bar, val in zip(bars2, u_ce):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)
        st.pyplot(fig)

        # Welfare theorems
        st.subheader("First Welfare Theorem")
        if fwt.holds:
            st.success(fwt.details)
        else:
            st.error(fwt.details)

        st.subheader("Second Welfare Theorem")
        if swt.holds:
            st.success(swt.details)
        else:
            st.warning(swt.details)


if __name__ == "__main__":
    main()
