# arrow-debreu-equilibrium

computing walrasian equilibria in exchange economies. the kind of thing that won arrow and debreu a nobel prize and still shows up in every micro theory class.

## what this is

given an economy with a bunch of agents, each with endowments and utility functions, find the price vector where all markets clear simultaneously. sounds simple until you realize it's a fixed-point problem in high-dimensional space.

implements three approaches:
- **scarf's algorithm** — the constructive proof that equilibria exist. pivots through simplices like a combinatorial simplex method
- **tâtonnement** — the classical "auctioneer" process. raise prices for goods in excess demand, lower for excess supply, iterate
- **edgeworth box** — 2-agent, 2-good case with full visualization. contract curves, offer curves, the whole thing

## running it

```bash
pip install -r requirements.txt
python main.py
```

## the math

an arrow-debreu exchange economy is a tuple (I, L, {u_i, e_i}) where I agents trade L goods. each agent i has a utility function u_i and an endowment vector e_i. a walrasian equilibrium is a price vector p* and allocation x* such that:

1. each agent maximizes utility given their budget: x_i* = argmax u_i(x) s.t. p·x ≤ p·e_i  
2. markets clear: Σx_i* = Σe_i

the first welfare theorem says any such equilibrium is pareto efficient. the second says any pareto efficient allocation can be supported as an equilibrium (with transfers). scarf showed these equilibria are computable, not just existential.
