# Game Theory Module

This document outlines the Game Theory formulation implemented in the **Sustainable Campus Resource Allocation Agent** project.

## Formulation

We model energy resource allocation under scarcity as a **non-cooperative, multi-player normal-form game**. 

### Players
1. **Classroom Agent** ($i = 0$)
2. **Laboratory Agent** ($i = 1$)
3. **Hostel Agent** ($i = 2$)

### Action Space (Strategies)
Each agent $i$ chooses a bidding strategy $s_i \in \{0, 1, 2\}$ representing the inflation/curtailment factor applied to their nominal energy requirement:
- **0: Conservative** ($0.7 \times$ nominal request): The agent curtails non-essential loads (e.g., dimming lights, minor temperature drift) to support grid stability.
- **1: Normal** ($1.0 \times$ nominal request): The agent requests exactly their expected operating demand.
- **2: Aggressive** ($1.3 \times$ nominal request): The agent inflates their request to secure a buffer, safeguarding comfort against potential grid brownouts.

### Utility Function
The utility $U_i$ for player $i$ represents a balance of operational comfort, cost penalty, and sustainability:

$$U_i = w_1 \cdot \text{Satisfaction}_i + w_2 \cdot \text{Sustainability}_i - w_3 \cdot \text{Cost Penalty}_i$$

Where:
- $\text{Satisfaction}_i = \frac{\text{Allocated}_i}{\text{Requested}_i}$ (clamped to $[0, 1]$).
- $\text{Sustainability}_i = \text{Green Energy Fraction} - \text{Waste Penalty}_i + 0.5$. If $\text{Allocated}_i > \text{Requested}_i$, the agent penalizes itself for wasted electricity.
- $\text{Cost Penalty}_i = \frac{\text{Allocated}_i \times \text{Price Rate}}{\text{Max Estimated Cost}}$.

## Scarcity Scenarios

The payoff matrix is generated dynamically on each simulation step depending on grid conditions:
- **Normal Demand**: Total grid capacity is larger than the sum of aggressive demands. The dominant strategy is for all players to request Aggressive ($1.3 \times$) to maximize comfort, resulting in a single Nash Equilibrium: `(Aggressive, Aggressive, Aggressive)`.
- **Scarcity Demand**: Grid capacity is tight. If multiple agents bid "Aggressive", the grid scales down allocations proportionally. Satisfaction falls, and costs rise, making `(Aggressive, Aggressive, Aggressive)` an unstable, low-utility trap (similar to a multi-player Prisoner's Dilemma).
- **Extreme Scarcity**: Grid capacity is extremely restricted. Players must play "Conservative" to prevent severe brownouts. The Nash Equilibrium shifts to cooperative conservation profiles (e.g., `(Conservative, Conservative, Conservative)`).

## Nash Equilibrium Solver

A strategy profile $(s^*_0, s^*_1, s^*_2)$ constitutes a **Pure Strategy Nash Equilibrium** if no player can unilaterally deviate to improve their utility:

$$U_i(s^*_i, s^*_{-i}) \geq U_i(s_i, s^*_{-i}) \quad \forall s_i \in \{0, 1, 2\}, \forall i$$

Our solver:
1. Performs a grid search over the $3 \times 3 \times 3$ payoff matrix.
2. Identifies all strategy profiles satisfying the Nash condition.
3. If multiple equilibria exist, it selects the **Pareto dominant** equilibrium (maximizing the sum of utilities/Social Welfare).
4. If no pure Nash Equilibrium exists, it falls back to the profile that maximizes **Social Welfare**:

$$\text{Social Welfare} = \sum_{i=0}^2 U_i$$
