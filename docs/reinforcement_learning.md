# Reinforcement Learning Module

This document outlines the Reinforcement Learning (RL) formulation implemented in the **Sustainable Campus Resource Allocation Agent** project.

## Formulation

The reinforcement learning agent operates as a **Grid Policy Selector** within a Markov Decision Process (MDP) context. Rather than making fine-grained allocation decisions directly, it observes the state of the campus microgrid and selects the optimal macro-allocation policy on each hour.

### State Space

The state represents the microgrid's current operational status. To implement tabular Q-learning, the continuous values are discretized:
1. **Available Grid Energy Capacity** (Discretized into 5 bins: Low, Below Average, Normal, Above Average, High).
2. **Pending Energy Requests** (Discretized into 5 bins: Low, Medium, High).
3. **Daily Budget Remaining** (Discretized into 4 bins: Depleted, Low, Medium, High).
4. **Hour of the Day** (Discretized into 4 bins: Morning, Afternoon, Evening, Night).

This yields a discrete state space of $5 \times 5 \times 4 \times 4 = 400$ possible states, represented as a tuple:
$$S = (s_{energy}, s_{requests}, s_{budget}, s_{hour})$$

### Action Space

On each step, the RL agent selects one of four macro policies:
- **Action 0: Proportional Sharing**: Allocates resources proportionally. Simple and fair under low-to-medium demand.
- **Action 1: Priority Allocation**: Lab equipment baseload protected, classrooms scaled, hostels curtailed first. Useful during peak hours.
- **Action 2: Game-Theoretic Nash Equilibrium**: Solves the non-cooperative game to enforce strategic coordination.
- **Action 3: Eco Conservation Mode**: Limits total allocated energy to a maximum of 80% capacity to maximize conservation and save budget.

### Reward Function

The reward function balances agent satisfaction, microgrid conservation efficiency, and budget conservation:

$$Reward = (1.5 \times \text{Average Satisfaction}) + (1.0 \times \text{Sustainability Score}) + \text{Penalty}_{budget}$$

Where:
- **Average Satisfaction** is the mean comfort score of the Classrooms, Labs, and Hostels.
- **Sustainability Score** evaluates energy savings and green power fraction.
- $\text{Penalty}_{budget}$ is a penalty ($-2.0$) applied if the microgrid runs out of daily budget before the day ends.

## Q-Learning Update Rule

The agent implements standard tabular Q-learning. The Q-table updates according to the Bellman Equation:

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ R + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$

Where:
- $\alpha$ (learning rate) $= 0.1$
- $\gamma$ (discount factor) $= 0.9$
- Epsilon-greedy exploration decays at a rate of $0.995$ per hour down to a minimum of $0.01$.

## Training Performance

Over multi-day simulations (e.g., 30 to 365 days), the RL policy selector:
1. Learns to play **Eco Conservation Mode** or **Priority Mode** during peak pricing hours to protect budgets.
2. Selects **Nash Equilibrium** or **Proportional Sharing** during daytime school hours to keep student comfort high.
3. Learns to shift loads away from critical brownout states.
