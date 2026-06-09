# Presentation Guide: Sustainable Campus Resource Allocation Agent

This guide is designed to help you present and demo your project for a class presentation, grading interview, or project review.

---

## ⏱️ Recommended Presentation Flow (10 Minutes)
*   **Slide 1: Title & Objective** (1 min) — Introduce the project and the problem.
*   **Slide 2: System Architecture** (2 mins) — Explain the multi-agent design.
*   **Slide 3: Game Theory & Nash Equilibrium** (2 mins) — Explain the allocation rules.
*   **Slide 4: Reinforcement Learning & ML** (2 mins) — Explain how the AI works.
*   **Slide 5: Live Demo** (2 mins) — Walk through the Streamlit app.
*   **Slide 6: Sustainability Impact (SDGs) & Q&A** (1 min) — Conclude with target goals.

---

## 📂 Slide-by-Slide Talking Points

### Slide 1: Title & The Problem
*   **Headline**: *Autonomous Microgrid Management under Scarcity.*
*   **Talking Points**:
    *   **The Conflict**: Shared smart grids have capacity limits (e.g., 1000 kWh). If Classrooms, Laboratories, and Hostels request power at the same time, it can overload the grid.
    *   **Goal**: Create an autonomous multi-agent AI system that dynamically manages requests and allocates energy fairly without grid failures.
    *   **SDG Mapping**: Aligned with **SDG 7 (Clean Energy)** and **SDG 12 (Responsible Consumption)** by maximizing local renewable energy usage and reducing wastage.

### Slide 2: Multi-Agent Architecture
*   **Headline**: *Who are the players?*
*   **Talking Points**:
    *   **Classroom Agent**: Predicts energy requirements based on schedule, occupancy, and HVAC load (calculated using deviations from a comfortable temperature).
    *   **Laboratory Agent**: Predicts equipment demand and rolling forecasts based on machine count and research baseloads.
    *   **Hostel Agent**: Models peak diurnal resident hours (morning/evening surges) and tracks efficiency.
    *   **Energy Manager Agent**: The microgrid coordinator. It collects requests and determines the hourly allocations based on a selected policy.

### Slide 3: Game Theory & Nash Equilibrium
*   **Headline**: *Resolving Grid Scarcity as a Strategic Game.*
*   **Talking Points**:
    *   **Strategies**: The agents choose one of three strategies: **Conservative** ($0.7 \times$ nominal request), **Normal** ($1.0 \times$), or **Aggressive** ($1.3 \times$).
    *   **Payoff Matrix**: We dynamically generate a $3 \times 3 \times 3$ payoff matrix based on occupancy and capacity.
    *   **Nash Equilibrium**: Under scarcity, playing "Aggressive" results in penalties (blackouts/brownouts). The solver calculates the **Pure Strategy Nash Equilibrium (PSNE)** so that agents coordinate their demands.
    *   **Fallback**: If no pure equilibrium exists, the manager defaults to the profile that maximizes **Social Welfare** (the sum of all utilities).

### Slide 4: AI & Reinforcement Learning
*   **Headline**: *Demand Prediction and RL Policy Selection.*
*   **Talking Points**:
    *   **Load Forecasting**: We trained a **Random Forest Regressor** on a 365-day dataset to predict building load based on features like hour, temperature, and holidays.
    *   **Reinforcement Learning**: We implemented a tabular **Q-Learning Agent** as a *microgrid policy selector*. 
    *   **MDP Formulation**:
        *   *State*: (Available capacity, pending requests, daily budget, hour).
        *   *Actions*: Selects the macro-policy on each hour: (Proportional, Priority, Game Theory Nash, or Eco Conservation).
        *   *Reward*: Based on average occupant satisfaction, low grid wastage, and a heavy penalty for running out of budget early in the day.

---

## 💻 Dashboard Demo Guide (Step-by-Step)

When sharing your screen to demo the Streamlit dashboard, guide your audience through these tabs:

1.  **Overview & SDGs**: Show the block diagram and explain how the agents represent independent buildings communicating through the microgrid.
2.  **Live Simulation**:
    *   Change the **Allocation Policy** dropdown (e.g., from *Equal Proportional* to *Reinforcement Learning*).
    *   Click **Run Simulation**.
    *   Show the Plotly charts: Point out how the "Total Allocated Power" is clamped at the "Grid Capacity Limit" line during peak hours.
3.  **AI Forecasts**: Select an agent (e.g., *laboratory*) and show the green (Actual) vs red (ML Predicted) line chart. Point out the high $R^2$ score in the comparison table showing that Random Forest outperforms Linear Regression.
4.  **Game Theory Solver**:
    *   Move the **Scarcity Capacity Limit** slider down to make energy scarce.
    *   Show how the solver updates in real-time to change the recommended strategies (e.g. telling Hostels or Classrooms to play "Conservative") to prevent grid collapse.
5.  **Sustainability Metrics**: Show the **Sustainability Index gauge chart** and point out the cumulative carbon ($tCO_2$) and financial savings chart.

---

## ❓ Potential Q&A Questions & How to Answer Them

*   **Q: Why did you use tabular Q-Learning instead of Deep RL (like DQN)?**
    *   *A*: Tabular Q-learning is lightweight, runs in milliseconds, is easy to implement, and doesn't require massive GPU resources. By discretizing the state space (capacity, requests, budget, hour), we keep the Q-table size small (400 states) which allows it to learn the optimal policy quickly within a 30-day simulation.
*   **Q: How does the game theory Nash Equilibrium prevent blackouts?**
    *   *A*: In scarcity conditions, if everyone requests maximum power, the grid fails. Our utility function penalizes agents for failed requests. The Nash Equilibrium solver finds a stable coordinate profile where agents with lower priorities voluntarily choose to play "Conservative" (curtailing non-essential load) to maintain grid safety for high-priority units (like research labs).
*   **Q: What is the target of the ML regressor model?**
    *   *A*: The regressor models target the expected energy load (in kWh) for each building type for the next hour. This allows agents to request a highly accurate amount of energy rather than making static guesses.
