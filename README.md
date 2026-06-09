# Sustainable Campus Resource Allocation Agent

An autonomous multi-agent AI system designed to allocate limited campus electricity grid capacity among Classrooms, Laboratories, and Hostels. The project maps to **SDG 7 (Affordable and Clean Energy)** and **SDG 12 (Responsible Consumption and Production)**, utilizing Demand Prediction, Game Theory, and Reinforcement Learning in a simple flat layout.

---

## Workspace Setup Recommendation

> [!IMPORTANT]
> This project is initialized in the following directory:
> `C:\Users\Admin\Desktop\sustainable-campus-resource-allocation-agent`
> **It is highly recommended to set this directory as your active workspace in your IDE or agent workspace settings.**

---

## Features

- **Multi-Agent Energy Management**: BaseAgent, ClassroomAgent, LaboratoryAgent, and HostelAgent communicate with a central EnergyManagerAgent.
- **Demand Forecasting**: Random Forest and Linear Regression models trained on seasonal load parameters (hour, day, occupancy, temp, holidays, exam periods).
- **Game Theory Negotiation**: Solves grid scarcity conflict by formulating a 3D payoff matrix across agent strategies (Conservative, Normal, Aggressive) and finding the Pure Strategy Nash Equilibrium (PSNE).
- **Reinforcement Learning**: Operates a Q-Learning controller that learns to select macro-allocation policies to maximize campus occupant comfort while avoiding blackout overloads and daily budget overruns.
- **Streamlit Web Dashboard**: An interactive, single-page interface to adjust grid supply, test weights, run multi-day simulations, inspect ML predictions, and track carbon footprint offsets.

---

## Repository Structure

```text
sustainable-campus-resource-allocation-agent/
│
├── data/
│   └── campus_energy_demand.csv       # Generated 365-day hourly dataset
│
├── models/
│   ├── classroom_demand_model.joblib  # Trained forecasting models
│   ├── laboratory_demand_model.joblib
│   └── hostel_demand_model.joblib
│
├── src/                       # Source code package (flat layout)
│   ├── __init__.py
│   ├── config.py              # App & Grid configurations (Pydantic)
│   ├── agents.py              # Agents and environment message bus
│   ├── game_theory.py         # Utility, payoff matrices, Nash solver
│   ├── ai.py                  # Dataset generator, regressor training, Q-learning
│   └── simulation.py          # CampusSimulation loop and metrics
│
├── tests/                     # Test Suite
│   ├── __init__.py
│   └── test_project.py        # Consolidated test suite
│
├── app.py                     # Streamlit Dashboard UI
├── requirements.txt           # Light dependencies
├── .gitignore                 # standard git ignore file
└── README.md                  # Project overview
```

---

## Local Installation

### Prerequisites
- Python 3.12 (or higher)
- `pip` (Python package installer)

### Setup Steps
1. Navigate to the project directory:
   ```bash
   cd "C:\Users\Admin\Desktop\sustainable-campus-resource-allocation-agent"
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

---

## Running the Project

### 1. Run the Streamlit Dashboard
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.
*(On first startup, the app automatically generates the dataset and trains the models in the background).*

### 2. Run the Unit Test Suite
```bash
python -m pytest tests/
```
