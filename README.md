# Sustainable Campus Resource Allocation Agent

An autonomous multi-agent AI system that dynamically allocates limited campus electricity resources among classrooms, laboratories, and hostels. The system maps to **SDG 7 (Affordable and Clean Energy)** and **SDG 12 (Responsible Consumption and Production)**, utilizing Demand Prediction, Game Theory, and Reinforcement Learning.

---

## Workspace Setup Recommendation

> [!IMPORTANT]
> This project is initialized in the following directory:
> `C:\Users\Admin\.gemini\antigravity\scratch\sustainable-campus-resource-allocation-agent`
> **It is highly recommended to set this directory as your active workspace in your IDE or agent workspace settings to ensure all file paths and tests run correctly.**

---

## Features

- **Autonomous Agent System**: Model-driven agents representing Classroom, Laboratory, and Hostel facilities, coordinated by a central Energy Manager.
- **Machine Learning Demand Prediction**: Auto-regressive demand forecasting models using Linear Regression and Random Forest Regressors, evaluated on MAE, RMSE, and $R^2$ scores.
- **Game Theory Negotiation**: Dynamic generation of 3D payoff matrices representing agent coordinate strategies (Conservative, Normal, Aggressive) solved for Pure Nash Equilibria.
- **Reinforcement Learning (Q-learning)**: A microgrid agent policy selector that learns optimal allocation actions to satisfy comfort constraints while minimizing cost, budget overruns, and energy wastage.
- **Sustainability KPI Tracker**: Direct translation of microgrid load reductions into metric tons of CO2 offset and financial cost savings.
- **Interactive Streamlit Dashboard**: Beautiful dark-theme GUI featuring live simulators, game solvers, ML forecasts, and sustainability charts.

---

## Repository Structure

```text
sustainable-campus-resource-allocation-agent/
│
├── .github/
│   └── workflows/
│       └── ci.yml             # Github Actions CI pipeline (Linting & pytest)
│
├── docs/                      # Documentation
│   ├── architecture.md
│   ├── game_theory.md
│   └── reinforcement_learning.md
│
├── src/
│   └── sustainable_campus/    # Source code package
│       ├── __init__.py
│       ├── config.py          # App & Grid configuration settings
│       ├── app.py             # Streamlit Dashboard UI
│       │
│       ├── agents/            # Multi-Agent modules
│       │   ├── __init__.py
│       │   ├── base_agent.py
│       │   ├── classroom_agent.py
│       │   ├── laboratory_agent.py
│       │   ├── hostel_agent.py
│       │   ├── energy_manager_agent.py
│       │   └── communication.py
│       │
│       ├── game_theory/       # Game Theory modules
│       │   ├── __init__.py
│       │   ├── utility.py
│       │   ├── payoff_matrix.py
│       │   └── nash_equilibrium.py
│       │
│       ├── ai/                # AI and ML modules
│       │   ├── __init__.py
│       │   ├── dataset_generator.py
│       │   ├── demand_prediction.py
│       │   └── q_learning.py
│       │
│       └── simulation/        # Simulation Orchestration
│           ├── __init__.py
│           ├── environment.py
│           └── metrics.py
│
├── tests/                     # Unit Test Suite
│   ├── __init__.py
│   ├── test_agents.py
│   ├── test_prediction.py
│   ├── test_rl.py
│   └── test_simulation.py
│
├── .gitignore
├── Dockerfile                 # Docker setup
├── LICENSE                    # MIT License
└── requirements.txt           # Package requirements
```

---

## Local Installation

### Prerequisites
- Python 3.12
- `pip` (Python package installer)

### Setup Steps
1. Navigate to the project root directory:
   ```bash
   cd C:\Users\Admin\.gemini\antigravity\scratch\sustainable-campus-resource-allocation-agent
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
The dashboard automatically generates the synthetic energy dataset and trains demand prediction models on its first startup.
```bash
streamlit run src/sustainable_campus/app.py
```
After running, open `http://localhost:8501` in your browser.

### 2. Run the Unit Test Suite
Verify that all modular code builds and passes tests with coverage:
```bash
pytest --cov=src --cov-report=term-missing
```

---

## Docker Support

Build and run the containerized Streamlit application:
```bash
# Build Docker Image
docker build -t sustainable-campus-agent .

# Run Docker Container
docker run -p 8501:8501 sustainable-campus-agent
```
Access the dashboard at `http://localhost:8501`.

---

## Sustainability Impact (SDGs)

### SDG 7: Affordable and Clean Energy
- **Target 7.3**: Double the global rate of improvement in energy efficiency.
- **Implementation**: The Energy Manager scales grid resource sharing to avoid microgrid overload and wastage, maximizing local green energy (solar/wind) consumption.

### SDG 12: Responsible Consumption and Production
- **Target 12.2**: Achieve the sustainable management and efficient use of natural resources.
- **Implementation**: Curtailment and load shifting reduce peak fossil-fuel electricity usage, lowering total CO2 emissions and optimizing energy efficiency.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
