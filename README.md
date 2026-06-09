# AI Smart Grid Campus Energy Resource Allocator

An Intelligent Smart Grid Energy Allocation AI Agent designed to solve power capacity allocation conflicts among campus building zones. The project maps to **SDG 7 (Affordable and Clean Energy)** and **SDG 11 (Sustainable Cities and Communities)**, utilizing Game Theory and an agent decision-making cognitive lifecycle (Perceive, Reason, Decide, Act).

---

## Workspace Setup Recommendation

> [!IMPORTANT]
> This project is initialized in the following directory:
> `C:\Users\Admin\Desktop\sustainable-campus-resource-allocation-agent`
> **It is highly recommended to set this directory as your active workspace in your IDE or agent workspace settings.**

---

## Features

- **Agent Cognitive Lifecycle**: Modeled decision-making split into four distinct stages:
  1. **Perception**: Observes environment state, reading building zone loads and available capacity.
  2. **Reasoning**: Evaluates electrical base utility score (load urgency curves) based on demand levels, occupancy weights, and priority scores.
  3. **Decision-making**: Resolves conflicts by calculating the Pure Strategy Nash Equilibrium (PSNE) using dominant strategy reduction.
  4. **Action**: Deploys electrical power capacity blocks (e.g. 1 MW blocks) to selected building zones.
- **Payoff Matrix Analysis**: Dynamic payoffs computed for different profile decisions (All Delay, Nash Equilibrium, Surge Competition) demonstrating microgrid stability.
- **Streamlit Web Dashboard**: Interactive, single-page interface to tweak parameters and instantly run the optimizer.

---

## Repository Structure

```text
sustainable-campus-resource-allocation-agent/
│
├── data/
│   └── zones.csv              # Campus Building Zones database
│
├── src/                       # Source code package
│   ├── __init__.py
│   ├── agent.py               # AI Agent (Perceive, Reason, Decide, Act)
│   ├── environment.py         # Microgrid Environment
│   ├── game_theory.py         # Game Theory Nash Allocator
│   ├── models.py              # Zone Data Model
│   └── utils.py               # Payoff reporting helpers
│
├── tests/                     # Unit Test Suite
│   ├── __init__.py
│   └── test_agent.py          # Unit test suite
│
├── app.py                     # Streamlit Dashboard UI
├── main.py                    # Console entrypoint
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

### 1. Run the Console Simulator
Execute the simulation directly via command line:
```bash
python main.py
```
Options:
- `--capacity <int>`: Adjust available grid supply MW (default: 3).
- `--data <path>`: Load a custom zones CSV.

### 2. Run the Streamlit Dashboard
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

### 3. Run the Unit Test Suite
```bash
python -m pytest tests/
```
