import os
import shutil
import pytest
import pandas as pd
import numpy as np

from src.config import default_config, AppConfig
from src.agents import CampusEnvironment, ClassroomAgent, LaboratoryAgent, HostelAgent, EnergyManagerAgent
from src.game_theory import calculate_agent_utility, generate_payoff_matrix, find_pure_nash_equilibria, solve_game
from src.ai import generate_campus_dataset, train_and_save_models, evaluate_regression_model, QLearningAgent, Discretizer
from src.simulation import CampusSimulation, calculate_cumulative_metrics

TEMP_DIR = "temp_test_dir"
TEMP_DATA = os.path.join(TEMP_DIR, "test_demand.csv")
TEMP_MODELS = os.path.join(TEMP_DIR, "models")

@pytest.fixture(scope="module", autouse=True)
def setup_temp_dir():
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)
    yield
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)


def test_agent_requests_and_lifecycle():
    env = CampusEnvironment(
        total_capacity_kwh=default_config.grid.total_capacity_kwh,
        weights=default_config.weights
    )
    classroom = ClassroomAgent("classroom", env, default_config.classroom)
    lab = LaboratoryAgent("laboratory", env, default_config.laboratory)
    hostel = HostelAgent("hostel", env, default_config.hostel)
    manager = EnergyManagerAgent("energy_manager", env, default_config.grid, default_config.weights)

    env.set_step_state(
        step=0, hour=10, day=0, 
        occupancy={"classroom": 150, "laboratory": 25, "hostel": 800}, 
        temperature=24.0, is_holiday=False, is_exam_period=False
    )

    req_c = classroom.make_request()
    req_l = lab.make_request()
    req_h = hostel.make_request()

    assert req_c > 0
    assert req_l > 0
    assert req_h > 0

    manager.receive_requests()
    assert len(manager.current_requests) == 3

    allocations = manager.allocate_resources(policy="equal_proportion")
    assert sum(allocations.values()) <= default_config.grid.total_capacity_kwh

    classroom.update_state(allocations["classroom"])
    lab.update_state(allocations["laboratory"])
    hostel.update_state(allocations["hostel"])

    manager.receive_requests()
    metrics = manager.collect_confirms()

    assert "average_satisfaction" in metrics
    assert "sustainability_score" in metrics


def test_game_theory_solver():
    pm = generate_payoff_matrix(
        classroom_nominal=200.0,
        laboratory_nominal=150.0,
        hostel_nominal=400.0,
        grid_capacity=600.0,
        weights=default_config.weights,
        grid_settings=default_config.grid
    )
    assert pm.shape == (3, 3, 3, 3)

    eqs = find_pure_nash_equilibria(pm)
    assert isinstance(eqs, list)

    solution = solve_game(pm)
    assert "strategy_profile" in solution
    assert "payoffs" in solution
    assert "social_welfare" in solution


def test_dataset_and_predictions():
    df = generate_campus_dataset(TEMP_DATA)
    assert os.path.exists(TEMP_DATA)
    assert len(df) == 8760
    assert "classroom_demand" in df.columns

    results = train_and_save_models(TEMP_DATA, TEMP_MODELS)
    assert "classroom" in results
    assert "laboratory" in results
    assert "hostel" in results
    
    assert os.path.exists(os.path.join(TEMP_MODELS, "classroom_demand_model.joblib"))


def test_q_learning_and_discretizer():
    disc = Discretizer(capacity=1000.0)
    state_key = disc.get_state_key(
        available_energy=900.0,
        pending_requests=500.0,
        budget_remaining=300.0,
        hour=8
    )
    assert len(state_key) == 4
    assert all(isinstance(x, int) for x in state_key)

    agent = QLearningAgent(learning_rate=0.1, discount_factor=0.9, epsilon=0.2)
    action = agent.choose_action(state_key, explore=False)
    assert action in [0, 1, 2, 3]

    diff = agent.update(state_key, action, reward=1.5, next_state_key=(1, 1, 1, 1))
    assert diff >= 0.0


def test_simulations_runs():
    config = AppConfig()
    config.simulation.seed = 42
    
    policies = ["equal_proportion", "priority", "game_theoretic", "rl"]
    for policy in policies:
        sim = CampusSimulation(
            config=config,
            policy=policy,
            model_dir=None,
            rl_policy_path=os.path.join(TEMP_DIR, "qtable.json") if policy == "rl" else None
        )
        df = sim.run_simulation(steps=12)
        assert len(df) == 12
        metrics = calculate_cumulative_metrics(df)
        assert metrics["steps_simulated"] == 12
