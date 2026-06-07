import os
import pytest
import shutil
import pandas as pd
from src.sustainable_campus.config import AppConfig
from src.sustainable_campus.simulation.environment import CampusSimulation
from src.sustainable_campus.simulation.metrics import calculate_cumulative_metrics

TEMP_DIR = "temp_sim_test_dir"

@pytest.fixture(scope="module", autouse=True)
def setup_temp_dir():
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)
    yield
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)

def test_simulation_execution():
    config = AppConfig()
    config.simulation.seed = 42
    
    # We will test all 4 allocation policies
    policies = ["equal_proportion", "priority", "game_theoretic", "rl"]
    
    for policy in policies:
        rl_path = os.path.join(TEMP_DIR, "qtable.json") if policy == "rl" else None
        
        sim = CampusSimulation(
            config=config,
            policy=policy,
            model_dir=None,  # run with structural formulas fallback
            rl_policy_path=rl_path
        )
        
        # Run 24 hours simulation
        df = sim.run_simulation(steps=24)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 24
        assert "total_energy_consumed" in df.columns
        assert "sustainability_score" in df.columns
        
        metrics = calculate_cumulative_metrics(df)
        assert metrics["steps_simulated"] == 24
        assert metrics["total_energy_consumed_mwh"] >= 0.0
        assert metrics["average_satisfaction"] > 0.0
        assert metrics["average_sustainability_score"] >= 0.0
