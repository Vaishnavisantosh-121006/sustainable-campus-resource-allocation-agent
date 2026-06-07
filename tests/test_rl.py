import os
import pytest
import shutil
from src.sustainable_campus.ai.q_learning import QLearningAgent, Discretizer

TEMP_DIR = "temp_rl_test_dir"
POLICY_PATH = os.path.join(TEMP_DIR, "policy.json")

@pytest.fixture(scope="module", autouse=True)
def setup_temp_dir():
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)
    yield
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)

def test_discretizer():
    disc = Discretizer(capacity=1000.0)
    
    # State values
    state_key1 = disc.get_state_key(
        available_energy=900.0,
        pending_requests=500.0,
        budget_remaining=300.0,
        hour=8
    )
    
    # Assert dimensions are valid
    assert len(state_key1) == 4
    assert all(isinstance(x, int) for x in state_key1)
    
    # High requests should discretize into the highest bin (4)
    state_key2 = disc.get_state_key(
        available_energy=200.0,
        pending_requests=2000.0,
        budget_remaining=0.0,
        hour=23
    )
    assert state_key2[1] == 4 # requests clamped at highest bin index
    assert state_key2[2] == 0 # budget index 0
    assert state_key2[3] == 3 # hour bin index 3

def test_q_learning_agent():
    agent = QLearningAgent(
        learning_rate=0.1,
        discount_factor=0.9,
        epsilon=0.2,
        action_size=4
    )
    
    state1 = (2, 2, 2, 1)
    state2 = (1, 3, 2, 1)
    
    # Choose action (test epsilon exploration/exploitation)
    action = agent.choose_action(state1, explore=False)
    assert action in [0, 1, 2, 3]
    
    # Q-update
    diff = agent.update(state1, action, reward=1.5, next_state_key=state2)
    assert diff > 0.0
    
    # Assert Q-value updated
    q_vals = agent._get_q_values(state1)
    assert q_vals[action] != 0.0
    
    # Verify save and load
    agent.save_policy(POLICY_PATH)
    assert os.path.exists(POLICY_PATH)
    
    new_agent = QLearningAgent()
    new_agent.load_policy(POLICY_PATH)
    assert str(state1) in new_agent.q_table
    assert new_agent.q_table[str(state1)] == q_vals
