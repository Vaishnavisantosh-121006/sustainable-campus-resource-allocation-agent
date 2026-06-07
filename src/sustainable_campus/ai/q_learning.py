import os
import json
import numpy as np
from typing import Tuple, Dict, Any, List
from loguru import logger

class Discretizer:
    """
    Discretizes continuous state variables into integer bins for tabular Q-learning.
    """
    def __init__(self, capacity: float):
        self.capacity = capacity
        # Define bin thresholds
        self.energy_bins = np.linspace(0, capacity, 6)           # 5 bins
        self.request_bins = np.linspace(0, capacity * 1.5, 6)    # 5 bins
        self.budget_bins = np.linspace(0, 500.0, 5)             # 4 bins
        self.hour_bins = [6, 12, 18, 24]                         # 4 bins

    def get_state_key(self, available_energy: float, pending_requests: float, 
                      budget_remaining: float, hour: int) -> Tuple[int, int, int, int]:
        """
        Converts continuous state parameters into a discrete state tuple index.
        """
        e_bin = int(np.digitize(available_energy, self.energy_bins)) - 1
        r_bin = int(np.digitize(pending_requests, self.request_bins)) - 1
        b_bin = int(np.digitize(budget_remaining, self.budget_bins)) - 1
        
        h_bin = 0
        for idx, limit in enumerate(self.hour_bins):
            if hour < limit:
                h_bin = idx
                break
                
        # Clamp bounds
        e_bin = max(0, min(e_bin, 4))
        r_bin = max(0, min(r_bin, 4))
        b_bin = max(0, min(b_bin, 3))
        h_bin = max(0, min(h_bin, 3))
        
        return (e_bin, r_bin, b_bin, h_bin)

class QLearningAgent:
    """
    Tabular Q-learning agent designed to select optimal grid allocation policies.
    
    Actions:
    0: Proportional allocation
    1: Priority allocation (Laboratory > Classroom > Hostel)
    2: Game Theory (Nash Equilibrium allocation)
    3: Green Conservation mode (Allocates max 80% capacity proportionally)
    """
    def __init__(
        self,
        learning_rate: float = 0.1,
        discount_factor: float = 0.9,
        epsilon: float = 0.15,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        action_size: int = 4
    ):
        self.alpha: float = learning_rate
        self.gamma: float = discount_factor
        self.epsilon: float = epsilon
        self.epsilon_decay: float = epsilon_decay
        self.epsilon_min: float = epsilon_min
        self.action_size: int = action_size
        
        # Q-table represented as a dictionary mapping state-key string to action list
        self.q_table: Dict[str, List[float]] = {}

    def _get_q_values(self, state_key: Tuple[int, int, int, int]) -> List[float]:
        key_str = str(state_key)
        if key_str not in self.q_table:
            self.q_table[key_str] = [0.0] * self.action_size
        return self.q_table[key_str]

    def choose_action(self, state_key: Tuple[int, int, int, int], explore: bool = True) -> int:
        """
        Selects an action using an epsilon-greedy policy.
        """
        if explore and np.random.rand() < self.epsilon:
            action = int(np.random.randint(self.action_size))
            logger.debug(f"RL Agent exploring: chose action {action}")
            return action
            
        q_vals = self._get_q_values(state_key)
        best_action = int(np.argmax(q_vals))
        logger.debug(f"RL Agent exploiting: chose action {best_action} with Q-values {q_vals}")
        return best_action

    def update(
        self,
        state_key: Tuple[int, int, int, int],
        action: int,
        reward: float,
        next_state_key: Tuple[int, int, int, int]
    ) -> float:
        """
        Updates the Q-value for the (state, action) pair using the Bellman equation.
        """
        q_vals = self._get_q_values(state_key)
        next_q_vals = self._get_q_values(next_state_key)
        
        # Bellman update equation: Q(s, a) = Q(s, a) + alpha * [R + gamma * max Q(s', a') - Q(s, a)]
        old_val = q_vals[action]
        best_next_q = max(next_q_vals)
        new_val = old_val + self.alpha * (reward + self.gamma * best_next_q - old_val)
        
        self.q_table[str(state_key)][action] = float(new_val)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return float(abs(new_val - old_val))

    def save_policy(self, file_path: str) -> None:
        """
        Saves the learned Q-table policy to a JSON file.
        """
        dir_name = os.path.dirname(file_path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name)
            
        with open(file_path, 'w') as f:
            json.dump({
                "q_table": self.q_table,
                "epsilon": self.epsilon,
                "alpha": self.alpha,
                "gamma": self.gamma
            }, f, indent=4)
        logger.info(f"Learned RL policy saved to {file_path}")

    def load_policy(self, file_path: str) -> None:
        """
        Loads a saved Q-table policy from a JSON file.
        """
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                self.q_table = data.get("q_table", {})
                self.epsilon = data.get("epsilon", self.epsilon)
                self.alpha = data.get("alpha", self.alpha)
                self.gamma = data.get("gamma", self.gamma)
                logger.info(f"Learned RL policy loaded from {file_path}. Size of Q-table: {len(self.q_table)}")
            except Exception as e:
                logger.error(f"Failed to load RL policy: {e}")
        else:
            logger.warning(f"RL policy path not found: {file_path}. Starting with clean Q-table.")
