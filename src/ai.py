import os
import json
import numpy as np
import pandas as pd
import joblib
from typing import Dict, Tuple, Any, List
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ----------------------------------------------------
# Dataset Generator
# ----------------------------------------------------
def generate_campus_dataset(file_path: str) -> pd.DataFrame:
    """
    Generates a realistic synthetic campus energy demand dataset with 8760 hourly steps (365 days).
    """
    logger.info(f"Generating synthetic dataset at {file_path}")
    dir_name = os.path.dirname(file_path)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name)

    np.random.seed(42)
    n_hours = 8760
    
    timestamps = pd.date_range(start="2026-01-01 00:00:00", periods=n_hours, freq="h")
    hours = timestamps.hour
    day_of_weeks = timestamps.dayofweek
    day_of_years = timestamps.dayofyear
    
    is_holiday = []
    for day, dow in zip(day_of_years, day_of_weeks):
        is_break = (150 <= day <= 210) or (350 <= day <= 365)
        is_we = dow >= 5
        is_holiday.append(1 if (is_break or is_we) else 0)
    is_holiday = np.array(is_holiday)

    is_exam_period = np.array([
        1 if ((120 <= day <= 135) or (320 <= day <= 335)) else 0
        for day in day_of_years
    ])

    base_temp = 20.0 + 10.0 * np.sin(2 * np.pi * (day_of_years - 100) / 365.0)
    daily_temp = 5.0 * np.sin(2 * np.pi * (hours - 8) / 24.0)
    temperatures = base_temp + daily_temp + np.random.normal(0, 1.5, n_hours)

    classroom_occupancies = []
    lab_occupancies = []
    hostel_occupancies = []

    for hr, hol, exam in zip(hours, is_holiday, is_exam_period):
        if hol == 1:
            c_occ = int(np.random.poisson(10))
            l_occ = int(np.random.poisson(15))
            h_occ = int(np.random.poisson(1000))
        else:
            if 8 <= hr < 17:
                c_occ = int(np.random.poisson(280)) if not exam else int(np.random.poisson(320))
                l_occ = int(np.random.poisson(40))
                h_occ = int(np.random.poisson(200))
            elif 17 <= hr < 22:
                c_occ = int(np.random.poisson(50))
                l_occ = int(np.random.poisson(35))
                h_occ = int(np.random.poisson(800))
            else:
                c_occ = int(np.random.poisson(5))
                l_occ = int(np.random.poisson(10))
                h_occ = int(np.random.poisson(1100))
        
        classroom_occupancies.append(c_occ)
        lab_occupancies.append(l_occ)
        hostel_occupancies.append(h_occ)

    classroom_occupancies = np.array(classroom_occupancies)
    lab_occupancies = np.array(lab_occupancies)
    hostel_occupancies = np.array(hostel_occupancies)

    classroom_demands = []
    lab_demands = []
    hostel_demands = []
    
    prev_c_dem = 80.0
    prev_l_dem = 50.0
    prev_h_dem = 150.0

    for i in range(n_hours):
        hr = hours[i]
        temp = temperatures[i]
        hol = is_holiday[i]
        exam = is_exam_period[i]
        c_occ = classroom_occupancies[i]
        l_occ = lab_occupancies[i]
        h_occ = hostel_occupancies[i]

        # Classroom load
        temp_delta = abs(temp - 22.0)
        c_hvac = 40.0 + (temp_delta * 3.5)
        c_load = c_occ * 0.15 if (8 <= hr < 18 and hol == 0) else 5.0
        c_dem = (c_hvac + c_load) * (1.2 if exam else 1.0)
        c_dem = 0.7 * c_dem + 0.2 * prev_c_dem + np.random.normal(0, 5.0)
        c_dem = max(5.0, c_dem)
        classroom_demands.append(c_dem)
        prev_c_dem = c_dem

        # Lab load
        active_machines = min(50, max(5, int(l_occ * 0.8)))
        l_load = 30.0 + active_machines * 1.2
        l_mult = 1.3 if exam else 1.0
        if not (8 <= hr < 22):
            l_mult *= 0.6
        l_dem = l_load * l_mult
        l_dem = 0.8 * l_dem + 0.15 * prev_l_dem + np.random.normal(0, 4.0)
        l_dem = max(10.0, l_dem)
        lab_demands.append(l_dem)
        prev_l_dem = l_dem

        # Hostel load
        is_peak = (6 <= hr < 9) or (17 <= hr < 23)
        peak_mult = 1.8 if is_peak else 0.7
        h_load = h_occ * 0.2 * peak_mult * (1.3 if hol else 1.0)
        if exam and (23 <= hr or hr < 2):
            h_load *= 1.4
        h_hvac = 60.0 + (abs(temp - 21.0) * 3.0)
        h_dem = h_load + h_hvac
        h_dem = 0.75 * h_dem + 0.2 * prev_h_dem + np.random.normal(0, 10.0)
        h_dem = max(15.0, h_dem)
        hostel_demands.append(h_dem)
        prev_h_dem = h_dem

    df = pd.DataFrame({
        "timestamp": timestamps,
        "day_of_week": day_of_weeks,
        "hour": hours,
        "temperature": temperatures,
        "is_holiday": is_holiday,
        "is_exam_period": is_exam_period,
        
        "classroom_occupancy": classroom_occupancies,
        "classroom_prev_demand": [80.0] + classroom_demands[:-1],
        "classroom_demand": classroom_demands,
        
        "laboratory_occupancy": lab_occupancies,
        "laboratory_prev_demand": [50.0] + lab_demands[:-1],
        "laboratory_demand": lab_demands,
        
        "hostel_occupancy": hostel_occupancies,
        "hostel_prev_demand": [150.0] + hostel_demands[:-1],
        "hostel_demand": hostel_demands,
    })

    df.to_csv(file_path, index=False)
    logger.info(f"Dataset generated successfully with {n_hours} records.")
    return df

# ----------------------------------------------------
# Demand Predictor Model Trainer
# ----------------------------------------------------
def evaluate_regression_model(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2)
    }

def train_and_save_models(data_path: str, models_dir: str) -> Dict[str, Dict[str, Dict[str, float]]]:
    logger.info(f"Training prediction models using data from {data_path}")
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    df = pd.read_csv(data_path)
    agent_types = ["classroom", "laboratory", "hostel"]
    results = {}

    for agent in agent_types:
        occ_col = f"{agent}_occupancy"
        prev_col = f"{agent}_prev_demand"
        target_col = f"{agent}_demand"
        
        features = ["day_of_week", "hour", "temperature", "is_holiday", "is_exam_period", occ_col, prev_col]
        X = df[features].copy()
        X.columns = ["day_of_week", "hour", "temperature", "is_holiday", "is_exam_period", "occupancy", "prev_demand"]
        y = df[target_col].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 1. Linear Regression
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        lr_preds = lr.predict(X_test)
        lr_metrics = evaluate_regression_model(y_test, lr_preds)
        
        # 2. Random Forest
        rf = RandomForestRegressor(n_estimators=30, max_depth=10, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        rf_preds = rf.predict(X_test)
        rf_metrics = evaluate_regression_model(y_test, rf_preds)

        best_model = rf if rf_metrics["R2"] > lr_metrics["R2"] else lr
        best_model_name = "random_forest" if rf_metrics["R2"] > lr_metrics["R2"] else "linear_regression"
        
        model_filename = f"{agent}_demand_model.joblib"
        model_save_path = os.path.join(models_dir, model_filename)
        joblib.dump(best_model, model_save_path)

        results[agent] = {
            "linear_regression": lr_metrics,
            "random_forest": rf_metrics,
            "best_model": best_model_name
        }

    return results

# ----------------------------------------------------
# Reinforcement Learning - Tabular Q-learning
# ----------------------------------------------------
class Discretizer:
    def __init__(self, capacity: float):
        self.capacity = capacity
        self.energy_bins = np.linspace(0, capacity, 6)
        self.request_bins = np.linspace(0, capacity * 1.5, 6)
        self.budget_bins = np.linspace(0, 500.0, 5)
        self.hour_bins = [6, 12, 18, 24]

    def get_state_key(self, available_energy: float, pending_requests: float, 
                      budget_remaining: float, hour: int) -> Tuple[int, int, int, int]:
        e_bin = int(np.digitize(available_energy, self.energy_bins)) - 1
        r_bin = int(np.digitize(pending_requests, self.request_bins)) - 1
        b_bin = int(np.digitize(budget_remaining, self.budget_bins)) - 1
        
        h_bin = 0
        for idx, limit in enumerate(self.hour_bins):
            if hour < limit:
                h_bin = idx
                break
                
        e_bin = max(0, min(e_bin, 4))
        r_bin = max(0, min(r_bin, 4))
        b_bin = max(0, min(b_bin, 3))
        h_bin = max(0, min(h_bin, 3))
        return (e_bin, r_bin, b_bin, h_bin)

class QLearningAgent:
    """
    Tabular Q-learning agent to select microgrid energy allocation policies.
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
        self.q_table: Dict[str, List[float]] = {}

    def _get_q_values(self, state_key: Tuple[int, int, int, int]) -> List[float]:
        key_str = str(state_key)
        if key_str not in self.q_table:
            self.q_table[key_str] = [0.0] * self.action_size
        return self.q_table[key_str]

    def choose_action(self, state_key: Tuple[int, int, int, int], explore: bool = True) -> int:
        if explore and np.random.rand() < self.epsilon:
            return int(np.random.randint(self.action_size))
            
        q_vals = self._get_q_values(state_key)
        return int(np.argmax(q_vals))

    def update(
        self,
        state_key: Tuple[int, int, int, int],
        action: int,
        reward: float,
        next_state_key: Tuple[int, int, int, int]
    ) -> float:
        q_vals = self._get_q_values(state_key)
        next_q_vals = self._get_q_values(next_state_key)
        
        old_val = q_vals[action]
        best_next_q = max(next_q_vals)
        new_val = old_val + self.alpha * (reward + self.gamma * best_next_q - old_val)
        
        self.q_table[str(state_key)][action] = float(new_val)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return float(abs(new_val - old_val))

    def save_policy(self, file_path: str) -> None:
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

    def load_policy(self, file_path: str) -> None:
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                self.q_table = data.get("q_table", {})
                self.epsilon = data.get("epsilon", self.epsilon)
                self.alpha = data.get("alpha", self.alpha)
                self.gamma = data.get("gamma", self.gamma)
            except Exception as e:
                logger.error(f"Failed to load RL policy: {e}")
