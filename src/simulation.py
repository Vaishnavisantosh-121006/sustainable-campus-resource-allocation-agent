import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from loguru import logger

from src.config import AppConfig
from src.agents import CampusEnvironment, ClassroomAgent, LaboratoryAgent, HostelAgent, EnergyManagerAgent
from src.game_theory import generate_payoff_matrix, solve_game
from src.ai import QLearningAgent, Discretizer

# ----------------------------------------------------
# Cumulative Metrics Compilation Helper
# ----------------------------------------------------
def calculate_cumulative_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {}

    total_consumed_kwh = df["total_energy_consumed"].sum()
    total_saved_kwh = df["total_energy_saved"].sum()
    total_carbon_saved_kg = df["carbon_reduction_kg"].sum()
    total_cost_usd = df["total_cost_usd"].sum()
    total_cost_saved_usd = df["cost_saved_usd"].sum()
    
    avg_satisfaction = df["average_satisfaction"].mean()
    avg_efficiency = df["average_efficiency"].mean()
    avg_sustainability_score = df["sustainability_score"].mean()

    total_consumed_mwh = total_consumed_kwh / 1000.0
    total_saved_mwh = total_saved_kwh / 1000.0
    total_carbon_saved_tons = total_carbon_saved_kg / 1000.0

    return {
        "total_energy_consumed_mwh": float(total_consumed_mwh),
        "total_energy_saved_mwh": float(total_saved_mwh),
        "total_carbon_saved_tons": float(total_carbon_saved_tons),
        "total_cost_usd": float(total_cost_usd),
        "total_cost_saved_usd": float(total_cost_saved_usd),
        "average_satisfaction": float(avg_satisfaction),
        "average_efficiency": float(avg_efficiency),
        "average_sustainability_score": float(avg_sustainability_score),
        "steps_simulated": len(df)
    }

# ----------------------------------------------------
# Campus Simulation Orchestration Loop
# ----------------------------------------------------
class CampusSimulation:
    def __init__(
        self,
        config: AppConfig,
        policy: str = "equal_proportion",
        model_dir: Optional[str] = None,
        rl_policy_path: Optional[str] = None
    ):
        self.config: AppConfig = config
        self.policy: str = policy
        self.model_dir: Optional[str] = model_dir
        self.rl_policy_path: Optional[str] = rl_policy_path
        
        self.env = CampusEnvironment(
            total_capacity_kwh=self.config.grid.total_capacity_kwh,
            weights=self.config.weights
        )
        
        self.classroom = ClassroomAgent("classroom", self.env, self.config.classroom, model_dir)
        self.laboratory = LaboratoryAgent("laboratory", self.env, self.config.laboratory, model_dir)
        self.hostel = HostelAgent("hostel", self.env, self.config.hostel, model_dir)
        self.manager = EnergyManagerAgent("energy_manager", self.env, self.config.grid, self.config.weights)
        
        self.rl_agent = None
        self.discretizer = None
        self.daily_budget_usd = 4000.0
        self.remaining_budget_usd = self.daily_budget_usd
        
        if self.policy == "rl":
            self.rl_agent = QLearningAgent(
                learning_rate=self.config.rl.learning_rate,
                discount_factor=self.config.rl.discount_factor,
                epsilon=self.config.rl.epsilon,
                epsilon_decay=self.config.rl.epsilon_decay,
                epsilon_min=self.config.rl.epsilon_min
            )
            self.discretizer = Discretizer(self.config.grid.total_capacity_kwh)
            if self.rl_policy_path:
                self.rl_agent.load_policy(self.rl_policy_path)

    def generate_step_environment(self, step: int) -> Tuple[int, int, Dict[str, int], float, bool, bool]:
        hour = step % 24
        day = step // 24
        day_of_year = (day % 365) + 1
        day_of_week = day % 7
        
        is_break = (150 <= day_of_year <= 210) or (350 <= day_of_year <= 365)
        is_we = day_of_week >= 5
        is_holiday = is_break or is_we
        
        is_exam_period = (120 <= day_of_year <= 135) or (320 <= day_of_year <= 335)
        
        base_temp = 20.0 + 10.0 * np.sin(2 * np.pi * (day_of_year - 100) / 365.0)
        daily_temp = 5.0 * np.sin(2 * np.pi * (hour - 8) / 24.0)
        temperature = base_temp + daily_temp
        
        occupancies = {}
        if is_holiday:
            occupancies["classroom"] = int(np.random.poisson(10))
            occupancies["laboratory"] = int(np.random.poisson(15))
            occupancies["hostel"] = int(np.random.poisson(1000))
        else:
            if 8 <= hour < 17:
                occupancies["classroom"] = int(np.random.poisson(280)) if not is_exam_period else int(np.random.poisson(320))
                occupancies["laboratory"] = int(np.random.poisson(40))
                occupancies["hostel"] = int(np.random.poisson(200))
            elif 17 <= hour < 22:
                occupancies["classroom"] = int(np.random.poisson(50))
                occupancies["laboratory"] = int(np.random.poisson(35))
                occupancies["hostel"] = int(np.random.poisson(800))
            else:
                occupancies["classroom"] = int(np.random.poisson(5))
                occupancies["laboratory"] = int(np.random.poisson(10))
                occupancies["hostel"] = int(np.random.poisson(1100))
                
        return hour, day, occupancies, temperature, is_holiday, is_exam_period

    def run_simulation(self, steps: int) -> pd.DataFrame:
        logger.info(f"Starting campus simulation of {steps} hours using policy '{self.policy}'")
        np.random.seed(self.config.simulation.seed)
        
        for step in range(steps):
            if step % 24 == 0:
                self.remaining_budget_usd = self.daily_budget_usd

            hour, day, occupancies, temp, is_hol, is_exam = self.generate_step_environment(step)
            self.env.set_step_state(step, hour, day, occupancies, temp, is_hol, is_exam)
            
            req_c = self.classroom.make_request()
            req_l = self.laboratory.make_request()
            req_h = self.hostel.make_request()
            total_requested = req_c + req_l + req_h
            
            self.manager.receive_requests()
            
            active_policy = self.policy
            rl_action = None
            rl_state_key = None
            
            if self.policy == "rl" and self.rl_agent and self.discretizer:
                avail_energy = self.config.grid.total_capacity_kwh
                rl_state_key = self.discretizer.get_state_key(
                    available_energy=avail_energy,
                    pending_requests=total_requested,
                    budget_remaining=self.remaining_budget_usd,
                    hour=hour
                )
                rl_action = self.rl_agent.choose_action(rl_state_key, explore=True)
                if rl_action == 0:
                    active_policy = "equal_proportion"
                elif rl_action == 1:
                    active_policy = "priority"
                elif rl_action == 2:
                    active_policy = "game_theoretic"
                elif rl_action == 3:
                    active_policy = "eco_conservation"
            
            allocations = {}
            if active_policy == "game_theoretic":
                payoff_matrix = generate_payoff_matrix(
                    classroom_nominal=req_c,
                    laboratory_nominal=req_l,
                    hostel_nominal=req_h,
                    grid_capacity=self.config.grid.total_capacity_kwh,
                    weights=self.config.weights,
                    grid_settings=self.config.grid
                )
                solution = solve_game(payoff_matrix)
                strategies = [0.7, 1.0, 1.3]
                strat_c, strat_l, strat_h = solution["strategy_profile"]
                
                eq_req_c = req_c * strategies[strat_c]
                eq_req_l = req_l * strategies[strat_l]
                eq_req_h = req_h * strategies[strat_h]
                eq_total = eq_req_c + eq_req_l + eq_req_h
                
                capacity = self.config.grid.total_capacity_kwh
                if eq_total <= capacity:
                    allocations = {
                        "classroom": eq_req_c,
                        "laboratory": eq_req_l,
                        "hostel": eq_req_h
                    }
                else:
                    scale = capacity / eq_total
                    allocations = {
                        "classroom": eq_req_c * scale,
                        "laboratory": eq_req_l * scale,
                        "hostel": eq_req_h * scale
                    }
                self.manager.allocate_resources(policy="custom", custom_allocations=allocations)
            elif active_policy == "eco_conservation":
                eco_capacity = self.config.grid.total_capacity_kwh * 0.8
                allocations = {}
                if total_requested <= eco_capacity:
                    allocations = {
                        "classroom": req_c,
                        "laboratory": req_l,
                        "hostel": req_h
                    }
                else:
                    scale = eco_capacity / total_requested
                    allocations = {
                        "classroom": req_c * scale,
                        "laboratory": req_l * scale,
                        "hostel": req_h * scale
                    }
                self.manager.allocate_resources(policy="custom", custom_allocations=allocations)
            else:
                self.manager.allocate_resources(policy=active_policy)
            
            alloc_c = self.manager.current_allocations.get("classroom", 0.0)
            alloc_l = self.manager.current_allocations.get("laboratory", 0.0)
            alloc_h = self.manager.current_allocations.get("hostel", 0.0)
            
            self.classroom.update_state(alloc_c)
            self.laboratory.update_state(alloc_l)
            self.hostel.update_state(alloc_h)
            
            self.manager.receive_requests()
            metrics = self.manager.collect_confirms()
            
            cost = metrics.get("total_cost_usd", 0.0)
            self.remaining_budget_usd = max(0.0, self.remaining_budget_usd - cost)
            
            if self.policy == "rl" and self.rl_agent and self.discretizer and rl_state_key is not None and rl_action is not None:
                next_hour = (step + 1) % 24
                next_day = (step + 1) // 24
                _, _, next_occupancies, next_temp, next_is_hol, next_is_exam = self.generate_step_environment(step + 1)
                
                next_total_req = total_requested
                next_rl_state_key = self.discretizer.get_state_key(
                    available_energy=self.config.grid.total_capacity_kwh,
                    pending_requests=next_total_req,
                    budget_remaining=self.remaining_budget_usd,
                    hour=next_hour
                )
                
                sat = metrics["average_satisfaction"]
                sust = metrics["sustainability_score"]
                
                budget_penalty = 0.0
                if self.remaining_budget_usd <= 0:
                    budget_penalty = -2.0
                    
                reward = (sat * 1.5) + (sust * 1.0) + budget_penalty
                self.rl_agent.update(rl_state_key, rl_action, reward, next_rl_state_key)

        if self.policy == "rl" and self.rl_agent and self.rl_policy_path:
            self.rl_agent.save_policy(self.rl_policy_path)
            
        return self.env.get_history_df()
