import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from loguru import logger
from ..config import AppConfig
from ..agents.communication import CampusEnvironment
from ..agents.classroom_agent import ClassroomAgent
from ..agents.laboratory_agent import LaboratoryAgent
from ..agents.hostel_agent import HostelAgent
from ..agents.energy_manager_agent import EnergyManagerAgent
from ..game_theory.payoff_matrix import generate_payoff_matrix
from ..game_theory.nash_equilibrium import solve_game
from ..ai.q_learning import QLearningAgent, Discretizer

class CampusSimulation:
    """
    Orchestrates the multi-agent campus simulation.
    Runs a simulation loop using a specific allocation policy:
    - equal_proportion: Standard proportional grid sharing.
    - priority: Prioritizes Laboratory > Classroom > Hostel.
    - game_theoretic: Allocates according to Nash Equilibrium.
    - rl: Reinforcement Learning policy selector.
    """
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
        
        # Initialize communication environment
        self.env = CampusEnvironment(
            total_capacity_kwh=self.config.grid.total_capacity_kwh,
            weights=self.config.weights
        )
        
        # Register agents
        self.classroom = ClassroomAgent("classroom", self.env, self.config.classroom, model_dir)
        self.laboratory = LaboratoryAgent("laboratory", self.env, self.config.laboratory, model_dir)
        self.hostel = HostelAgent("hostel", self.env, self.config.hostel, model_dir)
        self.manager = EnergyManagerAgent("energy_manager", self.env, self.config.grid, self.config.weights)
        
        # Initialize RL Agent if active
        self.rl_agent = None
        self.discretizer = None
        self.daily_budget_usd = 4000.0  # daily microgrid budget
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
        """
        Generates realistic environmental values for the given step.
        """
        hour = step % 24
        day = step // 24
        day_of_year = (day % 365) + 1
        day_of_week = day % 7
        
        # Holidays
        is_break = (150 <= day_of_year <= 210) or (350 <= day_of_year <= 365)
        is_we = day_of_week >= 5
        is_holiday = is_break or is_we
        
        # Exam periods
        is_exam_period = (120 <= day_of_year <= 135) or (320 <= day_of_year <= 335)
        
        # Temperature
        base_temp = 20.0 + 10.0 * np.sin(2 * np.pi * (day_of_year - 100) / 365.0)
        daily_temp = 5.0 * np.sin(2 * np.pi * (hour - 8) / 24.0)
        temperature = base_temp + daily_temp
        
        # Occupancies
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
        """
        Runs the simulation for a given number of steps.
        """
        logger.info(f"Starting campus simulation of {steps} hours using policy '{self.policy}'")
        np.random.seed(self.config.simulation.seed)
        
        for step in range(steps):
            # Reset daily budget at midnight
            if step % 24 == 0:
                self.remaining_budget_usd = self.daily_budget_usd

            # 1. Update environmental conditions
            hour, day, occupancies, temp, is_hol, is_exam = self.generate_step_environment(step)
            self.env.set_step_state(step, hour, day, occupancies, temp, is_hol, is_exam)
            
            # 2. Agents sense state and submit requests to the manager
            req_c = self.classroom.make_request()
            req_l = self.laboratory.make_request()
            req_h = self.hostel.make_request()
            total_requested = req_c + req_l + req_h
            
            # Read requests in Energy Manager
            self.manager.receive_requests()
            
            # Determine allocation policy
            active_policy = self.policy
            rl_action = None
            rl_state_key = None
            
            if self.policy == "rl" and self.rl_agent and self.discretizer:
                # RL state components
                avail_energy = self.config.grid.total_capacity_kwh
                rl_state_key = self.discretizer.get_state_key(
                    available_energy=avail_energy,
                    pending_requests=total_requested,
                    budget_remaining=self.remaining_budget_usd,
                    hour=hour
                )
                # Select policy action
                rl_action = self.rl_agent.choose_action(rl_state_key, explore=True)
                if rl_action == 0:
                    active_policy = "equal_proportion"
                elif rl_action == 1:
                    active_policy = "priority"
                elif rl_action == 2:
                    active_policy = "game_theoretic"
                elif rl_action == 3:
                    active_policy = "eco_conservation"
            
            # 3. Perform allocation based on selected policy
            allocations = {}
            if active_policy == "game_theoretic":
                # Solve allocation as a Game Theory optimization
                payoff_matrix = generate_payoff_matrix(
                    classroom_nominal=req_c,
                    laboratory_nominal=req_l,
                    hostel_nominal=req_h,
                    grid_capacity=self.config.grid.total_capacity_kwh,
                    weights=self.config.weights,
                    grid_settings=self.config.grid
                )
                solution = solve_game(payoff_matrix)
                # Apply the Nash Equilibrium strategies (0.7x, 1.0x, or 1.3x)
                strategies = [0.7, 1.0, 1.3]
                strat_c, strat_l, strat_h = solution["strategy_profile"]
                
                # Apply strategies to compute game equilibrium requests
                eq_req_c = req_c * strategies[strat_c]
                eq_req_l = req_l * strategies[strat_l]
                eq_req_h = req_h * strategies[strat_h]
                eq_total = eq_req_c + eq_req_l + eq_req_h
                
                # Allocation must fit capacity
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
                
                # Manager executes custom allocations
                self.manager.allocate_resources(policy="custom", custom_allocations=allocations)
            elif active_policy == "eco_conservation":
                # Allocates strictly up to 80% capacity proportionally to save energy & costs
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
                # Standard manager policies (equal_proportion or priority)
                self.manager.allocate_resources(policy=active_policy)
            
            # 4. Agents receive allocations and calculate satisfaction
            alloc_c = self.manager.current_allocations.get("classroom", 0.0)
            alloc_l = self.manager.current_allocations.get("laboratory", 0.0)
            alloc_h = self.manager.current_allocations.get("hostel", 0.0)
            
            self.classroom.update_state(alloc_c)
            self.laboratory.update_state(alloc_l)
            self.hostel.update_state(alloc_h)
            
            # 5. Energy Manager collects confirmations, evaluates KPIs, and commits step
            self.manager.receive_requests()  # Read confirmations from inbox
            metrics = self.manager.collect_confirms()
            
            # Track budget depletion
            cost = metrics.get("total_cost_usd", 0.0)
            self.remaining_budget_usd = max(0.0, self.remaining_budget_usd - cost)
            
            # 6. Train RL Agent
            if self.policy == "rl" and self.rl_agent and self.discretizer and rl_state_key is not None and rl_action is not None:
                # Next step state components
                next_hour = (step + 1) % 24
                next_day = (step + 1) // 24
                _, _, next_occupancies, next_temp, next_is_hol, next_is_exam = self.generate_step_environment(step + 1)
                
                # Estimate next requests (simple baseline lookup)
                next_total_req = total_requested  # approximate next step request as current
                
                next_rl_state_key = self.discretizer.get_state_key(
                    available_energy=self.config.grid.total_capacity_kwh,
                    pending_requests=next_total_req,
                    budget_remaining=self.remaining_budget_usd,
                    hour=next_hour
                )
                
                # Reward Formulation
                sat = metrics["average_satisfaction"]
                sust = metrics["sustainability_score"]
                waste = metrics["total_energy_saved"]  # curtailment saving is rewarded for sustainability
                
                # Over-consumption penalty (if we exceeded budget)
                budget_penalty = 0.0
                if self.remaining_budget_usd <= 0:
                    budget_penalty = -2.0  # Heavy penalty for running out of budget early in the day
                    
                # We reward satisfaction and sustainability, while penalizing budget violation
                reward = (sat * 1.5) + (sust * 1.0) + budget_penalty
                
                # Q-learning update
                self.rl_agent.update(rl_state_key, rl_action, reward, next_rl_state_key)

        # Save policy if RL training was active
        if self.policy == "rl" and self.rl_agent and self.rl_policy_path:
            self.rl_agent.save_policy(self.rl_policy_path)
            
        return self.env.get_history_df()
