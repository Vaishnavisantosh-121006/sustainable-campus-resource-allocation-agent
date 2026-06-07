from typing import Any, Optional, Dict
import os
import joblib
import pandas as pd
from loguru import logger
from .base_agent import BaseAgent
from .communication import CampusEnvironment
from ..config import LaboratoryAgentSettings

class LaboratoryAgent(BaseAgent):
    """
    Laboratory Agent representing research laboratories and computer centers.
    Estimates equipment demand, forecasts future usage, and optimizes resource requests.
    """
    def __init__(self, agent_id: str, env: CampusEnvironment, settings: LaboratoryAgentSettings, model_dir: Optional[str] = None):
        super().__init__(agent_id, env)
        self.settings: LaboratoryAgentSettings = settings
        self.model_dir: Optional[str] = model_dir
        self.model: Optional[Any] = None
        self.current_request: float = 0.0
        self.current_satisfaction: float = 1.0
        self.active_machines: int = 0
        
        self.load_prediction_model()

    def load_prediction_model(self) -> None:
        if self.model_dir:
            model_path = os.path.join(self.model_dir, "laboratory_demand_model.joblib")
            if os.path.exists(model_path):
                try:
                    self.model = joblib.load(model_path)
                    logger.info(f"LaboratoryAgent loaded ML model from {model_path}")
                except Exception as e:
                    logger.warning(f"LaboratoryAgent failed to load ML model: {e}")

    def make_request(self) -> float:
        state = self.env.state
        hour = state.get("hour", 0)
        day = state.get("day", 0)
        occupancy = state.get("occupancy", {}).get(self.agent_id, 0)
        temperature = state.get("temperature", 25.0)
        is_holiday = state.get("is_holiday", False)
        is_exam_period = state.get("is_exam_period", False)

        # Estimate active machines based on occupancy (each person uses ~1 machine, capped by settings.machine_count)
        self.active_machines = min(self.settings.machine_count, max(5, int(occupancy * 0.8)))
        if is_holiday:
            self.active_machines = 2  # minimal standby machinery/servers

        # 1. Prediction using ML model if available
        if self.model is not None:
            try:
                prev_demand = self.request_history[-1] if self.request_history else 80.0
                features = pd.DataFrame([{
                    "day_of_week": day % 7,
                    "hour": hour,
                    "occupancy": occupancy,
                    "temperature": temperature,
                    "is_holiday": 1 if is_holiday else 0,
                    "is_exam_period": 1 if is_exam_period else 0,
                    "prev_demand": prev_demand
                }])
                pred = float(self.model.predict(features)[0])
                self.current_request = max(15.0, pred)
                logger.debug(f"LaboratoryAgent ML model predicted request: {self.current_request:.2f} kWh")
            except Exception as e:
                logger.warning(f"LaboratoryAgent ML prediction failed, falling back to formula: {e}")
                self.current_request = self._calculate_formula_demand(is_holiday, is_exam_period, hour)
        else:
            self.current_request = self._calculate_formula_demand(is_holiday, is_exam_period, hour)

        # Forecast future usage: simple 3-hour moving average forecast
        forecast = self.forecast_future_usage()

        # Send request message to EnergyManagerAgent
        self.send_message(
            recipient_id="energy_manager",
            message_type="REQUEST",
            content={
                "requested_energy": self.current_request,
                "forecast_next_hour": forecast,
                "active_machines": self.active_machines
            }
        )
        
        self.request_history.append(self.current_request)
        return self.current_request

    def _calculate_formula_demand(self, is_holiday: bool, is_exam_period: bool, hour: int) -> float:
        if is_holiday:
            return self.settings.baseload_kw  # servers running baseline
            
        machine_load = self.active_machines * self.settings.avg_power_per_machine_kw
        total_load = self.settings.baseload_kw + machine_load
        
        # Laboratories run 24/7 but have lower night usage
        if 8 <= hour < 22:
            multiplier = 1.3 if is_exam_period else 1.0
        else:
            multiplier = 0.6  # lower activity at night
            
        return total_load * multiplier

    def forecast_future_usage(self) -> float:
        """
        Simple forecast for the next hour based on historical usage.
        """
        if len(self.request_history) >= 3:
            return sum(self.request_history[-3:]) / 3.0
        elif len(self.request_history) > 0:
            return self.request_history[-1]
        return 75.0  # default baseline request forecast

    def update_state(self, allocation: float) -> float:
        self.allocation_history.append(allocation)
        self.current_satisfaction = self.calculate_satisfaction(allocation, self.current_request)
        self.satisfaction_history.append(self.current_satisfaction)
        
        # Send status update
        self.send_message(
            recipient_id="energy_manager",
            message_type="CONFIRM",
            content={
                "allocated_energy": allocation,
                "satisfaction": self.current_satisfaction,
                "active_machines": self.active_machines
            }
        )
        
        return self.current_satisfaction
