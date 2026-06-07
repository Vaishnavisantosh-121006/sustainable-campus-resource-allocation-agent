from typing import Any, Optional
import os
import joblib
import pandas as pd
from loguru import logger
from .base_agent import BaseAgent
from .communication import CampusEnvironment
from ..config import HostelAgentSettings

class HostelAgent(BaseAgent):
    """
    Hostel Agent representing residential halls.
    Monitors resident demand, models peak hours, requests resources, and tracks efficiency.
    """
    def __init__(self, agent_id: str, env: CampusEnvironment, settings: HostelAgentSettings, model_dir: Optional[str] = None):
        super().__init__(agent_id, env)
        self.settings: HostelAgentSettings = settings
        self.model_dir: Optional[str] = model_dir
        self.model: Optional[Any] = None
        self.current_request: float = 0.0
        self.current_satisfaction: float = 1.0
        self.efficiency_score: float = 1.0
        
        self.load_prediction_model()

    def load_prediction_model(self) -> None:
        if self.model_dir:
            model_path = os.path.join(self.model_dir, "hostel_demand_model.joblib")
            if os.path.exists(model_path):
                try:
                    self.model = joblib.load(model_path)
                    logger.info(f"HostelAgent loaded ML model from {model_path}")
                except Exception as e:
                    logger.warning(f"HostelAgent failed to load ML model: {e}")

    def make_request(self) -> float:
        state = self.env.state
        hour = state.get("hour", 0)
        day = state.get("day", 0)
        occupancy = state.get("occupancy", {}).get(self.agent_id, 0)
        temperature = state.get("temperature", 25.0)
        is_holiday = state.get("is_holiday", False)
        is_exam_period = state.get("is_exam_period", False)

        # 1. Prediction using ML model if available
        if self.model is not None:
            try:
                prev_demand = self.request_history[-1] if self.request_history else 200.0
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
                self.current_request = max(20.0, pred)
                logger.debug(f"HostelAgent ML model predicted request: {self.current_request:.2f} kWh")
            except Exception as e:
                logger.warning(f"HostelAgent ML prediction failed, falling back to formula: {e}")
                self.current_request = self._calculate_formula_demand(occupancy, temperature, is_holiday, is_exam_period, hour)
        else:
            self.current_request = self._calculate_formula_demand(occupancy, temperature, is_holiday, is_exam_period, hour)

        # Send request message to EnergyManagerAgent
        self.send_message(
            recipient_id="energy_manager",
            message_type="REQUEST",
            content={"requested_energy": self.current_request}
        )
        
        self.request_history.append(self.current_request)
        return self.current_request

    def _calculate_formula_demand(self, occupancy: int, temperature: float, is_holiday: bool, is_exam_period: bool, hour: int) -> float:
        # Peak hours: Morning (6-9) and Evening (17-23)
        is_peak_hour = (6 <= hour < 9) or (17 <= hour < 23)
        peak_mult = 1.8 if is_peak_hour else 0.7
        
        # Holiday effect: residents stay in the hostels during daytime
        holiday_mult = 1.3 if is_holiday else 1.0
        
        # Exam period effect: study rooms active, late night usage (23-2)
        exam_mult = 1.15 if is_exam_period else 1.0
        if is_exam_period and (23 <= hour or hour < 2):
            exam_mult = 1.5
            
        # Temperature effect on HVAC/heaters
        temp_delta = abs(temperature - 21.0)
        temp_mult = 1.0 + (temp_delta * 0.08)

        resident_load = occupancy * self.settings.base_power_per_resident_kw * peak_mult * holiday_mult * exam_mult * temp_mult
        appliance_load = self.settings.appliance_power_kw * (1.2 if is_peak_hour else 0.5)
        
        return resident_load + appliance_load

    def update_state(self, allocation: float) -> float:
        self.allocation_history.append(allocation)
        self.current_satisfaction = self.calculate_satisfaction(allocation, self.current_request)
        self.satisfaction_history.append(self.current_satisfaction)
        
        # Calculate Hostel Efficiency Score
        # If allocation perfectly matches request, efficiency is 1.0
        # If allocation is higher than request, it is waste. If it is lower, it is undersupplied (efficient, but at cost of satisfaction)
        if allocation <= 0:
            self.efficiency_score = 0.0
        elif allocation > self.current_request:
            waste = allocation - self.current_request
            self.efficiency_score = max(0.5, 1.0 - (waste / self.current_request))
        else:
            # Undersupply: highly efficient in energy conservation, but penalize slightly if extremely low
            self.efficiency_score = min(1.0, allocation / self.current_request + 0.1)
            
        self.send_message(
            recipient_id="energy_manager",
            message_type="CONFIRM",
            content={
                "allocated_energy": allocation,
                "satisfaction": self.current_satisfaction,
                "efficiency": self.efficiency_score
            }
        )
        
        return self.current_satisfaction
