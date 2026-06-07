from typing import Any, Optional
import os
import joblib
import pandas as pd
from loguru import logger
from .base_agent import BaseAgent
from .communication import CampusEnvironment
from ..config import ClassroomAgentSettings

class ClassroomAgent(BaseAgent):
    """
    Classroom Agent representing school/college classrooms.
    Predicts demand, requests energy, tracks occupancy satisfaction, and classroom utilization.
    """
    def __init__(self, agent_id: str, env: CampusEnvironment, settings: ClassroomAgentSettings, model_dir: Optional[str] = None):
        super().__init__(agent_id, env)
        self.settings: ClassroomAgentSettings = settings
        self.model_dir: Optional[str] = model_dir
        self.model: Optional[Any] = None
        self.utilization: float = 0.0
        self.current_request: float = 0.0
        self.current_satisfaction: float = 1.0
        
        # Load model if directory is provided and model exists
        self.load_prediction_model()

    def load_prediction_model(self) -> None:
        if self.model_dir:
            model_path = os.path.join(self.model_dir, "classroom_demand_model.joblib")
            if os.path.exists(model_path):
                try:
                    self.model = joblib.load(model_path)
                    logger.info(f"ClassroomAgent loaded ML model from {model_path}")
                except Exception as e:
                    logger.warning(f"ClassroomAgent failed to load ML model: {e}")

    def make_request(self) -> float:
        # Get environmental state
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
                # Prepare features as expected by the trained model
                # Features: ['day_of_week', 'hour', 'occupancy', 'temperature', 'is_holiday', 'is_exam_period', 'prev_demand']
                # Let's get prev_demand or fallback to a default
                prev_demand = self.request_history[-1] if self.request_history else 100.0
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
                # Ensure demand is non-negative
                self.current_request = max(10.0, pred)
                logger.debug(f"ClassroomAgent ML model predicted request: {self.current_request:.2f} kWh")
            except Exception as e:
                logger.warning(f"ClassroomAgent ML prediction failed, falling back to formula: {e}")
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
        # Base HVAC load depends on temperature (comfort range is 21-24C)
        temp_delta = abs(temperature - 22.5)
        hvac_mult = 1.0 + (temp_delta * 0.1)  # HVAC draws 10% more power per degree deviation
        
        # Classroom schedule multiplier (typically active 8:00 to 18:00)
        is_active_hours = 8 <= hour < 18
        
        if is_holiday:
            # Minimal stand-by power
            return 5.0
        
        base_hvac = self.settings.hvac_power_kw * hvac_mult
        
        if is_active_hours:
            occupancy_load = occupancy * self.settings.base_power_per_person_kw
            schedule_factor = 1.2 if is_exam_period else 1.0
            return (base_hvac + occupancy_load) * schedule_factor
        else:
            # Night time stand-by power
            return 10.0

    def update_state(self, allocation: float) -> float:
        self.allocation_history.append(allocation)
        self.current_satisfaction = self.calculate_satisfaction(allocation, self.current_request)
        self.satisfaction_history.append(self.current_satisfaction)
        
        # Calculate classroom utilization: actual occupancy relative to base capacity
        max_capacity = self.settings.base_occupancy
        current_occ = self.env.state.get("occupancy", {}).get(self.agent_id, 0)
        self.utilization = min(1.0, current_occ / max_capacity) if max_capacity > 0 else 0.0

        # Send status update back to manager
        self.send_message(
            recipient_id="energy_manager",
            message_type="CONFIRM",
            content={
                "allocated_energy": allocation,
                "satisfaction": self.current_satisfaction,
                "utilization": self.utilization
            }
        )
        
        return self.current_satisfaction
