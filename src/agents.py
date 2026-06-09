import os
import joblib
import pandas as pd
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from loguru import logger

from src.config import ClassroomAgentSettings, LaboratoryAgentSettings, HostelAgentSettings, GridSettings, AgentWeights

# ----------------------------------------------------
# Message Data Model & Campus Environment Message Bus
# ----------------------------------------------------
class Message(BaseModel):
    sender_id: str
    recipient_id: str
    message_type: str  # e.g., "REQUEST", "ALLOCATION", "CONFIRM"
    content: Dict[str, Any]
    step: int

class CampusEnvironment:
    """
    Shared environment serving as a message bus and state coordinator for the agents.
    """
    def __init__(self, total_capacity_kwh: float, weights: Any):
        self.total_capacity_kwh: float = total_capacity_kwh
        self.weights = weights
        self.message_log: List[Message] = []
        self.current_step: int = 0
        self.agents: Dict[str, Any] = {}
        self.state: Dict[str, Any] = {
            "hour": 0,
            "day": 0,
            "occupancy": {},
            "temperature": 25.0,
            "is_holiday": False,
            "is_exam_period": False,
            "available_capacity": total_capacity_kwh,
            "allocations": {},
            "requests": {},
        }
        self.history: List[Dict[str, Any]] = []

    def register_agent(self, agent: Any) -> None:
        self.agents[agent.agent_id] = agent
        logger.info(f"Registered agent: {agent.agent_id}")

    def send_message(self, message: Message) -> None:
        self.message_log.append(message)
        logger.debug(f"[{message.step}] Message from {message.sender_id} to {message.recipient_id} | Type: {message.message_type}")
        
        # Route message if recipient exists
        if message.recipient_id in self.agents:
            self.agents[message.recipient_id].receive_message(message)
        elif message.recipient_id == "all":
            for agent_id, agent in self.agents.items():
                if agent_id != message.sender_id:
                    agent.receive_message(message)

    def set_step_state(self, step: int, hour: int, day: int, occupancy: Dict[str, int], 
                       temperature: float, is_holiday: bool, is_exam_period: bool) -> None:
        self.current_step = step
        self.state["hour"] = hour
        self.state["day"] = day
        self.state["occupancy"] = occupancy
        self.state["temperature"] = temperature
        self.state["is_holiday"] = is_holiday
        self.state["is_exam_period"] = is_exam_period
        self.state["available_capacity"] = self.total_capacity_kwh
        self.state["allocations"] = {}
        self.state["requests"] = {}

    def record_allocation(self, agent_id: str, requested: float, allocated: float, satisfaction: float) -> None:
        self.state["allocations"][agent_id] = allocated
        self.state["requests"][agent_id] = requested
        self.state["available_capacity"] -= allocated

    def commit_step(self, sustainability_metrics: Dict[str, Any]) -> None:
        record = {
            "step": self.current_step,
            "hour": self.state["hour"],
            "day": self.state["day"],
            "temperature": self.state["temperature"],
            "is_holiday": self.state["is_holiday"],
            "is_exam_period": self.state["is_exam_period"],
            "total_capacity": self.total_capacity_kwh,
            "available_capacity": self.state["available_capacity"],
            **{f"req_{k}": v for k, v in self.state["requests"].items()},
            **{f"alloc_{k}": v for k, v in self.state["allocations"].items()},
            **sustainability_metrics
        }
        self.history.append(record)

    def get_history_df(self) -> pd.DataFrame:
        if not self.history:
            return pd.DataFrame()
        return pd.DataFrame(self.history)

# ----------------------------------------------------
# Base and Specialized Agents
# ----------------------------------------------------
class BaseAgent(ABC):
    """
    Abstract base class for all campus agents.
    """
    def __init__(self, agent_id: str, env: CampusEnvironment):
        self.agent_id: str = agent_id
        self.env: CampusEnvironment = env
        self.inbox: List[Message] = []
        self.satisfaction_history: List[float] = []
        self.request_history: List[float] = []
        self.allocation_history: List[float] = []
        
        # Register self to environment
        self.env.register_agent(self)

    def receive_message(self, message: Message) -> None:
        self.inbox.append(message)
        logger.debug(f"Agent {self.agent_id} received message from {message.sender_id}")

    def send_message(self, recipient_id: str, message_type: str, content: Dict[str, Any]) -> None:
        msg = Message(
            sender_id=self.agent_id,
            recipient_id=recipient_id,
            message_type=message_type,
            content=content,
            step=self.env.current_step
        )
        self.env.send_message(msg)

    def clear_inbox(self) -> None:
        self.inbox.clear()

    def calculate_satisfaction(self, allocated: float, requested: float) -> float:
        if requested <= 0:
            return 1.0
        # Satisfaction is allocation ratio, clamped between 0.0 and 1.0
        sat = min(1.0, max(0.0, allocated / requested))
        # Add penalty for extreme under-allocation
        if sat < 0.5:
            sat = sat * 0.8  # disproportionate discomfort when getting less than half
        return sat

    @abstractmethod
    def make_request(self) -> float:
        """
        Calculates and returns the agent's energy demand request for the current step.
        """
        pass

    @abstractmethod
    def update_state(self, allocation: float) -> float:
        """
        Updates agent metrics based on the allocated energy and returns the satisfaction score.
        """
        pass


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
                self.current_request = max(10.0, pred)
            except Exception as e:
                logger.warning(f"ClassroomAgent ML prediction failed: {e}")
                self.current_request = self._calculate_formula_demand(occupancy, temperature, is_holiday, is_exam_period, hour)
        else:
            self.current_request = self._calculate_formula_demand(occupancy, temperature, is_holiday, is_exam_period, hour)

        # Send request message
        self.send_message(
            recipient_id="energy_manager",
            message_type="REQUEST",
            content={"requested_energy": self.current_request}
        )
        
        self.request_history.append(self.current_request)
        return self.current_request

    def _calculate_formula_demand(self, occupancy: int, temperature: float, is_holiday: bool, is_exam_period: bool, hour: int) -> float:
        temp_delta = abs(temperature - 22.5)
        hvac_mult = 1.0 + (temp_delta * 0.1)
        is_active_hours = 8 <= hour < 18
        
        if is_holiday:
            return 5.0
        
        base_hvac = self.settings.hvac_power_kw * hvac_mult
        if is_active_hours:
            occupancy_load = occupancy * self.settings.base_power_per_person_kw
            schedule_factor = 1.2 if is_exam_period else 1.0
            return (base_hvac + occupancy_load) * schedule_factor
        else:
            return 10.0

    def update_state(self, allocation: float) -> float:
        self.allocation_history.append(allocation)
        self.current_satisfaction = self.calculate_satisfaction(allocation, self.current_request)
        self.satisfaction_history.append(self.current_satisfaction)
        
        max_capacity = self.settings.base_occupancy
        current_occ = self.env.state.get("occupancy", {}).get(self.agent_id, 0)
        self.utilization = min(1.0, current_occ / max_capacity) if max_capacity > 0 else 0.0

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


class LaboratoryAgent(BaseAgent):
    """
    Laboratory Agent representing research laboratories.
    Estimates equipment demand, forecasts usage, and optimizes resource requests.
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

        self.active_machines = min(self.settings.machine_count, max(5, int(occupancy * 0.8)))
        if is_holiday:
            self.active_machines = 2

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
            except Exception as e:
                logger.warning(f"LaboratoryAgent ML prediction failed: {e}")
                self.current_request = self._calculate_formula_demand(is_holiday, is_exam_period, hour)
        else:
            self.current_request = self._calculate_formula_demand(is_holiday, is_exam_period, hour)

        forecast = self.forecast_future_usage()

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
            return self.settings.baseload_kw
            
        machine_load = self.active_machines * self.settings.avg_power_per_machine_kw
        total_load = self.settings.baseload_kw + machine_load
        
        if 8 <= hour < 22:
            multiplier = 1.3 if is_exam_period else 1.0
        else:
            multiplier = 0.6
        return total_load * multiplier

    def forecast_future_usage(self) -> float:
        if len(self.request_history) >= 3:
            return sum(self.request_history[-3:]) / 3.0
        elif len(self.request_history) > 0:
            return self.request_history[-1]
        return 75.0

    def update_state(self, allocation: float) -> float:
        self.allocation_history.append(allocation)
        self.current_satisfaction = self.calculate_satisfaction(allocation, self.current_request)
        self.satisfaction_history.append(self.current_satisfaction)
        
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


class HostelAgent(BaseAgent):
    """
    Hostel Agent representing student hostels.
    Monitors occupancy demands, models peak hours, and requests energy.
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
            except Exception as e:
                logger.warning(f"HostelAgent ML prediction failed: {e}")
                self.current_request = self._calculate_formula_demand(occupancy, temperature, is_holiday, is_exam_period, hour)
        else:
            self.current_request = self._calculate_formula_demand(occupancy, temperature, is_holiday, is_exam_period, hour)

        self.send_message(
            recipient_id="energy_manager",
            message_type="REQUEST",
            content={"requested_energy": self.current_request}
        )
        self.request_history.append(self.current_request)
        return self.current_request

    def _calculate_formula_demand(self, occupancy: int, temperature: float, is_holiday: bool, is_exam_period: bool, hour: int) -> float:
        is_peak_hour = (6 <= hour < 9) or (17 <= hour < 23)
        peak_mult = 1.8 if is_peak_hour else 0.7
        holiday_mult = 1.3 if is_holiday else 1.0
        
        exam_mult = 1.15 if is_exam_period else 1.0
        if is_exam_period and (23 <= hour or hour < 2):
            exam_mult = 1.5
            
        temp_delta = abs(temperature - 21.0)
        temp_mult = 1.0 + (temp_delta * 0.08)

        resident_load = occupancy * self.settings.base_power_per_resident_kw * peak_mult * holiday_mult * exam_mult * temp_mult
        appliance_load = self.settings.appliance_power_kw * (1.2 if is_peak_hour else 0.5)
        return resident_load + appliance_load

    def update_state(self, allocation: float) -> float:
        self.allocation_history.append(allocation)
        self.current_satisfaction = self.calculate_satisfaction(allocation, self.current_request)
        self.satisfaction_history.append(self.current_satisfaction)
        
        if allocation <= 0:
            self.efficiency_score = 0.0
        elif allocation > self.current_request:
            waste = allocation - self.current_request
            self.efficiency_score = max(0.5, 1.0 - (waste / self.current_request))
        else:
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


class EnergyManagerAgent(BaseAgent):
    """
    Energy Manager Agent acting as the grid operator and resource allocator.
    """
    def __init__(self, agent_id: str, env: CampusEnvironment, grid_settings: GridSettings, weights: AgentWeights):
        super().__init__(agent_id, env)
        self.grid_settings: GridSettings = grid_settings
        self.weights: AgentWeights = weights
        self.current_requests: Dict[str, float] = {}
        self.current_allocations: Dict[str, float] = {}
        self.agent_metadata: Dict[str, Dict[str, Any]] = {}
        
    def make_request(self) -> float:
        return 0.0

    def receive_requests(self) -> None:
        self.current_requests.clear()
        self.agent_metadata.clear()
        
        for msg in self.inbox:
            if msg.message_type == "REQUEST":
                req_energy = msg.content.get("requested_energy", 0.0)
                self.current_requests[msg.sender_id] = req_energy
                self.agent_metadata[msg.sender_id] = msg.content
        self.clear_inbox()

    def allocate_resources(self, policy: str = "equal_proportion", custom_allocations: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        capacity = self.grid_settings.total_capacity_kwh
        total_requested = sum(self.current_requests.values())
        
        if custom_allocations is not None:
            self.current_allocations = custom_allocations.copy()
            total_allocated = sum(self.current_allocations.values())
            if total_allocated > capacity:
                scale = capacity / total_allocated
                for k in self.current_allocations:
                    self.current_allocations[k] *= scale
        else:
            if policy == "equal_proportion":
                if total_requested <= capacity:
                    self.current_allocations = self.current_requests.copy()
                else:
                    scale = capacity / total_requested
                    for agent_id, req in self.current_requests.items():
                        self.current_allocations[agent_id] = req * scale
            elif policy == "priority":
                priority_order = ["laboratory", "classroom", "hostel"]
                remaining_capacity = capacity
                self.current_allocations = {aid: 0.0 for aid in self.current_requests}
                
                for p_id in priority_order:
                    matched_agents = [aid for aid in self.current_requests if p_id in aid.lower()]
                    for aid in matched_agents:
                        req = self.current_requests[aid]
                        alloc = min(req, remaining_capacity)
                        self.current_allocations[aid] = alloc
                        remaining_capacity -= alloc
            else:
                raise ValueError(f"Unknown allocation policy: {policy}")

        # Dispatch allocations
        for agent_id, alloc in self.current_allocations.items():
            self.send_message(
                recipient_id=agent_id,
                message_type="ALLOCATION",
                content={"allocated_energy": alloc}
            )
            req = self.current_requests.get(agent_id, 0.0)
            self.env.record_allocation(agent_id, req, alloc, 0.0)
            
        return self.current_allocations

    def update_state(self, allocation: float) -> float:
        return 1.0

    def collect_confirms(self) -> Dict[str, Any]:
        satisfactions = {}
        efficiencies = {}
        
        for msg in self.inbox:
            if msg.message_type == "CONFIRM":
                sender = msg.sender_id
                satisfactions[sender] = msg.content.get("satisfaction", 1.0)
                if "efficiency" in msg.content:
                    efficiencies[sender] = msg.content.get("efficiency", 1.0)
        self.clear_inbox()
        
        total_allocated = sum(self.current_allocations.values())
        total_requested = sum(self.current_requests.values())
        energy_saved = max(0.0, total_requested - total_allocated)
        
        green_consumed = total_allocated * self.grid_settings.green_energy_fraction
        carbon_avoided_kg = (energy_saved + green_consumed) * self.grid_settings.co2_kg_per_kwh
        
        hour = self.env.state.get("hour", 0)
        is_peak_hour = (6 <= hour < 9) or (17 <= hour < 23)
        price_rate = self.grid_settings.peak_price_per_kwh if is_peak_hour else self.grid_settings.normal_price_per_kwh
        
        total_cost = total_allocated * price_rate
        cost_saved = energy_saved * price_rate
        
        avg_sat = sum(satisfactions.values()) / len(satisfactions) if satisfactions else 1.0
        avg_eff = sum(efficiencies.values()) / len(efficiencies) if efficiencies else 1.0
        
        sustainability_score = (
            self.weights.satisfaction * avg_sat +
            self.weights.sustainability * avg_eff +
            self.weights.cost * (green_consumed / max(1.0, total_allocated))
        )
        
        metrics = {
            "total_energy_consumed": total_allocated,
            "total_energy_saved": energy_saved,
            "carbon_reduction_kg": carbon_avoided_kg,
            "total_cost_usd": total_cost,
            "cost_saved_usd": cost_saved,
            "average_satisfaction": avg_sat,
            "average_efficiency": avg_eff,
            "sustainability_score": sustainability_score,
        }
        
        self.env.commit_step(metrics)
        return metrics
