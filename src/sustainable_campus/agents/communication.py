from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from loguru import logger
import pandas as pd

class Message(BaseModel):
    sender_id: str
    recipient_id: str
    message_type: str  # e.g., "REQUEST", "ALLOCATION", "NEGOTIATE", "CONFIRM"
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
