from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from loguru import logger
from .communication import CampusEnvironment, Message

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
