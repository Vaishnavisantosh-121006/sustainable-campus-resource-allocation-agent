from typing import Dict, Any, List, Optional
from loguru import logger
from .base_agent import BaseAgent
from .communication import CampusEnvironment, Message
from ..config import GridSettings, AgentWeights

class EnergyManagerAgent(BaseAgent):
    """
    Energy Manager Agent acting as the grid operator and allocator.
    Receives energy requests, evaluates utilities, allocates energy, and tracks sustainability metrics.
    """
    def __init__(self, agent_id: str, env: CampusEnvironment, grid_settings: GridSettings, weights: AgentWeights):
        # We pass self as env since it registers. But EnergyManager is also registered.
        super().__init__(agent_id, env)
        self.grid_settings: GridSettings = grid_settings
        self.weights: AgentWeights = weights
        self.current_requests: Dict[str, float] = {}
        self.current_allocations: Dict[str, float] = {}
        self.agent_metadata: Dict[str, Dict[str, Any]] = {}
        
    def make_request(self) -> float:
        # Energy Manager does not request energy
        return 0.0

    def receive_requests(self) -> None:
        """
        Collects all REQUEST messages from the inbox.
        """
        self.current_requests.clear()
        self.agent_metadata.clear()
        
        for msg in self.inbox:
            if msg.message_type == "REQUEST":
                req_energy = msg.content.get("requested_energy", 0.0)
                self.current_requests[msg.sender_id] = req_energy
                self.agent_metadata[msg.sender_id] = msg.content
        
        # Clear inbox after reading requests for this step
        self.clear_inbox()
        logger.info(f"EnergyManager received requests: {self.current_requests}")

    def allocate_resources(self, policy: str = "equal_proportion", custom_allocations: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Allocates limited capacity among agents using different policies.
        """
        capacity = self.grid_settings.total_capacity_kwh
        total_requested = sum(self.current_requests.values())
        
        # Enforce budget/capacity constraints
        if custom_allocations is not None:
            # Policy decided externally (e.g., Nash Equilibrium or RL Agent)
            self.current_allocations = custom_allocations.copy()
            # Double check and scale down if it exceeds grid capacity
            total_allocated = sum(self.current_allocations.values())
            if total_allocated > capacity:
                logger.warning(f"Custom allocation ({total_allocated:.2f} kWh) exceeds capacity ({capacity:.2f} kWh). Scaling down.")
                scale = capacity / total_allocated
                for k in self.current_allocations:
                    self.current_allocations[k] *= scale
        else:
            if policy == "equal_proportion":
                if total_requested <= capacity:
                    # Allocate fully
                    self.current_allocations = self.current_requests.copy()
                else:
                    # Scale down proportionally
                    scale = capacity / total_requested
                    for agent_id, req in self.current_requests.items():
                        self.current_allocations[agent_id] = req * scale
            elif policy == "priority":
                # Priority: Laboratory > Classroom > Hostel
                priority_order = ["laboratory", "classroom", "hostel"]
                remaining_capacity = capacity
                self.current_allocations = {aid: 0.0 for aid in self.current_requests}
                
                # Assign according to priority order
                for p_id in priority_order:
                    # Match agents whose ID contains the priority string
                    matched_agents = [aid for aid in self.current_requests if p_id in aid.lower()]
                    for aid in matched_agents:
                        req = self.current_requests[aid]
                        alloc = min(req, remaining_capacity)
                        self.current_allocations[aid] = alloc
                        remaining_capacity -= alloc
            else:
                raise ValueError(f"Unknown allocation policy: {policy}")

        # Send ALLOCATION decisions back to agents
        for agent_id, alloc in self.current_allocations.items():
            self.send_message(
                recipient_id=agent_id,
                message_type="ALLOCATION",
                content={"allocated_energy": alloc}
            )
            # Record in shared environment
            req = self.current_requests.get(agent_id, 0.0)
            # satisfaction will be updated later when agents CONFIRM
            self.env.record_allocation(agent_id, req, alloc, 0.0)
            
        return self.current_allocations

    def update_state(self, allocation: float) -> float:
        # Self-update (not used as manager doesn't receive allocation)
        return 1.0

    def collect_confirms(self) -> Dict[str, Any]:
        """
        Collects confirmations and computes sustainability metrics for this step.
        """
        satisfactions = {}
        efficiencies = {}
        
        for msg in self.inbox:
            if msg.message_type == "CONFIRM":
                sender = msg.sender_id
                satisfactions[sender] = msg.content.get("satisfaction", 1.0)
                if "efficiency" in msg.content:
                    efficiencies[sender] = msg.content.get("efficiency", 1.0)
                
                # Update satisfaction in the environment record
                if sender in self.env.state["allocations"]:
                    # We can access environment directly
                    pass
        
        self.clear_inbox()
        
        # Calculate sustainability metrics
        total_allocated = sum(self.current_allocations.values())
        total_requested = sum(self.current_requests.values())
        
        # Energy saved compared to total requested demand (curtailment savings)
        energy_saved = max(0.0, total_requested - total_allocated)
        
        # Green energy consumed
        green_consumed = total_allocated * self.grid_settings.green_energy_fraction
        
        # Carbon reduction: CO2 saved from curtailed demand plus green energy offset
        carbon_avoided_kg = (energy_saved + green_consumed) * self.grid_settings.co2_kg_per_kwh
        
        # Grid price rate (normal vs peak)
        hour = self.env.state.get("hour", 0)
        is_peak_hour = (6 <= hour < 9) or (17 <= hour < 23)
        price_rate = self.grid_settings.peak_price_per_kwh if is_peak_hour else self.grid_settings.normal_price_per_kwh
        
        # Total electricity cost
        total_cost = total_allocated * price_rate
        # Cost savings from curtailment
        cost_saved = energy_saved * price_rate
        
        # Average satisfaction
        avg_sat = sum(satisfactions.values()) / len(satisfactions) if satisfactions else 1.0
        
        # Average efficiency
        avg_eff = sum(efficiencies.values()) / len(efficiencies) if efficiencies else 1.0
        
        # Overall sustainability score (weighted sum of satisfaction, efficiency, and carbon savings percentage)
        # sustainability score ranges between 0.0 and 1.0
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
        
        # Commit this step's history to environment
        self.env.commit_step(metrics)
        logger.info(f"Step {self.env.current_step} completed. Sustainability Score: {sustainability_score:.3f}")
        return metrics
