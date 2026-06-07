from typing import Any
from ..config import AgentWeights

def calculate_agent_utility(
    satisfaction: float,
    allocated_energy: float,
    requested_energy: float,
    cost: float,
    green_fraction: float,
    weights: AgentWeights
) -> float:
    """
    Calculates agent utility based on satisfaction, sustainability, and cost.
    Formula: Utility = (w1 * satisfaction) + (w2 * sustainability) - (w3 * cost_penalty)
    """
    # 1. Satisfaction term (w1)
    u_sat = satisfaction

    # 2. Sustainability term (w2): higher is better.
    # Penalize wastage (allocating more than requested) or reward using green energy
    waste = max(0.0, allocated_energy - requested_energy)
    waste_penalty = waste / max(1.0, requested_energy)
    
    # Sustainability index is green fraction minus waste penalty, clamped to [0, 1]
    u_sust = max(0.0, min(1.0, green_fraction - waste_penalty + 0.5))

    # 3. Cost penalty term (w3): lower energy spending is preferred by the agent's budget
    # Normalize cost. Let's assume a typical max cost of $150 per agent per hour.
    max_estimated_cost = 150.0
    u_cost_penalty = min(1.0, cost / max_estimated_cost)

    # Calculate final utility
    utility = (weights.satisfaction * u_sat) + (weights.sustainability * u_sust) - (weights.cost * u_cost_penalty)
    
    # Clamp utility to a reasonable range
    return float(max(-1.0, min(1.0, utility)))
