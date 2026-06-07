import numpy as np
from typing import Dict, Tuple, Any
from .utility import calculate_agent_utility
from ..config import AgentWeights, GridSettings

def generate_payoff_matrix(
    classroom_nominal: float,
    laboratory_nominal: float,
    hostel_nominal: float,
    grid_capacity: float,
    weights: AgentWeights,
    grid_settings: GridSettings
) -> np.ndarray:
    """
    Generates a 3D payoff matrix of shape (3, 3, 3, 3) for the three agents:
    Classroom (agent 0), Laboratory (agent 1), and Hostel (agent 2).
    
    Strategies:
    0: Conservative (70% of nominal demand)
    1: Normal (100% of nominal demand)
    2: Aggressive (130% of nominal demand)
    
    Returns:
    np.ndarray of shape (3, 3, 3, 3) where:
    payoff_matrix[s0, s1, s2, :] = [Classroom Utility, Laboratory Utility, Hostel Utility]
    """
    strategies = [0.7, 1.0, 1.3]
    payoff_matrix = np.zeros((3, 3, 3, 3))
    
    for s_c in range(3):
        req_c = classroom_nominal * strategies[s_c]
        for s_l in range(3):
            req_l = laboratory_nominal * strategies[s_l]
            for s_h in range(3):
                req_h = hostel_nominal * strategies[s_h]
                
                requests = [req_c, req_l, req_h]
                total_request = sum(requests)
                
                # Proportional Allocation logic
                allocations = [0.0, 0.0, 0.0]
                if total_request <= grid_capacity:
                    allocations = requests.copy()
                else:
                    scale = grid_capacity / total_request
                    allocations = [req * scale for req in requests]
                
                # Calculate prices
                # Assumes normal price for this matrix generation
                price_rate = grid_settings.normal_price_per_kwh
                
                # Compute utilities
                payoffs = []
                for i in range(3):
                    req = requests[i]
                    alloc = allocations[i]
                    satisfaction = alloc / req if req > 0 else 1.0
                    cost = alloc * price_rate
                    green_frac = grid_settings.green_energy_fraction
                    
                    utility = calculate_agent_utility(
                        satisfaction=satisfaction,
                        allocated_energy=alloc,
                        requested_energy=req,
                        cost=cost,
                        green_fraction=green_frac,
                        weights=weights
                    )
                    payoffs.append(utility)
                
                payoff_matrix[s_c, s_l, s_h, :] = payoffs
                
    return payoff_matrix
