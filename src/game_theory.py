import numpy as np
from typing import Dict, List, Tuple, Any
from loguru import logger
from src.config import AgentWeights, GridSettings

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
    u_sat = satisfaction

    waste = max(0.0, allocated_energy - requested_energy)
    waste_penalty = waste / max(1.0, requested_energy)
    u_sust = max(0.0, min(1.0, green_fraction - waste_penalty + 0.5))

    max_estimated_cost = 150.0
    u_cost_penalty = min(1.0, cost / max_estimated_cost)

    utility = (weights.satisfaction * u_sat) + (weights.sustainability * u_sust) - (weights.cost * u_cost_penalty)
    return float(max(-1.0, min(1.0, utility)))


def generate_payoff_matrix(
    classroom_nominal: float,
    laboratory_nominal: float,
    hostel_nominal: float,
    grid_capacity: float,
    weights: AgentWeights,
    grid_settings: GridSettings
) -> np.ndarray:
    """
    Generates a 3D payoff matrix of shape (3, 3, 3, 3) for:
    Classroom (agent 0), Laboratory (agent 1), and Hostel (agent 2).
    
    Strategies:
    0: Conservative (70% of nominal demand)
    1: Normal (100% of nominal demand)
    2: Aggressive (130% of nominal demand)
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
                
                allocations = [0.0, 0.0, 0.0]
                if total_request <= grid_capacity:
                    allocations = requests.copy()
                else:
                    scale = grid_capacity / total_request
                    allocations = [req * scale for req in requests]
                
                price_rate = grid_settings.normal_price_per_kwh
                
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


def find_pure_nash_equilibria(payoff_matrix: np.ndarray) -> List[Tuple[int, int, int]]:
    equilibria = []
    
    for c in range(3):
        for l in range(3):
            for h in range(3):
                current_payoffs = payoff_matrix[c, l, h, :]
                
                best_c_payoff = max(payoff_matrix[sc, l, h, 0] for sc in range(3))
                if current_payoffs[0] < best_c_payoff - 1e-5:
                    continue
                    
                best_l_payoff = max(payoff_matrix[c, sl, h, 1] for sl in range(3))
                if current_payoffs[1] < best_l_payoff - 1e-5:
                    continue
                    
                best_h_payoff = max(payoff_matrix[c, l, sh, 2] for sh in range(3))
                if current_payoffs[2] < best_h_payoff - 1e-5:
                    continue
                
                equilibria.append((c, l, h))
    return equilibria


def solve_game(payoff_matrix: np.ndarray) -> Dict[str, Any]:
    equilibria = find_pure_nash_equilibria(payoff_matrix)
    strategy_names = ["Conservative (0.7x)", "Normal (1.0x)", "Aggressive (1.3x)"]
    
    if equilibria:
        best_eq = equilibria[0]
        best_welfare = -100.0
        for eq in equilibria:
            welfare = float(np.sum(payoff_matrix[eq[0], eq[1], eq[2], :]))
            if welfare > best_welfare:
                best_welfare = welfare
                best_eq = eq
        
        selected_strategy = best_eq
        outcome_type = "Nash Equilibrium"
    else:
        best_profile = (1, 1, 1)
        best_welfare = -100.0
        for c in range(3):
            for l in range(3):
                for h in range(3):
                    welfare = float(np.sum(payoff_matrix[c, l, h, :]))
                    if welfare > best_welfare:
                        best_welfare = welfare
                        best_profile = (c, l, h)
                        
        selected_strategy = best_profile
        outcome_type = "Max Social Welfare (Fallback)"

    c_idx, l_idx, h_idx = selected_strategy
    payoffs = payoff_matrix[c_idx, l_idx, h_idx, :]
    
    return {
        "strategy_profile": selected_strategy,
        "strategy_names": {
            "classroom": strategy_names[c_idx],
            "laboratory": strategy_names[l_idx],
            "hostel": strategy_names[h_idx]
        },
        "payoffs": {
            "classroom": float(payoffs[0]),
            "laboratory": float(payoffs[1]),
            "hostel": float(payoffs[2])
        },
        "social_welfare": float(np.sum(payoffs)),
        "outcome_type": outcome_type,
        "all_equilibria": equilibria
    }
