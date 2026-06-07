import numpy as np
from typing import List, Tuple, Dict, Any
from loguru import logger

def find_pure_nash_equilibria(payoff_matrix: np.ndarray) -> List[Tuple[int, int, int]]:
    """
    Scans the payoff matrix to locate all pure-strategy Nash Equilibria.
    
    payoff_matrix is shape (3, 3, 3, 3).
    """
    equilibria = []
    
    for c in range(3):
        for l in range(3):
            for h in range(3):
                current_payoffs = payoff_matrix[c, l, h, :]
                
                # Check if Classroom (agent 0) has incentive to deviate
                best_c_payoff = max(payoff_matrix[sc, l, h, 0] for sc in range(3))
                if current_payoffs[0] < best_c_payoff - 1e-5:
                    continue
                    
                # Check if Laboratory (agent 1) has incentive to deviate
                best_l_payoff = max(payoff_matrix[c, sl, h, 1] for sl in range(3))
                if current_payoffs[1] < best_l_payoff - 1e-5:
                    continue
                    
                # Check if Hostel (agent 2) has incentive to deviate
                best_h_payoff = max(payoff_matrix[c, l, sh, 2] for sh in range(3))
                if current_payoffs[2] < best_h_payoff - 1e-5:
                    continue
                
                # If no one wants to deviate, it is a Nash Equilibrium!
                equilibria.append((c, l, h))
                
    return equilibria

def solve_game(payoff_matrix: np.ndarray) -> Dict[str, Any]:
    """
    Solves the allocation game and returns the recommended strategy profile.
    If multiple pure Nash Equilibria exist, selects the one maximizing social welfare.
    If no pure Nash Equilibria exist, returns the profile maximizing social welfare.
    """
    equilibria = find_pure_nash_equilibria(payoff_matrix)
    strategy_names = ["Conservative (0.7x)", "Normal (1.0x)", "Aggressive (1.3x)"]
    
    if equilibria:
        logger.info(f"Found {len(equilibria)} pure Nash Equilibria: {equilibria}")
        # Select the one with maximum social welfare (sum of utilities)
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
        logger.info("No pure Nash Equilibria found. Finding maximum social welfare outcome.")
        # Find maximum social welfare profile
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
