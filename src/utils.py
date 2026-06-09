"""
Utils module for the Smart Grid Energy Allocation AI Agent.
Provides helpers for grid formatting and payoff matrix reporting.
"""

from typing import List
from src.models import Zone


def format_currency(val: float) -> str:
    """
    Formats a numeric value for payoffs.
    """
    return f"{val:+.1f}"


def print_payoff_matrix_summary(zones: List[Zone], allocator) -> None:
    """
    Prints a summarized version of the payoff matrix for key action profiles
    to help users visualize the game theory payoffs under grid competition.
    """
    print("-" * 60)
    print("GRID PAYOFF MATRIX SUMMARY (Example strategic decisions)")
    print("-" * 60)
    print(f"{'Profile (A, B, C, D, E)':<30} | {'Payoffs (A, B, C, D, E)':<30}")
    print("-" * 60)
    
    profiles_to_show = [
        {"desc": "All Delay (Store Local Energy)", "prof": {z.zone_id: "DELAY" for z in zones}},
        {"desc": "Nash Equilibrium (A,C,E Request)", "prof": {"A": "REQUEST", "B": "DELAY", "C": "REQUEST", "D": "DELAY", "E": "REQUEST"}},
        {"desc": "All Request (Surge/Blackout)", "prof": {z.zone_id: "REQUEST" for z in zones}}
    ]

    for item in profiles_to_show:
        prof = item["prof"]
        payoffs = allocator.calculate_payoffs(prof)
        
        prof_str = ",".join([prof[z.zone_id][0] for z in zones])
        payoffs_str = ",".join([f"{int(round(payoffs[z.zone_id]))}" for z in zones])
        
        print(f"{item['desc']:<30} | ({prof_str}) -> Payoffs: ({payoffs_str})")
    print("-" * 60 + "\n")
