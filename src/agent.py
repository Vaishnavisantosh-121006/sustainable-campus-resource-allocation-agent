"""
AI Agent module for the Smart Grid Energy Allocation project.
Implements the WasteCollectionAgent class, structuring decision-making into
Perception, Reasoning, Decision-making, and Action phases.
"""

from typing import Dict, List
from src.environment import Environment
from src.game_theory import GameTheoryAllocator
from src.models import Zone


class WasteCollectionAgent:
    """
    An Intelligent Smart Grid Energy Allocation Agent.
    
    Structure:
        1. Perception: Observe building loads and priorities.
        2. Reasoning: Calculate base utility curves representing power urgency.
        3. Decision-making: Run game-theoretic solver to find Nash Equilibrium.
        4. Action: Deploy power capacity to selected building zones.
    """

    def __init__(self, environment: Environment):
        """
        Initializes the WasteCollectionAgent.
        
        Args:
            environment (Environment): The smart grid environment the agent operates in.
        """
        self.environment = environment
        
        self.perceived_zones: List[Zone] = []
        self.calculated_utilities: Dict[str, int] = {}
        self.available_trucks: int = 0  # Matches available capacity
        self.allocation_plan: Dict[str, int] = {}

    def perceive(self) -> None:
        """
        Perception Stage:
        Observes the environment to read building zone load demand and grid availability.
        """
        print("[PERCEPTION]\n")
        print("Reading building zone energy data...\n")
        
        self.perceived_zones = self.environment.get_all_zones()
        self.available_trucks = self.environment.total_trucks
        
        for zone in self.perceived_zones:
            print(f"Zone {zone.zone_id} -> Load Demand: {zone.demand_level}%")
        print()

    def reason(self) -> None:
        """
        Reasoning Stage:
        Analyzes the perceived data, calculating base utilities for each building zone
        representing grid urgency.
        """
        print("[REASONING]\n")
        print("Calculating electrical utilities...\n")
        
        self.calculated_utilities = {}
        for zone in self.perceived_zones:
            utility_score = int(round(zone.utility()))
            self.calculated_utilities[zone.zone_id] = utility_score
            print(f"Zone {zone.zone_id} Utility = {utility_score}")
        print()

    def decide(self) -> List[str]:
        """
        Decision Making Stage:
        Solves the energy allocation game by finding the Pure Strategy Nash Equilibrium (PSNE).
        
        Returns:
            List[str]: Ordered list of selected zone IDs.
        """
        print("[DECISION MAKING]\n")
        print(f"Available Grid Capacity: {self.available_trucks} MW\n")
        
        allocator = GameTheoryAllocator(self.perceived_zones, self.available_trucks)
        equilibrium_profile = allocator.find_nash_equilibrium()
        self.allocation_plan = allocator.allocate_resources(equilibrium_profile)
        
        allocated_zones = [
            z_id for z_id, count in self.allocation_plan.items() if count > 0
        ]
        
        # Sort by utility descending
        allocated_zones.sort(key=lambda z_id: self.calculated_utilities[z_id], reverse=True)
        
        print("Selected Zones for Power Distribution:\n")
        for i, z_id in enumerate(allocated_zones, start=1):
            print(f"{i}. Zone {z_id}")
        print()
        
        return allocated_zones

    def act(self, selected_zones: List[str]) -> None:
        """
        Action Stage:
        Generates and executes the final power allocation plan.
        
        Args:
            selected_zones (List[str]): Ordered list of selected zone IDs.
        """
        print("[ACTION]\n")
        print("Grid Power Allocation Plan Generated\n")
        
        mw_num = 1
        for z_id in selected_zones:
            allocated_count = self.allocation_plan.get(z_id, 0)
            for _ in range(allocated_count):
                if mw_num <= self.available_trucks:
                    print(f"1 MW Block {mw_num} -> Zone {z_id}")
                    mw_num += 1
                    
        print("\nPower Allocation Complete")

    def run(self) -> None:
        """
        Runs the complete agent lifecycle (Perceive -> Reason -> Decide -> Act).
        """
        print("=" * 50)
        print("SMART GRID ENERGY RESOURCE ALLOCATION AI AGENT")
        print("=" * 45 + "\n")
        
        self.perceive()
        self.reason()
        selected = self.decide()
        self.act(selected)
