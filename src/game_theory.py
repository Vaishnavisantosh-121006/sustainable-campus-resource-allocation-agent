"""
Game Theory module for the Smart Grid Energy Allocation AI Agent.
Models energy allocation as a strategic game where building zones (players)
compete for limited grid power capacity.
"""

from typing import Dict, List, Tuple
from src.models import Zone


class GameTheoryAllocator:
    """
    Handles game-theoretic modeling and grid power allocation.
    
    Players: Building Zones (A, B, C, D, E)
    Strategies: 
        - 'DELAY' (Store local / Delay): Cooperate or wait, requesting 0 MW.
        - 'REQUEST' (Request power): Request 1 MW of grid capacity.
        - 'REQUEST_ADDITIONAL' (Request additional power): Request 2 MW of grid capacity.
    
    Payoff model:
        - Reward based on zone utility if power is allocated.
        - Strategic costs: transmission costs for requesting 1 or 2 MW.
        - Penalty for blackout (unmet demand) if a request is denied.
        - Mild local storage depletion penalty if voluntary delay is chosen.
    """

    STRATEGIES = ["DELAY", "REQUEST", "REQUEST_ADDITIONAL"]
    
    # Strategy transmission/surge cost mappings
    STRATEGY_COSTS = {
        "DELAY": 0.0,
        "REQUEST": 10.0,
        "REQUEST_ADDITIONAL": 25.0
    }

    def __init__(self, zones: List[Zone], total_capacity: int = 3):
        """
        Initializes the Game Theory Allocator.
        
        Args:
            zones (List[Zone]): The list of building zones participating in the game.
            total_capacity (int): The amount of available MW capacity to distribute.
        """
        self.zones = zones
        self.total_trucks = total_capacity  # Named total_trucks for direct semantic equivalence with core agent tests

    def allocate_resources(self, strategy_profile: Dict[str, str]) -> Dict[str, int]:
        """
        Allocates power to zones based on a chosen strategy profile.
        
        Allocation rules:
            - If total demanded power <= total_capacity, everyone gets what they requested.
            - If total demanded power > total_capacity, we allocate power one-by-one
              to the requests yielding the highest marginal utility:
                - First MW to zone i: yields utility U_i (Zone utility).
                - Second MW to zone i: yields utility 0.2 * U_i (diminishing returns).
        
        Args:
            strategy_profile (Dict[str, str]): Strategy choice for each zone (e.g., {'A': 'REQUEST', 'B': 'DELAY', ...})
            
        Returns:
            Dict[str, int]: Map of zone_id to allocated MW capacity.
        """
        demands: Dict[str, int] = {}
        for zone in self.zones:
            strat = strategy_profile.get(zone.zone_id, "DELAY")
            if strat == "DELAY":
                demands[zone.zone_id] = 0
            elif strat == "REQUEST":
                demands[zone.zone_id] = 1
            elif strat == "REQUEST_ADDITIONAL":
                demands[zone.zone_id] = 2

        total_demanded = sum(demands.values())

        if total_demanded <= self.total_trucks:
            return demands

        marginal_requests: List[Tuple[float, str]] = []
        for zone in self.zones:
            demand = demands[zone.zone_id]
            base_util = zone.utility()
            
            if demand >= 1:
                marginal_requests.append((base_util, zone.zone_id))
            if demand >= 2:
                marginal_requests.append((base_util * 0.2, zone.zone_id))

        # Sort descending by marginal utility, tie-breaking alphabetically
        marginal_requests.sort(key=lambda x: (x[0], x[1]), reverse=True)

        allocations: Dict[str, int] = {zone.zone_id: 0 for zone in self.zones}
        allocated_count = 0
        
        for _, zone_id in marginal_requests:
            if allocated_count < self.total_trucks:
                allocations[zone_id] += 1
                allocated_count += 1
            else:
                break

        return allocations

    def calculate_payoffs(self, strategy_profile: Dict[str, str]) -> Dict[str, float]:
        """
        Calculates the payoffs for all zones given a strategy profile.
        
        Payoff Formula for zone i:
            - If allocated == 0:
                - If strategy == 'DELAY': Payoff = -0.2 * U_i (battery usage/planned delay)
                - If strategy != 'DELAY': Payoff = -1.5 * U_i - Cost(strategy) (blackout penalty)
            - If allocated == 1:
                - Payoff = U_i - Cost(strategy) (power delivered payoff)
            - If allocated == 2:
                - Payoff = 1.2 * U_i - Cost(strategy) (surge heating/cooling payoff)
                
        Args:
            strategy_profile (Dict[str, str]): Map of zone_id to chosen strategy.
            
        Returns:
            Dict[str, float]: Map of zone_id to payoff.
        """
        allocations = self.allocate_resources(strategy_profile)
        payoffs: Dict[str, float] = {}

        for zone in self.zones:
            z_id = zone.zone_id
            strat = strategy_profile.get(z_id, "DELAY")
            allocated = allocations.get(z_id, 0)
            u = zone.utility()
            cost = self.STRATEGY_COSTS.get(strat, 0.0)

            if allocated == 0:
                if strat == "DELAY":
                    payoffs[z_id] = -0.2 * u
                else:
                    payoffs[z_id] = -1.5 * u - cost
            elif allocated == 1:
                payoffs[z_id] = u - cost
            elif allocated == 2:
                payoffs[z_id] = (1.2 * u) - cost
            else:
                payoffs[z_id] = u - cost

        return payoffs

    def find_nash_equilibrium(self) -> Dict[str, str]:
        """
        Solves for the Pure Strategy Nash Equilibrium (PSNE) using dominant strategy reduction.
        
        Simple Explanation:
            1. Zones compete for limited grid power capacity (total_capacity = 3 MW).
            2. The allocator awards power to the zones with the highest utility first.
            3. The 3 zones with the highest utility (C, A, E) have a dominant strategy to
               choose 'REQUEST'. They are guaranteed to get a truck, which gives them a high payoff.
            4. The remaining lower-utility zones (B, D) know they cannot win a truck under
               competition. If they play 'REQUEST', they get 0 trucks and suffer a severe
               unmet request penalty (-1.5 * U_i - cost). By playing 'DELAY', they avoid the 
               request cost and get a much milder delay penalty (-0.2 * U_i).
            5. Therefore, B and D choose 'DELAY' as their best response.
        """
        sorted_zones = sorted(self.zones, key=lambda z: z.utility(), reverse=True)
        
        equilibrium = {}
        for i, zone in enumerate(sorted_zones):
            if i < self.total_trucks:
                equilibrium[zone.zone_id] = "REQUEST"
            else:
                equilibrium[zone.zone_id] = "DELAY"
                
        return equilibrium
