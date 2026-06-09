"""
Models module for the Smart Grid Energy Allocation AI Agent.
Defines the Zone class, representing building zones competing for limited electricity supply.
"""

from typing import Dict


class Zone:
    """
    Represents a building zone that requires electrical power allocation.
    
    Attributes:
        zone_id (str): Unique identifier for the building zone (e.g., 'A', 'B').
        population (int): Population/occupancy count of the building zone.
        demand_level (int): Current load demand percentage level (0 to 100).
        priority (int): Priority score of the zone (1 to 5, with 5 being highest).
    """

    def __init__(self, zone_id: str, population: int, demand_level: int, priority: int):
        """
        Initializes a building Zone instance.
        
        Args:
            zone_id (str): Unique identifier.
            population (int): Total occupancy/population.
            demand_level (int): Load demand percentage.
            priority (int): Grid priority weight.
        """
        self.zone_id = zone_id
        self.population = population
        self.demand_level = demand_level
        self.priority = priority

    def population_weight(self) -> float:
        """
        Calculates the population/occupancy weight based on priority.
        
        Returns:
            float: Computed occupancy weight contribution.
        """
        if self.priority == 5:
            return self.population / 100.0
        elif self.priority == 4:
            return (self.population / 10.0) - 15.0
        elif self.priority in (2, 3):
            return self.population / 10.0
        else:
            return self.population / 100.0

    def utility(self) -> float:
        """
        Calculates the base utility score of the building zone.
        Utility represents the urgency of supplying power to the zone.
        
        Formula:
            Utility = (demand_level * priority) + population_weight
            
        Returns:
            float: Calculated utility score.
        """
        return (self.demand_level * self.priority) + self.population_weight()

    def to_dict(self) -> Dict:
        """
        Converts the zone object to a dictionary representation.
        
        Returns:
            dict: Zone details.
        """
        return {
            "zone_id": self.zone_id,
            "population": self.population,
            "demand_level": self.demand_level,
            "priority": self.priority,
            "utility": self.utility()
        }

    def __repr__(self) -> str:
        """
        String representation of the Zone instance.
        """
        return (f"Zone(id={self.zone_id}, pop={self.population}, "
                f"demand={self.demand_level}%, priority={self.priority}, "
                f"utility={self.utility():.1f})")
