"""
Environment module for the Smart Grid Energy Allocation AI Agent.
Manages the grid state of building zones and the total electrical capacity available.
"""

import csv
import os
from typing import List
from src.models import Zone


class Environment:
    """
    Represents the Smart Grid environment managing building zones and total power capacity.
    
    Responsibilities:
        - Loading building zones from a CSV database.
        - Managing total grid electrical capacity bounds (default: 3 MW).
    """

    def __init__(self, data_filepath: str = None, total_capacity: int = 3):
        """
        Initializes the Smart Grid Environment.
        
        Args:
            data_filepath (str, optional): Path to the CSV containing load data.
            total_capacity (int, optional): Total electrical capacity in MW available for allocation. Defaults to 3.
        """
        self.zones: List[Zone] = []
        self.total_trucks = total_capacity  # Named total_trucks for direct semantic equivalence with core agent tests
        
        if data_filepath:
            self.load_zones(data_filepath)

    def load_zones(self, filepath: str) -> List[Zone]:
        """
        Loads building zone data from a CSV file.
        
        Args:
            filepath (str): The path to the CSV file.
            
        Returns:
            List[Zone]: The list of loaded zones.
            
        Raises:
            FileNotFoundError: If the CSV file does not exist.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Zone data file not found at: {filepath}")

        self.zones = []
        with open(filepath, mode="r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                zone = Zone(
                    zone_id=row["zone_id"].strip(),
                    population=int(row["population"]),
                    demand_level=int(row["demand_level"]),
                    priority=int(row["priority"])
                )
                self.zones.append(zone)
                
        return self.zones

    def get_zone_by_id(self, zone_id: str) -> Zone:
        """
        Retrieves a building Zone instance by its unique ID.
        
        Args:
            zone_id (str): The ID of the zone to find.
            
        Returns:
            Zone: The matching Zone instance, or None if not found.
        """
        for zone in self.zones:
            if zone.zone_id == zone_id:
                return zone
        return None

    def get_all_zones(self) -> List[Zone]:
        """
        Returns all registered building zones in the environment.
        
        Returns:
            List[Zone]: List of all Zone objects.
        """
        return self.zones

    def update_zone_demand(self, zone_id: str, new_demand_level: int) -> bool:
        """
        Updates the load demand level of a specific building zone.
        
        Args:
            zone_id (str): The target zone ID.
            new_demand_level (int): The new demand percentage level.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        zone = self.get_zone_by_id(zone_id)
        if zone:
            zone.demand_level = max(0, min(100, new_demand_level))
            return True
        return False

    def __repr__(self) -> str:
        """
        String representation of the Environment.
        """
        return f"Environment(total_zones={len(self.zones)}, available_capacity={self.total_trucks} MW)"
