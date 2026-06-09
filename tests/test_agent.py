"""
Unit tests for the Smart Grid Energy Allocation AI Agent project.
Verifies the utility calculations, payoff modeling, game-theoretic solver, and agent phases.
"""

import unittest
from src.models import Zone
from src.environment import Environment
from src.game_theory import GameTheoryAllocator
from src.agent import WasteCollectionAgent


class TestSmartGridAllocation(unittest.TestCase):
    """
    Test suite for resource allocation, payoffs, and agent lifecycle in the smart grid.
    """

    def setUp(self):
        """
        Set up shared test data before each test.
        """
        self.zones = [
            Zone(zone_id="A", population=1200, demand_level=85, priority=5),
            Zone(zone_id="B", population=900, demand_level=40, priority=2),
            Zone(zone_id="C", population=1500, demand_level=95, priority=5),
            Zone(zone_id="D", population=700, demand_level=60, priority=3),
            Zone(zone_id="E", population=1100, demand_level=75, priority=4)
        ]
        self.total_capacity = 3

    def test_zone_utility_calculations(self):
        """
        Verify that zone utility scores match the expected values precisely:
        Zone A -> 437
        Zone B -> 170
        Zone C -> 490
        Zone D -> 250
        Zone E -> 395
        """
        expected_utilities = {
            "A": 437.0,
            "B": 170.0,
            "C": 490.0,
            "D": 250.0,
            "E": 395.0
        }
        
        for zone in self.zones:
            calculated = zone.utility()
            expected = expected_utilities[zone.zone_id]
            self.assertEqual(
                calculated, 
                expected, 
                f"Zone {zone.zone_id} utility expected {expected}, got {calculated}"
            )

    def test_environment_capacity_availability(self):
        """
        Verifies environment successfully registers available grid capacity.
        """
        env = Environment(total_capacity=self.total_capacity)
        env.zones = self.zones
        self.assertEqual(env.total_trucks, 3)
        self.assertEqual(len(env.get_all_zones()), 5)

    def test_game_theory_allocator_demands(self):
        """
        Test the resource allocator allocation under full capacity vs over-capacity.
        """
        allocator = GameTheoryAllocator(self.zones, total_capacity=self.total_capacity)
        
        profile_under = {
            "A": "REQUEST",
            "B": "DELAY",
            "C": "REQUEST",
            "D": "DELAY",
            "E": "DELAY"
        }
        allocations_under = allocator.allocate_resources(profile_under)
        self.assertEqual(allocations_under["A"], 1)
        self.assertEqual(allocations_under["C"], 1)
        self.assertEqual(allocations_under["E"], 0)
        self.assertEqual(sum(allocations_under.values()), 2)
        
        profile_over = {
            "A": "REQUEST",
            "B": "REQUEST",
            "C": "REQUEST",
            "D": "DELAY",
            "E": "REQUEST"
        }
        allocations_over = allocator.allocate_resources(profile_over)
        self.assertEqual(allocations_over["C"], 1)
        self.assertEqual(allocations_over["A"], 1)
        self.assertEqual(allocations_over["E"], 1)
        self.assertEqual(allocations_over["B"], 0)
        self.assertEqual(sum(allocations_over.values()), 3)

    def test_nash_equilibrium_solver(self):
        """
        Verify that the Pure Strategy Nash Equilibrium (PSNE) leads to the 
        expected equilibrium profile (A, C, E playing REQUEST, B, D play DELAY).
        """
        allocator = GameTheoryAllocator(self.zones, total_capacity=self.total_capacity)
        eq_profile = allocator.find_nash_equilibrium()
        
        self.assertEqual(eq_profile["A"], "REQUEST")
        self.assertEqual(eq_profile["B"], "DELAY")
        self.assertEqual(eq_profile["C"], "REQUEST")
        self.assertEqual(eq_profile["D"], "DELAY")
        self.assertEqual(eq_profile["E"], "REQUEST")

    def test_agent_lifecycle_stages(self):
        """
        Ensures the Agent goes through the stages (perception, reasoning, decision) correctly.
        """
        env = Environment(total_capacity=self.total_capacity)
        env.zones = self.zones
        
        agent = WasteCollectionAgent(env)
        
        agent.perceive()
        self.assertEqual(len(agent.perceived_zones), 5)
        self.assertEqual(agent.available_trucks, 3)
        
        agent.reason()
        self.assertEqual(agent.calculated_utilities["A"], 437)
        self.assertEqual(agent.calculated_utilities["C"], 490)
        
        selected_zones = agent.decide()
        self.assertEqual(selected_zones, ["C", "A", "E"])
        
        agent.act(selected_zones)
        self.assertEqual(agent.allocation_plan["C"], 1)
        self.assertEqual(agent.allocation_plan["A"], 1)
        self.assertEqual(agent.allocation_plan["E"], 1)
        self.assertEqual(agent.allocation_plan["B"], 0)


if __name__ == "__main__":
    unittest.main()
