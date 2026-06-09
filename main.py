"""
Main entrypoint for the Smart Grid Energy Allocation AI Agent.
Runs the grid simulation.
"""

import os
import argparse
from src.environment import Environment
from src.agent import WasteCollectionAgent  # Class name remains WasteCollectionAgent for compatibility
from src.game_theory import GameTheoryAllocator
from src.utils import print_payoff_matrix_summary


def main():
    """
    Main execution flow.
    """
    parser = argparse.ArgumentParser(
        description="Smart Grid Campus Energy Allocation AI Agent (Game Theory Simulator)"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=os.path.join("data", "zones.csv"),
        help="Path to the building zones CSV data file"
    )
    parser.add_argument(
        "--capacity",
        type=int,
        default=3,
        help="Total electrical MW grid capacity available"
    )
    parser.add_argument(
        "--show-matrix",
        action="store_true",
        default=True,
        help="Prints a summary of the grid payoff matrix"
    )
    
    args = parser.parse_args()
    
    try:
        env = Environment(data_filepath=args.data, total_capacity=args.capacity)
    except FileNotFoundError as e:
        print(f"Error loading environment: {e}")
        print("Please ensure you are running the script from the project root directory.")
        return

    agent = WasteCollectionAgent(env)
    agent.run()
    
    if args.show_matrix:
        print()
        allocator = GameTheoryAllocator(env.get_all_zones(), env.total_trucks)
        print_payoff_matrix_summary(env.get_all_zones(), allocator)


if __name__ == "__main__":
    main()
