#!/usr/bin/env python3
"""
Test Runner for Autonomous Delivery Agent

Validates the implementation and ensures all components work correctly.
"""

import os
import sys
import json
import traceback
from delivery_agent import GridWorld, DeliveryAgent
from grid_generator import GridGenerator
import math

class TestRunner:
    """Runs validation tests for the delivery agent system."""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
    
    def run_test(self, test_name, test_func):
        """Run a single test and track results."""
        try:
            print(f"Running {test_name}... ", end="")
            test_func()
            print("PASSED")
            self.passed += 1
        except Exception as e:
            print(f"FAILED: {e}")
            self.failed += 1
            self.errors.append((test_name, str(e), traceback.format_exc()))
    
    def test_grid_generation(self):
        """Test grid generation functionality."""
        generator = GridGenerator()
        
        # Test small grid generation
        small_grid = generator.generate_small_grid()
        assert small_grid['width'] == 8
        assert small_grid['height'] == 8
        assert len(small_grid['grid']) == 8
        assert len(small_grid['grid'][0]) == 8
        assert small_grid['start'] == [0, 0]
        assert small_grid['goal'] == [7, 7]
        
        # Test medium grid generation
        medium_grid = generator.generate_medium_grid()
        assert medium_grid['width'] == 15
        assert medium_grid['height'] == 15
        
        # Test large grid generation
        large_grid = generator.generate_large_grid()
        assert large_grid['width'] == 25
        assert large_grid['height'] == 25
    
    def test_grid_world_creation(self):
        """Test GridWorld initialization."""
        # Test default grid creation
        grid_world = GridWorld(width=5, height=5)
        assert grid_world.width == 5
        assert grid_world.height == 5
        assert grid_world.start == (0, 0)
        assert grid_world.goal == (4, 4)
        
        # Test cost retrieval
        assert grid_world.get_cost((0, 0)) >= 1
        assert grid_world.get_cost((-1, 0)) == float('inf')  # Out of bounds
        
        # Test position validity
        assert grid_world.is_valid_position((0, 0))
        assert grid_world.is_valid_position((4, 4))
        assert not grid_world.is_valid_position((-1, 0))
        assert not grid_world.is_valid_position((5, 5))
    
    def test_bfs_algorithm(self):
        """Test BFS algorithm implementation."""
        grid_world = GridWorld(width=5, height=5)
        agent = DeliveryAgent(grid_world)
        
        path = agent.bfs()
        
        # Should find a path
        assert len(path) > 0
        assert path[0] == grid_world.start
        assert path[-1] == grid_world.goal
        
        # Check path validity
        for i in range(len(path) - 1):
            current = path[i]
            next_pos = path[i + 1]
            # Check that consecutive positions are adjacent
            distance = abs(current[0] - next_pos[0]) + abs(current[1] - next_pos[1])
            assert distance == 1
    
    def test_ucs_algorithm(self):
        """Test UCS algorithm implementation."""
        grid_world = GridWorld(width=5, height=5)
        agent = DeliveryAgent(grid_world)
        
        path = agent.uniform_cost_search()
        
        # Should find a path
        assert len(path) > 0
        assert path[0] == grid_world.start
        assert path[-1] == grid_world.goal
        
        # Should have calculated total cost
        assert agent.total_cost > 0
    
    def test_astar_algorithm(self):
        """Test A* algorithm implementation."""
        grid_world = GridWorld(width=5, height=5)
        agent = DeliveryAgent(grid_world)
        
        # Test Manhattan heuristic
        path_manhattan = agent.a_star('manhattan')
        assert len(path_manhattan) > 0
        assert path_manhattan[0] == grid_world.start
        assert path_manhattan[-1] == grid_world.goal
        
        # Test Euclidean heuristic
        agent.reset_stats()
        path_euclidean = agent.a_star('euclidean')
        assert len(path_euclidean) > 0
        assert path_euclidean[0] == grid_world.start
        assert path_euclidean[-1] == grid_world.goal
    
    def test_hill_climbing_algorithm(self):
        """Test Hill Climbing algorithm implementation."""
        grid_world = GridWorld(width=5, height=5)
        agent = DeliveryAgent(grid_world)
        
        path = agent.hill_climbing(max_restarts=3)
        
        # Hill climbing might not always find a solution
        # but should at least run without errors
        assert isinstance(path, list)
        if path:  # If path found
            assert path[0] == grid_world.start
    
    def test_heuristic_functions(self):
        """Test heuristic function implementations."""
        grid_world = GridWorld(width=5, height=5)
        agent = DeliveryAgent(grid_world)
        
        # Test Manhattan distance
        manhattan_dist = agent.manhattan_distance((0, 0), (3, 4))
        assert manhattan_dist == 7  # |0-3| + |0-4| = 7
        
        # Test Euclidean distance
        euclidean_dist = agent.euclidean_distance((0, 0), (3, 4))
        expected_euc = math.sqrt(3**2 + 4**2)  # Should be 5
        assert abs(euclidean_dist - expected_euc) < 0.001
    
    def test_dynamic_replanning(self):
        """Test dynamic replanning functionality."""
        # Create a simple grid
        grid_data = {
            'width': 5,
            'height': 5,
            'grid': [[1 for _ in range(5)] for _ in range(5)],
            'start': [0, 0],
            'goal': [4, 4],
            'static_obstacles': [],
            'moving_obstacles': []
        }
        
        # Save temporary grid
        with open('test_grid.json', 'w') as f:
            json.dump(grid_data, f)
        
        grid_world = GridWorld(grid_file='test_grid.json')
        agent = DeliveryAgent(grid_world)
        
        # Test that dynamic replanning runs without errors
        path = agent.dynamic_replan('astar')
        assert isinstance(path, list)
        
        # Clean up
        if os.path.exists('test_grid.json'):
            os.remove('test_grid.json')
    
    def test_moving_obstacles(self):
        """Test moving obstacle functionality."""
        grid_data = {
            'width': 5,
            'height': 5,
            'grid': [[1 for _ in range(5)] for _ in range(5)],
            'start': [0, 0],
            'goal': [4, 4],
            'static_obstacles': [],
            'moving_obstacles': [
                {
                    'id': 'test_vehicle',
                    'schedule': [[1, 1], [1, 2], [1, 3]]
                }
            ]
        }
        
        with open('test_moving_grid.json', 'w') as f:
            json.dump(grid_data, f)
        
        grid_world = GridWorld(grid_file='test_moving_grid.json')
        
        # Test obstacle positions at different times
        assert not grid_world.is_valid_position((1, 1), time=0)
        assert not grid_world.is_valid_position((1, 2), time=1)
        assert not grid_world.is_valid_position((1, 3), time=2)
        assert grid_world.is_valid_position((1, 1), time=3)  # Obstacle moved away
        
        # Clean up
        if os.path.exists('test_moving_grid.json'):
            os.remove('test_moving_grid.json')
    
    def test_grid_file_loading(self):
        """Test grid loading from file."""
        grid_data = {
            'width': 3,
            'height': 3,
            'grid': [[1, 1, 1], [1, 2, 1], [1, 1, 1]],
            'start': [0, 0],
            'goal': [2, 2],
            'static_obstacles': [[1, 1]],
            'moving_obstacles': []
        }
        
        with open('test_load_grid.json', 'w') as f:
            json.dump(grid_data, f)
        
        grid_world = GridWorld(grid_file='test_load_grid.json')
        
        assert grid_world.width == 3
        assert grid_world.height == 3
        assert grid_world.start == (0, 0)
        assert grid_world.goal == (2, 2)
        assert (1, 1) in grid_world.static_obstacles
        assert grid_world.get_cost((1, 1)) == 2
        
        # Clean up
        if os.path.exists('test_load_grid.json'):
            os.remove('test_load_grid.json')
    
    def run_all_tests(self):
        """Run all test cases."""
        print("=" * 60)
        print("Running Autonomous Delivery Agent Test Suite")
        print("=" * 60)
        
        tests = [
            ("Grid Generation", self.test_grid_generation),
            ("Grid World Creation", self.test_grid_world_creation),
            ("BFS Algorithm", self.test_bfs_algorithm),
            ("UCS Algorithm", self.test_ucs_algorithm),
            ("A* Algorithm", self.test_astar_algorithm),
            ("Hill Climbing Algorithm", self.test_hill_climbing_algorithm),
            ("Heuristic Functions", self.test_heuristic_functions),
            ("Dynamic Replanning", self.test_dynamic_replanning),
            ("Moving Obstacles", self.test_moving_obstacles),
            ("Grid File Loading", self.test_grid_file_loading)
        ]
        
        for test_name, test_func in tests:
            self.run_test(test_name, test_func)
        
        # Print summary
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        print(f"Passed: {self.passed}")
        print(f"Failed: {self.failed}")
        print(f"Total:  {self.passed + self.failed}")
        
        if self.failed > 0:
            print(f"\nErrors encountered:")
            for test_name, error, traceback_info in self.errors:
                print(f"\n{test_name}:")
                print(f"  Error: {error}")
            return 1
        else:
            print("\nAll tests passed successfully!")
            return 0

def main():
    """Main test runner function."""
    runner = TestRunner()
    return runner.run_all_tests()

if __name__ == "__main__":
    sys.exit(main())
