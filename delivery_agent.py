#!/usr/bin/env python3
"""
CSA2001 - Fundamentals of AI and ML
Project 1: Autonomous Delivery Agent

Main implementation of the delivery agent with multiple pathfinding algorithms.
Author: Abhishek Shandilya
Date: 22/09/2025
"""

import heapq
import random
import math
import time
from collections import deque
from typing import List, Tuple, Dict, Optional, Set
import json
import argparse

class GridWorld:
    """Represents the 2D grid environment for the delivery agent."""
    
    def __init__(self, grid_file: str = None, width: int = 10, height: int = 10):
        """
        Initialize the grid world.
        
        Args:
            grid_file: Path to grid file (JSON format)
            width: Grid width if creating new grid
            height: Grid height if creating new grid
        """
        self.width = width
        self.height = height
        self.grid = []
        self.start = (0, 0)
        self.goal = (0, 0)
        self.moving_obstacles = []
        self.static_obstacles = set()
        
        if grid_file:
            self.load_from_file(grid_file)
        else:
            self.create_default_grid()
    
    def load_from_file(self, filename: str):
        """Load grid configuration from JSON file."""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            self.width = data['width']
            self.height = data['height']
            self.grid = data['grid']
            self.start = tuple(data['start'])
            self.goal = tuple(data['goal'])
            self.static_obstacles = set(tuple(pos) for pos in data.get('static_obstacles', []))
            
            # Load moving obstacles with their schedules
            self.moving_obstacles = []
            for obs in data.get('moving_obstacles', []):
                self.moving_obstacles.append({
                    'id': obs['id'],
                    'schedule': [tuple(pos) for pos in obs['schedule']],
                    'current_time': 0
                })
                
        except FileNotFoundError:
            print(f"Grid file {filename} not found. Creating default grid.")
            self.create_default_grid()
        except json.JSONDecodeError:
            print(f"Invalid JSON in {filename}. Creating default grid.")
            self.create_default_grid()
    
    def create_default_grid(self):
        """Create a default grid with varying terrain costs."""
        self.grid = [[1 for _ in range(self.width)] for _ in range(self.height)]
        
        # Add some terrain variations
        for i in range(self.height):
            for j in range(self.width):
                if random.random() < 0.1:  # 10% chance of difficult terrain
                    self.grid[i][j] = 3
                elif random.random() < 0.2:  # 20% chance of moderate terrain
                    self.grid[i][j] = 2
        
        # Set start and goal
        self.start = (0, 0)
        self.goal = (self.height - 1, self.width - 1)
        
        # Ensure start and goal have cost 1
        self.grid[self.start[0]][self.start[1]] = 1
        self.grid[self.goal[0]][self.goal[1]] = 1
    
    def get_cost(self, pos: Tuple[int, int]) -> int:
        """Get the movement cost for a given position."""
        row, col = pos
        if 0 <= row < self.height and 0 <= col < self.width:
            return self.grid[row][col]
        return float('inf')  # Out of bounds
    
    def is_valid_position(self, pos: Tuple[int, int], time: int = 0) -> bool:
        """Check if a position is valid (within bounds and not blocked)."""
        row, col = pos
        
        # Check bounds
        if not (0 <= row < self.height and 0 <= col < self.width):
            return False
        
        # Check static obstacles
        if pos in self.static_obstacles:
            return False
        
        # Check moving obstacles at given time
        for obstacle in self.moving_obstacles:
            if time < len(obstacle['schedule']) and obstacle['schedule'][time] == pos:
                return False
        
        return True
    
    def get_neighbors(self, pos: Tuple[int, int], time: int = 0) -> List[Tuple[int, int]]:
        """Get valid neighboring positions (4-connected)."""
        row, col = pos
        neighbors = []
        
        # 4-connected movement: up, down, left, right
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        for dr, dc in directions:
            new_pos = (row + dr, col + dc)
            if self.is_valid_position(new_pos, time + 1):
                neighbors.append(new_pos)
        
        return neighbors
    
    def update_moving_obstacles(self, time: int):
        """Update positions of moving obstacles based on time."""
        for obstacle in self.moving_obstacles:
            if time < len(obstacle['schedule']):
                obstacle['current_time'] = time

class DeliveryAgent:
    """Autonomous delivery agent with multiple pathfinding algorithms."""
    
    def __init__(self, grid_world: GridWorld):
        """Initialize the delivery agent."""
        self.grid_world = grid_world
        self.current_pos = grid_world.start
        self.goal_pos = grid_world.goal
        self.path = []
        self.total_cost = 0
        self.nodes_expanded = 0
        self.search_time = 0
    
    def reset_stats(self):
        """Reset search statistics."""
        self.nodes_expanded = 0
        self.search_time = 0
        self.total_cost = 0
    
    def manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """Calculate Manhattan distance heuristic."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def euclidean_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calculate Euclidean distance heuristic."""
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def bfs(self) -> List[Tuple[int, int]]:
        """Breadth-First Search implementation."""
        start_time = time.time()
        self.reset_stats()
        
        queue = deque([(self.current_pos, [self.current_pos])])
        visited = {self.current_pos}
        
        while queue:
            current, path = queue.popleft()
            self.nodes_expanded += 1
            
            if current == self.goal_pos:
                self.search_time = time.time() - start_time
                self.total_cost = len(path) - 1  # Number of moves
                return path
            
            for neighbor in self.grid_world.get_neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        self.search_time = time.time() - start_time
        return []  # No path found
    
    def uniform_cost_search(self) -> List[Tuple[int, int]]:
        """Uniform Cost Search implementation."""
        start_time = time.time()
        self.reset_stats()
        
        # Priority queue: (cost, position, path)
        heap = [(0, self.current_pos, [self.current_pos])]
        visited = {self.current_pos: 0}
        
        while heap:
            cost, current, path = heapq.heappop(heap)
            self.nodes_expanded += 1
            
            if current == self.goal_pos:
                self.search_time = time.time() - start_time
                self.total_cost = cost
                return path
            
            # Skip if we've found a better path to this position
            if current in visited and visited[current] < cost:
                continue
            
            for neighbor in self.grid_world.get_neighbors(current):
                new_cost = cost + self.grid_world.get_cost(neighbor)
                
                if neighbor not in visited or visited[neighbor] > new_cost:
                    visited[neighbor] = new_cost
                    heapq.heappush(heap, (new_cost, neighbor, path + [neighbor]))
        
        self.search_time = time.time() - start_time
        return []  # No path found
    
    def a_star(self, heuristic='manhattan') -> List[Tuple[int, int]]:
        """A* Search implementation with admissible heuristic."""
        start_time = time.time()
        self.reset_stats()
        
        if heuristic == 'manhattan':
            h_func = self.manhattan_distance
        else:
            h_func = self.euclidean_distance
        
        # Priority queue: (f_score, g_score, position, path)
        heap = [(h_func(self.current_pos, self.goal_pos), 0, self.current_pos, [self.current_pos])]
        visited = {self.current_pos: 0}
        
        while heap:
            f_score, g_score, current, path = heapq.heappop(heap)
            self.nodes_expanded += 1
            
            if current == self.goal_pos:
                self.search_time = time.time() - start_time
                self.total_cost = g_score
                return path
            
            # Skip if we've found a better path to this position
            if current in visited and visited[current] < g_score:
                continue
            
            for neighbor in self.grid_world.get_neighbors(current):
                new_g_score = g_score + self.grid_world.get_cost(neighbor)
                
                if neighbor not in visited or visited[neighbor] > new_g_score:
                    visited[neighbor] = new_g_score
                    h_score = h_func(neighbor, self.goal_pos)
                    f_score = new_g_score + h_score
                    heapq.heappush(heap, (f_score, new_g_score, neighbor, path + [neighbor]))
        
        self.search_time = time.time() - start_time
        return []  # No path found
    
    def hill_climbing(self, max_restarts: int = 10) -> List[Tuple[int, int]]:
        """Hill Climbing with random restarts for local search."""
        start_time = time.time()
        self.reset_stats()
        best_path = []
        best_cost = float('inf')
        
        for restart in range(max_restarts):
            # Start from random position or current position
            if restart == 0:
                start_pos = self.current_pos
            else:
                # Random restart
                start_pos = (random.randint(0, self.grid_world.height - 1),
                           random.randint(0, self.grid_world.width - 1))
                if not self.grid_world.is_valid_position(start_pos):
                    continue
            
            current_pos = start_pos
            path = [current_pos]
            visited = {current_pos}
            
            while current_pos != self.goal_pos:
                self.nodes_expanded += 1
                neighbors = self.grid_world.get_neighbors(current_pos)
                
                if not neighbors:
                    break  # Stuck
                
                # Find neighbor with best heuristic value
                best_neighbor = None
                best_heuristic = float('inf')
                
                for neighbor in neighbors:
                    if neighbor not in visited:
                        h_val = self.manhattan_distance(neighbor, self.goal_pos)
                        if h_val < best_heuristic:
                            best_heuristic = h_val
                            best_neighbor = neighbor
                
                if best_neighbor is None:
                    break  # No unvisited neighbors
                
                # Check if this is an improvement
                current_heuristic = self.manhattan_distance(current_pos, self.goal_pos)
                if best_heuristic >= current_heuristic:
                    break  # Local maximum reached
                
                current_pos = best_neighbor
                path.append(current_pos)
                visited.add(current_pos)
                
                # Prevent infinite loops
                if len(path) > self.grid_world.width * self.grid_world.height:
                    break
            
            # Calculate path cost
            if current_pos == self.goal_pos:
                path_cost = sum(self.grid_world.get_cost(pos) for pos in path[1:])
                if path_cost < best_cost:
                    best_cost = path_cost
                    best_path = path
        
        self.search_time = time.time() - start_time
        self.total_cost = best_cost if best_cost != float('inf') else 0
        return best_path
    
    def dynamic_replan(self, algorithm: str = 'a_star') -> List[Tuple[int, int]]:
        """
        Demonstrate dynamic replanning when obstacles appear.
        
        Args:
            algorithm: Algorithm to use for replanning ('a_star', 'ucs', 'bfs')
        """
        print("=== Dynamic Replanning Demo ===")
        
        # Get initial path
        if algorithm == 'a_star':
            initial_path = self.a_star()
        elif algorithm == 'ucs':
            initial_path = self.uniform_cost_search()
        else:
            initial_path = self.bfs()
        
        print(f"Initial path found with {len(initial_path)} steps")
        print(f"Path: {initial_path[:5]}..." if len(initial_path) > 5 else f"Path: {initial_path}")
        
        # Simulate moving along the path and encountering dynamic obstacles
        current_step = 0
        executed_path = [self.current_pos]
        
        while current_step < len(initial_path) - 1 and self.current_pos != self.goal_pos:
            next_pos = initial_path[current_step + 1]
            
            # Simulate a dynamic obstacle appearing (random chance)
            if random.random() < 0.3 and current_step > 2:  # 30% chance after step 2
                # Add a temporary obstacle
                obstacle_pos = next_pos
                print(f"Dynamic obstacle appeared at {obstacle_pos}!")
                self.grid_world.static_obstacles.add(obstacle_pos)
                
                # Replan from current position
                old_current = self.current_pos
                self.current_pos = executed_path[-1]
                
                print("Replanning path...")
                if algorithm == 'a_star':
                    new_path = self.a_star()
                elif algorithm == 'ucs':
                    new_path = self.uniform_cost_search()
                else:
                    new_path = self.bfs()
                
                if new_path:
                    print(f"New path found with {len(new_path)} steps")
                    initial_path = new_path
                    current_step = 0
                    self.current_pos = old_current
                    
                    # Remove the temporary obstacle for next iteration
                    self.grid_world.static_obstacles.discard(obstacle_pos)
                else:
                    print("No alternative path found!")
                    break
            else:
                # Move to next position
                self.current_pos = next_pos
                executed_path.append(next_pos)
                current_step += 1
        
        return executed_path

def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(description='Autonomous Delivery Agent')
    parser.add_argument('--algorithm', choices=['bfs', 'ucs', 'astar', 'hill', 'dynamic'], 
                       default='astar', help='Pathfinding algorithm to use')
    parser.add_argument('--grid', type=str, help='Grid file to load')
    parser.add_argument('--heuristic', choices=['manhattan', 'euclidean'], 
                       default='manhattan', help='Heuristic for A*')
    parser.add_argument('--demo', action='store_true', help='Run dynamic replanning demo')
    
    args = parser.parse_args()
    
    # Create grid world
    if args.grid:
        grid_world = GridWorld(grid_file=args.grid)
    else:
        grid_world = GridWorld(width=10, height=10)
    
    # Create agent
    agent = DeliveryAgent(grid_world)
    
    print(f"Grid size: {grid_world.width}x{grid_world.height}")
    print(f"Start: {grid_world.start}, Goal: {grid_world.goal}")
    
    # Run selected algorithm
    if args.demo or args.algorithm == 'dynamic':
        path = agent.dynamic_replan()
    elif args.algorithm == 'bfs':
        path = agent.bfs()
    elif args.algorithm == 'ucs':
        path = agent.uniform_cost_search()
    elif args.algorithm == 'astar':
        path = agent.a_star(args.heuristic)
    elif args.algorithm == 'hill':
        path = agent.hill_climbing()
    else:
        print(f"Unknown algorithm: {args.algorithm}")
        return
    
    # Print results
    if path:
        print(f"\nPath found!")
        print(f"Path length: {len(path)} steps")
        print(f"Total cost: {agent.total_cost}")
        print(f"Nodes expanded: {agent.nodes_expanded}")
        print(f"Search time: {agent.search_time:.4f} seconds")
        if len(path) <= 20:  # Only print full path for short paths
            print(f"Path: {path}")
        else:
            print(f"Path preview: {path[:5]} ... {path[-5:]}")
    else:
        print("No path found!")

if __name__ == "__main__":
    main()
