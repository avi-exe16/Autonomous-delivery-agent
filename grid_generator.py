#!/usr/bin/env python3
"""
Grid Generator for Autonomous Delivery Agent

Generates test grids of different sizes and complexities for algorithm evaluation.
"""

import json
import random
import os
from typing import List, Tuple, Dict

class GridGenerator:
    """Generates grid configurations for testing."""
    
    def __init__(self):
        self.grids_dir = 'grids'
        os.makedirs(self.grids_dir, exist_ok=True)
    
    def generate_small_grid(self) -> Dict:
        """Generate a small 8x8 grid for quick testing."""
        width, height = 8, 8
        grid = [[1 for _ in range(width)] for _ in range(height)]
        
        # Add some terrain variations
        for i in range(height):
            for j in range(width):
                if random.random() < 0.15:  # 15% difficult terrain
                    grid[i][j] = 3
                elif random.random() < 0.25:  # 25% moderate terrain
                    grid[i][j] = 2
        
        # Add a few static obstacles
        static_obstacles = []
        obstacle_count = random.randint(3, 6)
        for _ in range(obstacle_count):
            pos = (random.randint(1, height-2), random.randint(1, width-2))
            if pos not in [(0, 0), (height-1, width-1)]:
                static_obstacles.append(pos)
        
        return {
            'width': width,
            'height': height,
            'grid': grid,
            'start': [0, 0],
            'goal': [height-1, width-1],
            'static_obstacles': static_obstacles,
            'moving_obstacles': []
        }
    
    def generate_medium_grid(self) -> Dict:
        """Generate a medium 15x15 grid with moderate complexity."""
        width, height = 15, 15
        grid = [[1 for _ in range(width)] for _ in range(height)]
        
        # Add terrain variations
        for i in range(height):
            for j in range(width):
                rand = random.random()
                if rand < 0.08:  # 8% very difficult terrain
                    grid[i][j] = 4
                elif rand < 0.2:  # 12% difficult terrain
                    grid[i][j] = 3
                elif rand < 0.35:  # 15% moderate terrain
                    grid[i][j] = 2
        
        # Add static obstacles
        static_obstacles = []
        obstacle_count = random.randint(8, 15)
        for _ in range(obstacle_count):
            pos = (random.randint(1, height-2), random.randint(1, width-2))
            if pos not in [(0, 0), (height-1, width-1)]:
                static_obstacles.append(pos)
        
        # Add one moving obstacle
        moving_obstacles = [{
            'id': 'vehicle_1',
            'schedule': self.generate_moving_schedule(width, height, 20)
        }]
        
        return {
            'width': width,
            'height': height,
            'grid': grid,
            'start': [0, 0],
            'goal': [height-1, width-1],
            'static_obstacles': static_obstacles,
            'moving_obstacles': moving_obstacles
        }
    
    def generate_large_grid(self) -> Dict:
        """Generate a large 25x25 grid with high complexity."""
        width, height = 25, 25
        grid = [[1 for _ in range(width)] for _ in range(height)]
        
        # Add terrain variations with patterns
        for i in range(height):
            for j in range(width):
                rand = random.random()
                # Create some "difficulty zones"
                if (i > height//3 and i < 2*height//3) or (j > width//3 and j < 2*width//3):
                    if rand < 0.12:
                        grid[i][j] = 4
                    elif rand < 0.25:
                        grid[i][j] = 3
                    elif rand < 0.4:
                        grid[i][j] = 2
                else:
                    if rand < 0.05:
                        grid[i][j] = 3
                    elif rand < 0.15:
                        grid[i][j] = 2
        
        # Add more static obstacles
        static_obstacles = []
        obstacle_count = random.randint(20, 30)
        for _ in range(obstacle_count):
            pos = (random.randint(1, height-2), random.randint(1, width-2))
            if pos not in [(0, 0), (height-1, width-1)]:
                static_obstacles.append(pos)
        
        # Add multiple moving obstacles
        moving_obstacles = []
        for i in range(3):
            moving_obstacles.append({
                'id': f'vehicle_{i+1}',
                'schedule': self.generate_moving_schedule(width, height, 30)
            })
        
        return {
            'width': width,
            'height': height,
            'grid': grid,
            'start': [0, 0],
            'goal': [height-1, width-1],
            'static_obstacles': static_obstacles,
            'moving_obstacles': moving_obstacles
        }
    
    def generate_dynamic_grid(self) -> Dict:
        """Generate a grid specifically for dynamic replanning testing."""
        width, height = 12, 12
        grid = [[1 for _ in range(width)] for _ in range(height)]
        
        # Simpler terrain for easier dynamic testing
        for i in range(height):
            for j in range(width):
                if random.random() < 0.1:
                    grid[i][j] = 2
        
        # Fewer static obstacles to allow for dynamic ones
        static_obstacles = []
        obstacle_count = random.randint(3, 6)
        for _ in range(obstacle_count):
            pos = (random.randint(2, height-3), random.randint(2, width-3))
            if pos not in [(0, 0), (height-1, width-1)]:
                static_obstacles.append(pos)
        
        # Multiple moving obstacles for dynamic scenarios
        moving_obstacles = []
        for i in range(4):
            moving_obstacles.append({
                'id': f'dynamic_vehicle_{i+1}',
                'schedule': self.generate_moving_schedule(width, height, 25, predictable=False)
            })
        
        return {
            'width': width,
            'height': height,
            'grid': grid,
            'start': [1, 1],  # Slightly different start
            'goal': [height-2, width-2],  # Slightly different goal
            'static_obstacles': static_obstacles,
            'moving_obstacles': moving_obstacles
        }
    
    def generate_moving_schedule(self, width: int, height: int, 
                               steps: int, predictable: bool = True) -> List[List[int]]:
        """Generate a movement schedule for a moving obstacle."""
        schedule = []
        
        if predictable:
            # Generate a predictable path (e.g., back and forth or circular)
            if random.choice([True, False]):
                # Back and forth movement
                start_row = random.randint(1, height-2)
                for step in range(steps):
                    col = 1 + (step % (2 * (width-3)))
                    if col >= width-1:
                        col = 2*(width-2) - col
                    schedule.append([start_row, col])
            else:
                # Circular movement
                center_row, center_col = height//2, width//2
                radius = min(3, min(center_row-1, center_col-1))
                for step in range(steps):
                    angle = (step * 2 * 3.14159) / 8  # 8-step cycle
                    row = int(center_row + radius * math.cos(angle))
                    col = int(center_col + radius * math.sin(angle))
                    row = max(0, min(height-1, row))
                    col = max(0, min(width-1, col))
                    schedule.append([row, col])
        else:
            # Random movement for dynamic testing
            current_pos = [random.randint(1, height-2), random.randint(1, width-2)]
            schedule.append(current_pos[:])
            
            for _ in range(steps-1):
                # Random walk with some bias toward staying in bounds
                moves = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]  # Include staying still
                dr, dc = random.choice(moves)
                new_row = max(1, min(height-2, current_pos[0] + dr))
                new_col = max(1, min(width-2, current_pos[1] + dc))
                current_pos = [new_row, new_col]
                schedule.append(current_pos[:])
        
        return schedule
    
    def generate_all_grids(self):
        """Generate all test grids and save them."""
        print("Generating test grids...")
        
        # Set seed for reproducible results
        random.seed(42)
        
        grids = {
            'small_grid.json': self.generate_small_grid(),
            'medium_grid.json': self.generate_medium_grid(),
            'large_grid.json': self.generate_large_grid(),
            'dynamic_grid.json': self.generate_dynamic_grid()
        }
        
        for filename, grid_data in grids.items():
            filepath = os.path.join(self.grids_dir, filename)
            with open(filepath, 'w') as f:
                json.dump(grid_data, f, indent=2)
            print(f"Generated {filepath}")
            print(f"  Size: {grid_data['width']}x{grid_data['height']}")
            print(f"  Static obstacles: {len(grid_data['static_obstacles'])}")
            print(f"  Moving obstacles: {len(grid_data['moving_obstacles'])}")
            print()
        
        print("All grids generated successfully!")

if __name__ == "__main__":
    import math  # Add this import for circular movement
    
    generator = GridGenerator()
    generator.generate_all_grids()
