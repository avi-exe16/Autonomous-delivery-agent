#!/usr/bin/env python3
"""
Visualization tools for the Autonomous Delivery Agent

Provides grid visualization and path animation capabilities.
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from delivery_agent import GridWorld, DeliveryAgent

class GridVisualizer:
    """Visualizes grid environments and agent paths."""
    
    def __init__(self, grid_world: GridWorld):
        self.grid_world = grid_world
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        
        # Color mappings
        self.terrain_colors = {
            1: 'white',    # Normal terrain
            2: 'lightyellow',  # Moderate terrain
            3: 'orange',   # Difficult terrain
            4: 'red'       # Very difficult terrain
        }
    
    def plot_grid(self, path=None, current_pos=None, obstacles=None, title="Grid World"):
        """Plot the grid environment with optional path and agent position."""
        self.ax.clear()
        
        # Create grid visualization
        grid_data = np.zeros((self.grid_world.height, self.grid_world.width))
        
        for i in range(self.grid_world.height):
            for j in range(self.grid_world.width):
                grid_data[i, j] = self.grid_world.get_cost((i, j))
        
        # Create colored grid
        colored_grid = np.zeros((self.grid_world.height, self.grid_world.width, 3))
        for i in range(self.grid_world.height):
            for j in range(self.grid_world.width):
                cost = grid_data[i, j]
                if cost in self.terrain_colors:
                    color = self.terrain_colors[cost]
                    rgb = plt.cm.colors.to_rgb(color)
                    colored_grid[i, j] = rgb
                else:
                    colored_grid[i, j] = (0.5, 0.5, 0.5)  # Gray for unknown
        
        # Mark static obstacles
        for obstacle in self.grid_world.static_obstacles:
            i, j = obstacle
            colored_grid[i, j] = (0.2, 0.2, 0.2)  # Dark gray for obstacles
        
        # Mark moving obstacles at current time
        for obstacle in self.grid_world.moving_obstacles:
            if obstacle['schedule']:
                current_obstacle_pos = obstacle['schedule'][obstacle['current_time'] % len(obstacle['schedule'])]
                i, j = current_obstacle_pos
                colored_grid[i, j] = (0.8, 0.2, 0.2)  # Red for moving obstacles
        
        self.ax.imshow(colored_grid, origin='upper')
        
        # Mark start and goal
        start_i, start_j = self.grid_world.start
        goal_i, goal_j = self.grid_world.goal
        self.ax.plot(start_j, start_i, 'gs', markersize=15, label='Start')  # Green square
        self.ax.plot(goal_j, goal_i, 'r*', markersize=15, label='Goal')    # Red star
        
        # Plot path if provided
        if path:
            path_j = [pos[1] for pos in path]
            path_i = [pos[0] for pos in path]
            self.ax.plot(path_j, path_i, 'b-', linewidth=2, alpha=0.7, label='Path')
            self.ax.plot(path_j, path_i, 'bo', markersize=4, alpha=0.5)
        
        # Mark current position if provided
        if current_pos:
            curr_i, curr_j = current_pos
            self.ax.plot(curr_j, curr_i, 'mo', markersize=10, label='Current Position')
        
        self.ax.set_xticks(range(self.grid_world.width))
        self.ax.set_yticks(range(self.grid_world.height))
        self.ax.grid(which='both', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
        self.ax.set_title(title)
        self.ax.legend()
        
        plt.tight_layout()
    
    def animate_path(self, path, interval=500, save_path=None):
        """Animate the agent following a path."""
        fig, ax = plt.subplots(figsize=(10, 10))
        
        def animate(frame):
            ax.clear()
            current_pos = path[frame] if frame < len(path) else path[-1]
            self.plot_grid(path[:frame+1], current_pos, 
                          title=f"Path Animation (Step {frame}/{len(path)-1})")
        
        anim = animation.FuncAnimation(fig, animate, frames=len(path), 
                                     interval=interval, repeat=False)
        
        if save_path:
            anim.save(save_path, writer='pillow', fps=1000/interval)
        
        plt.show()
        return anim

def visualize_algorithm_comparison(results_file='experimental_results.json'):
    """Visualize algorithm performance comparison."""
    import json
    import pandas as pd
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    df = pd.DataFrame(results)
    successful_runs = df[df['path_found'] == True]
    
    # Create comparison plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Search Time by Algorithm
    algorithms = successful_runs['algorithm'].unique()
    for algo in algorithms:
        algo_data = successful_runs[successful_runs['algorithm'] == algo]
        ax1.bar(str(algo), algo_data['search_time'].mean(), 
               yerr=algo_data['search_time'].std(), capsize=5, label=algo)
    
    ax1.set_ylabel('Search Time (seconds)')
    ax1.set_title('Average Search Time by Algorithm')
    ax1.legend()
    
    # Plot 2: Nodes Expanded by Algorithm
    for algo in algorithms:
        algo_data = successful_runs[successful_runs['algorithm'] == algo]
        ax2.bar(str(algo), algo_data['nodes_expanded'].mean(), 
               yerr=algo_data['nodes_expanded'].std(), capsize=5, label=algo)
    
    ax2.set_ylabel('Nodes Expanded')
    ax2.set_title('Average Nodes Expanded by Algorithm')
    ax2.legend()
    
    # Plot 3: Path Cost by Algorithm
    for algo in algorithms:
        algo_data = successful_runs[successful_runs['algorithm'] == algo]
        ax3.bar(str(algo), algo_data['total_cost'].mean(), 
               yerr=algo_data['total_cost'].std(), capsize=5, label=algo)
    
    ax3.set_ylabel('Path Cost')
    ax3.set_title('Average Path Cost by Algorithm')
    ax3.legend()
    
    # Plot 4: Success Rate by Algorithm
    for algo in algorithms:
        algo_data = df[df['algorithm'] == algo]
        success_rate = algo_data['path_found'].mean() * 100
        ax4.bar(str(algo), success_rate, label=f'{success_rate:.1f}%')
    
    ax4.set_ylabel('Success Rate (%)')
    ax4.set_title('Success Rate by Algorithm')
    
    plt.tight_layout()
    plt.savefig('plots/algorithm_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Example usage
    grid_world = GridWorld(width=8, height=8)
    visualizer = GridVisualizer(grid_world)
    
    agent = DeliveryAgent(grid_world)
    path = agent.a_star()
    
    visualizer.plot_grid(path, title="A* Pathfinding Result")
    plt.show()
