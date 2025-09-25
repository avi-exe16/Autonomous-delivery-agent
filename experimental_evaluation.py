#!/usr/bin/env python3
"""
Experimental Evaluation of Pathfinding Algorithms

Compares BFS, UCS, A*, and Hill Climbing across different grid configurations.
Generates performance metrics and visualizations.
"""

import json
import time
import os
from typing import Dict, List
import matplotlib.pyplot as plt
import pandas as pd
from delivery_agent import GridWorld, DeliveryAgent

class ExperimentRunner:
    """Runs experiments and collects performance data."""
    
    def __init__(self):
        self.results = []
    
    def run_algorithm_test(self, grid_file: str, algorithm: str, 
                          heuristic: str = 'manhattan') -> Dict:
        """Run a single algorithm test and return performance metrics."""
        grid_world = GridWorld(grid_file=grid_file)
        agent = DeliveryAgent(grid_world)
        
        # Run the specified algorithm
        start_time = time.time()
        
        if algorithm == 'bfs':
            path = agent.bfs()
        elif algorithm == 'ucs':
            path = agent.uniform_cost_search()
        elif algorithm == 'astar':
            path = agent.a_star(heuristic)
        elif algorithm == 'hill_climbing':
            path = agent.hill_climbing(max_restarts=5)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        # Collect metrics
        result = {
            'grid_file': grid_file,
            'algorithm': algorithm,
            'heuristic': heuristic if algorithm == 'astar' else None,
            'path_found': len(path) > 0,
            'path_length': len(path) if path else 0,
            'total_cost': agent.total_cost,
            'nodes_expanded': agent.nodes_expanded,
            'search_time': agent.search_time,
            'grid_size': grid_world.width * grid_world.height,
            'success': path[-1] == grid_world.goal if path else False
        }
        
        return result
    
    def run_comprehensive_evaluation(self):
        """Run comprehensive evaluation across all algorithms and grids."""
        grid_files = [
            'grids/small_grid.json',
            'grids/medium_grid.json',
            'grids/large_grid.json'
        ]
        
        algorithms = ['bfs', 'ucs', 'astar', 'hill_climbing']
        heuristics = ['manhattan', 'euclidean']  # For A* only
        
        print("Running comprehensive evaluation...")
        print("=" * 50)
        
        for grid_file in grid_files:
            if not os.path.exists(grid_file):
                print(f"Warning: {grid_file} not found, skipping...")
                continue
                
            print(f"\nTesting on {grid_file}")
            
            for algorithm in algorithms:
                if algorithm == 'astar':
                    # Test both heuristics for A*
                    for heuristic in heuristics:
                        print(f"  Running {algorithm} with {heuristic} heuristic...")
                        try:
                            result = self.run_algorithm_test(grid_file, algorithm, heuristic)
                            self.results.append(result)
                            print(f"    Success: {result['success']}, Cost: {result['total_cost']}, "
                                  f"Nodes: {result['nodes_expanded']}, Time: {result['search_time']:.4f}s")
                        except Exception as e:
                            print(f"    Error: {e}")
                else:
                    print(f"  Running {algorithm}...")
                    try:
                        result = self.run_algorithm_test(grid_file, algorithm)
                        self.results.append(result)
                        print(f"    Success: {result['success']}, Cost: {result['total_cost']}, "
                              f"Nodes: {result['nodes_expanded']}, Time: {result['search_time']:.4f}s")
                    except Exception as e:
                        print(f"    Error: {e}")
    
    def save_results(self, filename: str = 'experimental_results.json'):
        """Save experimental results to JSON file."""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved to {filename}")
    
    def generate_performance_plots(self):
        """Generate performance comparison plots."""
        if not self.results:
            print("No results to plot!")
            return
        
        # Convert results to DataFrame
        df = pd.DataFrame(self.results)
        df = df[df['path_found'] == True]  # Only successful runs
        
        # Create plots directory
        os.makedirs('plots', exist_ok=True)
        
        # Plot 1: Search Time Comparison
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        for algorithm in df['algorithm'].unique():
            alg_data = df[df['algorithm'] == algorithm]
            if algorithm == 'astar':
                for heuristic in alg_data['heuristic'].unique():
                    if pd.notna(heuristic):
                        heur_data = alg_data[alg_data['heuristic'] == heuristic]
                        plt.plot(heur_data['grid_size'], heur_data['search_time'], 
                                marker='o', label=f"A* ({heuristic})")
            else:
                plt.plot(alg_data['grid_size'], alg_data['search_time'], 
                        marker='o', label=algorithm.upper())
        
        plt.xlabel('Grid Size (cells)')
        plt.ylabel('Search Time (seconds)')
        plt.title('Search Time vs Grid Size')
        plt.legend()
        plt.yscale('log')
        
        # Plot 2: Nodes Expanded Comparison
        plt.subplot(1, 2, 2)
        for algorithm in df['algorithm'].unique():
            alg_data = df[df['algorithm'] == algorithm]
            if algorithm == 'astar':
                for heuristic in alg_data['heuristic'].unique():
                    if pd.notna(heuristic):
                        heur_data = alg_data[alg_data['heuristic'] == heuristic]
                        plt.plot(heur_data['grid_size'], heur_data['nodes_expanded'], 
                                marker='o', label=f"A* ({heuristic})")
            else:
                plt.plot(alg_data['grid_size'], alg_data['nodes_expanded'], 
                        marker='o', label=algorithm.upper())
        
        plt.xlabel('Grid Size (cells)')
        plt.ylabel('Nodes Expanded')
        plt.title('Nodes Expanded vs Grid Size')
        plt.legend()
        plt.yscale('log')
        
        plt.tight_layout()
        plt.savefig('plots/performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 3: Path Cost Comparison
        plt.figure(figsize=(10, 6))
        algorithms = df['algorithm'].unique()
        grid_types = ['small_grid.json', 'medium_grid.json', 'large_grid.json']
        
        x_pos = range(len(grid_types))
        width = 0.15
        
        for i, algorithm in enumerate(algorithms):
            costs = []
            for grid_type in grid_types:
                grid_data = df[(df['algorithm'] == algorithm) & 
                              (df['grid_file'].str.contains(grid_type.split('_')[0]))]
                if not grid_data.empty:
                    if algorithm == 'astar':
                        # Use Manhattan heuristic results for A*
                        manhattan_data = grid_data[grid_data['heuristic'] == 'manhattan']
                        if not manhattan_data.empty:
                            costs.append(manhattan_data['total_cost'].iloc[0])
                        else:
                            costs.append(grid_data['total_cost'].iloc[0])
                    else:
                        costs.append(grid_data['total_cost'].iloc[0])
                else:
                    costs.append(0)
            
            plt.bar([x + i * width for x in x_pos], costs, width, 
                   label=algorithm.upper() if algorithm != 'astar' else 'A* (Manhattan)')
        
        plt.xlabel('Grid Type')
        plt.ylabel('Total Path Cost')
        plt.title('Path Cost Comparison Across Grid Types')
        plt.xticks([x + width * 1.5 for x in x_pos], ['Small', 'Medium', 'Large'])
        plt.legend()
        plt.savefig('plots/path_cost_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Performance plots saved to 'plots/' directory")
    
    def generate_summary_table(self):
        """Generate summary performance table."""
        if not self.results:
            print("No results to summarize!")
            return
        
        df = pd.DataFrame(self.results)
        successful_runs = df[df['path_found'] == True]
        
        # Group by algorithm and grid type
        summary = []
        
        for grid_file in successful_runs['grid_file'].unique():
            grid_name = os.path.basename(grid_file).replace('.json', '').replace('_', ' ').title()
            grid_data = successful_runs[successful_runs['grid_file'] == grid_file]
            
            for algorithm in grid_data['algorithm'].unique():
                alg_data = grid_data[grid_data['algorithm'] == algorithm]
                
                if algorithm == 'astar':
                    # Include both heuristics
                    for heuristic in alg_data['heuristic'].unique():
                        if pd.notna(heuristic):
                            heur_data = alg_data[alg_data['heuristic'] == heuristic]
                            if not heur_data.empty:
                                summary.append({
                                    'Grid': grid_name,
                                    'Algorithm': f"A* ({heuristic.capitalize()})",
                                    'Path Cost': heur_data['total_cost'].iloc[0],
                                    'Nodes Expanded': heur_data['nodes_expanded'].iloc[0],
                                    'Search Time (s)': f"{heur_data['search_time'].iloc[0]:.4f}",
                                    'Path Length': heur_data['path_length'].iloc[0]
                                })
                else:
                    if not alg_data.empty:
                        summary.append({
                            'Grid': grid_name,
                            'Algorithm': algorithm.upper(),
                            'Path Cost': alg_data['total_cost'].iloc[0],
                            'Nodes Expanded': alg_data['nodes_expanded'].iloc[0],
                            'Search Time (s)': f"{alg_data['search_time'].iloc[0]:.4f}",
                            'Path Length': alg_data['path_length'].iloc[0]
                        })
        
        summary_df = pd.DataFrame(summary)
        
        # Save to CSV
        summary_df.to_csv('performance_summary.csv', index=False)
        
        # Print formatted table
        print("\n" + "="*80)
        print("PERFORMANCE SUMMARY")
        print("="*80)
        print(summary_df.to_string(index=False))
        print("="*80)
        
        return summary_df

def test_dynamic_replanning():
    """Test dynamic replanning capabilities."""
    print("\n" + "="*50)
    print("DYNAMIC REPLANNING TEST")
    print("="*50)
    
    if not os.path.exists('grids/dynamic_grid.json'):
        print("Dynamic grid not found. Please run grid_generator.py first.")
        return
    
    grid_world = GridWorld(grid_file='grids/dynamic_grid.json')
    agent = DeliveryAgent(grid_world)
    
    print(f"Testing dynamic replanning on {grid_world.width}x{grid_world.height} grid")
    print(f"Start: {grid_world.start}, Goal: {grid_world.goal}")
    print(f"Moving obstacles: {len(grid_world.moving_obstacles)}")
    
    # Test replanning with different algorithms
    algorithms = ['astar', 'ucs', 'bfs']
    
    for algorithm in algorithms:
        print(f"\n--- Testing {algorithm.upper()} with dynamic replanning ---")
        
        # Reset agent position
        agent.current_pos = grid_world.start
        
        # Run dynamic replanning
        try:
            executed_path = agent.dynamic_replan(algorithm)
            
            if executed_path and executed_path[-1] == grid_world.goal:
                print(f"SUCCESS: Reached goal with {algorithm.upper()}")
                print(f"Executed path length: {len(executed_path)}")
            else:
                print(f"FAILED: Could not reach goal with {algorithm.upper()}")
                
        except Exception as e:
            print(f"ERROR during {algorithm} replanning: {e}")

def main():
    """Main experimental evaluation."""
    print("Autonomous Delivery Agent - Experimental Evaluation")
    print("=" * 60)
    
    # Check if required packages are available
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
        plotting_available = True
    except ImportError:
        print("Warning: matplotlib and/or pandas not available. Plots will not be generated.")
        plotting_available = False
    
    # Initialize experiment runner
    runner = ExperimentRunner()
    
    # Run comprehensive evaluation
    runner.run_comprehensive_evaluation()
    
    # Save raw results
    runner.save_results()
    
    # Generate summary table
    runner.generate_summary_table()
    
    # Generate plots if possible
    if plotting_available and runner.results:
        runner.generate_performance_plots()
    
    # Test dynamic replanning
    test_dynamic_replanning()
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    print("Files generated:")
    print("  - experimental_results.json (raw results)")
    print("  - performance_summary.csv (summary table)")
    if plotting_available:
        print("  - plots/performance_comparison.png")
        print("  - plots/path_cost_comparison.png")

if __name__ == "__main__":
    main()
