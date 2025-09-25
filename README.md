# Autonomous Delivery Agent - AI/ML Project

An intelligent delivery agent that navigates 2D grid environments using various pathfinding algorithms with dynamic replanning capabilities.

## Project Overview

This project implements an autonomous delivery agent that can:
- Navigate 2D grid environments with static obstacles and varying terrain costs
- Handle dynamic moving obstacles with predictable schedules
- Use multiple pathfinding algorithms (BFS, UCS, A*, Hill Climbing)
- Perform dynamic replanning when obstacles appear during execution
- Compare algorithm performance across different map configurations

## Features

### Pathfinding Algorithms
- **BFS (Breadth-First Search)**: Complete and optimal for uniform costs
- **UCS (Uniform Cost Search)**: Optimal for non-uniform terrain costs
- **A***: Informed search with admissible heuristics (Manhattan/Euclidean)
- **Hill Climbing**: Local search with random restarts

### Environment Features
- 4-connected grid movement
- Variable terrain costs (1-4)
- Static obstacles
- Dynamic moving obstacles with predictable schedules
- Dynamic obstacle appearance during execution

## Project Structure
autonomous-delivery-agent/
├── delivery_agent.py # Main agent implementation
├── grid_generator.py # Test grid generation
├── experimental_evaluation.py # Performance evaluation
├── test_runner.py # Unit tests
├── requirements.txt # Dependencies
├── grids/ # Generated test maps
├── plots/ # Performance visualizations
└── report.md # Project report

text

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/autonomous-delivery-agent.git
cd autonomous-delivery-agent
Install dependencies:

bash
pip install -r requirements.txt
Usage
Basic Agent Execution
bash
# Run A* with Manhattan heuristic on default grid
python delivery_agent.py --algorithm astar

# Run UCS on custom grid
python delivery_agent.py --algorithm ucs --grid grids/medium_grid.json

# Run with Euclidean heuristic
python delivery_agent.py --algorithm astar --heuristic euclidean

# Run dynamic replanning demo
python delivery_agent.py --algorithm dynamic --grid grids/dynamic_grid.json
Generate Test Grids
bash
python grid_generator.py
Run Comprehensive Evaluation
bash
python experimental_evaluation.py
Run Tests
bash
python test_runner.py
Algorithms Comparison
Algorithm	Completeness	Optimality	Time Complexity	Space Complexity	Best Use Case
BFS	Yes	Yes (uniform cost)	O(b^d)	O(b^d)	Small grids, uniform cost
UCS	Yes	Yes	O(b^(C*/ε))	O(b^(C*/ε))	Variable terrain costs
A*	Yes	Yes*	O(b^d)	O(b^d)	Most scenarios with good heuristic
Hill Climbing	No	No	O(b)	O(b)	Large maps, quick solutions
*with admissible heuristic

File Formats
Grid JSON Format
json
{
  "width": 10,
  "height": 10,
  "grid": [[1, 1, 3, ...], ...],
  "start": [0, 0],
  "goal": [9, 9],
  "static_obstacles": [[1, 1], [2, 2], ...],
  "moving_obstacles": [
    {
      "id": "vehicle_1",
      "schedule": [[1, 1], [1, 2], [1, 3], ...]
    }
  ]
}
Results
Performance metrics include:

Path cost (total movement cost)

Nodes expanded (search efficiency)

Search time (runtime performance)

Success rate (completeness)

Run python experimental_evaluation.py to generate comprehensive results.

Contributors
Abhishek Shandilya

License
This project is for educational purposes as part of CSA2001 - Fundamentals of AI and ML.
