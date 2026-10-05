from dataclasses import dataclass

@dataclass
class Config:
    rows: int = 15
    cols: int = 15
    num_warehouses: int = 6
    num_delivery_points: int = 12
    num_parcels: int = 25
    num_obstacles: int = 28
    truck_capacity: int = 5

    max_steps: int = 2500

    # Q-learning
    alpha: float = 0.15
    gamma: float = 0.95
    epsilon_start: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.995

    # Planner weights
    distance_weight: float = 1.0
    delivery_weight: float = 0.55
    priority_weight: float = 3.0
    capacity_penalty: float = 8.0

    seed: int = 42
