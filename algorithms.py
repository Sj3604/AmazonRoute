import heapq
from typing import Dict, Iterable, Optional
from models import Position

MOVES = {
    0: (-1, 0),  # up
    1: (1, 0),   # down
    2: (0, -1),  # left
    3: (0, 1),   # right
}

def manhattan(a: Position, b: Position) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def neighbors(position: Position, rows: int, cols: int, obstacles: set[Position]):
    r, c = position
    for dr, dc in MOVES.values():
        nr, nc = r + dr, c + dc
        nxt = (nr, nc)
        if 0 <= nr < rows and 0 <= nc < cols and nxt not in obstacles:
            yield nxt

def astar(start: Position, goal: Position, rows: int, cols: int,
          obstacles: set[Position]) -> Optional[list[Position]]:
    """A* shortest path. Returns positions after start, ending at goal."""
    if start == goal:
        return []

    frontier = []
    counter = 0
    heapq.heappush(frontier, (manhattan(start, goal), counter, start))
    came_from: Dict[Position, Optional[Position]] = {start: None}
    cost_so_far = {start: 0}

    while frontier:
        _, _, current = heapq.heappop(frontier)

        if current == goal:
            path = []
            node = goal
            while node != start:
                path.append(node)
                node = came_from[node]
            path.reverse()
            return path

        for nxt in neighbors(current, rows, cols, obstacles):
            new_cost = cost_so_far[current] + 1
            if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                cost_so_far[nxt] = new_cost
                counter += 1
                priority = new_cost + manhattan(nxt, goal)
                heapq.heappush(frontier, (priority, counter, nxt))
                came_from[nxt] = current

    return None

class MinHeapTaskQueue:
    """Small explicit DSA priority queue used by the task planner."""
    def __init__(self):
        self._heap = []
        self._counter = 0

    def push(self, score: float, item):
        self._counter += 1
        heapq.heappush(self._heap, (score, self._counter, item))

    def pop(self):
        return heapq.heappop(self._heap)[2]

    def __bool__(self):
        return bool(self._heap)
