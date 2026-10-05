# Project Notes

## What the project is solving

This is a hybrid autonomous logistics problem.

The truck has to:

1. locate useful parcels across multiple warehouses,
2. respect truck capacity,
3. choose which parcel to handle next,
4. travel around obstacles,
5. pick up the parcel,
6. travel to the assigned delivery point,
7. deliver it,
8. update the world,
9. re-plan.

## Why not make Q-learning choose everything?

A monolithic Q-learning action space would have to encode movement, warehouse selection, parcel selection, pickup, delivery, capacity, and a changing number of objects. That makes the learning problem unnecessarily large.

Instead:

- DSA handles discrete planning/search.
- RL learns navigation behavior.

This is a clean hybrid architecture.

## Complexity

For A* on a grid with `V` reachable cells and `E` movement edges, the practical complexity is approximately `O(E log V)` with a binary heap.

The task planner evaluates waiting parcels and performs shortest-path calculations. If there are `P` waiting parcels, it performs approximately `O(P)` candidate evaluations per replan, with each candidate using A*.

The Q-learning update is `O(1)` per navigation step.

## Extension ideas

After this version is stable, the project can be upgraded with:

- multiple trucks,
- battery/fuel constraints,
- traffic,
- dynamic road closures,
- parcel deadlines,
- delivery-time windows,
- stochastic travel time,
- prioritized emergency parcels,
- Double Q-learning,
- DQN,
- PPO,
- multi-agent RL,
- centralized fleet planner,
- route comparison against greedy/A*/OR baselines.
