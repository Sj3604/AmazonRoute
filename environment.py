import random
from collections import deque
from models import Parcel, Warehouse, DeliveryPoint, Truck, ParcelStatus
from algorithms import astar, MOVES
from config import Config

class DeliveryEnvironment:
    def __init__(self, config: Config | None = None, seed: int | None = None):
        self.config = config or Config()
        self.seed = self.config.seed if seed is None else seed
        self.rng = random.Random(self.seed)
        self.reset()

    def reset(self):
        self.rows = self.config.rows
        self.cols = self.config.cols
        self.obstacles = set()
        self.warehouses = {}
        self.delivery_points = {}
        self.parcels = {}
        self.truck = None
        self.steps = 0
        self.total_distance = 0
        self.total_reward = 0.0
        self.delivered_count = 0
        self.pickups = 0
        self.invalid_actions = 0
        self.collisions = 0
        self.replans = 0
        self._generate_world()
        return self

    def _random_free_cell(self, reserved):
        while True:
            p = (self.rng.randrange(self.rows), self.rng.randrange(self.cols))
            if p not in reserved and p not in self.obstacles:
                return p

    def _generate_world(self):
        reserved = set()
        truck_pos = self._random_free_cell(reserved)
        reserved.add(truck_pos)

        for wid in range(self.config.num_warehouses):
            p = self._random_free_cell(reserved)
            reserved.add(p)
            self.warehouses[wid] = Warehouse(wid, p)

        for did in range(self.config.num_delivery_points):
            p = self._random_free_cell(reserved)
            reserved.add(p)
            self.delivery_points[did] = DeliveryPoint(did, p)

        # Add obstacles while preserving connectivity to important nodes.
        attempts = 0
        while len(self.obstacles) < self.config.num_obstacles and attempts < 5000:
            attempts += 1
            p = (self.rng.randrange(self.rows), self.rng.randrange(self.cols))
            if p in reserved or p in self.obstacles:
                continue
            candidate = self.obstacles | {p}
            important = [truck_pos] + [w.position for w in self.warehouses.values()] + \
                        [d.position for d in self.delivery_points.values()]
            if all(self._reachable(a, b, candidate) for a, b in
                   zip(important, important[1:])):
                self.obstacles.add(p)

        self.truck = Truck(truck_pos, self.config.truck_capacity)

        for pid in range(self.config.num_parcels):
            wid = self.rng.randrange(self.config.num_warehouses)
            did = self.rng.randrange(self.config.num_delivery_points)
            priority = self.rng.choices([1, 2, 3], weights=[5, 3, 1])[0]
            parcel = Parcel(pid, wid, did, priority)
            self.parcels[pid] = parcel
            self.warehouses[wid].parcel_ids.append(pid)

    def _reachable(self, start, goal, obstacles):
        if start == goal:
            return True
        q = deque([start])
        seen = {start}
        while q:
            cur = q.popleft()
            r, c = cur
            for dr, dc in MOVES.values():
                nxt = (r + dr, c + dc)
                if not (0 <= nxt[0] < self.rows and 0 <= nxt[1] < self.cols):
                    continue
                if nxt in obstacles or nxt in seen:
                    continue
                if nxt == goal:
                    return True
                seen.add(nxt)
                q.append(nxt)
        return False

    def move(self, action):
        dr, dc = MOVES[action]
        nr = self.truck.position[0] + dr
        nc = self.truck.position[1] + dc
        target = (nr, nc)

        self.steps += 1

        if not (0 <= nr < self.rows and 0 <= nc < self.cols) or target in self.obstacles:
            self.collisions += 1
            self.invalid_actions += 1
            reward = -12.0
            self.total_reward += reward
            return reward, False

        self.truck.position = target
        self.total_distance += 1
        reward = -1.0
        self.total_reward += reward
        return reward, False

    def pickup(self, parcel_id):
        self.steps += 1
        parcel = self.parcels[parcel_id]
        warehouse = self.warehouses[parcel.warehouse_id]

        if parcel.status != ParcelStatus.WAITING:
            self.invalid_actions += 1
            return -8.0, False

        if self.truck.position != warehouse.position:
            self.invalid_actions += 1
            return -8.0, False

        if self.truck.remaining_capacity <= 0:
            self.invalid_actions += 1
            return -10.0, False

        parcel.status = ParcelStatus.LOADED
        self.truck.loaded_parcels.append(parcel_id)
        self.pickups += 1
        return 15.0, False

    def deliver(self, parcel_id):
        self.steps += 1
        parcel = self.parcels[parcel_id]
        destination = self.delivery_points[parcel.destination_id]

        if parcel.status != ParcelStatus.LOADED:
            self.invalid_actions += 1
            return -8.0, False

        if self.truck.position != destination.position:
            self.invalid_actions += 1
            return -8.0, False

        parcel.status = ParcelStatus.DELIVERED
        self.truck.loaded_parcels.remove(parcel_id)
        self.delivered_count += 1
        reward = 60.0 + 10.0 * parcel.priority

        done = self.delivered_count == len(self.parcels)
        if done:
            reward += 500.0

        return reward, done

    def all_delivered(self):
        return self.delivered_count == len(self.parcels)

    def nearest_target_path_from(self, start, target):
        return astar(start, target, self.rows, self.cols, self.obstacles)

    def nearest_target_path(self, target):
        return self.nearest_target_path_from(self.truck.position, target)

    def metrics(self):
        return {
            "delivered": self.delivered_count,
            "total_parcels": len(self.parcels),
            "delivery_rate": self.delivered_count / len(self.parcels),
            "distance": self.total_distance,
            "steps": self.steps,
            "reward": self.total_reward,
            "pickups": self.pickups,
            "invalid_actions": self.invalid_actions,
            "collisions": self.collisions,
            "replans": self.replans,
            "capacity": self.truck.capacity,
        }
