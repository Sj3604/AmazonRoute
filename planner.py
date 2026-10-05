from algorithms import astar, manhattan, MinHeapTaskQueue
from models import ParcelStatus

class TaskPlanner:
    """
    Chooses the next parcel/task.
    This is deliberately separate from the RL navigation policy.
    """

    def __init__(self, env):
        self.env = env

    def candidate_parcels(self):
        for parcel in self.env.parcels.values():
            if parcel.status != ParcelStatus.WAITING:
                continue
            warehouse = self.env.warehouses[parcel.warehouse_id]
            if self.env.truck.remaining_capacity <= 0:
                # When full, the planner will first deliver loaded parcels.
                continue
            yield parcel, warehouse

    def choose_loaded_delivery(self):
        """Choose which loaded parcel to deliver next."""
        if not self.env.truck.loaded_parcels:
            return None

        queue = MinHeapTaskQueue()
        for pid in self.env.truck.loaded_parcels:
            parcel = self.env.parcels[pid]
            dest = self.env.delivery_points[parcel.destination_id]
            distance = manhattan(self.env.truck.position, dest.position)
            # Higher priority gets a lower effective score.
            score = distance - self.env.config.priority_weight * parcel.priority
            queue.push(score, parcel)
        return queue.pop()

    def choose_next_pickup(self):
        """Select a useful parcel from the currently reachable warehouses."""
        if self.env.truck.remaining_capacity <= 0:
            return None

        queue = MinHeapTaskQueue()

        for parcel, warehouse in self.candidate_parcels():
            path_to_wh = astar(
                self.env.truck.position,
                warehouse.position,
                self.env.config.rows,
                self.env.config.cols,
                self.env.obstacles,
            )
            if path_to_wh is None:
                continue

            destination = self.env.delivery_points[parcel.destination_id]
            path_after_pickup = astar(
                warehouse.position,
                destination.position,
                self.env.config.rows,
                self.env.config.cols,
                self.env.obstacles,
            )
            if path_after_pickup is None:
                continue

            score = (
                self.env.config.distance_weight * len(path_to_wh)
                + self.env.config.delivery_weight * len(path_after_pickup)
                - self.env.config.priority_weight * parcel.priority
            )

            # Small bonus for using the remaining capacity efficiently.
            if self.env.truck.remaining_capacity == 1:
                score += self.env.config.capacity_penalty

            queue.push(score, parcel)

        return queue.pop() if queue else None

    def choose_task(self):
        """
        Hierarchical planning:
        1. If loaded parcels exist and the truck is full, deliver.
        2. If the truck has room, compare pickup opportunities with delivery.
        3. Prefer delivery when it is clearly competitive.
        """
        if not self.env.truck.loaded_parcels:
            return ("pickup", self.choose_next_pickup())

        if self.env.truck.remaining_capacity == 0:
            return ("delivery", self.choose_loaded_delivery())

        pickup = self.choose_next_pickup()
        delivery = self.choose_loaded_delivery()

        if pickup is None:
            return ("delivery", delivery)
        if delivery is None:
            return ("pickup", pickup)

        warehouse = self.env.warehouses[pickup.warehouse_id]
        pickup_dist = manhattan(self.env.truck.position, warehouse.position)
        destination = self.env.delivery_points[delivery.destination_id]
        delivery_dist = manhattan(self.env.truck.position, destination.position)

        # Avoid making the truck endlessly collect without delivering.
        if delivery_dist <= pickup_dist + 2:
            return ("delivery", delivery)
        return ("pickup", pickup)
