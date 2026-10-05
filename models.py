from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple

Position = Tuple[int, int]

class ParcelStatus(Enum):
    WAITING = "waiting"
    LOADED = "loaded"
    DELIVERED = "delivered"

@dataclass
class Parcel:
    parcel_id: int
    warehouse_id: int
    destination_id: int
    priority: int = 1
    status: ParcelStatus = ParcelStatus.WAITING

@dataclass
class Warehouse:
    warehouse_id: int
    position: Position
    parcel_ids: list[int] = field(default_factory=list)

@dataclass
class DeliveryPoint:
    destination_id: int
    position: Position

@dataclass
class Truck:
    position: Position
    capacity: int
    loaded_parcels: list[int] = field(default_factory=list)

    @property
    def remaining_capacity(self) -> int:
        return self.capacity - len(self.loaded_parcels)
