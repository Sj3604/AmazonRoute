from __future__ import annotations

import json
import random
import threading
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from config import Config
from environment import DeliveryEnvironment
from planner import TaskPlanner
from rl_agent import QLearningNavigator

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
HOST = "127.0.0.1"
PORT = 8765


class DashboardController:
    # Stateful bridge between the browser and the hybrid DSA + RL engine.
    def __init__(self):
        self.lock = threading.RLock()
        self.env = None
        self.agent = None
        self.planner = None
        self.active_task = None
        self.active_path = []
        self.completed_path = []
        self.log = []
        self.last_event = "Waiting for a scenario."
        self.mode = "Hybrid Q-learning + A* safety"
        self.training_status = "Not trained"
        self.config = None

    @staticmethod
    def _parse_position(value, rows, cols):
        """Parse a single grid position robustly.

        Accepted examples:
          6,6
          6, 6
          (6,6)
          [6,6]
          W1: 6,6
          Warehouse W1 = 6,6
        The dashboard UI sends one position per line, so labels are allowed
        for convenience but the final coordinate must still be row,column.
        """
        if isinstance(value, str):
            raw = value.strip()
            # Remove common labels/punctuation users may paste from examples.
            raw = raw.replace("(", " ").replace(")", " ")
            raw = raw.replace("[", " ").replace("]", " ")
            if ":" in raw:
                raw = raw.rsplit(":", 1)[1].strip()
            elif "=" in raw:
                raw = raw.rsplit("=", 1)[1].strip()

            # Extract exactly two integer coordinates, allowing spaces after comma.
            import re
            match = re.search(r"(-?\d+)\s*,\s*(-?\d+)", raw)
            if not match:
                raise ValueError(
                    f"Invalid position {value!r}. Use row,column, for example 6,6."
                )
            r, c = int(match.group(1)), int(match.group(2))
        else:
            try:
                parts = list(value)
            except TypeError:
                raise ValueError(
                    f"Invalid position {value!r}. Use row,column, for example 6,6."
                ) from None
            if len(parts) != 2:
                raise ValueError(
                    f"Invalid position {value!r}. Use row,column, for example 6,6."
                )
            try:
                r, c = int(parts[0]), int(parts[1])
            except (TypeError, ValueError):
                raise ValueError(
                    f"Invalid position {value!r}. Use row,column, for example 6,6."
                ) from None

        if not (0 <= r < rows and 0 <= c < cols):
            raise ValueError(
                f"Position ({r},{c}) is outside the {rows}x{cols} grid. "
                f"Valid rows and columns are 0 to {rows - 1} and 0 to {cols - 1}."
            )
        return r, c

    def _parse_positions(self, values, prefix, rows, cols):
        if not isinstance(values, list) or not values:
            raise ValueError(f"At least one {prefix} position is required.")
        positions = [self._parse_position(v, rows, cols) for v in values]
        if len(set(positions)) != len(positions):
            raise ValueError(f"{prefix} positions must be unique.")
        return positions

    def _make_custom_env(self, payload):
        rows = int(payload.get("rows", 15))
        cols = int(payload.get("cols", 15))
        if rows < 5 or cols < 5 or rows > 40 or cols > 40:
            raise ValueError("Grid dimensions must be between 5 and 40.")

        num_parcels = int(payload.get("num_parcels", 25))
        capacity = int(payload.get("capacity", 5))
        obstacles = int(payload.get("num_obstacles", max(10, min(28, rows * cols // 8))))
        seed = int(payload.get("seed", 42))
        if num_parcels < 1 or capacity < 1:
            raise ValueError("Parcels and capacity must be positive.")
        if obstacles < 0 or obstacles >= rows * cols:
            raise ValueError("Obstacle count is outside the valid grid range.")

        truck_pos = self._parse_position(payload.get("truck_position", "0,0"), rows, cols)
        warehouses = self._parse_positions(payload.get("warehouses", []), "Warehouse", rows, cols)
        deliveries = self._parse_positions(payload.get("deliveries", []), "Delivery", rows, cols)

        reserved = {truck_pos, *warehouses, *deliveries}
        if len(reserved) != 1 + len(warehouses) + len(deliveries):
            raise ValueError("Truck, warehouses and delivery points cannot overlap.")

        cfg = Config(
            rows=rows,
            cols=cols,
            num_warehouses=len(warehouses),
            num_delivery_points=len(deliveries),
            num_parcels=num_parcels,
            num_obstacles=obstacles,
            truck_capacity=capacity,
            seed=seed,
        )

        env = DeliveryEnvironment(cfg, seed=seed)
        env.obstacles = set()

        # Rebuild obstacles around user-specified nodes while checking
        # connectivity after every candidate obstacle.
        important = [truck_pos, *warehouses, *deliveries]
        rng = random.Random(seed)
        attempts = 0
        while len(env.obstacles) < obstacles and attempts < 10000:
            attempts += 1
            p = (rng.randrange(rows), rng.randrange(cols))
            if p in reserved or p in env.obstacles:
                continue
            candidate = env.obstacles | {p}
            if all(env._reachable(a, b, candidate) for a, b in zip(important, important[1:])):
                env.obstacles.add(p)

        env.truck.position = truck_pos
        for i, p in enumerate(warehouses):
            env.warehouses[i].position = p
        for i, p in enumerate(deliveries):
            env.delivery_points[i].position = p
        return env

    def initialize(self, payload):
        with self.lock:
            self.env = self._make_custom_env(payload)
            self.config = self.env.config
            self.agent = QLearningNavigator(
                alpha=self.config.alpha,
                gamma=self.config.gamma,
                epsilon=0.0,
                epsilon_min=0.0,
                epsilon_decay=1.0,
                seed=self.config.seed,
            )
            qpath = OUTPUTS / "q_table.pkl"
            if qpath.exists():
                self.agent.load(qpath)
                self.mode = "Hybrid Q-learning + A* safety"
                self.training_status = "Loaded trained Q-table"
            else:
                self.mode = "Hybrid Q-learning + A* safety"
                self.training_status = "No Q-table found"
            self.planner = TaskPlanner(self.env)
            self.active_task = None
            self.active_path = []
            self.completed_path = [list(self.env.truck.position)]
            self.log = []
            self.last_event = (
                f"Scenario ready: {len(self.env.warehouses)} warehouses, "
                f"{len(self.env.delivery_points)} delivery points, "
                f"{len(self.env.parcels)} parcels."
            )
            self._add_log("SYSTEM", self.last_event)
            return self.state()

    def train(self, episodes=300):
        with self.lock:
            if self.config is None:
                raise ValueError("Apply a scenario before training.")
            from train import train
            episodes = max(1, min(int(episodes), 5000))
            train(episodes=episodes, seed=self.config.seed)
            self.agent.load(OUTPUTS / "q_table.pkl")
            self.mode = "Hybrid Q-learning + A* safety"
            self.training_status = f"Trained for {episodes} episodes"
            self._add_log("RL", f"Loaded Q-table after {episodes} training episodes.")
            return self.state()

    def _add_log(self, source, message):
        self.log.append({"source": source, "message": message})
        self.log = self.log[-12:]

    @staticmethod
    def _action_name(action):
        return {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}.get(action, f"ACTION {action}")

    def _select_task(self):
        task_type, item = self.planner.choose_task()
        if item is None:
            self.active_task = None
            return False
        if task_type == "pickup":
            target = self.env.warehouses[item.warehouse_id].position
            label = f"Go to Warehouse W{item.warehouse_id + 1}"
            detail = f"Pick up P{item.parcel_id + 1}"
        else:
            target = self.env.delivery_points[item.destination_id].position
            label = f"Deliver P{item.parcel_id + 1} to D{item.destination_id + 1}"
            detail = f"Drop at D{item.destination_id + 1}"

        path = self.env.nearest_target_path(target)
        if path is None:
            self._add_log("DSA", f"No feasible path for {label}; replanning.")
            return False

        self.active_task = {
            "type": task_type,
            "parcel_id": item.parcel_id,
            "warehouse_id": item.warehouse_id,
            "destination_id": item.destination_id,
            "target": list(target),
            "label": label,
            "detail": detail,
            "reason": f"Nearest feasible task; path distance {len(path)}",
            "planned_distance": len(path),
        }
        self.active_path = [list(self.env.truck.position)] + [list(p) for p in path]
        self.env.replans += 1
        self._add_log("DSA", f"{label} | {self.active_task['reason']}")
        return True

    def step(self):
        with self.lock:
            if self.env is None:
                raise ValueError("Initialize a scenario first.")
            if self.env.all_delivered():
                self.last_event = "All parcels have been delivered."
                return self.state()

            if self.active_task is None:
                if not self._select_task():
                    self.last_event = "No feasible task remains."
                    return self.state()

            target = tuple(self.active_task["target"])
            if self.env.truck.position == target:
                pid = self.active_task["parcel_id"]
                if self.active_task["type"] == "pickup":
                    self.env.pickup(pid)
                    event = f"Picked up P{pid + 1} at W{self.active_task['warehouse_id'] + 1}."
                else:
                    self.env.deliver(pid)
                    event = f"Delivered P{pid + 1} at D{self.active_task['destination_id'] + 1}."
                self._add_log("TRUCK", event)
                self.last_event = event
                self.active_task = None
                self.active_path = []
                return self.state()

            # The Q-table is learned on the training environment and can propose
            # an action that is invalid or cyclic on a user-defined obstacle map.
            # Keep RL as the decision maker, but verify every proposed move against
            # the current A* shortest path before applying it. This preserves the
            # hybrid RL + DSA design while guaranteeing progress toward the target.
            state = self.agent.state(self.env, target)
            remaining = self.env.nearest_target_path(target)
            if remaining is None:
                self._add_log("DSA", f"Target {target} became unreachable; replanning.")
                self.active_task = None
                self.active_path = []
                self.last_event = "Target unreachable; replanning."
                return self.state()

            from algorithms import MOVES, manhattan
            before = self.env.truck.position
            current_distance = len(remaining)
            rl_action = self.agent.choose_action(state, greedy=True)
            dr, dc = MOVES[rl_action]
            candidate = (before[0] + dr, before[1] + dc)

            in_bounds = (
                0 <= candidate[0] < self.env.rows
                and 0 <= candidate[1] < self.env.cols
            )
            not_obstacle = candidate not in self.env.obstacles
            candidate_remaining = None
            makes_progress = False
            if in_bounds and not_obstacle and candidate != before:
                candidate_remaining = self.env.nearest_target_path_from(candidate, target)
                makes_progress = (
                    candidate_remaining is not None
                    and len(candidate_remaining) < current_distance
                )

            valid_candidate = in_bounds and not_obstacle and candidate != before and makes_progress

            # A valid RL action must make strict progress according to the actual
            # current map. Otherwise use the next A* step instead of allowing a
            # learned policy from another map to oscillate forever.
            if valid_candidate:
                action = rl_action
                source = "RL"
            else:
                next_cell = remaining[0]
                dr = next_cell[0] - before[0]
                dc = next_cell[1] - before[1]
                action = next(k for k, v in MOVES.items() if v == (dr, dc))
                source = "A* safety"

                rl_name = self._action_name(rl_action)
                astar_name = self._action_name(action)
                if not in_bounds:
                    correction_reason = "the RL move leaves the grid"
                elif not not_obstacle:
                    correction_reason = "the RL move is blocked by an obstacle"
                elif candidate_remaining is None:
                    correction_reason = "the RL move leads to an unreachable state"
                elif not makes_progress:
                    correction_reason = "the RL move does not reduce the path distance to the target"
                else:
                    correction_reason = "the RL move is not feasible on the current map"

                self._add_log(
                    "DSA",
                    f"RL proposed: {rl_name} → A* corrected: {astar_name} | Reason: {correction_reason}."
                )

            self.env.move(action)
            after = self.env.truck.position
            if after != before:
                self.completed_path.append(list(after))
            else:
                self._add_log("RL", "Invalid movement; replanning from current cell.")

            self.last_event = f"{source}: moved from {before} to {after}."
            return self.state()

    def run(self, max_steps=5000):
        with self.lock:
            for _ in range(max_steps):
                if self.env is None or self.env.all_delivered():
                    break
                self.step()
            return self.state()

    def state(self):
        if self.env is None:
            return {"ready": False}

        parcels = list(self.env.parcels.values())
        at_warehouse = sum(p.status.value == "waiting" for p in parcels)
        in_transit = sum(p.status.value == "loaded" for p in parcels)
        delivered = sum(p.status.value == "delivered" for p in parcels)
        remaining = len(parcels) - delivered
        target = self.active_task["target"] if self.active_task else None
        task = self.active_task.copy() if self.active_task else None
        if task:
            task["distance_now"] = abs(self.env.truck.position[0] - target[0]) + abs(
                self.env.truck.position[1] - target[1]
            )

        return {
            "ready": True,
            "grid": {"rows": self.env.rows, "cols": self.env.cols},
            "truck": {
                "position": list(self.env.truck.position),
                "capacity": self.env.truck.capacity,
                "carrying": [p + 1 for p in self.env.truck.loaded_parcels],
            },
            "warehouses": [
                {"id": w.warehouse_id + 1, "position": list(w.position),
                 "parcels": sum(1 for p in parcels if p.warehouse_id == w.warehouse_id and p.status.value == "waiting")}
                for w in self.env.warehouses.values()
            ],
            "deliveries": [
                {"id": d.destination_id + 1, "position": list(d.position)}
                for d in self.env.delivery_points.values()
            ],
            "obstacles": [list(p) for p in sorted(self.env.obstacles)],
            "parcels": [
                {"id": p.parcel_id + 1, "warehouse": p.warehouse_id + 1,
                 "destination": p.destination_id + 1, "priority": p.priority,
                 "status": p.status.value}
                for p in parcels
            ],
            "task": task,
            "planned_route": self.active_path,
            "completed_route": self.completed_path,
            "last_event": self.last_event,
            "mode": self.mode,
            "training_status": self.training_status,
            "metrics": {
                "total": len(parcels),
                "at_warehouses": at_warehouse,
                "in_transit": in_transit,
                "delivered": delivered,
                "remaining": remaining,
                "distance": self.env.total_distance,
                "steps": self.env.steps,
                "replans": self.env.replans,
                "success_rate": delivered / len(parcels) * 100 if parcels else 0,
            },
            "log": self.log,
        }


CONTROLLER = DashboardController()

HTML = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AmazonRoute RL — Hybrid DSA + Reinforcement Learning</title>
<style>
:root{--bg:#020b18;--panel:#07172a;--line:#123b61;--blue:#1299ff;--green:#70f34c;--orange:#ffab2e;--red:#ff4557;--text:#dcecff;--muted:#7fa0c2}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 55% 20%,#09213b 0,#020b18 48%,#010711 100%);color:var(--text);font-family:Inter,Segoe UI,Arial,sans-serif}
header{height:70px;border-bottom:1px solid #17436b;display:flex;align-items:center;padding:0 28px;gap:24px}.brand{font-size:27px;font-weight:800;letter-spacing:-1px}.brand span{color:var(--orange)}.sep{height:34px;width:1px;background:#29608c}.title{font-size:15px;color:#b7d3ef}.tag{margin-left:auto;color:#7fc9ff;font-size:13px;font-style:italic}
.layout{display:grid;grid-template-columns:minmax(650px,1fr) 450px;gap:16px;padding:16px;max-width:1540px;margin:auto}.card{background:linear-gradient(180deg,#071a2f,#061326);border:1px solid #155080;border-radius:10px;box-shadow:0 10px 30px #0006}.main{padding:10px}.panel{padding:13px}.section-title{font-weight:700;color:#d9efff;font-size:14px;margin-bottom:10px}
#grid{display:grid;gap:1px;background:#163a59;border:1px solid #1a527f;aspect-ratio:1/1;width:100%;position:relative}.cell{position:relative;background:linear-gradient(135deg,#08233b,#061a2c);min-width:0;min-height:0}.cell:after{content:"";position:absolute;inset:0;border:1px solid #17425f55}.coord{position:absolute;top:2px;left:3px;font-size:8px;color:#4d7494}
.entity{position:absolute;z-index:3;inset:14%;display:flex;align-items:center;justify-content:center;border-radius:7px;font-weight:800;text-shadow:0 2px 4px #000;font-size:clamp(8px,1vw,13px)}.truck-float{position:absolute;z-index:10;width:clamp(22px,5.2%,48px);height:clamp(18px,4.2%,38px);display:flex;align-items:center;justify-content:center;background:#f0a21a;color:#111;font-size:clamp(7px,.8vw,11px);font-weight:900;clip-path:polygon(8% 24%,62% 24%,78% 40%,94% 40%,94% 78%,65% 78%,60% 91%,30% 91%,24% 78%,6% 78%,6% 24%);text-shadow:none;transform:translate(-50%,-50%);transition:left .22s linear,top .22s linear;box-shadow:0 0 12px #f0a21a88}.truck-float.bump{animation:truckBump .32s ease}@keyframes truckBump{50%{scale:1.18}}.truck{background:#f0a21a;color:#111;clip-path:polygon(10% 25%,65% 25%,80% 42%,95% 42%,95% 78%,65% 78%,60% 90%,30% 90%,25% 78%,5% 78%,5% 25%);border-radius:0}.warehouse{background:#1278d6;border:2px solid #72c9ff;border-radius:7px}.delivery{background:#53e63c;border:2px solid #b8ffad;border-radius:50%;color:#06220a}.obstacle{background:linear-gradient(145deg,#6d7f91,#33485e);box-shadow:inset 0 0 12px #172334;border-radius:2px}.target{outline:3px solid var(--red);outline-offset:2px}.route-plan{background:#128fff88}.route-done{background:#57ef3a66}
.legend{display:flex;gap:16px;flex-wrap:wrap;padding:9px 4px;color:#9cb9d5;font-size:11px}.legend i{display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:5px;vertical-align:-2px}
.right{display:flex;flex-direction:column;gap:10px}.step{border:1px solid #1d5885;border-radius:9px;padding:13px;background:#081d33}.stepnum{display:inline-flex;width:28px;height:28px;border-radius:50%;background:#087dd8;align-items:center;justify-content:center;font-weight:800;margin-right:9px}.step h2{display:inline;font-size:16px}.muted{color:var(--muted);font-size:12px}.kv{display:grid;grid-template-columns:1fr 1fr;gap:9px;font-size:12px}.kv div{border-bottom:1px solid #143b5b;padding-bottom:5px}.kv b{color:#d8edff;float:right}.bar{height:8px;background:#0d3152;border-radius:10px;overflow:hidden;margin-top:5px}.bar>span{display:block;height:100%;background:#49e6c8}
.stats{display:grid;grid-template-columns:repeat(4,1fr)}.stat{padding:9px;border:1px solid #12446c;text-align:center}.stat:first-child{border-radius:7px 0 0 7px}.stat:last-child{border-radius:0 7px 7px 0}.stat b{display:block;font-size:17px}.stat span{font-size:10px;color:#91abc6}.plan{max-height:280px;overflow:auto}.planrow{display:flex;gap:10px;padding:9px 2px;border-bottom:1px solid #123652;font-size:12px}.num{width:24px;height:24px;border-radius:50%;border:1px solid #2a6c9d;display:flex;align-items:center;justify-content:center;color:#a9d9ff}.active{background:#0b5c9c66;border-color:#1fa4ff}
.controls{display:grid;grid-template-columns:1fr 1fr;gap:8px}.controls button{background:#0a3a61;color:#dff3ff;border:1px solid #1a6a9e;padding:9px;border-radius:7px;cursor:pointer}.controls button:hover{background:#0d527f}.controls .primary{background:#0b82d7}.form{display:grid;grid-template-columns:1fr 1fr;gap:8px}.form label{font-size:11px;color:#8caaca}.form input,.form textarea{width:100%;margin-top:4px;background:#041323;border:1px solid #1b496d;color:#d8ecff;border-radius:6px;padding:7px}.form textarea{height:50px;resize:vertical}.full{grid-column:1/-1}.log{max-height:150px;overflow:auto;font:11px Consolas,monospace;color:#9ec2df}.log div{padding:3px 0}.source{color:#57d5ff}.footer{padding:10px 16px 18px;color:#6689a9;font-size:11px;text-align:center}
@media(max-width:1100px){.layout{grid-template-columns:1fr}.right{display:grid;grid-template-columns:1fr 1fr}}
/* Light theme */
body.light{--bg:#f3f6fa;--panel:#ffffff;--line:#cbd7e3;--text:#172334;--muted:#5c7087;background:linear-gradient(135deg,#eef3f8,#f9fbfd 55%,#e9f0f7);color:var(--text)}
body.light header{border-bottom-color:#c8d5e2;background:#ffffffcc}.light .title{color:#42566d}.light .tag{color:#0876b9}.light .card{background:linear-gradient(180deg,#ffffff,#f6f9fc);border-color:#c7d5e2;box-shadow:0 8px 24px #18344a18}.light #grid{background:#b9cad9;border-color:#9eb3c5}.light .cell{background:linear-gradient(135deg,#f8fbfd,#edf3f8)}.light .cell:after{border-color:#c5d3df88}.light .coord{color:#7890a6}.light .section-title{color:#223449}.light .legend{color:#536a80}.light .step{background:#f8fbfd;border-color:#c8d6e2}.light .kv div{border-bottom-color:#d9e3eb}.light .kv b{color:#20364c}.light .bar{background:#d9e5ee}.light .controls button{background:#e8f2fa;color:#16466b;border-color:#9dbbd0}.light .controls button:hover{background:#d8eaf7}.light .controls .primary{background:#0b82d7;color:#fff}.light .form label{color:#5c7087}.light .form input,.light .form textarea{background:#fff;border-color:#b8cbd9;color:#1c3045}.light .log{color:#506a82}.light .footer{color:#6d8296}.light .stats .stat{border-color:#c8d6e2}.light .stats .stat span{color:#61778c}.light .planrow{border-bottom-color:#d9e3eb}.light .num{border-color:#8eafc8;color:#41627d}.light .active{background:#dceffd;border-color:#1595e8}.light .route-done{background:#27a844 !important;box-shadow:inset 0 0 0 2px #198f38,0 0 5px #27a84466}.light .route-plan{background:#1689e8 !important;box-shadow:inset 0 0 0 2px #0b6fbd}
.theme-toggle{margin-left:8px;border:1px solid #2a638d;background:#0a2944;color:#dff3ff;border-radius:18px;padding:7px 12px;cursor:pointer;font-size:12px;white-space:nowrap}.theme-toggle:hover{filter:brightness(1.12)}.light .theme-toggle{background:#edf5fa;color:#204c6d;border-color:#aac2d3}
</style>
</head>
<body>
<header><div class="brand">Amazon<span>Route RL</span></div><div class="sep"></div><div class="title">Hybrid Reinforcement Learning &amp; DSA for Autonomous Multi-Warehouse Delivery</div><div class="tag">Smarter Routing, Faster Deliveries</div><button class="theme-toggle" id="themeBtn" onclick="toggleTheme()">☀ Light mode</button></header>
<div class="layout">
<section class="card main">
  <div id="grid"></div>
  <div class="legend">
    <span><i style="background:#f0a21a"></i>Truck</span><span><i style="background:#1278d6"></i>Warehouse</span><span><i style="background:#53e63c;border-radius:50%"></i>Delivery</span><span><i style="background:#657789"></i>Obstacle</span><span><i style="background:#128fff"></i>Planned route</span><span><i style="background:#57ef3a"></i>Completed route</span><span><i style="background:#ff4557;border-radius:50%"></i>Current target</span>
  </div>
  <div class="card panel">
    <div class="section-title">Scenario Configuration — user-defined locations</div>
    <div class="form">
      <label>Grid rows<input id="rows" value="15"></label><label>Grid columns<input id="cols" value="15"></label>
      <label>Truck start (row,col)<input id="truck" value="6,6"></label><label>Capacity<input id="capacity" value="5"></label>
      <label class="full">Warehouse positions — one per line (row,column)<textarea id="wh">1,2
2,10
6,12
10,2</textarea></label>
      <label class="full">Delivery/drop positions — one per line (row,column)<textarea id="del">3,1
5,13
8,11
11,13
13,7</textarea></label>
      <label>Parcels<input id="parcels" value="25"></label><label>Obstacles<input id="obstacles" value="20"></label>
    </div>
    <div class="controls" style="margin-top:9px"><button class="primary" onclick="initialize()">Apply scenario</button><button onclick="train()">Train Q-learning</button><button onclick="step()">Next step</button><button onclick="runAll()">Run live</button><button onclick="togglePause()" id="pauseBtn">Pause</button><label style="grid-column:1/-1;font-size:11px;color:#8caaca">Animation speed <input id="speed" type="range" min="80" max="1000" value="260" style="width:100%"><span id="speedVal">260 ms/step</span></label></div>
  </div>
</section>
<aside class="right">
  <div class="card panel"><div class="step"><span class="stepnum" id="stepnum">1</span><h2 id="headline">Configure your scenario</h2><p class="muted" id="subhead">Enter warehouse and delivery coordinates, then apply the scenario.</p></div></div>
  <div class="card panel"><div class="section-title">Task Details <span class="muted">(DSA Planner)</span></div><div class="kv" id="taskkv"></div></div>
  <div class="card panel"><div class="section-title">Truck Status</div><div class="kv" id="truckkv"></div></div>
  <div class="card panel"><div class="section-title">Parcels Overview</div><div class="stats" id="stats"></div></div>
  <div class="card panel"><div class="section-title">Step-by-Step Plan <span class="muted">(DSA + RL)</span></div><div class="plan" id="plan"></div></div>
  <div class="card panel"><div class="section-title">Real-time Log</div><div class="log" id="log"></div></div>
</aside>
</div>
<div class="footer">DSA selects the task • A* verifies feasible paths • Q-learning controls navigation when a trained policy is available</div>
<script>
let S=null;
let previousS=null;
let running=false;
let paused=false;
const $=id=>document.getElementById(id);
function applyTheme(theme){
  document.body.classList.toggle('light',theme==='light');
  const b=$('themeBtn');
  if(b) b.textContent=theme==='light'?'☾ Dark mode':'☀ Light mode';
  try{localStorage.setItem('amazonroute-theme',theme)}catch(e){}
}
function toggleTheme(){applyTheme(document.body.classList.contains('light')?'dark':'light')}
(function(){let t='dark';try{t=localStorage.getItem('amazonroute-theme')||'dark'}catch(e){}applyTheme(t)})();
function parseLines(id){return $(id).value.split(/\n/).map(x=>x.trim()).filter(Boolean)}
async function api(path,body){const r=await fetch(path,{method:body?'POST':'GET',headers:{'Content-Type':'application/json'},body:body?JSON.stringify(body):undefined});const j=await r.json();if(!r.ok)throw new Error(j.error||'Request failed');return j}
function cfg(){return {rows:+$('rows').value,cols:+$('cols').value,truck_position:$('truck').value,capacity:+$('capacity').value,num_parcels:+$('parcels').value,num_obstacles:+$('obstacles').value,warehouses:parseLines('wh'),deliveries:parseLines('del'),seed:42}}
function sleep(ms){return new Promise(resolve=>setTimeout(resolve,ms))}
function animationDelay(){return Math.max(80,Math.min(1000,+$('speed').value||260))}
$('speed').addEventListener('input',()=>{$('speedVal').textContent=$('speed').value+' ms/step'})
async function initialize(){stopRun();try{previousS=null;S=await api('/api/initialize',cfg());render(true)}catch(e){alert(e.message)}}
async function train(){try{S=await api('/api/train',{episodes:300});render(true)}catch(e){alert(e.message)}}
async function step(){if(running) return;try{previousS=S;S=await api('/api/step',{});render(true);flashTruck();}catch(e){alert(e.message)}}
function togglePause(){if(!running)return;paused=!paused;$('pauseBtn').textContent=paused?'Resume':'Pause'}
function stopRun(){running=false;paused=false;if($('pauseBtn'))$('pauseBtn').textContent='Pause'}
async function runAll(){
  if(!S||!S.ready){alert('Apply a scenario first.');return}
  if(running){togglePause();return}
  running=true;paused=false;$('pauseBtn').textContent='Pause';
  try{
    let guard=0;
    while(running && S.ready && S.metrics.remaining>0 && guard<10000){
      while(paused && running) await sleep(80);
      if(!running) break;
      previousS=S;
      S=await api('/api/step',{});
      render(true);
      flashTruck();
      guard++;
      await sleep(animationDelay());
    }
  }catch(e){alert(e.message)}
  finally{stopRun();render(false)}
}
function key(r,c){return `${r},${c}`}
function renderGrid(){
 if(!S.ready)return;
 const g=$('grid');g.style.gridTemplateColumns=`repeat(${S.grid.cols},1fr)`;
 // Keep the moving truck as an overlay so CSS can animate its position between cells.
 const oldTruck=g.querySelector('.truck-float');
 g.innerHTML='';
 const obs=new Set(S.obstacles.map(x=>key(x[0],x[1]))), plan=new Set((S.planned_route||[]).map(x=>key(x[0],x[1]))), done=new Set((S.completed_route||[]).map(x=>key(x[0],x[1])));
 const wh=new Map(S.warehouses.map(w=>[key(...w.position),w])), de=new Map(S.deliveries.map(d=>[key(...d.position),d]));
 for(let r=0;r<S.grid.rows;r++)for(let c=0;c<S.grid.cols;c++){
  const el=document.createElement('div');el.className='cell';const k=key(r,c);el.innerHTML=`<span class="coord">${r},${c}</span>`;
  if(done.has(k))el.classList.add('route-done');if(plan.has(k))el.classList.add('route-plan');
  if(obs.has(k)){const x=document.createElement('div');x.className='entity obstacle';el.appendChild(x)}
  if(wh.has(k)){const x=document.createElement('div');x.className='entity warehouse';x.textContent='W'+wh.get(k).id;el.appendChild(x)}
  if(de.has(k)){const x=document.createElement('div');x.className='entity delivery';x.textContent='D'+de.get(k).id;el.appendChild(x)}
  if(S.task&&key(...S.task.target)===k)el.classList.add('target');
  g.appendChild(el)
 }
 const truck=document.createElement('div');truck.className='truck-float';truck.id='truckFloat';truck.textContent='TRK';
 positionTruck(truck,S.truck.position);g.appendChild(truck);
 // If a previous truck exists, briefly place it at the old position before transitioning.
 if(previousS&&previousS.truck&&previousS.truck.position.toString()!==S.truck.position.toString()){
   positionTruck(truck,previousS.truck.position);requestAnimationFrame(()=>requestAnimationFrame(()=>positionTruck(truck,S.truck.position)));
 }
}
function positionTruck(el,pos){if(!el||!S)return;el.style.left=`${((pos[1]+0.5)/S.grid.cols)*100}%`;el.style.top=`${((pos[0]+0.5)/S.grid.rows)*100}%`}
function flashTruck(){const t=$('truckFloat');if(!t)return;t.classList.remove('bump');void t.offsetWidth;t.classList.add('bump');setTimeout(()=>t.classList.remove('bump'),340)}
function eventKind(){
 if(!S||!S.last_event)return '';
 const x=S.last_event.toLowerCase();
 if(x.includes('picked up'))return 'pickup';
 if(x.includes('delivered'))return 'delivery';
 if(x.includes('moved'))return 'move';
 if(x.includes('scenario ready'))return 'ready';
 return 'task';
}
function render(animate=false){
 if(!S)return;
 renderGrid();if(!S.ready)return;
 const t=S.task;
 const kind=eventKind();
 let headline=t?t.label:(S.metrics.remaining?'Select next task':'All deliveries complete');
 if(kind==='pickup') headline='Pickup completed';
 if(kind==='delivery') headline='Delivery completed';
 if(kind==='move'&&t) headline=t.label;
 $('headline').textContent=headline;
 $('subhead').textContent=t?`${t.detail} • ${S.mode}`:`${S.last_event} • ${S.mode}`;
 $('stepnum').textContent=Math.max(1,S.metrics.steps);
 $('taskkv').innerHTML=t?`<div>Next Action<b>${t.label}</b></div><div>Reason<b>${t.reason}</b></div><div>Target<b>(${t.target[0]}, ${t.target[1]})</b></div><div>Distance now<b>${t.distance_now}</b></div>`:`<div>Planner<b>${S.metrics.remaining?'Selecting':'Complete'}</b></div>`;
 const pct=S.truck.capacity?S.truck.carrying.length/S.truck.capacity*100:0;
 $('truckkv').innerHTML=`<div>Location<b>(${S.truck.position[0]}, ${S.truck.position[1]})</b></div><div>Capacity<b>${S.truck.carrying.length} / ${S.truck.capacity}</b><div class="bar"><span style="width:${pct}%"></span></div></div><div>Carrying<b>${S.truck.carrying.length?S.truck.carrying.map(x=>'P'+x).join(', '):'None'}</b></div><div>Mode<b>${S.mode}</b></div>`;
 $('stats').innerHTML=[['At warehouses',S.metrics.at_warehouses],['In transit',S.metrics.in_transit],['Delivered',S.metrics.delivered],['Remaining',S.metrics.remaining]].map(x=>`<div class="stat"><b>${x[1]}</b><span>${x[0]}</span></div>`).join('');
 $('plan').innerHTML=[['1',`Start at (${S.completed_route[0][0]}, ${S.completed_route[0][1]})`],['2',t?t.label:'DSA selects next task'],['3',t?`Navigate to target (${t.target[0]}, ${t.target[1]})`:'Waiting for task'],['4',kind==='pickup'?'Pickup completed':kind==='delivery'?'Delivery completed':t?`Complete ${t.detail}`:'—'],['5',`Replan • ${S.metrics.replans} replans so far`]].map((x,i)=>`<div class="planrow ${i===2&&t?'active':''}"><div class="num">${x[0]}</div><div>${x[1]}</div></div>`).join('');
 $('log').innerHTML=S.log.map(x=>`<div><span class="source">[${x.source}]</span> ${x.message}</div>`).join('');
}
initialize();
</script>
</body></html>'''


class Handler(BaseHTTPRequestHandler):
    def _json(self, payload, status=200):
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            data = HTML.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        elif path == "/api/state":
            self._json(CONTROLLER.state())
        else:
            self._json({"error": "Not found"}, 404)

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            path = urlparse(self.path).path
            if path == "/api/initialize":
                result = CONTROLLER.initialize(payload)
            elif path == "/api/train":
                result = CONTROLLER.train(payload.get("episodes", 300))
            elif path == "/api/step":
                result = CONTROLLER.step()
            elif path == "/api/run":
                result = CONTROLLER.run(payload.get("max_steps", 5000))
            else:
                self._json({"error": "Not found"}, 404)
                return
            self._json(result)
        except Exception as exc:
            traceback.print_exc()
            self._json({"error": str(exc)}, 400)

    def log_message(self, fmt, *args):
        pass


def main():
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    url = f"http://{HOST}:{PORT}"
    print(f"AmazonRoute RL dashboard: {url}")
    print("Press Ctrl+C to stop.")
    webbrowser.open(url)
    server.serve_forever()


if __name__ == "__main__":
    main()
