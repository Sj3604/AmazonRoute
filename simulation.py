from algorithms import astar
from planner import TaskPlanner
from models import ParcelStatus

def navigate_to_target(env, agent, target, training=True, max_moves=1000):
    """Use Q-learning to navigate until target is reached."""
    if env.truck.position == target:
        return 0.0, True, 0

    planned = astar(env.truck.position, target, env.rows, env.cols, env.obstacles)
    if planned is None:
        return -100.0, False, 0

    total = 0.0
    moves = 0

    while env.truck.position != target and moves < max_moves:
        state = agent.state(env, target)
        action = agent.choose_action(state, greedy=not training)
        reward, _ = env.move(action)
        next_state = agent.state(env, target)

        # Shaping: encourage movement toward target.
        old_dist = abs(state[0]) + abs(state[1])
        new_dist = abs(next_state[0]) + abs(next_state[1])
        shaped = reward + 0.6 * (old_dist - new_dist)

        # A valid arrival gets a small navigation bonus.
        if env.truck.position == target:
            shaped += 8.0

        if training:
            agent.update(state, action, shaped, next_state, False)

        total += shaped
        moves += 1

        if env.steps >= env.config.max_steps:
            break

    return total, env.truck.position == target, moves

def run_episode(env, agent, training=True):
    planner = TaskPlanner(env)
    total_reward = 0.0
    action_log = []

    while not env.all_delivered() and env.steps < env.config.max_steps:
        task_type, item = planner.choose_task()
        env.replans += 1

        if item is None:
            break

        if task_type == "pickup":
            warehouse = env.warehouses[item.warehouse_id]
            target = warehouse.position
            task_label = f"PICKUP P{item.parcel_id} at W{item.warehouse_id}"
        else:
            destination = env.delivery_points[item.destination_id]
            target = destination.position
            task_label = f"DELIVER P{item.parcel_id} at D{item.destination_id}"

        nav_reward, reached, moves = navigate_to_target(
            env, agent, target, training=training,
            max_moves=max(50, env.config.rows * env.config.cols * 2)
        )
        total_reward += nav_reward

        if not reached:
            break

        if task_type == "pickup":
            reward, done = env.pickup(item.parcel_id)
        else:
            reward, done = env.deliver(item.parcel_id)

        total_reward += reward
        action_log.append({
            "step": env.steps,
            "task": task_label,
            "moves": moves,
            "reward": reward,
            "delivered": env.delivered_count,
        })

        if done:
            break

    if training:
        agent.decay()

    return total_reward, env.metrics(), action_log
