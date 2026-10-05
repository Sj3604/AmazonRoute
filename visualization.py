import matplotlib.pyplot as plt
from models import ParcelStatus

def draw_environment(env, show_labels=True):
    fig, ax = plt.subplots(figsize=(10, 10))

    ax.set_xlim(-0.5, env.cols - 0.5)
    ax.set_ylim(env.rows - 0.5, -0.5)
    ax.set_xticks(range(env.cols))
    ax.set_yticks(range(env.rows))
    ax.grid(True, alpha=0.25)

    # Obstacles
    for r, c in env.obstacles:
        ax.add_patch(plt.Rectangle((c - .45, r - .45), .9, .9, fill=True))

    # Warehouses
    for wid, wh in env.warehouses.items():
        r, c = wh.position
        ax.scatter(c, r, s=220, marker="s", label="Warehouse" if wid == 0 else None)
        if show_labels:
            ax.text(c, r, f"W{wid}", ha="center", va="center", fontsize=8)

    # Delivery points
    for did, dp in env.delivery_points.items():
        r, c = dp.position
        ax.scatter(c, r, s=170, marker="D", label="Delivery" if did == 0 else None)
        if show_labels:
            ax.text(c, r, f"D{did}", ha="center", va="center", fontsize=7)

    # Truck
    r, c = env.truck.position
    ax.scatter(c, r, s=320, marker="*", label="Truck", zorder=5)

    ax.set_title(
        f"AmazonRoute RL | Delivered {env.delivered_count}/{len(env.parcels)} | "
        f"Distance {env.total_distance}"
    )
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    return fig
