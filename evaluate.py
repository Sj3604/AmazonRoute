import json
from pathlib import Path
import matplotlib.pyplot as plt

from config import Config
from environment import DeliveryEnvironment
from rl_agent import QLearningNavigator
from simulation import run_episode
from visualization import draw_environment

def evaluate(seed=2026):
    out = Path("outputs")
    out.mkdir(exist_ok=True)

    config = Config(seed=seed)
    env = DeliveryEnvironment(config, seed=seed)
    agent = QLearningNavigator(
        alpha=config.alpha,
        gamma=config.gamma,
        epsilon=0.0,
        epsilon_min=0.0,
        epsilon_decay=1.0,
        seed=seed,
    )

    qpath = out / "q_table.pkl"
    if qpath.exists():
        agent.load(qpath)
        print("Loaded trained Q-table.")
    else:
        print("No Q-table found. Run training first; evaluating an untrained navigator.")

    reward, metrics, log = run_episode(env, agent, training=False)

    result = {
        "reward": reward,
        **metrics,
        "tasks": log,
    }

    with open(out / "evaluation.json", "w") as f:
        json.dump(result, f, indent=2)

    fig = draw_environment(env, show_labels=True)
    fig.savefig(out / "evaluation.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    print("\nEvaluation")
    print("----------")
    for k, v in metrics.items():
        print(f"{k:18s}: {v}")

    print(f"\nVisualization saved to {out/'evaluation.png'}")
    return result

if __name__ == "__main__":
    evaluate()
