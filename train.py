import argparse
from pathlib import Path
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import Config
from environment import DeliveryEnvironment
from rl_agent import QLearningNavigator
from simulation import run_episode

def train(episodes=600, seed=42):
    config = Config(seed=seed)
    out = Path("outputs")
    out.mkdir(exist_ok=True)

    agent = QLearningNavigator(
        alpha=config.alpha,
        gamma=config.gamma,
        epsilon=config.epsilon_start,
        epsilon_min=config.epsilon_min,
        epsilon_decay=config.epsilon_decay,
        seed=seed,
    )

    rows = []
    best = -1

    for episode in range(1, episodes + 1):
        env = DeliveryEnvironment(config, seed=seed + episode)
        reward, metrics, _ = run_episode(env, agent, training=True)

        row = {
            "episode": episode,
            "reward": reward,
            **metrics,
            "epsilon": agent.epsilon,
        }
        rows.append(row)

        score = metrics["delivered"]
        if score > best:
            best = score

        if episode == 1 or episode % 50 == 0:
            print(
                f"Episode {episode:4d} | "
                f"delivered {metrics['delivered']:2d}/{metrics['total_parcels']} | "
                f"distance {metrics['distance']:4d} | "
                f"reward {reward:8.1f} | "
                f"epsilon {agent.epsilon:.3f}"
            )

    agent.save(out / "q_table.pkl")

    with open(out / "training_metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    plt.figure(figsize=(10, 5))
    rewards = [r["reward"] for r in rows]
    window = min(25, len(rewards))
    moving = [
        sum(rewards[max(0, i-window+1):i+1]) /
        len(rewards[max(0, i-window+1):i+1])
        for i in range(len(rewards))
    ]
    plt.plot(rewards, alpha=0.25, label="episode reward")
    plt.plot(moving, label=f"{window}-episode moving average")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("AmazonRoute RL Training")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "training_rewards.png", dpi=160)
    plt.close()

    print(f"\nTraining complete. Q-table saved to {out/'q_table.pkl'}")
    return agent

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=600)
    args = parser.parse_args()
    train(args.episodes)
