import argparse

from train import train
from evaluate import evaluate


def main():
    parser = argparse.ArgumentParser(description="AmazonRoute RL")
    parser.add_argument(
        "--mode",
        choices=["train", "evaluate", "demo"],
        default="demo",
    )
    parser.add_argument("--episodes", type=int, default=600)
    args = parser.parse_args()

    if args.mode == "train":
        train(args.episodes)
    elif args.mode == "evaluate":
        evaluate()
    else:
        from app import main as dashboard_main
        dashboard_main()


if __name__ == "__main__":
    main()
