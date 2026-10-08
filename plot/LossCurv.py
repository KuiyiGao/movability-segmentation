import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def parse_args():
    parser = argparse.ArgumentParser(description="Plot scalars from the TensorBoard logs written by src/train.py")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--tags", nargs="+", default=["Loss/train", "Loss/val"])
    parser.add_argument("--output", default="training_loss.png")
    return parser.parse_args()


def main():
    args = parse_args()
    if not Path(args.log_dir).is_dir():
        raise SystemExit(f"Log directory not found: {args.log_dir}")
    events = EventAccumulator(args.log_dir, size_guidance={"scalars": 0})
    events.Reload()
    available = events.Tags()["scalars"]
    missing = [tag for tag in args.tags if tag not in available]
    if missing:
        raise SystemExit(f"Tags {missing} not found in {args.log_dir}; available: {available}")
    series = {}
    for tag in args.tags:
        scalars = events.Scalars(tag)
        steps = [s.step for s in scalars]
        if any(later <= earlier for earlier, later in zip(steps, steps[1:])):
            raise SystemExit(f"{tag} repeats epochs in {args.log_dir}, so it probably holds several runs; give each run its own --log-dir")
        series[tag] = (steps, [s.value for s in scalars])
    plt.figure(figsize=(8, 4))
    for tag, (steps, values) in series.items():
        plt.plot(steps, values, label=tag)
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
    plt.close()
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
