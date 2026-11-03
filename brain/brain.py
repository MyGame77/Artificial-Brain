import argparse
import os
import sys

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from network import NeuralNetwork
from experiments.brain_memory import (
    load_task,
    save_task,
    list_memories,
)


TASKS = {
    "and": {
        "inputs": 2,
        "data": [
            ([0.0, 0.0], [0.0]),
            ([0.0, 1.0], [0.0]),
            ([1.0, 0.0], [0.0]),
            ([1.0, 1.0], [1.0]),
        ],
    },
    "or": {
        "inputs": 2,
        "data": [
            ([0.0, 0.0], [0.0]),
            ([0.0, 1.0], [1.0]),
            ([1.0, 0.0], [1.0]),
            ([1.0, 1.0], [1.0]),
        ],
    },
    "xor": {
        "inputs": 2,
        "data": [
            ([0.0, 0.0], [0.0]),
            ([0.0, 1.0], [1.0]),
            ([1.0, 0.0], [1.0]),
            ([1.0, 1.0], [0.0]),
        ],
    },
    "not": {
        "inputs": 1,
        "data": [
            ([0.0], [1.0]),
            ([1.0], [0.0]),
        ],
    },
}


def train_task(name):
    task = TASKS[name]

    brain = NeuralNetwork(
        inputs=task["inputs"],
        hidden_layers=[8, 4],
        outputs=1,
        seed=101 + list(TASKS).index(name),
    )

    print(f"\nTraining {name.upper()}...")
    print("Architecture:", brain.describe())

    brain.train(
        task["data"],
        epochs=20000,
        learning_rate=0.5,
    )

    save_task(name, brain)
    print(f"{name.upper()} training and saving complete.")


def predict_task(name, values):
    brain = load_task(name)

    if brain is None:
        print(f"No saved {name.upper()} memory found.")
        print(f"Train it first: python brain.py train {name}")
        return

    if len(values) != brain.input_count:
        print(
            f"This task expects {brain.input_count} input value(s)."
        )
        return

    if any(value not in (0.0, 1.0) for value in values):
        print("Use only 0 and 1 for these logic experiments.")
        return

    output = brain.predict(values)[0]
    result = 1 if output >= 0.5 else 0

    print(f"\nTask: {name.upper()}")
    print("Architecture:", brain.describe())
    print("Input:", values)
    print(f"Raw output: {output:.6f}")
    print("Prediction:", result)


def main():
    parser = argparse.ArgumentParser(
        description="Artificial Brain v0.6"
    )

    commands = parser.add_subparsers(
        dest="command",
        required=True,
    )

    train_parser = commands.add_parser(
        "train",
        help="Train and save a task",
    )
    train_parser.add_argument(
        "task",
        choices=TASKS.keys(),
    )

    predict_parser = commands.add_parser(
        "predict",
        help="Predict using saved memory",
    )
    predict_parser.add_argument(
        "task",
        choices=TASKS.keys(),
    )
    predict_parser.add_argument(
        "inputs",
        nargs="+",
        type=float,
    )

    commands.add_parser(
        "memories",
        help="List saved task memories",
    )

    args = parser.parse_args()

    if args.command == "train":
        train_task(args.task)

    elif args.command == "predict":
        predict_task(args.task, args.inputs)

    elif args.command == "memories":
        list_memories()


if __name__ == "__main__":
    main()