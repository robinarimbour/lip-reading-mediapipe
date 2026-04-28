
"""
Script: realtime_demo.py

Description:
Runs real-time lip reading using webcam input.

Usage:
    python -m scripts.realtime_demo --model_dir models/v1
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse

from src.modeling.model_loader import load_artifacts
from src.modeling.realtime import realtime_inference


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run real-time lip reading demo"
    )

    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to trained model directory",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("[INFO] Loading model...")
    model, idx_to_word = load_artifacts(args.model_dir)

    print("[INFO] Starting real-time inference...")
    print("[INFO] Press 'q' to quit")

    realtime_inference(model, idx_to_word)


if __name__ == "__main__":
    main()
