
"""
Script: predict_video.py

Description:
Runs inference on a video using a trained lip reading model.

Usage:
    python -m scripts.predict_video --model_dir models/v1 --mode grid \
        --video_path path/to/video.mpg --align_path path/to/file.align

Modes:
    grid   -> predict GRID sentence
    play   -> visualize predictions on video
    word   -> predict single word video
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse

from src.modeling.model_loader import load_artifacts
from src.modeling.predictor import (
    predict_grid_video,
    play_grid_video_with_predictions,
    predict_video
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run lip reading inference on video"
    )

    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to trained model directory",
    )

    parser.add_argument(
        "--video_path",
        type=str,
        required=True,
        help="Path to input video file",
    )

    parser.add_argument(
        "--align_path",
        type=str,
        help="Path to align file (required for GRID mode)",
    )

    parser.add_argument(
        "--mode",
        choices=["grid", "play", "word"],
        default="word",
        help="Inference mode",
    )

    parser.add_argument(
        "--speed",
        type=int,
        default=40,
        help="Playback speed for visualization",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("[INFO] Loading model...")
    artifacts = load_artifacts(args.model_dir)
    model, idx_to_word = artifacts

    print("[INFO] Running inference...")

    if args.mode == "grid":
        if not args.align_path:
            print("[ERROR] --align_path required for grid mode")
            return

        predict_grid_video(
            model, idx_to_word, 
            args.align_path,
            args.video_path
        )

    elif args.mode == "play":
        if not args.align_path:
            print("[ERROR] --align_path required for play mode")
            return

        play_grid_video_with_predictions(
            model, idx_to_word, 
            args.align_path,
            args.video_path,
            playback_speed=args.speed
        )

    elif args.mode == "word":
        predict_video(
            model, idx_to_word, 
            args.video_path
        )

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
