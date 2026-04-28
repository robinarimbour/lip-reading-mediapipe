
"""
Script: visualize_landmarks.py

Description:
Visualize lip landmarks either on video or from saved NumPy arrays.

Usage:
    python -m scripts.visualize_landmarks --video path/to/video.mp4
    python -m scripts.visualize_landmarks --npy path/to/file.npy --frame 10

Controls:
    Press 'q' to close the video window.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse
import numpy as np

from src.utils.visualization import visualize_on_video, plot_landmarks


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize lip landmarks on video or from .npy file"
    )

    parser.add_argument(
        "--video",
        type=str,
        help="Path to input video file"
    )

    parser.add_argument(
        "--npy",
        type=str,
        help="Path to .npy file containing landmark sequence"
    )

    parser.add_argument(
        "--speed",
        type=int,
        default=40,
        help="Playback speed for video visualization (default: 40)"
    )

    parser.add_argument(
        "--frame",
        type=int,
        default=0,
        help="Frame index to visualize for .npy input (default: 0)"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if not args.video and not args.npy:
        print("[ERROR] Please provide either --video or --npy")
        return

    # ---- VIDEO VISUALIZATION ----
    if args.video:
        if not os.path.exists(args.video):
            print(f"[ERROR] Video not found: {args.video}")
            return

        print(f"[INFO] Visualizing video: {args.video}")
        print("[INFO] Press 'q' to close the video window.")
        visualize_on_video(args.video, playback_speed=args.speed)

    # ---- NPY LANDMARK VISUALIZATION ----
    if args.npy:
        if not os.path.exists(args.npy):
            print(f"[ERROR] File not found: {args.npy}")
            return

        print(f"[INFO] Loading landmarks from: {args.npy}")
        landmarks = np.load(args.npy)
        
        print(f"[INFO] Shape: {landmarks.shape}")
        plot_landmarks(landmarks, frame_idx=args.frame)


if __name__ == "__main__":
    main()
