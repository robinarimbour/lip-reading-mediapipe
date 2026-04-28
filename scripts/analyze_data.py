
"""
Script: analyze_data.py

Description:
Analyzes the dataset to compute word segment frame statistics.
Useful for determining sequence length for model training.

Usage:
    # All speakers
    python -m scripts.analyze_data --data_path path/to/grid

    # Specific speakers
    python -m scripts.analyze_data --data_path path/to/grid --speakers s1 s2

    # Range of speakers
    python -m scripts.analyze_data --data_path path/to/grid --speakers s1-s20

    # exclude speakers
    python -m scripts.analyze_data --data_path data/grid --exclude s8 s21
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import argparse

from src.utils.dataset import get_all_speakers, parse_speakers
from src.analysis.dataset_analysis import analyze_word_frame_lengths


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze dataset to compute word frame statistics"
    )

    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to GRID dataset root folder",
    )

    parser.add_argument(
        "--speakers",
        nargs="+",
        help="Speakers (e.g., s1 s2 or s1-s20)",
    )

    parser.add_argument(
        "--exclude",
        nargs="+",
        help="Speakers to exclude (e.g., s8 s21)"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    all_speakers = get_all_speakers(args.data_path)
    
    if args.exclude:
        exclude_set = set(args.exclude)
        all_speakers = [s for s in all_speakers if s not in exclude_set]
    
    speakers = parse_speakers(args.speakers, all_speakers)

    if not speakers:
        print("[ERROR] No valid speakers selected.")
        return

    print(f"[INFO] Running dataset analysis on {len(speakers)} speakers...")

    stats = analyze_word_frame_lengths(speakers, args.data_path)

    if not stats:
        print("[ERROR] No data found.")
        return

    print("\n--- Dataset Frame Statistics ---")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key.upper():>8}: {value:.2f}")
        else:
            print(f"{key.upper():>8}: {value}")

    print("\n[INFO] Analysis complete.")


if __name__ == "__main__":
    main()
