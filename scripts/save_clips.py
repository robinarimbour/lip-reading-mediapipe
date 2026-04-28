
"""
Script: save_clips.py

Description:
Splits videos into word-level clips and saves them.

Usage:
    # All speakers
    python -m scripts.save_clips --data_path path/to/grid

    # Specific speakers
    python -m scripts.save_clips --data_path path/to/grid --speakers s1 s2

    # Range of speakers
    python -m scripts.save_clips --data_path path/to/grid --speakers s1-s20

    # Custom output + samples
    python -m scripts.save_clips --data_path path/to/grid --num_samples 100 --output_dir data/landmarks
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse

from src.utils.dataset import get_all_speakers, parse_speakers
from src.preprocessing.pipeline import process_dataset
from src.preprocessing.video import save_frames_as_video


def parse_args():
    parser = argparse.ArgumentParser(
        description="Save word-level video clips from dataset"
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
        help="Speakers to exclude",
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of clips per speaker (default: use all)",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory to save landmarks (default: use config)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    all_speakers = get_all_speakers(args.data_path)

    if args.exclude:
        all_speakers = [s for s in all_speakers if s not in set(args.exclude)]

    speakers = parse_speakers(args.speakers, all_speakers)

    if not speakers:
        print("[ERROR] No valid speakers selected.")
        return

    print(f"[INFO] Using {len(speakers)} speakers")

    if args.num_samples:
        print(f"[INFO] Samples per speaker: {args.num_samples}")

    if args.output_dir:
        print(f"[INFO] Output directory: {args.output_dir}")

    process_dataset(
        speakers=speakers,
        data_path=args.data_path,
        split="clips",
        process_fn=save_frames_as_video,
        num_samples=args.num_samples,
        output_dir=args.output_dir,
    )

    print("[INFO] Clip extraction complete.")


if __name__ == "__main__":
    main()
