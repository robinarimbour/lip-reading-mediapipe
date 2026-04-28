
from pathlib import Path
from tqdm import tqdm
import numpy as np

from src.config import ALIGN_PATH, FPS
from src.preprocessing.align import read_align_file
from src.preprocessing.video import time_to_frame


def analyze_word_frame_lengths(speakers, data_path):
    """
    Compute statistics of word segment lengths (in frames) across the dataset.

    Args:
        speakers (list[str]): List of speaker IDs (e.g., ["s1", "s2", ...])

    Returns:
        dict: Statistics including max, mean, and percentiles
    """
    overall_max_frames = 0
    lengths= []

    for speaker in tqdm(speakers, desc="Speakers"):
        speaker_dir = Path(data_path) / speaker

        if not speaker_dir.exists():
            print(f"[WARNING] Speaker '{speaker}' not found. Skipping.")
            continue
        
        align_dir = speaker_dir / ALIGN_PATH

        if not align_dir.exists():
            print(f"[WARNING] Align folder missing for '{speaker}'. Skipping.")
            continue

        align_files = list(align_dir.glob("*.align"))

        for align_path in tqdm(align_files, desc=f"{speaker}", leave=False):
            segments = read_align_file(str(align_path))
            
            for start, end, _ in segments:
                start_frame = time_to_frame(start, FPS)
                end_frame = time_to_frame(end, FPS)
                length = max(0, end_frame - start_frame)
                
                overall_max_frames = max(overall_max_frames, length)
                lengths.append(length)

    if not lengths:
        print("[ERROR] No valid segments found.")
        return 0

    lengths_np = np.array(lengths)

    return {
        "max": int(overall_max_frames),
        "mean": float(lengths_np.mean()),
        "median": float(np.percentile(lengths_np, 50)),
        "p75": float(np.percentile(lengths_np, 75)),
        "p90": float(np.percentile(lengths_np, 90)),
        "p95": float(np.percentile(lengths_np, 95)),
        "p99": float(np.percentile(lengths_np, 99)),
        "count": int(len(lengths_np)),
    }
