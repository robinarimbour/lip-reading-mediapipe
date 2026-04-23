
from pathlib import Path
from tqdm import tqdm

from src.config import *
from src.preprocessing.align import read_align_file
from src.preprocessing.video import time_to_frame


# -----------------------------
# FIND WORD WITH MAX FRAMES
# -----------------------------
def max_word_frames(speakers):
    """
    Computes the maximum number of frames across all word segments in the dataset.
    """
    overall_max_frames = 0

    for speaker in tqdm(speakers, desc="Speakers"):
        # print(f"--- Processing Speaker: {speaker} ---")
        
        # Define paths using Pathlib / for cleaner syntax
        speaker_dir = Path(DATASET_PATH) / speaker
        align_dir = speaker_dir / ALIGN_PATH
        
        # Loop all .align files
        for align_path in tqdm(list(align_dir.iterdir()), desc=f"Videos ({speaker})", leave=False):
            segments = read_align_file(str(align_path))
            for start, end, word in segments:
                start_frame = time_to_frame(start, FPS)
                end_frame = time_to_frame(end, FPS)
                overall_max_frames = max(overall_max_frames, end_frame - start_frame)

    print(f"\n--- Final Result ---")
    print(f"The longest word in your dataset is {overall_max_frames} frames.")
