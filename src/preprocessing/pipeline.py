
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm
import random

random.seed(42)

from src.config import OUTPUT_DIR, ALIGN_PATH, VIDEO_EXTENSION, FPS, MAX_LEN
from .align import read_align_file
from .video import load_video_frames, time_to_frame
from .landmarks import extract_lip_landmarks


def pad_sequence(seq, max_len=MAX_LEN):
    """
    Pads or truncates landmark sequences to a fixed length for model input.
    """
    if seq.shape[0] >= max_len:
        return seq[:max_len]
    
    # Pre-allocate array of zeros
    padded_seq = np.zeros((max_len, seq.shape[1]), dtype=seq.dtype)
    # Copy the sequence into the start of the pre-allocated array
    padded_seq[:seq.shape[0], :] = seq

    return padded_seq


def process_word_segment(frames, output_dir, word, speaker, clip_name):
    """
    Processes a word-level video segment into padded landmarks and saves it.
    """
    if len(frames) == 0:
        return
    
    landmarks = extract_lip_landmarks(frames)

    if landmarks is None or len(landmarks) == 0:
        return

    padded_landmarks = pad_sequence(landmarks)

    word_dir = Path(output_dir) / word
    word_dir.mkdir(parents=True, exist_ok=True)

    save_path = word_dir / f"{word}_{speaker}_{clip_name}.npy"
    np.save(save_path, padded_landmarks)


def split_video(video_path, segments, process_fn, output_dir, speaker, clip_name):
    """
    Splits a video into word segments and applies a processing function.
    """
    frames = load_video_frames(video_path)

    if not frames:
        return

    for start, end, word in segments:

        start_frame = time_to_frame(start, FPS)
        end_frame = time_to_frame(end, FPS)

        if start_frame >= end_frame:
            continue

        word_frames = frames[start_frame:end_frame]

        process_fn(word_frames, output_dir, word, speaker, clip_name)


def process_dataset(speakers, data_path, split, process_fn, num_samples=None, output_dir=None):
    """
    Runs the full preprocessing pipeline over all speakers and videos.
    """

    if not output_dir:
        output_dir = OUTPUT_DIR

    # Final output directory
    output_dir = Path(output_dir) / split
    os.makedirs(output_dir, exist_ok=True)

    for speaker in tqdm(speakers, desc="Speakers"):
        speaker_dir = Path(data_path) / speaker

        if not speaker_dir.exists():
            print(f"Skipping: Speaker {speaker} not found.")
            continue

        align_dir = speaker_dir / ALIGN_PATH

        if not align_dir.exists():
            print(f"Skipping: Align folder not found for {speaker}")
            continue

        all_align_files = list(align_dir.iterdir())

        if not all_align_files:
            print(f"Skipping: No align files for {speaker}")
            continue

        if num_samples:
            selected_align_files = random.sample(
                all_align_files,
                min(num_samples, len(all_align_files))
            )
        else:
            selected_align_files = all_align_files

        for align_path in tqdm(selected_align_files, desc=f"{speaker}", leave=False):
            try:
                # Gets 'bbaf2n' from 'bbaf2n.align'
                clip_name = align_path.stem
                video_path = speaker_dir / f"{clip_name}{VIDEO_EXTENSION}"
                
                if not video_path.exists():
                    print(f"Skipping: Video {video_path.name} not found.")
                    continue
                
                segments = read_align_file(str(align_path))
                
                split_video(
                    str(video_path),
                    segments,
                    process_fn,
                    output_dir,
                    speaker,
                    clip_name
                )
        
            except Exception as e:
                print(f"\n❌ Error processing file: {align_path}")
                print(f"Error: {e}")
