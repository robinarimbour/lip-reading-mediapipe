
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm

from src.config import *
from .align import read_align_file
from .video import load_video_frames, time_to_frame
from .landmarks import extract_lip_landmarks


# -----------------------------
# PAD SEQUENCE TO MAX LEN
# -----------------------------
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


# -----------------------------
# CONVERT WORD SEGMENTS INTO PADDED LANDMARKS
# -----------------------------
def process_word_segment(frames, word, clip_name, split_type):
    """
    Processes a word-level video segment into padded landmarks and saves it.
    """
    landmarks = extract_lip_landmarks(frames)
    padded_landmarks = pad_sequence(landmarks)

    # Create folder per word
    word_dir = Path(OUTPUT_DIR) / split_type / word
    word_dir.mkdir(parents=True, exist_ok=True)

    # Save as .npy
    save_path = word_dir / f"{clip_name}_{word}.npy"
    np.save(save_path, padded_landmarks)

    # print(f"Saved: {save_path}, shape: {padded_landmarks.shape}")


# -----------------------------
# SPLIT VIDEO INTO WORD CLIPS
# -----------------------------
def split_video(video_path, segments, clip_name, is_train=True):
    """
    Splits a video into word segments and processes each segment individually.
    """
    # Load frames from video
    frames = load_video_frames(video_path)

    # Determine the split folder
    split_type = "train" if is_train else "test"

    # Process each word segment
    for start, end, word in segments:
        start_frame = time_to_frame(start, FPS)
        end_frame = time_to_frame(end, FPS)
        # print(start_frame, end_frame)

        word_frames = frames[start_frame:end_frame]
        process_word_segment(word_frames, word, clip_name, split_type)


# -----------------------------
# PROCESS THE DATASET
# -----------------------------
def process_dataset(speakers, is_train):
    """
    Runs the full preprocessing pipeline over all speakers and videos.
    """
    # Ensure output dir exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for speaker in tqdm(speakers, desc="Speakers"):
        print(f"--- Processing Speaker: {speaker} ---")
        
        # Define paths using Pathlib / for cleaner syntax
        speaker_dir = Path(DATASET_PATH) / speaker
        align_dir = speaker_dir / ALIGN_PATH
        
        # Loop all .align files
        for align_path in tqdm(list(align_dir.iterdir()), desc=f"Videos ({speaker})", leave=False):
            clip_name = align_path.stem  # Gets 'bbaf2n' from 'bbaf2n.align'
            video_path = speaker_dir / f"{clip_name}{VIDEO_EXTENSION}"
            
            if not video_path.exists():
                print(f"Skipping: Video {video_path.name} not found.")
                continue
            
            # print(f"Processing Clip: {clip_name}")
            segments = read_align_file(str(align_path))
            split_video(str(video_path), segments, clip_name, is_train)
