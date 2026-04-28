
import os
import cv2

from src.config import FPS


def time_to_frame(time_val, fps):
    """
    Converts timestamp values from alignment files into frame indices.
    GRID timestamps are in units of 1/25000 seconds.
    """
    return int((time_val / 25000) * fps)


def load_video_frames(video_path):
    """
    Loads all frames from a video file into memory.

    Returns:
        List of frames (BGR format). Returns empty list if failed.
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"[ERROR] Failed to open video: {video_path}")
        return []

    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()

    if not frames:
        print(f"[WARNING] No frames read from: {video_path}")

    return frames


def save_frames_as_video(frames, output_dir, word, speaker, clip_name):
    """
    Saves frames as a playable video using a reliable codec.
    """
    if not frames:
        return

    word_dir = os.path.join(output_dir, word)
    os.makedirs(word_dir, exist_ok=True)

    output_path = os.path.join(word_dir, f"{word}_{speaker}_{clip_name}.avi")

    height, width, _ = frames[0].shape

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(output_path, fourcc, FPS, (width, height))

    if not out.isOpened():
        print("[ERROR] Failed to initialize video writer.")
        return

    for frame in frames:
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height))

        out.write(frame)

    out.release()
