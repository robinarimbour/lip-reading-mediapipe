
import cv2


# -----------------------------
# CONVERT TIME → FRAME INDEX
# -----------------------------
def time_to_frame(time_val, fps):
    """
    Converts timestamp values from alignment files into frame indices.
    """
    seconds = time_val / 25000
    return int(seconds * fps)


# -----------------------------
# LOAD FRAMES FROM VIDEO
# -----------------------------
def load_video_frames(video_path):
    """
    Loads all frames from a video file into memory.
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error opening video")
        return
    
    # total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # print(f"Total frames: {total_frames}")

    # Read all frames once
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    # print("Read frames:", len(frames))
    return frames
