
import cv2
import numpy as np
import matplotlib.pyplot as plt

from src.preprocessing.landmarks import *


# -----------------------------
# VISUALIZE LANDMARKS ON VIDEO
# -----------------------------
def visualize_on_video(video_path):
    """
    Displays lip landmarks overlaid on video frames for debugging.
    """
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]

            for idx in LIP_LANDMARKS + [ANCHOR_IDX] + [LEFT_FACE_IDX] + [RIGHT_FACE_IDX]:
                lm = landmarks.landmark[idx]
                x, y = int(lm.x * w), int(lm.y * h)

                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        cv2.imshow("Lip Landmarks", frame)
        if cv2.waitKey(40) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# -----------------------------
# VISUALIZE LIP LANDMARKS FOR A FRAME
# -----------------------------
def plot_landmarks(landmarks, frame_idx=0):
    """
    Plots normalized lip landmarks for a given frame.
    """
    frame = landmarks[frame_idx]
    points = frame.reshape(-1, 2)

    plt.figure(figsize=(5, 5))

    # scatter points
    plt.scatter(points[:, 0], points[:, 1],  c='blue', s=10)

    # connect points
    plt.plot(points[:, 0], points[:, 1], c='gray', alpha=0.5)

    plt.gca().invert_yaxis()
    plt.title(f"Normalized Lip Shape: {frame_idx}")
    plt.xlabel("Normalized X")
    plt.ylabel("Normalized Y")
    plt.grid(True)
    plt.show()
