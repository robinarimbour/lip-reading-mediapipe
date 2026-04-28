
import cv2
import matplotlib.pyplot as plt

from src.preprocessing.landmarks import (
    face_mesh,
    LIP_LANDMARKS,
    LEFT_FACE_IDX,
    RIGHT_FACE_IDX,
)


def visualize_on_video(video_path, playback_speed=40):
    """
    Displays lip landmarks overlaid on video frames for debugging.

    Args:
        video_path (str): Path to input video
        playback_speed (int): Delay between frames (lower = faster)

    Controls:
        Press 'q' to close the window.
    """
    if not video_path:
        raise ValueError("video_path must be provided")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]

            for idx in LIP_LANDMARKS + [LEFT_FACE_IDX] + [RIGHT_FACE_IDX]:
                lm = landmarks.landmark[idx]
                x, y = int(lm.x * w), int(lm.y * h)

                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        cv2.imshow("Lip Landmarks (Press 'q' to quit)", frame)

        if cv2.waitKey(playback_speed) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def plot_landmarks(landmarks, frame_idx=0):
    """
    Plots normalized lip landmarks for a given frame.

    Args:
        landmarks (np.ndarray): Shape (T, features)
        frame_idx (int): Frame index to visualize
    """
    total_frames = len(landmarks)

    if frame_idx < 0:
        print(f"[WARNING] frame_idx {frame_idx} < 0. Using 0 instead.")
        frame_idx = 0
    elif frame_idx >= total_frames:
        print(f"[WARNING] frame_idx {frame_idx} out of range. Using last frame ({total_frames - 1}).")
        frame_idx = total_frames - 1

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
