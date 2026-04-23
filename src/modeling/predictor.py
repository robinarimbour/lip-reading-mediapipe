
import numpy as np
import cv2

from src.config import FPS
from src.preprocessing.align import read_align_file
from src.preprocessing.video import time_to_frame, load_video_frames
from src.preprocessing.landmarks import extract_lip_landmarks
from src.preprocessing.pipeline import pad_sequence


# -----------------------------
# GET PREDICTIONS FROM VIDEO
# -----------------------------
def predict_video(model, idx_to_word, mean, std, align_path, video_path):
    """
    Performs word-level predictions on a video using alignment segments and returns predictions with accuracy.
    """
    segments = read_align_file(align_path)
    frames = load_video_frames(video_path)

    correct = 0
    total = 0
    predictions = []
    for start, end, word_true in segments:
        start_f = time_to_frame(start, FPS)
        end_f = time_to_frame(end, FPS)

        word_frames = frames[start_f:end_f]

        landmarks = extract_lip_landmarks(word_frames)
        seq = pad_sequence(landmarks)

        seq = (seq - mean) / std
        seq = np.expand_dims(seq, axis=0)

        pred = model.predict(seq, verbose=0)

        word_pred = idx_to_word[np.argmax(pred)]
        confidence = np.max(pred)

        print(f"True: {word_true} | Pred: {word_pred} ({confidence:.2f})")
        
        if word_pred == word_true:
            correct += 1
        total += 1

        predictions.append({
            "start": start_f,
            "end": end_f,
            "true": word_true,
            "pred": word_pred,
            "conf": confidence
        })

    print("\nAccuracy:", correct / total)

    return predictions


# -----------------------------
# DISPLAY PREDICTIONS ON VIDEO
# -----------------------------
def play_video_with_predictions(model, idx_to_word, mean, std, align_path, video_path):
    """
    Plays a video with overlaid predicted words and confidence scores for each segment.
    """
    predictions = predict_video(
        model, idx_to_word, mean, std,
        align_path, video_path
    )

    # 🎥 Play video with overlay
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Find current segment
        current_text = ""
        true_text = ""
        for seg in predictions:
            if seg["start"] <= frame_idx < seg["end"]:
                current_text = f"{seg['pred']} ({seg['conf']:.2f})"
                true_text = f"True: {seg['true']}"
                break

        # Draw text
        cv2.putText(frame, current_text,
                    (50, 50),
                    cv2.FONT_HERSHEY_COMPLEX,
                    1,
                    (0, 255, 0) if seg['pred'] == seg['true'] else (0, 0, 255),
                    2)
        
        cv2.putText(frame, true_text,
                    (50, 90),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.8,
                    (255, 0, 0),
                    2)

        cv2.imshow("Lip Reading Prediction", frame)

        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
