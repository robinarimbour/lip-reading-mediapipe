
import cv2
import numpy as np

from src.config import MAX_LEN
from src.preprocessing.landmarks import extract_frame_landmarks
from src.preprocessing.pipeline import pad_sequence


def realtime_inference(model, idx_to_word):
    """
    Perform real-time lip reading using webcam input.

    Press 'r' to start recording a sequence.
    Press 'q' to quit.
    """

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Cannot access webcam")
        return

    sequence = []
    recording = False
    prediction_text = ""
    last_valid_width = 1.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        key = cv2.waitKey(1) & 0xFF

        # Start recording
        if key == ord('r'):
            recording = True
            sequence = []
            prediction_text = "Recording..."

        # Capture frames
        if recording:
            coords, last_valid_width = extract_frame_landmarks(
                frame, last_valid_width, draw=True
            )

            if coords is not None:
                sequence.append(coords)
            else:
                prediction_text = "Face not detected"

            # Stop when enough frames collected
            if len(sequence) >= MAX_LEN:
                recording = False

                seq = pad_sequence(np.array(sequence))
                seq = np.expand_dims(seq, axis=0)

                pred = model.predict(seq, verbose=0)

                word = idx_to_word[np.argmax(pred)]
                confidence = float(np.max(pred))

                prediction_text = f"{word} ({confidence:.2f})"

        # Display text
        cv2.putText(frame, prediction_text,
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2)

        # Instructions
        cv2.putText(frame, "Press 'r' to record | 'q' to quit",
                    (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2)

        cv2.imshow("Lip Reading Webcam", frame)

        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
