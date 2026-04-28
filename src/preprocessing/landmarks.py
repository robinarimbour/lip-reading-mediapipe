
import numpy as np
import cv2
import mediapipe as mp


# -----------------------------
# MEDIAPIPE SETUP
# -----------------------------
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=False
)


# -----------------------------
# FACE MESH LANDMARKS
# -----------------------------
LIP_LANDMARKS = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    291, 308, 324, 318, 402, 317, 14, 87, 178, 88,
    95, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    415, 310, 311, 312, 13, 82, 81, 42, 183, 78
]

LEFT_FACE_IDX = 234 
RIGHT_FACE_IDX = 454


def extract_frame_landmarks(frame, last_valid_width, draw=False):
    """
    Extracts and normalizes lip landmarks from a single frame using face mesh.
    """
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return None, last_valid_width

    face_landmarks = results.multi_face_landmarks[0]

    # ---- Vectorized lip points ----
    lip_points = np.array([
        [face_landmarks.landmark[idx].x, face_landmarks.landmark[idx].y]
        for idx in LIP_LANDMARKS
    ])

    # 1. Lip centroid (anchor)
    centroid_x, centroid_y = lip_points.mean(axis=0)

    # 2. Face width (scale)
    p1 = face_landmarks.landmark[LEFT_FACE_IDX]
    p2 = face_landmarks.landmark[RIGHT_FACE_IDX]

    dx = p2.x - p1.x
    dy = p2.y - p1.y

    face_width = np.sqrt(dx**2 + dy**2)

    if face_width < 1e-3:
        face_width = last_valid_width
    else:
        last_valid_width = face_width

    # 3. Rotation (tilt correction)
    angle = np.arctan2(dy, dx)
    cos_a = np.cos(-angle)
    sin_a = np.sin(-angle)

    # ---- Normalize all points ----
    normalized = []

    for (x, y), idx in zip(lip_points, LIP_LANDMARKS):
        if draw:
            px = int(x * w)
            py = int(y * h)
            cv2.circle(frame, (px, py), 2, (0, 255, 0), -1)

        # Translate
        x -= centroid_x
        y -= centroid_y

        # Rotate
        x_rot = x * cos_a - y * sin_a
        y_rot = x * sin_a + y * cos_a

        # Scale
        norm_x = x_rot / face_width
        norm_y = y_rot / face_width

        normalized.extend([norm_x, norm_y])

    return np.array(normalized), last_valid_width


def extract_lip_landmarks(frames):
    """
    Converts a sequence of frames into a sequence of normalized lip landmark vectors.
    """
    sequence = []
    last_valid_width = 1.0

    for frame in frames:
        coords, last_valid_width = extract_frame_landmarks(
            frame, last_valid_width
        )

        if coords is not None:
            sequence.append(coords)
    
    if not sequence:
        return None
    
    return np.array(sequence, dtype=np.float32)
