
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
    refine_landmarks=True
)

# -----------------------------
# FACE MESH LANDMARKS
# -----------------------------
# Lip landmark indices
LIP_LANDMARKS = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    291, 308, 324, 318, 402, 317, 14, 87, 178, 88,
    95, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    415, 310, 311, 312, 13, 82, 81, 42, 183, 78
]

# Nose tip for centering
ANCHOR_IDX = 4

# Indices for scaling (left-most and right-most face points)
LEFT_FACE_IDX = 234 
RIGHT_FACE_IDX = 454


# -----------------------------
# EXTRACT LANDMARKS FROM FRAMES
# -----------------------------
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

    # 1. Lip centroid (anchor)
    lip_points = []
    for idx in LIP_LANDMARKS:
        lm = face_landmarks.landmark[idx]
        lip_points.append([lm.x, lm.y])

    lip_points = np.array(lip_points)
    centroid_x, centroid_y = lip_points.mean(axis=0)

    # 2. Face width (scale)
    p1 = face_landmarks.landmark[LEFT_FACE_IDX]
    p2 = face_landmarks.landmark[RIGHT_FACE_IDX]

    dx = p2.x - p1.x
    dy = p2.y - p1.y

    face_width = np.sqrt(dx**2 + dy**2)

    if face_width < 0.01:
        face_width = last_valid_width
    else:
        last_valid_width = face_width

    # 3. Rotation (tilt correction)
    angle = np.arctan2(dy, dx)
    cos_a = np.cos(-angle)
    sin_a = np.sin(-angle)

    # 4. Lip coordinates
    lip_coords = []

    for idx in LIP_LANDMARKS:
        lm = face_landmarks.landmark[idx]

        # ---- DRAW LANDMARK ----
        if draw:
            px = int(lm.x * w)
            py = int(lm.y * h)
            cv2.circle(frame, (px, py), 2, (0, 255, 0), -1)

        # Translate
        x = lm.x - centroid_x
        y = lm.y - centroid_y

        # Rotate
        x_rot = x * cos_a - y * sin_a
        y_rot = x * sin_a + y * cos_a

        # Scale
        norm_x = x_rot / face_width
        norm_y = y_rot / face_width

        lip_coords.extend([norm_x, norm_y])

    return lip_coords, last_valid_width


def extract_lip_landmarks(frames):
    """
    Converts a sequence of frames into a sequence of normalized lip landmark vectors.
    """
    sequence = []
    last_valid_width = 1.0

    for frame in frames:
        coords, last_valid_width = extract_frame_landmarks(frame, last_valid_width)

        if coords is not None:
            sequence.append(coords)
        else:
            # fallback (same as before)
            if len(sequence) > 0:
                sequence.append(sequence[-1])
            else:
                sequence.append([0.0] * (len(LIP_LANDMARKS) * 2))

    return np.array(sequence)
