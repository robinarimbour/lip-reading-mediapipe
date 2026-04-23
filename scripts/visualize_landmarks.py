
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from dotenv import load_dotenv
load_dotenv()

import numpy as np
from src.utils.visualization import visualize_on_video, plot_landmarks


if __name__ == "__main__":
    # Visualization (optional)
    # video_path = os.getenv("TEST_VIDEO_PATH", None)
    # visualize_on_video(video_path)

    npy_path = os.getenv("TEST_NPY_PATH", None)
    landmarks = np.load(npy_path)
    print("Shape:", landmarks.shape)
    plot_landmarks(landmarks)
