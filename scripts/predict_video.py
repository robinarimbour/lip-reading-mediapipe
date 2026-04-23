
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from dotenv import load_dotenv
load_dotenv()

from src.config import MODEL_DIR
from src.modeling.model_loader import load_artifacts
from src.modeling.predictor import predict_video, play_video_with_predictions


if __name__ == "__main__":
    ALIGN_PATH = os.getenv("TEST_ALIGN_PATH", None)
    VIDEO_PATH = os.getenv("TEST_VIDEO_PATH", None)
    model, idx_to_word, mean, std = load_artifacts(MODEL_DIR)

    # Predict video
    predict_video(
        model, idx_to_word, mean, std,
        ALIGN_PATH, VIDEO_PATH
    )

    # Display predictions on video
    # play_video_with_predictions(
    #     model, idx_to_word, mean, std,
    #     ALIGN_PATH, VIDEO_PATH
    # )

