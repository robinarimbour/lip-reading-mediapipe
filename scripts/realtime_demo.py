
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from dotenv import load_dotenv
load_dotenv()

from src.config import MODEL_DIR
from src.modeling.model_loader import load_artifacts
from src.modeling.realtime import realtime_inference


if __name__ == "__main__":
    model, idx_to_word, mean, std = load_artifacts(MODEL_DIR)

    realtime_inference(model, idx_to_word, mean, std)
