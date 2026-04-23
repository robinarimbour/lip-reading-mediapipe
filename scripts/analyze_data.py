
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from dotenv import load_dotenv
load_dotenv()

from src.config import TRAINING_SET
from src.analysis.dataset_analysis import max_word_frames


if __name__ == "__main__":
    # Find word with max frames
    max_word_frames(TRAINING_SET[:3])
