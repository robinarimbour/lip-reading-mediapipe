
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from dotenv import load_dotenv
load_dotenv()

from src.config import TRAINING_SET
from src.preprocessing.pipeline import process_dataset


if __name__ == "__main__":
    # Process words into landmarks
    process_dataset(TRAINING_SET[:2], True)
    