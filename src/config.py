
import os

from dotenv import load_dotenv
load_dotenv()

# Paths
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "data/landmarks")
MODEL_DIR = os.getenv("MODEL_DIR", "models/v1")
ALIGN_PATH = "align"
VIDEO_EXTENSION = ".mpg"

# Model Config
FPS = 25
MAX_LEN = 16
