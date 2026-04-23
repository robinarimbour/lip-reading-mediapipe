
import os

# Paths
DATASET_PATH = os.getenv("DATASET_PATH", "data/grid-corpus")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "data/landmarks")
ALIGN_PATH = "align"
VIDEO_EXTENSION = ".mpg"
MODEL_DIR = "models/v0"

# Data split
TRAINING_SET = ['s1_processed', 's2_processed', 's3_processed', 's4_processed', 
                's5_processed', 's6_processed', 's7_processed', 's8_processed']
TESTING_SET = ['s9_processed', 's10_processed']

# Model Config
FPS = 25
MAX_LEN = 30
