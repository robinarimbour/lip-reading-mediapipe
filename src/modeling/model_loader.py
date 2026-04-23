
from pathlib import Path
import pickle
from tensorflow.keras.models import load_model


def load_artifacts(model_dir):
    """
    Loads the trained model, label mappings, and normalization parameters for inference.
    """
    model_dir = Path(model_dir)

    model_path = model_dir / "lip_reading_model.h5"
    label_map_path = model_dir / "label_map.pkl"
    norm_path = model_dir / "norm.pkl"
    
    model = load_model(model_path)

    with open(label_map_path, "rb") as f:
        word_to_idx = pickle.load(f)

    idx_to_word = {v: k for k, v in word_to_idx.items()}

    with open(norm_path, "rb") as f:
        mean, std = pickle.load(f)
    
    return model, idx_to_word, mean, std
