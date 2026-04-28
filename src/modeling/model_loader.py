
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
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    if not label_map_path.exists():
        raise FileNotFoundError(f"Label map not found: {label_map_path}")

    # Load model
    model = load_model(model_path)

    # Load label mapping
    with open(label_map_path, "rb") as f:
        word_to_idx = pickle.load(f)

    idx_to_word = {v: k for k, v in word_to_idx.items()}
    
    return model, idx_to_word
