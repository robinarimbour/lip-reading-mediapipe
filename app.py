
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import streamlit as st
import tempfile

from src.config import MODEL_DIR
from src.modeling.model_loader import load_artifacts
from src.modeling.predictor import predict_video


# -------------------------------
# UI CONFIG
# -------------------------------
st.set_page_config(page_title="Lip Reading App", layout="centered")

st.title("Lip Reading using MediaPipe")
st.markdown("Upload a video and get predicted speech from lip movements.")


# -------------------------------
# LOAD RESOURCES
# -------------------------------
@st.cache_resource
def load_model_cached():
    return load_artifacts(MODEL_DIR)

model, idx_to_word = load_model_cached()


# -------------------------------
# FILE UPLOAD
# -------------------------------
video_file = st.file_uploader("Upload Video")

if video_file is not None:
    col_video, col_pred = st.columns([2, 1])

    with col_video:
        st.subheader("🎥 Video")
        st.video(video_file)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        temp_file.write(video_file.read())
        temp_video_path = temp_file.name

    try:
        with st.spinner("Processing video... ⏳"):
            results = predict_video(
                model, idx_to_word,
                temp_video_path
            )

        with col_pred:
            st.subheader("📊 Prediction")

            if results:
                predicted_word = results["prediction"]
                confidence = results["confidence"]

                st.metric("Word", predicted_word)
                st.metric("Confidence", f"{confidence * 100:.2f}%")

                st.progress(float(confidence))
            else:
                st.warning("No prediction could be made.")

    except Exception as e:
        st.error(f"Prediction failed: {e}")

    finally:
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
