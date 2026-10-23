import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "animal_classifier.keras"
LABELS_PATH = BASE_DIR / "class_names.json"
IMG_SIZE = (180, 180)

st.set_page_config(
    page_title="Animal Detection AI",
    page_icon="🐾",
    layout="centered",
)

st.markdown(
    """
    <style>
    .main-title {text-align:center; font-size:2.35rem; font-weight:800;}
    .subtitle {text-align:center; color:#64748b; margin-bottom:1.4rem;}
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown('<div class="main-title">🐾 Animal Detection AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Upload an animal image and let the AI model predict its class.</div>',
    unsafe_allow_html=True,
)

with st.expander("Supported animals"):
    st.write("Cats · Dogs · Elephants · Lion · Tigers")

if not MODEL_PATH.exists() or not LABELS_PATH.exists():
    st.error(
        "The trained model files are missing. Add both "
        "`animal_classifier.keras` and `class_names.json` beside `app.py` "
        "in your GitHub repository, then redeploy."
    )
    st.stop()

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_data
def load_class_names():
    with LABELS_PATH.open("r", encoding="utf-8") as f:
        names = json.load(f)
    if not isinstance(names, list) or not names:
        raise ValueError("class_names.json must contain a non-empty JSON list.")
    return names

try:
    model = load_model()
    class_names = load_class_names()
except Exception as exc:
    st.error(f"Could not load the model: {exc}")
    st.stop()

uploaded_file = st.file_uploader(
    "Choose an animal picture",
    type=["jpg", "jpeg", "png"],
    help="Upload a clear picture in JPG, JPEG, or PNG format.",
)

if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Your uploaded image", use_container_width=True)

        if st.button("🔍 Detect Animal", type="primary", use_container_width=True):
            with st.spinner("Analyzing image..."):
                resized = image.resize(IMG_SIZE)
                array = np.asarray(resized, dtype=np.float32)
                array = np.expand_dims(array, axis=0)
                predictions = model.predict(array, verbose=0)[0]

                if len(predictions) != len(class_names):
                    st.error(
                        "The number of model outputs does not match class_names.json. "
                        "Please check the class names saved with your Kaggle model."
                    )
                    st.stop()

                best_index = int(np.argmax(predictions))
                predicted_label = class_names[best_index]
                score = float(predictions[best_index]) * 100

            st.success(f"Predicted animal: **{predicted_label}**")
            st.metric("Model score", f"{score:.2f}%")
            st.subheader("Scores for all animal classes")
            for label, value in zip(class_names, predictions):
                st.write(f"**{label}** — {float(value) * 100:.2f}%")
                st.progress(float(np.clip(value, 0, 1)))

            st.caption(
                "Model scores are not guaranteed confidence levels. "
                "Prediction quality depends on the training data."
            )
    except Exception as exc:
        st.error(f"Unable to process this image: {exc}")

st.divider()
st.caption("Animal Detection AI · Built with Streamlit and TensorFlow")
