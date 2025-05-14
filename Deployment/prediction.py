import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load model sekali saja (cached)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("best_model.h5")

model = load_model()
class_names = ['fire', 'non-fire', 'smoke']  # Ganti sesuai klasemu

# Upload image
uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    img_resized = img.resize((150, 150))
    img_array = np.expand_dims(np.array(img_resized) / 255.0, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    # Output
    st.success(f"✅ Prediction: **{predicted_class}**")
    st.write(f"Confidence: **{confidence:.2%}**")