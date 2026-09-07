import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Page config
st.set_page_config(page_title="AI vs Real Image Detector", page_icon="🖼️", layout="centered")

st.title("🖼️ AI / FAKE vs REAL Image Detector")
st.write("Upload an image to test whether it is Real or AI-generated.")

# Load Trained Model
@st.cache_resource
def load_detection_model():
    # model.keras-এর জায়গায় আপনার মডেলে আসল নাম থাকলে সেটি দিন
    return tf.keras.models.load_model("model.keras") 

try:
    model = load_detection_model()
except Exception as e:
    st.error(f"Error loading model. Please ensure 'model.keras' is in the folder: {e}")
    st.stop()

# File Uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocessing (matching 32x32 target size)
    img_resized = img.resize((32, 32))
    img_array = np.array(img_resized)

    if img_array.ndim == 2:
        img_array = np.stack((img_array,)*3, axis=-1)
    elif img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]

    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    with st.spinner("Analyzing image..."):
        result = model.predict(img_array)
        confidence = float(result[0][0])

    st.write("---")

    if confidence > 0.5:
        score = confidence * 100
        st.success("**Prediction: REAL Image**")
        st.metric(label="Confidence Level", value=f"{score:.2f}%")
    else:
        score = (1 - confidence) * 100
        st.error("**Prediction: AI / FAKE Image**")
        st.metric(label="Confidence Level", value=f"{score:.2f}%")
