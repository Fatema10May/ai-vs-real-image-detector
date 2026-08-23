import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.title("🤖 AI vs Real Image Detector")
st.write("Upload an image to check whether it is Real or AI Generated.")

@st.cache_resource
def load_keras_model():
    return tf.keras.models.load_model('models/ai_real_detector.keras', compile=False)

model = load_keras_model()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    # Preprocessing
    img = image.resize((32, 32))
    img_array = np.array(img, dtype=np.float32) / 255.0  # Normalization
    img_array = np.expand_dims(img_array, axis=0)
    
    # Prediction
    raw_pred = model.predict(img_array)[0][0]
    
    st.markdown("---")
    
    # উল্টো লেবেল লজিক (raw_pred < 0.5 হলে REAL)
    if raw_pred < 0.5:
        confidence = (1 - raw_pred) * 100
        st.success(f"### Result: REAL Image")
        st.info(f"Confidence: {confidence:.2f}%")
    else:
        confidence = raw_pred * 100
        st.error(f"### Result: AI / FAKE Image")
        st.info(f"Confidence: {confidence:.2f}%")