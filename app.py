import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.resnet_v2 import preprocess_input

# Load Model
#@st.cache_resource
def load_model():
    return tf.keras.models.load_model('models/ai_real_detector.keras')

model = load_model()

st.title("AI vs Real Image Detector")
st.write("Upload an image to check if it's Real or AI-generated.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)

    with st.spinner('Analyzing image...'):
        # Image Preprocessing for ResNet50V2
        img = image.convert('RGB')
        img = img.resize((128, 128))
        
        img_array = np.array(img, dtype=np.float32)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        # Prediction
        raw_pred = model.predict(img_array)[0][0]

        # Output Logic
        if raw_pred < 0.5:
            st.success(f"Prediction: REAL Image (Confidence: {(1-raw_pred)*100:.2f}%)")
        else:
            st.error(f"Prediction: AI / FAKE Image (Confidence: {raw_pred*100:.2f}%)")
