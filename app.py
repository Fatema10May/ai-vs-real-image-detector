import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# Page Configuration
st.set_page_config(
    page_title="AI vs Real Image Detector",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 AI vs Real Image Detector")
st.write("Upload an image to check whether it is **REAL** or **AI-Generated / FAKE**.")

# Load the Trained Model
@st.cache_resource
def load_keras_model():
    # Path to your saved model
    model = tf.keras.models.load_model('models/ai_real_detector.keras')
    return model

try:
    model = load_keras_model()
except Exception as e:
    st.error(f"Error loading model: {e}")

# File Uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    with st.spinner('Analyzing image...'):
        # 1. Image Preprocessing
        # Ensure image is RGB (converts PNG with alpha channels or grayscale to 3 channels)
        img = image.convert('RGB')
        
        # Resize image to model's input size (32x32)
        img = img.resize((32, 32))
        
        # Convert to numpy array and scale pixels to [0, 1]
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        # Expand dimensions to match model batch input: shape becomes (1, 32, 32, 3)
        img_array = np.expand_dims(img_array, axis=0)
        
        # 2. Model Prediction
        raw_pred = model.predict(img_array)[0][0]
        
        # 3. Decision Logic & Confidence Calculation
        # Assuming raw_pred < 0.5 represents REAL, and >= 0.5 represents FAKE
        if raw_pred < 0.5:
            label = "REAL"
            confidence = (1.0 - raw_pred) * 100.0
            st.success(f"Result: {label} Image")
        else:
            label = "AI-Generated / FAKE"
            confidence = raw_pred * 100.0
            st.warning(f"Result: {label} Image")
            
        # 4. Display Confidence Score
        st.info(f"Confidence: {confidence:.2f}%")
