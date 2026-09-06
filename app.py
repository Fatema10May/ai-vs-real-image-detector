from PIL import Image
import numpy as np
import streamlit as st
import tensorflow as tf


st.title("AI vs Real Image Detector")
st.write("Upload an image to check whether it's AI-generated or Real.")


@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model("ai_real_detector.keras")


model = load_my_model()

uploaded_file = st.file_uploader(
    "Choose an image...", type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:
    # ১. স্ক্রিনে ছবিটি দেখানো
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write("Classifying...")

    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((128, 128))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    
    raw_pred = model.predict(img_array)[0][0]

    if raw_pred > 0.5:
        confidence = raw_pred * 100
        st.success(f"Result: **Real Image** (Confidence: {confidence:.2f}%)")
    else:
        confidence = (1 - raw_pred) * 100
        st.error(
            f"Result: **AI Generated / Fake Image** (Confidence: {confidence:.2f}%)"
        )
