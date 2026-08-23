# 🤖 AI vs Real Image Detector

A Deep Learning-based web application that classifies images as either **Real** or **AI-Generated / Fake** using TensorFlow, Keras, and Streamlit.

## Features
- **High Accuracy:** CNN-based model trained on a balanced dataset achieving ~93% classification accuracy.
- **Real-Time Classification:** Upload any image (`.png`, `.jpg`, `.jpeg`) for instant analysis.
- **Confidence Score:** Displays probability score for the model's prediction.
- **Interactive UI:** Simple and user-friendly interface built with Streamlit.

---

## Tech Stack & Tools
- **Language:** Python
- **Frameworks:** TensorFlow, Keras
- **Web App:** Streamlit
- **Libraries:** NumPy, Pillow (PIL), Scikit-learn, Seaborn, Matplotlib

---

## Project Structure
```text
AI & Real Images_Detection/
│
├── models/
│   └── ai_real_detector.keras  # Saved Trained Keras Model
├── notebooks/
│   ├── 01_Data_Preprocessing.ipynb
│   ├── 02_Model_Training.ipynb
│   └── 04_Evaluation.ipynb     # Model Evaluation & Confusion Matrix
├── app.py                      # Streamlit Web Application Code
├── requirements.txt            # Project Dependencies
└── README.md                   # Project Documentation