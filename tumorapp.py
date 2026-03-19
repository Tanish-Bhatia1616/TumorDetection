import streamlit as st
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from tensorflow.keras.models import load_model
import os

# Load your trained model
model = load_model("C:/Users/Hp/OneDrive/Desktop/Deep learning/mri_model.h5")

# Ensure class labels match training order
class_labels = sorted(os.listdir("C:/Users/Hp/OneDrive/Desktop/Deep learning/MRI images/Training"))

def predict_image(uploaded_file, model):
    img = load_img(uploaded_file, target_size=(128,128))  # file-like object works here
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions)
    confidence_score = np.max(predictions)

    return class_labels[predicted_class_index], confidence_score

# Streamlit UI
st.title("🧠 MRI Tumor Detection")
uploaded_file = st.file_uploader("Upload an MRI image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded MRI", use_column_width=True)
    label, confidence = predict_image(uploaded_file, model)
    if label == "notumor":
        st.success(f"No tumor detected (Confidence: {confidence*100:.2f}%)")
    else:
        st.error(f"Tumor detected: {label} (Confidence: {confidence*100:.2f}%)")