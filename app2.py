import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Page config
st.set_page_config(page_title="🌿 Plant Disease Classifier", page_icon="🌿", layout="centered")

# Custom CSS styling
st.markdown("""
    <style>
        .main {
            background-color: #f0fff0;
        }
        .title {
            font-size: 36px;
            font-weight: bold;
            color: #2e7d32;
            text-align: center;
            margin-bottom: 20px;
        }
        .footer {
            text-align: center;
            font-size: 14px;
            color: gray;
            margin-top: 50px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🌱 Plant Disease Classifier</div>', unsafe_allow_html=True)
st.write("Upload up to **2 images** of plant leaves to detect possible diseases.")

# Load model
model = tf.keras.models.load_model("plant_disease_model_compressed .h5")
class_names = ['Healthy 🌿', 'Powdery ❄️', 'Rust 🍂']

# Multiple file upload
uploaded_files = st.file_uploader("📤 Choose up to 2 leaf images...", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    if len(uploaded_files) > 2:
        st.warning("⚠️ Please upload only 2 images at most.")
    else:
        cols = st.columns(len(uploaded_files))  # Create 1 or 2 columns

        for i, uploaded_file in enumerate(uploaded_files):
            with cols[i]:
                img = Image.open(uploaded_file).convert('RGB')
                st.image(img, caption=f"🖼️ Image {i+1}", use_container_width=True)

                st.write("🧠 Classifying...")
                img_resized = img.resize((128, 128))
                img_array = np.array(img_resized) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                prediction = model.predict(img_array)
                predicted_index = np.argmax(prediction)
                predicted_class = class_names[predicted_index]
                confidence = prediction[0][predicted_index] * 100

                st.success(f"✅ Prediction: **{predicted_class}**")
                st.info(f"📊 Confidence: **{confidence:.2f}%**")

                st.write("🔍 Probability for each class:")
                for j, class_name in enumerate(class_names):
                    st.write(f"- {class_name}: {prediction[0][j]*100:.2f}%")

st.markdown('<div class="footer">Made with ❤️ for farmers and plant lovers</div>', unsafe_allow_html=True)
