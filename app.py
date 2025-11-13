import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

# ---------------------------------------------------------
# 🧠 App Configuration
# ---------------------------------------------------------
st.set_page_config(page_title="Fish Image Classification", page_icon="🐟", layout="centered")
st.title("🐟 Multiclass Fish Image Classification")
st.write("Upload an image of a fish and let the AI predict its category using a trained deep learning model.")

# ---------------------------------------------------------
# 🔹 Load Model
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("best_fish_model.h5")  # ✅ Change this to your model name if different
    return model

model = load_model()

# ---------------------------------------------------------
# 📚 Class Names
# ---------------------------------------------------------
# ⚠️ You can automatically load your dataset folder names if you want:
# dataset_dir = "data/train"
# class_names = sorted(os.listdir(dataset_dir))
# OR manually define if you know your categories:
class_names = ['Black_Sea_Sprat', 'Gilt_Head_Bream', 'Hourse_Mackerel',
               'Red_Mullet', 'Sea_Bass', 'Striped_Red_Mullet', 'Trout']

# ---------------------------------------------------------
# 📁 File Upload
# ---------------------------------------------------------
uploaded_file = st.file_uploader("Upload a fish image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and show image
    image = Image.open(uploaded_file).convert("RGB")

    # ---------------------------------------------------------
    # 🧩 Preprocess Image
    # ---------------------------------------------------------
    target_size = model.input_shape[1:3]  # automatically use model input size (e.g., 224x224)
    img = image.resize(target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # ---------------------------------------------------------
    # 🔮 Model Prediction
    # ---------------------------------------------------------
    predictions = model.predict(img_array)[0]
    sorted_indices = np.argsort(predictions)[::-1]

    # ---------------------------------------------------------
    # 🐟 Draw Predicted Name on Image
    # ---------------------------------------------------------
    top_index = sorted_indices[0]
    top_class = class_names[top_index] if top_index < len(class_names) else f"Class {top_index}"
    top_confidence = predictions[top_index] * 100
    label = f"{top_class} ({top_confidence:.2f}%)"

    # Draw label box and text on image
    draw = ImageDraw.Draw(image)
    box_width = 10 + len(label) * 10
    draw.rectangle([(10, 10), (box_width, 50)], fill=(0, 0, 0))
    draw.text((15, 15), label, fill="white")

    # Display image with overlay
    st.image(image, caption=f"Predicted: {label}", use_container_width=True)

    # ---------------------------------------------------------
    # 📊 Show Detailed Results
    # ---------------------------------------------------------
    st.subheader("📊 Prediction Results (Top 5):")

    num_classes = min(len(class_names), len(predictions))
    for i in sorted_indices[:min(5, num_classes)]:
        name = class_names[i] if i < len(class_names) else f"Class {i}"
        st.write(f"{name}: **{predictions[i]*100:.2f}%**")

    st.success(f"✅ Predicted Fish: {top_class} ({top_confidence:.2f}%)")

# ---------------------------------------------------------
# ℹ️ Footer
# ---------------------------------------------------------
st.markdown("---")
st.caption("Developed by **Pasupathi Raj** | Powered by TensorFlow & Streamlit 🧠")
