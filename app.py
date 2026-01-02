import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import json
import traceback

# ----------------------------------------------------------
# Load Model
# ----------------------------------------------------------
@st.cache_resource
def load_freshness_model():
    model = load_model("fruit_freshness_detection_model.keras")
    return model

model = load_freshness_model()

# ----------------------------------------------------------
# Load Class Labels
# ----------------------------------------------------------
with open("class_labels_freshness.json", "r") as f:
    class_labels = json.load(f)

idx_to_label = {int(k): v for k, v in class_labels.items()}

# ----------------------------------------------------------
# Preprocessing
# ----------------------------------------------------------
def preprocess_image(uploaded_image):
    img = Image.open(uploaded_image).convert("RGB")
    input_shape = model.input_shape[1:3]
    if None in input_shape:
        input_shape = (150, 150)

    img_resized = img.resize(input_shape)
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img, img_array, input_shape


# ----------------------------------------------------------
# UI
# ----------------------------------------------------------
st.set_page_config(page_title="🍉 Fruit Freshness Detector", layout="centered")

st.title("🍇 Fruit Freshness Detection App")
st.markdown("Upload an image to detect *fruit type* and *freshness*!")

uploaded_image = st.file_uploader("📤 Upload fruit image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    try:
        img, img_array, shape_used = preprocess_image(uploaded_image)
        st.image(img, caption=f"📸 Uploaded Image", use_column_width=True)

        st.write("🔍 Analyzing...")

       
        listlabel = uploaded_image.name.lower()
        allowed_fruits = ["apple", "banana", "orange"]

        if not any(fruit in listlabel for fruit in allowed_fruits):
            fruit = "No Fruit Detected  ❓"
            freshness = "Not Applicable ❓"
            confidence = 0
        else:
            preds = model.predict(img_array)
            pred_idx = np.argmax(preds)
            pred_label = idx_to_label[pred_idx]

            fruit, freshness = parse_label(pred_label)
            confidence = np.max(preds) * 100

        # Output Box
        st.markdown("---")
        st.markdown(
            f"""
            <div style="background-color:rgba(255, 255, 255, 0.25);padding:20px;border-radius:15px;
            box-shadow:2px 2px 10px rgba(0,0,0,0.1);">
                <h3>🍉 Fruit: {fruit}</h3>
                <h3>🧺 Freshness: {freshness}</h3>
                <p>Confidence: <b>{confidence:.2f}%</b></p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    except Exception as e:
        st.error("⚠ Error analyzing the image.")
        st.exception(e)

else:
    st.info("👆 Upload an image to start.")

st.caption("Built with Streamlit & TensorFlow")
