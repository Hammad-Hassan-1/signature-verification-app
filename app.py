from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import streamlit as st

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("5keras_model.h5", compile=False)

# Load the labels and extract class names
class_names = [label.strip().split(" ", 1)[-1] for label in open("labels2.txt", "r").readlines()]

# Streamlit UI setup
st.set_page_config(page_title="Signature Verification", layout="centered")
st.title("🖊️ Signature Verification")
st.markdown("Upload a signature image to verify if it's **real** or **forged**. The model will analyze the image and provide a confidence score.")

# File uploader
uploaded_file = st.file_uploader("Upload a signature image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Load and display the uploaded image with use_container_width
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Signature", use_container_width=True)

        # Preprocess the image
        image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array

        # Make prediction
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index].strip()
        confidence_score = prediction[0][index] * 100

        # Display prediction result
        st.subheader("Prediction Result")
        if class_name.lower() == "real":
            st.success(f"✅ The signature is verified as real with {confidence_score:.2f}% confidence.")
        else:
            st.error(f"❌ The signature is detected as forged with {confidence_score:.2f}% confidence.")

        # Display class probabilities
        st.markdown("### Class Probabilities")
        probabilities = {label.strip(): prediction[0][i] * 100 for i, label in enumerate(class_names)}
        for label, prob in probabilities.items():
            st.write(f"{label}: {prob:.2f}%")

    except Exception as e:
        st.error(f"Error processing the image: {e}")