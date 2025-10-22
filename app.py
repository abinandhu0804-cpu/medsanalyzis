import streamlit as st
from PIL import Image
import numpy as np
import qrcode
from io import BytesIO
from ultralytics import YOLO

st.set_page_config(page_title="Meds Analyzer", page_icon="💊")

st.title("💊 Meds Analyzer")
st.write("Scan the QR code below on your phone to open this app and upload a photo of your medicine.")

# -----------------------------
# QR Code generation
# -----------------------------
app_url = "https://medsanalyz.streamlit.app/"  # Replace with your deployed URL
qr = qrcode.QRCode(box_size=6, border=2)
qr.add_data(app_url)
qr.make(fit=True)
img_qr = qr.make_image(fill_color="black", back_color="white")

# Display QR code
st.image(img_qr, caption="Scan to open on your phone", use_container_width=True)

# -----------------------------
# Load YOLO model
# -----------------------------
@st.cache_resource
def load_model():
    model_path = "models/best_pill_model.pt"  # Make sure this exists in your repo
    model = YOLO(model_path)
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")

# -----------------------------
# File uploader
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload a medicine image (jpg/png)", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None and model is not None:
    try:
        # Open uploaded image
        image = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(image)

        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Predict with YOLO
        results = model.predict(img_array)

        # Display results
        st.write("### Detection Results")
        for r in results:
            st.write(r.boxes)  # Show bounding boxes
            st.write(r.names)  # Show detected classes

        # Optional: Draw boxes on image
        annotated_img = results[0].plot()
        st.image(annotated_img, caption="Detected Pills", use_container_width=True)

    except Exception as e:
        st.error(f"Error processing image: {e}")
