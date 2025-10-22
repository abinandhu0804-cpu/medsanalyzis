import streamlit as st
from ultralytics import YOLO
from PIL import Image
import io
import qrcode
import numpy as np

# ---------- PAGE SETUP ----------
st.set_page_config(
    page_title="Meds Analyzer",
    page_icon="💊",
    layout="centered"
)

st.title("💊 Meds Analyzer")
st.write("Scan the QR code below on your phone to open this app and upload a photo of your medicine.")

# ---------- GENERATE QR CODE ----------
app_url = "https://medsanalyz.streamlit.app/"  # Replace with your deployed URL
qr = qrcode.QRCode(box_size=6, border=2)
qr.add_data(app_url)
qr.make(fit=True)
img_qr = qr.make_image(fill_color="black", back_color="white")
st.image(img_qr, caption="Scan to open on your phone", use_container_width=True)

st.markdown("---")

# ---------- FILE UPLOAD ----------
uploaded_file = st.file_uploader(
    "Take a photo of your medicine", 
    type=["png", "jpg", "jpeg"], 
    accept_multiple_files=False
)

# ---------- LOAD MODEL ----------
@st.cache_resource
def load_model():
    try:
        model = YOLO("models/best_pill_model.pt")  # Make sure this path exists
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# ---------- PREDICTION ----------
if uploaded_file:
    model = load_model()
    if model:
        img_bytes = uploaded_file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        with st.spinner("Analyzing pill..."):
            try:
                results = model.predict(image)

                for result in results:
                    boxes = result.boxes
                    if boxes is not None and len(boxes) > 0:
                        st.success(f"Pill detected! Number of pills: {len(boxes)}")
                        annotated_image = result.plot()
                        st.image(annotated_image, caption="Detection Result", use_container_width=True)
                    else:
                        st.warning("No pill detected! Have you taken your dose?")
            except Exception as e:
                st.error(f"Error during prediction: {e}")
