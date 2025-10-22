import streamlit as st
import qrcode
import tempfile
import numpy as np
from PIL import Image
from ultralytics import YOLO

# -----------------------------
# YOLO model initialization
# -----------------------------
@st.cache_resource
def load_model():
    model = YOLO("models/best_pill_model.pt")  # Replace with your YOLO model path
    return model

model = load_model()

# -----------------------------
# App layout
# -----------------------------
st.set_page_config(page_title="Meds Analyzer", page_icon="💊")
st.title("💊 Meds Analyzer")
st.write("Scan the QR code below on your phone to upload a photo of your medicine:")

# -----------------------------
# Generate QR code pointing to the same app URL
# -----------------------------
app_url = st.secrets.get("APP_URL", "https://medsanalyz.streamlit.app/")  # Replace with your deployed URL
qr = qrcode.QRCode(box_size=6, border=2)
qr.add_data(app_url)
qr.make(fit=True)
img_qr = qr.make_image(fill_color="black", back_color="white")
st.image(img_qr, use_container_width=False)

st.write("---")
st.header("Upload your medicine photo")

# -----------------------------
# File uploader for camera or gallery
# -----------------------------
uploaded_file = st.file_uploader(
    "Take a photo or upload an image", 
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=False
)

if uploaded_file is not None:
    # Open and display the image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    st.write("Analyzing your medicine... 💊")

    # Convert to numpy array for YOLO
    img_np = np.array(image)

    # Run YOLO detection
    results = model.predict(img_np)

    # Display results
    for r in results:
        annotated_img = r.plot()
        st.image(annotated_img, caption="Detection Result", use_container_width=True)

        if len(r.boxes.xyxy) > 0:
            st.success(f"✅ Pill detected! Total pills in image: {len(r.boxes.xyxy)}")
            st.info("Reminder: Check if you have taken today's dose.")
        else:
            st.warning("⚠ No pills detected! Did you forget your dose?")
