import streamlit as st
from PIL import Image
from ultralytics import YOLO
import io

st.set_page_config(page_title="Meds Analyzer", page_icon="💊", layout="centered")

st.title("💊 Meds Analyzer")
st.write("Take a photo of your medicine to verify today's dose.")

# Initialize YOLO model
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")  # small pre-trained YOLOv8 model

model = load_model()

# File uploader for mobile camera capture
uploaded_file = st.file_uploader("📷 Upload your pill image", type=["jpg", "jpeg", "png"], accept_multiple_files=False)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    st.write("Analyzing...")
    # Convert PIL Image to bytes for YOLO
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='PNG')
    img_bytes.seek(0)

    # Run YOLO detection
    results = model.predict(img_bytes)

    # Display results
    for r in results:
        annotated_img = r.plot()
        st.image(annotated_img, caption="Detection Result", use_container_width=True)

        # Example feedback (you can replace with your pill database logic)
        if len(r.boxes.xyxy) > 0:
            st.success(f"✅ Pill detected! Total pills in image: {len(r.boxes.xyxy)}")
            st.info("Reminder: Check if you have taken today's dose.")
        else:
            st.warning("⚠ No pills detected! Did you forget your dose?")

st.markdown("---")
st.write("App works on mobile — just take a photo and upload!")
