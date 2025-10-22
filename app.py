import streamlit as st
from ultralytics import YOLO
from PIL import Image
import io
import qrcode

st.set_page_config(page_title="Meds Analyzer", layout="centered")

st.title("Meds Analyzer")
st.write("Scan the QR code below to open this page on another device and upload your medicine photo.")

# --- Generate QR code for current URL ---
url = st.secrets.get("APP_URL", "https://medsanalyz.streamlit.app")  # Optional: set your app URL in Streamlit secrets
qr = qrcode.QRCode(box_size=6, border=2)
qr.add_data(url)
qr.make(fit=True)
img_qr = qr.make_image(fill_color="black", back_color="white")
st.image(img_qr, caption="Scan this QR code", use_container_width=True)

# --- Cache the YOLO model ---
@st.cache_resource
def load_model():
    model = YOLO("models/best_pill_model.pt")  # Use cleaned, compatible YOLO model
    return model

model = load_model()

st.write("---")
st.subheader("Upload your medicine photo:")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    # Convert to bytes for YOLO
    img_bytes = io.BytesIO()
    image.save(img_bytes, format="PNG")
    img_bytes = img_bytes.getvalue()

    # Run prediction
    try:
        results = model.predict(img_bytes)
        # Show results image
        result_img = results[0].plot()
        st.image(result_img, caption="Pill Detection Result", use_container_width=True)

        # Display detected pills info
        if results[0].boxes:
            st.success(f"Detected {len(results[0].boxes)} pill(s) in the image.")
        else:
            st.warning("No pills detected. Make sure the image is clear and taken from above.")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
