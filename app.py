import streamlit as st
from PIL import Image
from ultralytics import YOLO

# Load your trained pill detection model
model = YOLO("models/best_pill_model.pt")

st.title("Meds Analyzer")
st.write("Upload your medicine photo to check doses.")

uploaded_file = st.file_uploader("Take a photo of your pill", type=["jpg","png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)
    
    # Run YOLO inference
    results = model.predict(img)
    
    # Display results
    if results[0].boxes:
        for i, box in enumerate(results[0].boxes):
            pill_class = results[0].names[box.cls[0].item()]
            st.success(f"Pill {i+1}: {pill_class}")
    else:
        st.warning("No pill detected. Please try again.")
