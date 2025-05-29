import streamlit as st
import cv2
import numpy as np
from PIL import Image

def blur_background(image):
    # Convert PIL image to OpenCV format
    image = np.array(image.convert('RGB'))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Simple full image blur for demo (replace with your AI blur code)
    blurred = cv2.GaussianBlur(image, (21, 21), 0)

    # Convert back to RGB for Streamlit display
    blurred = cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB)
    return blurred

st.title("Background Blur AI Web App")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Original Image', use_column_width=True)

    if st.button("Blur Background"):
        result = blur_background(image)
        st.image(result, caption='Blurred Background', use_column_width=True)
