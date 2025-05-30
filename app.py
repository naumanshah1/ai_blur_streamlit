import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
import io

st.title("AI Background Blur App")

# Load Mediapipe Selfie Segmentation model
mp_selfie_segmentation = mp.solutions.selfie_segmentation
segment = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

def blur_background(image, blur_intensity):
    # Convert PIL to OpenCV format
    image_np = np.array(image.convert('RGB'))
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Get segmentation mask
    results = segment.process(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    mask = results.segmentation_mask

    condition = np.stack((mask,) * 3, axis=-1) > 0.6

    # Blur background with intensity (kernel size)
    blurred_image = cv2.GaussianBlur(image_bgr, (blur_intensity, blur_intensity), 0)

    # Combine original + blurred using mask
    output_image = np.where(condition, image_bgr, blurred_image)

    # Convert back to RGB for display
    return cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)

# Main app
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Original", use_column_width=True)

    blur_intensity = st.slider(
        "Select Blur Intensity",
        min_value=1,
        max_value=99,
        value=55,
        step=2,
        help="Select the intensity of background blur. Higher values mean more blur."
    )

    if st.button("Blur Background"):
        result = blur_background(image, blur_intensity)
        st.image(result, caption="AI Blurred Background", use_column_width=True)

        # Convert result (numpy array) back to PIL Image for download
        result_pil = Image.fromarray(result)
        buf = io.BytesIO()
        result_pil.save(buf, format="PNG")
        byte_im = buf.getvalue()

        st.download_button(
            label="Download Blurred Image",
            data=byte_im,
            file_name="blurred_image.png",
            mime="image/png"
        )

