import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
import io

# Title
st.title("AI Background Blur & Enhancement App")

# Load Mediapipe Selfie Segmentation model
mp_selfie_segmentation = mp.solutions.selfie_segmentation
segment = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

# Load ESPCN super-resolution model
sr = cv2.dnn_superres.DnnSuperResImpl_create()
sr.readModel("espcn_x4.pb")  # Make sure this file is in the same directory
sr.setModel("espcn", 4)      # Use ESPCN model with scale factor 4

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Function to blur background
def blur_background(image, blur_intensity):
    image_np = np.array(image.convert('RGB'))
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    results = segment.process(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    mask = results.segmentation_mask
    condition = np.stack((mask,) * 3, axis=-1) > 0.6

    blurred_image = cv2.GaussianBlur(image_bgr, (blur_intensity, blur_intensity), 0)
    output_image = np.where(condition, image_bgr, blurred_image)

    return cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)

# Function to enhance image
def enhance_image(image):
    image_np = np.array(image.convert('RGB'))
    enhanced = sr.upsample(image_np)
    return enhanced

# Main app logic
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Original", use_column_width=True)

    # Blur Section
    blur_intensity = st.slider(
        "Select Blur Intensity",
        min_value=1,
        max_value=99,
        value=55,
        step=2,
        help="Higher values mean more blur."
    )

    if st.button("Blur Background"):
        result = blur_background(image, blur_intensity)
        st.image(result, caption="AI Blurred Background", use_column_width=True)

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

    # Enhance Section
    if st.button("Enhance Image"):
        enhanced = enhance_image(image)
        st.image(enhanced, caption="Enhanced Image (HD)", use_column_width=True)

        enhanced_pil = Image.fromarray(enhanced)
        buf = io.BytesIO()
        enhanced_pil.save(buf, format="PNG")
        byte_im = buf.getvalue()

        st.download_button(
            label="Download Enhanced Image",
            data=byte_im,
            file_name="enhanced_image.png",
            mime="image/png"
        )
