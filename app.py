import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
import io

st.title("AI Background Blur & Enhancement App")

# Load segmentation model
mp_selfie_segmentation = mp.solutions.selfie_segmentation
segment = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

# Load super-resolution model
sr = cv2.dnn_superres.DnnSuperResImpl_create()
sr.readModel("espcn_x4.pb")  # Make sure this file exists in the same directory
sr.setModel("espcn", 4)

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

def blur_background(image, blur_intensity):
    image_np = np.array(image.convert('RGB'))
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    results = segment.process(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    mask = results.segmentation_mask
    condition = np.stack((mask,) * 3, axis=-1) > 0.6

    blurred_image = cv2.GaussianBlur(image_bgr, (blur_intensity, blur_intensity), 0)
    output_image = np.where(condition, image_bgr, blurred_image)

    return cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)

def enhance_image(image):
    image_np = np.array(image.convert('RGB'))
    enhanced = sr.upsample(image_np)
    return enhanced

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Original", use_column_width=True)

    blur_intensity = st.slider("Select Blur Intensity", 1, 99, 55, 2)

    if st.button("Blur Background"):
        result = blur_background(image, blur_intensity)
        st.image(result, caption="AI Blurred Background", use_column_width=True)
        result_pil = Image.fromarray(result)
        buf = io.BytesIO()
        result_pil.save(buf, format="PNG")
        st.download_button("Download Blurred Image", buf.getvalue(), "blurred_image.png", "image/png")

    if st.button("Enhance Image"):
        enhanced = enhance_image(image)
        st.image(enhanced, caption="Enhanced Image (HD)", use_column_width=True)
        enhanced_pil = Image.fromarray(enhanced)
        buf = io.BytesIO()
        enhanced_pil.save(buf, format="PNG")
        st.download_button("Download Enhanced Image", buf.getvalue(), "enhanced_image.png", "image/png")
