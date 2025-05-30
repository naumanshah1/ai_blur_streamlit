import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
import gradio as gr

# Initialize Mediapipe segmentation
mp_selfie_segmentation = mp.solutions.selfie_segmentation
segment = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

# Load super-resolution model
sr = cv2.dnn_superres.DnnSuperResImpl_create()
sr.readModel("espcn_x4.pb")  # Put this file in your working dir
sr.setModel("espcn", 4)

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

def process_image(image, blur_intensity, mode):
    if mode == "Blur Background":
        return blur_background(image, blur_intensity)
    elif mode == "Enhance Image":
        return enhance_image(image)
    else:
        return image

iface = gr.Interface(
    fn=process_image,
    inputs=[
        gr.Image(type="pil"),
        gr.Slider(minimum=1, maximum=99, step=2, value=55, label="Blur Intensity"),
        gr.Radio(choices=["Blur Background", "Enhance Image"], label="Select Mode"),
    ],
    outputs=gr.Image(type="numpy"),
    title="AI Background Blur & Enhancement App",
    description="Upload an image and choose to blur the background or enhance the image to HD."
)

if __name__ == "__main__":
    iface.launch()
