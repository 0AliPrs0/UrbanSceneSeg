import streamlit as st
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

MODEL_PATH = "../models/best_model.pth"
INPUT_H, INPUT_W = 96, 256
NUM_CLASSES = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

COLORS = np.array([
    (0, 0, 0), (111, 74, 0), (81, 0, 81), (128, 64, 128), (244, 35, 232),
    (250, 170, 160), (230, 150, 140), (70, 70, 70), (102, 102, 156),
    (190, 153, 153), (180, 165, 180), (150, 100, 100), (150, 120, 90),
    (153, 153, 153), (250, 170, 30), (220, 220, 0), (107, 142, 35),
    (152, 251, 152), (70, 130, 180), (220, 20, 60), (255, 0, 0),
    (0, 0, 142), (0, 0, 70), (0, 60, 100), (0, 0, 90), (0, 0, 110),
    (0, 80, 100), (0, 0, 230), (119, 11, 32), (0, 0, 142)
], dtype=np.uint8)

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        model = torch.load(MODEL_PATH, map_location=DEVICE)
        model.eval()
        st.success("Model loaded successfully!")
        return model
    except Exception:
        return None

def preprocess(image_pil):
    image_np = np.array(image_pil).astype(np.float32)
    image_np = image_np / 255.0
    image_np = np.transpose(image_np, (2, 0, 1))
    image_tensor = torch.from_numpy(image_np).unsqueeze(0)
    image_tensor = F.interpolate(image_tensor, size=(INPUT_H, INPUT_W), mode='bilinear', align_corners=False)
    return image_tensor.to(DEVICE)

def postprocess(prediction_tensor):
    prediction_np = prediction_tensor.squeeze(0).cpu().numpy()
    mask_np = np.argmax(prediction_np, axis=0)
    colored_mask = np.zeros((INPUT_H, INPUT_W, 3), dtype=np.uint8)
    for class_id in range(NUM_CLASSES):
        colored_mask[mask_np == class_id] = COLORS[class_id]
    return Image.fromarray(colored_mask)

st.set_page_config(page_title="Vehicle Segmentation - PyTorch", layout="wide")
st.title("🚗 Vehicle Segmentation Predictor (PyTorch)")

uploaded_file = st.file_uploader("Select your Image", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    input_image = Image.open(uploaded_file).convert('RGB')
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Input Image")
        st.image(input_image, use_column_width=True)
    
    model = load_model()
    
    if model is not None:
        with st.spinner("Predicting..."):
            input_tensor = preprocess(input_image)
            with torch.no_grad():
                output = model(input_tensor)
            mask_image = postprocess(output)
        
        with col2:
            st.subheader("Segmentation Mask")
            st.image(mask_image, use_column_width=True)
        
        st.download_button(
            label="Download Mask",
            data=mask_image.tobytes(),
            file_name="segmentation_mask_pytorch.png",
            mime="image/png"
        )
    else:
        st.error("Model could not be loaded. Check model path and file integrity.")

else:
    st.info("Upload an image to begin prediction.")
