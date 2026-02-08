import streamlit as st
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import os
import time
import base64
from datetime import datetime

MODEL_PATH = os.path.join("models", "best_model.pth")
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

CLASS_NAMES = [
    "Background", "Road", "Sidewalk", "Building", "Wall", "Fence", "Pole",
    "Traffic Light", "Traffic Sign", "Vegetation", "Terrain", "Sky", "Person",
    "Rider", "Car", "Truck", "Bus", "Train", "Motorcycle", "Bicycle",
    "Dynamic", "Static", "Parking", "Rail Track", "Bridge", "Tunnel",
    "Water", "Void", "Lane Marking", "Crosswalk"
]

st.set_page_config(
    page_title="UrbanScene Segmentation Dashboard",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def add_bg_from_local(image_file):
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{b64_encoded}");
            background-size: cover;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

local_css("src/style.css")

def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        model = torch.load(MODEL_PATH, map_location=DEVICE)
        model.eval()
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
    return Image.fromarray(colored_mask), mask_np

def calculate_class_distribution(mask_np):
    unique, counts = np.unique(mask_np, return_counts=True)
    total_pixels = mask_np.size
    distribution = {CLASS_NAMES[u]: (counts[i] / total_pixels * 100) for i, u in enumerate(unique)}
    return distribution

st.sidebar.title("🏙️ UrbanScene Segmentation")
st.sidebar.markdown("---")
st.sidebar.info("Upload an urban scene image to get a detailed semantic segmentation mask.")

uploaded_file = st.sidebar.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    input_image = Image.open(uploaded_file).convert('RGB')
    st.sidebar.success("Image uploaded successfully!")
    
    model = load_model()
    if model is None:
        st.sidebar.error("Model not found. Please check the model path.")
    else:
        st.sidebar.success("Model loaded successfully!")
        
        with st.spinner("Generating segmentation mask..."):
            start_time = time.time()
            input_tensor = preprocess(input_image)
            with torch.no_grad():
                output = model(input_tensor)
            mask_image, mask_np = postprocess(output)
            inference_time = time.time() - start_time
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📷 Input Image")
            st.image(input_image, caption=f"Original - {input_image.size[0]}x{input_image.size[1]}", use_column_width=True)
        
        with col2:
            st.subheader("🎨 Predicted Mask")
            st.image(mask_image, caption=f"Segmentation - {INPUT_H}x{INPUT_W}", use_column_width=True)
        
        st.markdown("---")
        st.subheader("📊 Class Distribution")
        distribution = calculate_class_distribution(mask_np)
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.write("**Percentage per class:**")
            for cls, perc in distribution.items():
                st.write(f"{cls}: {perc:.2f}%")
        
        with col4:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(6, 4))
            classes = list(distribution.keys())
            percentages = list(distribution.values())
            ax.barh(classes, percentages, color='steelblue')
            ax.set_xlabel('Percentage (%)')
            ax.set_title('Class Distribution in Mask')
            st.pyplot(fig)
        
        st.markdown("---")
        col5, col6, col7 = st.columns(3)
        
        with col5:
            st.metric("Inference Time", f"{inference_time:.3f} seconds")
        
        with col6:
            st.metric("Model Device", str(DEVICE).upper())
        
        with col7:
            unique_classes = len(distribution)
            st.metric("Unique Classes Found", unique_classes)
        
        st.markdown("---")
        st.subheader("💾 Download Results")
        
        col8, col9 = st.columns(2)
        
        with col8:
            img_bytes = mask_image.tobytes()
            st.download_button(
                label="Download Mask (PNG)",
                data=img_bytes,
                file_name=f"segmentation_mask_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                mime="image/png"
            )
        
        with col9:
            import json
            dist_json = json.dumps(distribution, indent=2)
            st.download_button(
                label="Download Distribution (JSON)",
                data=dist_json,
                file_name=f"class_distribution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        st.markdown("---")
        st.subheader("ℹ️ Class Color Legend")
        legend_cols = st.columns(4)
        for idx, (cls_name, color) in enumerate(zip(CLASS_NAMES, COLORS)):
            with legend_cols[idx % 4]:
                st.markdown(
                    f"<div style='display: flex; align-items: center; margin-bottom: 5px;'>"
                    f"<div style='width: 20px; height: 20px; background-color: rgb{tuple(color)}; margin-right: 10px;'></div>"
                    f"<span>{cls_name}</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )
else:
    st.title("🏙️ Welcome to UrbanScene Segmentation Dashboard")
    st.markdown("""
    This interactive dashboard performs **semantic segmentation** on urban scene images using a deep learning model.
    
    ### How to use:
    1.  **Upload an image** using the file uploader in the sidebar.
    2.  The model will automatically generate a segmentation mask.
    3.  Explore the results: view the mask, class distribution, and download the outputs.
    
    ### Features:
    - Supports JPG, PNG, JPEG formats.
    - Real‑time inference with performance metrics.
    - Detailed class distribution visualization.
    - Download mask and distribution data.
    - Interactive color legend for class mapping.
    """)
    st.image("https://images.unsplash.com/photo-1541339907198-e08756dedf3f?w=800&auto=format&fit=crop", use_column_width=True)
    st.caption("Example urban scene (source: Unsplash)")

st.sidebar.markdown("---")
st.sidebar.markdown("Built with ❤️ using PyTorch & Streamlit")
