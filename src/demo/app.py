import streamlit as st
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import os
import time
import base64
from datetime import datetime
import json
import matplotlib.pyplot as plt

MODEL_PATH = os.path.join("models", "best_model.pth")
INPUT_H, INPUT_W = 96, 256
NUM_CLASSES = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

COLORS = np.array(
    [
        (0, 0, 0),
        (111, 74, 0),
        (81, 0, 81),
        (128, 64, 128),
        (244, 35, 232),
        (250, 170, 160),
        (230, 150, 140),
        (70, 70, 70),
        (102, 102, 156),
        (190, 153, 153),
        (180, 165, 180),
        (150, 100, 100),
        (150, 120, 90),
        (153, 153, 153),
        (250, 170, 30),
        (220, 220, 0),
        (107, 142, 35),
        (152, 251, 152),
        (70, 130, 180),
        (220, 20, 60),
        (255, 0, 0),
        (0, 0, 142),
        (0, 0, 70),
        (0, 60, 100),
        (0, 0, 90),
        (0, 0, 110),
        (0, 80, 100),
        (0, 0, 230),
        (119, 11, 32),
        (0, 0, 142),
    ],
    dtype=np.uint8,
)

CLASS_NAMES = [
    "Background",
    "Road",
    "Sidewalk",
    "Building",
    "Wall",
    "Fence",
    "Pole",
    "Traffic Light",
    "Traffic Sign",
    "Vegetation",
    "Terrain",
    "Sky",
    "Person",
    "Rider",
    "Car",
    "Truck",
    "Bus",
    "Train",
    "Motorcycle",
    "Bicycle",
    "Dynamic",
    "Static",
    "Parking",
    "Rail Track",
    "Bridge",
    "Tunnel",
    "Water",
    "Void",
    "Lane Marking",
    "Crosswalk",
]

# ========== Page Config ==========
st.set_page_config(
    page_title="UrbanScene Segmentation — CityScapes",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ========== Custom Dark CSS ==========
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# Load the dark theme CSS
local_css("src/style.css")


# ========== Header with Logo ==========
def render_header():
    col_logo, col_title = st.columns([1, 5])
    with col_logo:
        # Ensure the logo file exists in assets/logo_car.png
        logo_path = "assets/logo_car.png"
        if os.path.exists(logo_path):
            logo = Image.open(logo_path)
            st.image(logo, width=80)
        else:
            st.markdown("🚗")  # fallback
    with col_title:
        st.markdown(
            "<h1 style='margin-bottom: 0;'>UrbanScene Segmentation — CityScapes Dataset</h1>",
            unsafe_allow_html=True,
        )
    st.markdown("---")


# ========== Model Loading ==========
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        model = torch.load(MODEL_PATH, map_location=DEVICE)
        model.eval()
        return model
    except Exception:
        return None


# ========== Preprocessing ==========
def preprocess(image_pil):
    image_np = np.array(image_pil).astype(np.float32)
    image_np = image_np / 255.0
    image_np = np.transpose(image_np, (2, 0, 1))
    image_tensor = torch.from_numpy(image_np).unsqueeze(0)
    image_tensor = F.interpolate(
        image_tensor, size=(INPUT_H, INPUT_W), mode="bilinear", align_corners=False
    )
    return image_tensor.to(DEVICE)


# ========== Postprocessing ==========
def postprocess(prediction_tensor):
    prediction_np = prediction_tensor.squeeze(0).cpu().numpy()
    mask_np = np.argmax(prediction_np, axis=0)
    colored_mask = np.zeros((INPUT_H, INPUT_W, 3), dtype=np.uint8)
    for class_id in range(NUM_CLASSES):
        colored_mask[mask_np == class_id] = COLORS[class_id]
    return Image.fromarray(colored_mask), mask_np


# ========== Class Distribution ==========
def calculate_class_distribution(mask_np):
    unique, counts = np.unique(mask_np, return_counts=True)
    total_pixels = mask_np.size
    distribution = {
        CLASS_NAMES[u]: (counts[i] / total_pixels * 100) for i, u in enumerate(unique)
    }
    return distribution


# ========== Sidebar ==========
st.sidebar.title("🏙️ UrbanScene Segmentation")
st.sidebar.markdown("---")
st.sidebar.info(
    "Upload an urban scene image to get a detailed semantic segmentation mask."
)

uploaded_file = st.sidebar.file_uploader(
    "Choose an image...", type=["png", "jpg", "jpeg"]
)

# ========== Main Content ==========
render_header()

if uploaded_file is not None:
    input_image = Image.open(uploaded_file).convert("RGB")
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

        # ========== Input & Output Images ==========
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📷 Input Image")
            st.image(
                input_image,
                caption=f"Original — {input_image.size[0]}x{input_image.size[1]}",
                use_column_width=True,
            )
        with col2:
            st.subheader("🎨 Predicted Mask")
            st.image(
                mask_image,
                caption=f"Segmentation — {INPUT_H}x{INPUT_W}",
                use_column_width=True,
            )

        st.markdown("---")

        # ========== Class Distribution ==========
        st.subheader("📊 Class Distribution")
        distribution = calculate_class_distribution(mask_np)

        col3, col4 = st.columns(2)
        with col3:
            st.write("**Percentage per class:**")
            for cls, perc in distribution.items():
                st.write(f"{cls}: {perc:.2f}%")

        with col4:
            fig, ax = plt.subplots(figsize=(6, 4))
            classes = list(distribution.keys())
            percentages = list(distribution.values())
            ax.barh(classes, percentages, color="#4a90e2")
            ax.set_xlabel("Percentage (%)", color="white")
            ax.set_title("Class Distribution in Mask", color="white")
            ax.tick_params(colors="white")
            ax.spines["bottom"].set_color("white")
            ax.spines["left"].set_color("white")
            ax.spines["top"].set_color("none")
            ax.spines["right"].set_color("none")
            ax.set_facecolor("#1e1e1e")
            fig.patch.set_facecolor("#1e1e1e")
            st.pyplot(fig)

        st.markdown("---")

        # ========== Metrics ==========
        st.subheader("⚙️ Inference Metrics")
        col5, col6, col7 = st.columns(3)
        with col5:
            st.metric("Inference Time", f"{inference_time:.3f} seconds")
        with col6:
            st.metric("Model Device", str(DEVICE).upper())
        with col7:
            unique_classes = len(distribution)
            st.metric("Unique Classes Found", unique_classes)

        st.markdown("---")

        # ========== Download ==========
        st.subheader("💾 Download Results")
        col8, col9 = st.columns(2)
        with col8:
            img_bytes = mask_image.tobytes()
            st.download_button(
                label="Download Mask (PNG)",
                data=img_bytes,
                file_name=f"segmentation_mask_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                mime="image/png",
            )
        with col9:
            dist_json = json.dumps(distribution, indent=2)
            st.download_button(
                label="Download Distribution (JSON)",
                data=dist_json,
                file_name=f"class_distribution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
            )

        st.markdown("---")

        # ========== Color Legend ==========
        st.subheader("🎨 Class Color Legend")
        legend_cols = st.columns(4)
        for idx, (cls_name, color) in enumerate(zip(CLASS_NAMES, COLORS)):
            with legend_cols[idx % 4]:
                st.markdown(
                    f"""
                    <div style='display: flex; align-items: center; margin-bottom: 10px;'>
                        <div style='width: 24px; height: 24px; background-color: rgb{tuple(color)}; border-radius: 4px; margin-right: 10px;'></div>
                        <span style='color: #e0e0e0;'>{cls_name}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
else:
    # ========== Minimal Welcome ==========
    st.markdown(
        """
    ## Welcome to UrbanScene Segmentation
    Upload an urban scene image via the sidebar to see the segmentation results.
    """
    )
    st.info("ℹ️ This dashboard uses a pre‑trained model on the CityScapes dataset.")

st.sidebar.markdown("---")
st.sidebar.markdown("Built with ❤️ using PyTorch & Streamlit")
