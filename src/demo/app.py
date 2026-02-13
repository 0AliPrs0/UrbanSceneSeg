import sys
from pathlib import Path
import io
import os
import time
import json
from datetime import datetime
import streamlit as st
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models


# ============================================================
# Paths (robust to where you run streamlit from)
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # .../UrbanSceneSeg
MODEL_PATH = PROJECT_ROOT / "models" / "best_unet_cityscapes.pth"
CSS_PATH = PROJECT_ROOT / "src" / "demo" / "style.css"
LOGO_PATH = PROJECT_ROOT / "assets" / "logo_car.png"


# ============================================================
# Config (from seg.ipynb)
# - Cityscapes 19 trainIds (0..18)
# - Input resized to 96x256
# ============================================================
NUM_CLASSES = 19
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)


def pick_size_for_inference(pil_img, max_w=1024):
    w, h = pil_img.size
    scale = max_w / float(w)  # scale to max_w, keep aspect ratio
    new_w = int(w * scale)
    new_h = int(h * scale)

    # divisible by 32 (for resnet)
    new_w = max(32, (new_w // 32) * 32)
    new_h = max(32, (new_h // 32) * 32)
    return new_h, new_w


# Cityscapes trainId label set used in seg.ipynb (19 classes)
CLASS_NAMES = [
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic light",
    "traffic sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
]

# Matching colors from seg.ipynb (trainId -> color)
COLORS = np.array(
    [
        (128, 64, 128),  # road
        (244, 35, 232),  # sidewalk
        (70, 70, 70),  # building
        (102, 102, 156),  # wall
        (190, 153, 153),  # fence
        (153, 153, 153),  # pole
        (250, 170, 30),  # traffic light
        (220, 220, 0),  # traffic sign
        (107, 142, 35),  # vegetation
        (152, 251, 152),  # terrain
        (70, 130, 180),  # sky
        (220, 20, 60),  # person
        (255, 0, 0),  # rider
        (0, 0, 142),  # car
        (0, 0, 70),  # truck
        (0, 60, 100),  # bus
        (0, 80, 100),  # train
        (0, 0, 230),  # motorcycle
        (119, 11, 32),  # bicycle
    ],
    dtype=np.uint8,
)


# ============================================================
# Model architecture (matches seg.ipynb: UNetTL with ResNet34 encoder)
# NOTE: We set weights=None to avoid runtime downloads; checkpoint overwrites weights anyway.
# ============================================================
class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class UNetTL(nn.Module):
    """
    From seg.ipynb:
      resnet34 encoder (up to layer4, removing avgpool/fc)
      upsample path: 512->256->128->64->32, then 1x1 to n_classes
      (No skip-connections in that notebook definition.)
    """

    def __init__(self, n_classes: int = 19):
        super().__init__()
        resnet = tv_models.resnet34(weights=None)
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])  # (B,512,H/32,W/32)

        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(256, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(128, 128)

        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(64, 64)

        self.up4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(32, 32)

        self.outc = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.up1(x)
        x = self.conv1(x)
        x = self.up2(x)
        x = self.conv2(x)
        x = self.up3(x)
        x = self.conv3(x)
        x = self.up4(x)
        x = self.conv4(x)
        x = self.outc(x)
        return x


def tta_predict_logits(model, x: torch.Tensor, base_h: int, base_w: int):
    # multi-scale + horizontal flip, then average logits
    scales = [0.75, 1.0, 1.25]
    acc = None
    n = 0

    for s in scales:
        h = max(32, int((base_h * s) // 32) * 32)
        w = max(32, int((base_w * s) // 32) * 32)

        xs = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)

        # normal
        lg = model(xs)
        lg = F.interpolate(
            lg, size=(base_h, base_w), mode="bilinear", align_corners=False
        )

        # flip
        xflip = torch.flip(xs, dims=[3])
        lgf = model(xflip)
        lgf = torch.flip(lgf, dims=[3])
        lgf = F.interpolate(
            lgf, size=(base_h, base_w), mode="bilinear", align_corners=False
        )

        if acc is None:
            acc = lg + lgf
        else:
            acc = acc + lg + lgf
        n += 2

    return acc / n


def _strip_module_prefix(state_dict: dict) -> dict:
    """Handle checkpoints saved from DataParallel (keys start with 'module.')."""
    if not state_dict:
        return state_dict
    any_module = any(k.startswith("module.") for k in state_dict.keys())
    if not any_module:
        return state_dict
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(f"Model file not found at: {MODEL_PATH}")
        return None

    try:
        sd = torch.load(str(MODEL_PATH), map_location="cpu")

        # If it's a checkpoint dict, pull the right key
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        elif isinstance(sd, dict) and "model_state_dict" in sd:
            sd = sd["model_state_dict"]

        if not isinstance(sd, dict):
            st.error(
                "Loaded checkpoint is not a state_dict/dict. Re-save as model.state_dict()."
            )
            return None

        sd = _strip_module_prefix(sd)

        model = UNetTL(n_classes=NUM_CLASSES)
        model.load_state_dict(sd, strict=True)
        model.to(DEVICE)
        model.eval()
        return model

    except Exception as e:
        st.exception(e)
        return None


# ============================================================
# UI helpers
# ============================================================
st.set_page_config(
    page_title="UrbanScene Segmentation — Cityscapes (19 classes)",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def local_css(path: Path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass


local_css(CSS_PATH)


def render_header():
    col_logo, col_title = st.columns([1, 5])
    with col_logo:
        if LOGO_PATH.exists():
            st.image(Image.open(LOGO_PATH), width=80)
        else:
            st.markdown("🚗")
    with col_title:
        st.markdown(
            "<h1 style='margin-bottom: 0;'>UrbanScene Segmentation — Cityscapes (19 classes)</h1>",
            unsafe_allow_html=True,
        )
    st.markdown("---")


def letterbox_resize(pil_img: Image.Image, target_h: int, target_w: int):
    """
    Resize with aspect ratio kept + pad to (target_w,target_h).
    Returns: padded PIL image, and (pad_left, pad_top, new_w, new_h)
    """
    w, h = pil_img.size
    scale = min(target_w / w, target_h / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    resized = pil_img.resize((new_w, new_h), Image.BILINEAR)

    pad_left = (target_w - new_w) // 2
    pad_top = (target_h - new_h) // 2

    canvas = Image.new("RGB", (target_w, target_h), (0, 0, 0))
    canvas.paste(resized, (pad_left, pad_top))
    return canvas, (pad_left, pad_top, new_w, new_h)


def undo_letterbox_mask(mask_np_orig: np.ndarray, meta, orig_size):
    pad_left, pad_top, new_w, new_h = meta
    cropped = mask_np_orig[pad_top : pad_top + new_h, pad_left : pad_left + new_w]
    # resize back to original image size
    return np.array(
        Image.fromarray(cropped.astype(np.uint8)).resize(orig_size, Image.NEAREST),
        dtype=np.int32,
    )


def softmax_logits(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=1)


# ============================================================
# Pre/Post processing (aligned with seg.ipynb)
# ============================================================
def preprocess(image_pil: Image.Image, target_h: int, target_w: int):
    padded, meta = letterbox_resize(image_pil, target_h, target_w)

    x = np.array(padded).astype(np.float32) / 255.0
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return x, meta


def postprocess(logits: torch.Tensor, conf_thr: float = 0.55):
    """
    logits: [B,C,H,W] یا [C,H,W]
    خروجی:
      mask_np: [H,W] in padded/model space (values are class ids)
      mask_image: #Color image of mask in padded/model space
    """
    if logits.dim() == 4:
        logits = logits.squeeze(0)  # [C,H,W]

    probs = torch.softmax(logits, dim=0)  # [C,H,W]

    top2 = torch.topk(probs, k=2, dim=0)
    top2_prob = top2.values  # [2,H,W]
    top2_idx = top2.indices  # [2,H,W]

    top1_prob = top2_prob[0]  # [H,W]
    top1_idx = top2_idx[0]  # [H,W]
    top2_idx2 = top2_idx[1]  # [H,W]

    # If top1_prob >= conf_thr, keep top1_idx; else use top2_idx2 (the second-best class)
    chosen = torch.where(top1_prob >= conf_thr, top1_idx, top2_idx2)

    mask_np = chosen.detach().cpu().numpy().astype(np.int32)

    h, w = mask_np.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    for cid in range(NUM_CLASSES):
        colored[mask_np == cid] = COLORS[cid]

    return Image.fromarray(colored), mask_np


def calculate_class_distribution(mask_np: np.ndarray):
    unique, counts = np.unique(mask_np, return_counts=True)
    total = mask_np.size
    dist = {}
    for u, c in zip(unique, counts):
        u = int(u)
        name = CLASS_NAMES[u] if 0 <= u < len(CLASS_NAMES) else f"class_{u}"
        dist[name] = float(c) / float(total) * 100.0
    return dist


# ============================================================
# Sidebar
# ============================================================
st.sidebar.title("🏙️ UrbanScene Segmentation")
st.sidebar.markdown("---")
st.sidebar.info(
    "Upload an urban scene image (PNG/JPG) to get semantic segmentation (Cityscapes 19 classes)."
)

uploaded_file = st.sidebar.file_uploader(
    "Choose an image...", type=["png", "jpg", "jpeg"]
)


# ============================================================
# Main
# ============================================================
render_header()

# Debug info (optional; comment out later)
with st.expander("Debug info"):
    st.write("CWD:", os.getcwd())
    st.write("__file__:", str(Path(__file__).resolve()))
    st.write("MODEL_PATH:", str(MODEL_PATH))
    st.write("MODEL EXISTS?:", MODEL_PATH.exists())
    st.write("DEVICE:", str(DEVICE))

if uploaded_file is None:
    st.markdown(
        "## Welcome\nUpload an image via the sidebar to see segmentation results."
    )
    st.info(
        "ℹ️ This dashboard uses a UNet-style decoder with a ResNet34 encoder, trained on Cityscapes (19 trainId classes)."
    )
    st.stop()

input_image = Image.open(uploaded_file).convert("RGB")
st.sidebar.success("Image uploaded successfully!")

model = load_model()
if model is None:
    st.sidebar.error("Model failed to load. See error details above.")
    st.stop()

st.sidebar.success("Model loaded successfully!")

# Choose a better size for inference (larger and divisible by 32)
target_h, target_w = pick_size_for_inference(input_image, max_w=1024)
conf_thr = st.sidebar.slider("Confidence threshold", 0.30, 0.95, 0.55, 0.05)


with st.spinner("Generating segmentation mask..."):
    start_time = time.time()

    input_tensor, lb_meta = preprocess(input_image, target_h, target_w)

    with torch.inference_mode():
        logits = tta_predict_logits(model, input_tensor, target_h, target_w)

        # ---- 6 lines diagnostics (on logits) ----
        probs = torch.softmax(logits, dim=1)[0]  # [C,H,W]
        mean_prob = probs.mean(dim=(1, 2)).detach().cpu().numpy()
        top = np.argsort(-mean_prob)[:5]
        st.write(
            "Top mean-prob classes:",
            [(int(i), CLASS_NAMES[i], float(mean_prob[i])) for i in top],
        )
        # ------------------------------------------

    # 1) ماسک در فضای padded/model
    mask_image, mask_np = postprocess(logits, conf_thr=conf_thr)

    # 2) برگردوندن ماسک به اندازه تصویر اصلی (undo letterbox)
    mask_np_orig = undo_letterbox_mask(mask_np, lb_meta, input_image.size)

    # 3) رنگ‌آمیزی ماسک نهایی (روی اندازه اصلی)
    colored_orig = np.zeros(
        (mask_np_orig.shape[0], mask_np_orig.shape[1], 3), dtype=np.uint8
    )
    for cid in range(NUM_CLASSES):
        colored_orig[mask_np_orig == cid] = COLORS[cid]
    mask_image_vis = Image.fromarray(colored_orig)

    inference_time = time.time() - start_time


# (اختیاری) ببین چند کلاس واقعاً تولید شده
st.write("Unique class ids:", np.unique(mask_np_orig))


# Input & Output
col1, col2 = st.columns(2)
with col1:
    st.subheader("📷 Input Image")
    st.image(
        input_image,
        caption=f"Original — {input_image.size[0]}x{input_image.size[1]}",
        use_container_width=True,
    )
with col2:
    st.subheader("🎨 Predicted Mask")
    st.image(
        mask_image_vis,
        caption=f"Segmentation (model: {target_w}x{target_h})",
        use_container_width=True,
    )

st.markdown("---")

# Class distribution
st.subheader("📊 Class Distribution")
distribution = calculate_class_distribution(mask_np_orig)

col3, col4 = st.columns(2)
with col3:
    st.write("**Percentage per class:**")
    for cls, perc in sorted(distribution.items(), key=lambda x: -x[1]):
        st.write(f"{cls}: {perc:.2f}%")

with col4:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    classes = list(distribution.keys())
    percentages = list(distribution.values())
    ax.barh(classes, percentages)
    ax.set_xlabel("Percentage (%)")
    ax.set_title("Class Distribution in Mask")
    st.pyplot(fig)

st.markdown("---")

# Metrics
st.subheader("⚙️ Inference Metrics")
col5, col6, col7 = st.columns(3)
with col5:
    st.metric("Inference Time", f"{inference_time:.3f} seconds")
with col6:
    st.metric("Model Device", str(DEVICE).upper())
with col7:
    st.metric("Unique Classes Found", str(len(distribution)))

st.markdown("---")

# Download results (proper PNG encoding)
st.subheader("💾 Download Results")
col8, col9 = st.columns(2)

with col8:
    buf = io.BytesIO()
    mask_image_vis.save(buf, format="PNG")
    st.download_button(
        label="Download Mask (PNG)",
        data=buf.getvalue(),
        file_name=f"segmentation_mask_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
        mime="image/png",
    )

with col9:
    dist_json = json.dumps(distribution, indent=2, ensure_ascii=False)
    st.download_button(
        label="Download Distribution (JSON)",
        data=dist_json,
        file_name=f"class_distribution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
    )

st.markdown("---")

# Legend
st.subheader("🎨 Class Color Legend")
legend_cols = st.columns(4)
for idx, (cls_name, color) in enumerate(zip(CLASS_NAMES, COLORS)):
    with legend_cols[idx % 4]:
        st.markdown(
            f"""
            <div style='display: flex; align-items: center; margin-bottom: 10px;'>
                <div style='width: 24px; height: 24px; background-color: rgb{tuple(color)}; border-radius: 4px; margin-right: 10px;'></div>
                <span style='color: #e0e0e0;'>{idx}: {cls_name}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.sidebar.markdown("---")
st.sidebar.markdown("Built with ❤️ using PyTorch & Streamlit")
