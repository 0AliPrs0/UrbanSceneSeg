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


# ============================================================
# Paths (robust)
# ============================================================
THIS_FILE = Path(__file__).resolve()

# تلاش می‌کنیم ریشه پروژه را پیدا کنیم (UrbanSceneSeg)
# با فرض اینکه app.py داخل src/demo باشد:
# UrbanSceneSeg/src/demo/app.py  -> parents[2] = UrbanSceneSeg
PROJECT_ROOT = THIS_FILE.parents[2]

# اگر ساختار پروژه‌ات متفاوت است، این خط را تنظیم کن
# حالت رایج شما: UrbanSceneSeg/models/model_full.pth
MODEL_PATH = PROJECT_ROOT / "models" / "model_full.pth"

# اگر مدل داخل src/demo/models است، این را جایگزین کن:
# MODEL_PATH = PROJECT_ROOT / "src" / "demo" / "models" / "model_full.pth"

CSS_PATH = PROJECT_ROOT / "src" / "demo" / "style.css"
LOGO_PATH = PROJECT_ROOT / "assets" / "logo_car.png"


# ============================================================
# Config
# ============================================================
NUM_CLASSES = 19
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pick_size_for_inference(pil_img: Image.Image, max_w=1024):
    w, h = pil_img.size
    scale = max_w / float(w)
    new_w = int(w * scale)
    new_h = int(h * scale)

    # divisible by 32 (SegNet pools)
    new_w = max(32, (new_w // 32) * 32)
    new_h = max(32, (new_h // 32) * 32)
    return new_h, new_w


# Cityscapes trainId label set (19 classes)
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

# Matching colors
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
# Model (SegNet MTAN)
# ============================================================
class SegNet(nn.Module):
    def __init__(self):
        super().__init__()
        filter = [64, 128, 256, 512, 512]
        self.class_nb = NUM_CLASSES

        self.encoder_block = nn.ModuleList([self.conv_layer([3, filter[0]])])
        self.decoder_block = nn.ModuleList([self.conv_layer([filter[0], filter[0]])])
        for i in range(4):
            self.encoder_block.append(self.conv_layer([filter[i], filter[i + 1]]))
            self.decoder_block.append(self.conv_layer([filter[i + 1], filter[i]]))

        self.conv_block_enc = nn.ModuleList([self.conv_layer([filter[0], filter[0]])])
        self.conv_block_dec = nn.ModuleList([self.conv_layer([filter[0], filter[0]])])
        for i in range(4):
            if i == 0:
                self.conv_block_enc.append(
                    self.conv_layer([filter[i + 1], filter[i + 1]])
                )
                self.conv_block_dec.append(self.conv_layer([filter[i], filter[i]]))
            else:
                self.conv_block_enc.append(
                    nn.Sequential(
                        self.conv_layer([filter[i + 1], filter[i + 1]]),
                        self.conv_layer([filter[i + 1], filter[i + 1]]),
                    )
                )
                self.conv_block_dec.append(
                    nn.Sequential(
                        self.conv_layer([filter[i], filter[i]]),
                        self.conv_layer([filter[i], filter[i]]),
                    )
                )

        self.encoder_att = nn.ModuleList(
            [nn.ModuleList([self.att_layer([filter[0], filter[0], filter[0]])])]
        )
        self.decoder_att = nn.ModuleList(
            [nn.ModuleList([self.att_layer([2 * filter[0], filter[0], filter[0]])])]
        )
        self.encoder_block_att = nn.ModuleList(
            [self.conv_layer([filter[0], filter[1]])]
        )
        self.decoder_block_att = nn.ModuleList(
            [self.conv_layer([filter[0], filter[0]])]
        )

        for j in range(2):
            if j < 2:
                self.encoder_att.append(
                    nn.ModuleList([self.att_layer([filter[0], filter[0], filter[0]])])
                )
                self.decoder_att.append(
                    nn.ModuleList(
                        [self.att_layer([2 * filter[0], filter[0], filter[0]])]
                    )
                )
            for i in range(4):
                self.encoder_att[j].append(
                    self.att_layer([2 * filter[i + 1], filter[i + 1], filter[i + 1]])
                )
                self.decoder_att[j].append(
                    self.att_layer([filter[i + 1] + filter[i], filter[i], filter[i]])
                )

        for i in range(4):
            if i < 3:
                self.encoder_block_att.append(
                    self.conv_layer([filter[i + 1], filter[i + 2]])
                )
                self.decoder_block_att.append(
                    self.conv_layer([filter[i + 1], filter[i]])
                )
            else:
                self.encoder_block_att.append(
                    self.conv_layer([filter[i + 1], filter[i + 1]])
                )
                self.decoder_block_att.append(
                    self.conv_layer([filter[i + 1], filter[i + 1]])
                )

        self.pred_task1 = self.conv_layer([filter[0], self.class_nb], pred=True)
        self.pred_task2 = self.conv_layer([filter[0], 1], pred=True)

        self.down_sampling = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.up_sampling = nn.MaxUnpool2d(kernel_size=2, stride=2)

        self.logsigma = nn.Parameter(torch.FloatTensor([-0.5, -0.5, -0.5]))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def conv_layer(self, channel, pred=False):
        if not pred:
            return nn.Sequential(
                nn.Conv2d(channel[0], channel[1], kernel_size=3, padding=1),
                nn.BatchNorm2d(channel[1]),
                nn.ReLU(inplace=True),
            )
        return nn.Sequential(
            nn.Conv2d(channel[0], channel[0], kernel_size=3, padding=1),
            nn.Conv2d(channel[0], channel[1], kernel_size=1, padding=0),
        )

    def att_layer(self, channel):
        return nn.Sequential(
            nn.Conv2d(channel[0], channel[1], kernel_size=1, padding=0),
            nn.BatchNorm2d(channel[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel[1], channel[2], kernel_size=1, padding=0),
            nn.BatchNorm2d(channel[2]),
            nn.Sigmoid(),
        )

    def forward(self, x):
        g_encoder, g_decoder, g_maxpool, g_upsampl, indices = (
            [0] * 5 for _ in range(5)
        )
        for i in range(5):
            g_encoder[i], g_decoder[-i - 1] = ([0] * 2 for _ in range(2))

        atten_encoder, atten_decoder = ([0] * 3 for _ in range(2))
        for i in range(2):
            atten_encoder[i], atten_decoder[i] = ([0] * 5 for _ in range(2))
        for i in range(2):
            for j in range(5):
                atten_encoder[i][j], atten_decoder[i][j] = ([0] * 3 for _ in range(2))

        for i in range(5):
            if i == 0:
                g_encoder[i][0] = self.encoder_block[i](x)
                g_encoder[i][1] = self.conv_block_enc[i](g_encoder[i][0])
                g_maxpool[i], indices[i] = self.down_sampling(g_encoder[i][1])
            else:
                g_encoder[i][0] = self.encoder_block[i](g_maxpool[i - 1])
                g_encoder[i][1] = self.conv_block_enc[i](g_encoder[i][0])
                g_maxpool[i], indices[i] = self.down_sampling(g_encoder[i][1])

        for i in range(5):
            if i == 0:
                g_upsampl[i] = self.up_sampling(g_maxpool[-1], indices[-i - 1])
                g_decoder[i][0] = self.decoder_block[-i - 1](g_upsampl[i])
                g_decoder[i][1] = self.conv_block_dec[-i - 1](g_decoder[i][0])
            else:
                g_upsampl[i] = self.up_sampling(g_decoder[i - 1][-1], indices[-i - 1])
                g_decoder[i][0] = self.decoder_block[-i - 1](g_upsampl[i])
                g_decoder[i][1] = self.conv_block_dec[-i - 1](g_decoder[i][0])

        for i in range(2):
            for j in range(5):
                if j == 0:
                    atten_encoder[i][j][0] = self.encoder_att[i][j](g_encoder[j][0])
                    atten_encoder[i][j][1] = atten_encoder[i][j][0] * g_encoder[j][1]
                    atten_encoder[i][j][2] = self.encoder_block_att[j](
                        atten_encoder[i][j][1]
                    )
                    atten_encoder[i][j][2] = F.max_pool2d(
                        atten_encoder[i][j][2], kernel_size=2, stride=2
                    )
                else:
                    atten_encoder[i][j][0] = self.encoder_att[i][j](
                        torch.cat((g_encoder[j][0], atten_encoder[i][j - 1][2]), dim=1)
                    )
                    atten_encoder[i][j][1] = atten_encoder[i][j][0] * g_encoder[j][1]
                    atten_encoder[i][j][2] = self.encoder_block_att[j](
                        atten_encoder[i][j][1]
                    )
                    atten_encoder[i][j][2] = F.max_pool2d(
                        atten_encoder[i][j][2], kernel_size=2, stride=2
                    )

            for j in range(5):
                if j == 0:
                    atten_decoder[i][j][0] = F.interpolate(
                        atten_encoder[i][-1][-1],
                        scale_factor=2,
                        mode="bilinear",
                        align_corners=True,
                    )
                    atten_decoder[i][j][0] = self.decoder_block_att[-j - 1](
                        atten_decoder[i][j][0]
                    )
                    atten_decoder[i][j][1] = self.decoder_att[i][-j - 1](
                        torch.cat((g_upsampl[j], atten_decoder[i][j][0]), dim=1)
                    )
                    atten_decoder[i][j][2] = atten_decoder[i][j][1] * g_decoder[j][-1]
                else:
                    atten_decoder[i][j][0] = F.interpolate(
                        atten_decoder[i][j - 1][2],
                        scale_factor=2,
                        mode="bilinear",
                        align_corners=True,
                    )
                    atten_decoder[i][j][0] = self.decoder_block_att[-j - 1](
                        atten_decoder[i][j][0]
                    )
                    atten_decoder[i][j][1] = self.decoder_att[i][-j - 1](
                        torch.cat((g_upsampl[j], atten_decoder[i][j][0]), dim=1)
                    )
                    atten_decoder[i][j][2] = atten_decoder[i][j][1] * g_decoder[j][-1]

        t1_pred = F.log_softmax(self.pred_task1(atten_decoder[0][-1][-1]), dim=1)
        t2_pred = self.pred_task2(atten_decoder[1][-1][-1])
        return [t1_pred, t2_pred], self.logsigma


# ============================================================
# Checkpoint helpers
# ============================================================
def _strip_module_prefix(state_dict: dict) -> dict:
    if not state_dict:
        return state_dict
    if any(k.startswith("module.") for k in state_dict.keys()):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def _try_extract_state_dict(ckpt):
    """
    ckpt می‌تواند:
      - خود مدل (nn.Module)
      - state_dict (dict)
      - dict شامل کلیدهایی مثل state_dict / model_state_dict / model
    """
    if isinstance(ckpt, nn.Module):
        return ckpt, None  # model itself

    if isinstance(ckpt, dict):
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            return None, ckpt["state_dict"]
        if "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
            return None, ckpt["model_state_dict"]
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            return None, ckpt["model"]
        # ممکن است خود dict همان state_dict باشد
        return None, ckpt

    return None, None


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(f"Model file not found at: {MODEL_PATH}")
        return None

    try:
        # PyTorch 2.6+: اگر فایل "مدل کامل" باشد باید weights_only=False شود
        ckpt = torch.load(str(MODEL_PATH), map_location="cpu", weights_only=False)

        # حالت 1) فایل، خود مدل کامل است
        if isinstance(ckpt, nn.Module):
            model = ckpt.to(DEVICE)
            model.eval()
            return model

        # حالت 2) dict (state_dict یا checkpoint dict)
        if isinstance(ckpt, dict):
            if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
                sd = ckpt["state_dict"]
            elif "model_state_dict" in ckpt and isinstance(
                ckpt["model_state_dict"], dict
            ):
                sd = ckpt["model_state_dict"]
            elif "model" in ckpt and isinstance(ckpt["model"], dict):
                sd = ckpt["model"]
            else:
                sd = ckpt  # شاید خودش state_dict باشد

            if not isinstance(sd, dict):
                st.error("Checkpoint dict does not contain a valid state_dict.")
                return None

            sd = _strip_module_prefix(sd)

            model = SegNet()
            model.load_state_dict(sd, strict=True)
            model.to(DEVICE)
            model.eval()
            return model

        st.error(f"Unsupported checkpoint type: {type(ckpt)}")
        return None

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


def undo_letterbox_mask(mask_np_model: np.ndarray, meta, orig_size):
    pad_left, pad_top, new_w, new_h = meta
    cropped = mask_np_model[pad_top : pad_top + new_h, pad_left : pad_left + new_w]
    return np.array(
        Image.fromarray(cropped.astype(np.uint8)).resize(orig_size, Image.NEAREST),
        dtype=np.int32,
    )


def preprocess(image_pil: Image.Image, target_h: int, target_w: int):
    padded, meta = letterbox_resize(image_pil, target_h, target_w)
    x = np.array(padded).astype(np.float32) / 255.0
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    return x, meta


def postprocess(logits: torch.Tensor):
    if logits.dim() == 4:
        logits = logits.squeeze(0)  # [C,H,W]
    mask_np = logits.argmax(dim=0).detach().cpu().numpy().astype(np.int32)

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

with st.expander("Debug info"):
    st.write("CWD:", os.getcwd())
    st.write("__file__:", str(THIS_FILE))
    st.write("PROJECT_ROOT:", str(PROJECT_ROOT))
    st.write("MODEL_PATH:", str(MODEL_PATH))
    st.write("MODEL EXISTS?:", MODEL_PATH.exists())
    st.write("DEVICE:", str(DEVICE))
    if MODEL_PATH.exists():
        try:
            ckpt = torch.load(str(MODEL_PATH), map_location="cpu")
            st.write("Checkpoint type:", str(type(ckpt)))
            if isinstance(ckpt, dict):
                st.write("Checkpoint dict keys (top-level):", list(ckpt.keys())[:30])
        except Exception as e:
            st.write("Checkpoint inspect failed:", str(e))

if uploaded_file is None:
    st.markdown(
        "## Welcome\nUpload an image via the sidebar to see segmentation results."
    )
    st.info("ℹ️ This dashboard uses the MTAN architecture trained on Cityscapes.")
    st.stop()

input_image = Image.open(uploaded_file).convert("RGB")
st.sidebar.success("Image uploaded successfully!")

model = load_model()
if model is None:
    st.sidebar.error("Model failed to load. See error details above.")
    st.stop()

st.sidebar.success("Model loaded successfully!")


# Fix: function bug (new_w variable)
def pick_size_for_inference_fixed(pil_img, max_w=1024):
    w, h = pil_img.size
    scale = max_w / float(w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    new_w = max(32, (new_w // 32) * 32)
    new_h = max(32, (new_h // 32) * 32)
    return new_h, new_w


target_h, target_w = pick_size_for_inference_fixed(input_image, max_w=1024)

with st.spinner("Generating segmentation mask..."):
    start_time = time.time()

    input_tensor, lb_meta = preprocess(input_image, target_h, target_w)

    with torch.inference_mode():
        predictions, _ = model(input_tensor)
        logits = predictions[0]  # [1, 19, H, W]

        probs = torch.softmax(logits, dim=1)[0]
        mean_prob = probs.mean(dim=(1, 2)).detach().cpu().numpy()
        top = np.argsort(-mean_prob)[:5]
        st.write(
            "Top mean-prob classes:",
            [(int(i), CLASS_NAMES[i], float(mean_prob[i])) for i in top],
        )

    mask_image_model, mask_np_model = postprocess(logits)
    mask_np_orig = undo_letterbox_mask(mask_np_model, lb_meta, input_image.size)

    colored_orig = np.zeros(
        (mask_np_orig.shape[0], mask_np_orig.shape[1], 3), dtype=np.uint8
    )
    for cid in range(NUM_CLASSES):
        colored_orig[mask_np_orig == cid] = COLORS[cid]
    mask_image_vis = Image.fromarray(colored_orig)

    inference_time = time.time() - start_time

st.write("Unique class ids:", np.unique(mask_np_orig))

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

st.subheader("⚙️ Inference Metrics")
col5, col6, col7 = st.columns(3)
with col5:
    st.metric("Inference Time", f"{inference_time:.3f} seconds")
with col6:
    st.metric("Model Device", str(DEVICE).upper())
with col7:
    st.metric("Unique Classes Found", str(len(distribution)))

st.markdown("---")

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
