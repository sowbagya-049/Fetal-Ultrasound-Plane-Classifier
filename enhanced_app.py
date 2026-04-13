"""
Fetal Ultrasound Plane Classifier - Enhanced Streamlit App
Author: Sowbagya VS | Reg: 2303717673722049
Model: EfficientNet-B0 (best_efficientnet.pth)
"""

import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import time

# ── Constants ─────────────────────────────────────────────────────────────────
CLASSES = [
    'Fetal Abdomen', 'Fetal Brain', 'Fetal Femur',
    'Fetal Thorax',  'Maternal Cervix', 'Other'
]

CLASS_INFO = {
    'Fetal Abdomen': '🟡 Abdominal circumference view — key growth marker.',
    'Fetal Brain':   '🔵 Standard brain plane — checks neural development.',
    'Fetal Femur':   '🟢 Femur length — used to estimate gestational age.',
    'Fetal Thorax':  '🟠 Thoracic view — evaluates heart and lung position.',
    'Maternal Cervix': '🔴 Cervical length — screens for preterm birth risk.',
    'Other':         '⚪ Non-standard or intermediate plane.',
}

DEVICE = torch.device("cpu")

# ── Model ─────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    model = efficientnet_b0(weights=None)
    in_f  = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_f, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 6),
    )
    # Try multiple paths for the weights file
    for path in ["best_efficientnet.pth",
                 "/mnt/user-data/uploads/best_efficientnet.pth"]:
        if os.path.exists(path):
            ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
            state = ckpt.get("model_state_dict", ckpt)
            model.load_state_dict(state)
            break
    model.eval()
    return model

# ── Transforms ────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fetal Ultrasound Classifier",
    page_icon="🤱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/ultrasound.png", width=80)
    st.title("ℹ️ About")
    st.markdown("""
    **Model:** EfficientNet-B0  
    **Dataset:** FETAL_PLANES_DB (Zenodo)  
    **Classes:** 6 anatomical planes  
    **Test Accuracy:** 90.53%  
    **Macro AUC:** 0.990  
    """)
    st.divider()
    st.markdown("**Class Descriptions**")
    for cls, desc in CLASS_INFO.items():
        st.markdown(f"- {desc}")
    st.divider()
    st.caption("Coimbatore Institute of Technology | MSc DCS")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<h1 style='text-align:center; color:#1a6fa8;'>
    🤱 Fetal Ultrasound Plane Classifier
</h1>
<p style='text-align:center; color:#555;'>
    Upload a grayscale fetal ultrasound image to identify the anatomical plane using AI.
</p>
<hr/>
""", unsafe_allow_html=True)

# ── Load Model ────────────────────────────────────────────────────────────────
with st.spinner("Loading AI model..."):
    model = load_model()
st.success("✅ Model loaded successfully!", icon="🤖")

# ── Upload ────────────────────────────────────────────────────────────────────
st.subheader("📤 Upload Ultrasound Image")
uploaded = st.file_uploader(
    "Supported formats: PNG, JPG, JPEG",
    type=["png", "jpg", "jpeg"],
    help="Upload a grayscale or RGB fetal ultrasound image."
)

if uploaded:
    img = Image.open(uploaded).convert("RGB")

    col1, col2 = st.columns([1, 1], gap="large")

    # ── Left: Image ──────────────────────────────────────────────────────────
    with col1:
        st.subheader("🖼️ Uploaded Image")
        st.image(img, caption=f"File: {uploaded.name}", use_column_width=True)
        st.caption(f"Size: {img.size[0]}×{img.size[1]} px | Mode: {img.mode}")

    # ── Inference ────────────────────────────────────────────────────────────
    tensor = transform(img).unsqueeze(0).to(DEVICE)
    t0 = time.time()
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1).squeeze().numpy()
    latency = (time.time() - t0) * 1000
    pred = int(probs.argmax())

    # ── Right: Results ───────────────────────────────────────────────────────
    with col2:
        st.subheader("🔬 Prediction Result")

        # Badge
        confidence = probs[pred] * 100
        color = "green" if confidence >= 75 else "orange" if confidence >= 50 else "red"
        st.markdown(f"""
        <div style='background:{color};padding:16px;border-radius:12px;text-align:center;'>
            <h2 style='color:white;margin:0;'>{CLASSES[pred]}</h2>
            <p style='color:white;font-size:22px;margin:4px 0;'>{confidence:.1f}% confidence</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"> {CLASS_INFO[CLASSES[pred]]}")
        st.caption(f"⏱️ Inference time: {latency:.1f} ms")
        st.divider()

        # All class probabilities
        st.subheader("📊 All Class Probabilities")
        for i, (cls, p) in enumerate(zip(CLASSES, probs)):
            bar_color = "🟩" if i == pred else "⬜"
            st.progress(float(p), text=f"{bar_color} {cls}: {p*100:.1f}%")

    # ── Chart ────────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("📈 Confidence Chart")

    fig, ax = plt.subplots(figsize=(9, 3.5))
    colors = ["#1a6fa8" if i == pred else "#c9dcea" for i in range(len(CLASSES))]
    bars = ax.barh(CLASSES, probs * 100, color=colors, edgecolor="white", height=0.6)
    ax.set_xlabel("Confidence (%)", fontsize=11)
    ax.set_title("Per-Class Confidence", fontsize=13, fontweight="bold")
    ax.set_xlim(0, 105)
    for bar, p in zip(bars, probs):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f"{p*100:.1f}%", va="center", fontsize=9)
    fig.tight_layout()
    st.pyplot(fig)

    # ── Disclaimer ───────────────────────────────────────────────────────────
    st.info(
        "⚕️ **Clinical Disclaimer:** This tool is for research and educational purposes only. "
        "Results should not be used as a substitute for professional medical diagnosis.",
        icon="⚠️"
    )

else:
    # Placeholder
    st.markdown("""
    <div style='text-align:center;padding:60px;background:#f0f6ff;border-radius:16px;border:2px dashed #1a6fa8;'>
        <h3 style='color:#1a6fa8;'>👆 Upload an ultrasound image to get started</h3>
        <p style='color:#888;'>The AI will identify which anatomical plane is shown.</p>
    </div>
    """, unsafe_allow_html=True)