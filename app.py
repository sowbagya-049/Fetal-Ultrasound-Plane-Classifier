
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

CLASSES = ['Fetal abdomen', 'Fetal brain', 'Fetal femur',
           'Fetal thorax',  'Maternal cervix', 'Other']

DEVICE  = torch.device("cpu")

MODEL_PATHS = [
    "fetal_classifier_final.pth",
    "/content/fetal_classifier_final.pth",
    "/content/drive/MyDrive/fetal_classifier_final.pth"
]

def get_model_path():
    for path in MODEL_PATHS:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(" Model file not found!")

@st.cache_resource
def load_model():
    from torchvision.models import resnet50

    model = resnet50(weights=None)
    in_f = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_f, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 6)
    )

    model_path = get_model_path()
    ckpt = torch.load(model_path, map_location=DEVICE)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225]),
])

st.set_page_config(
    page_title="Fetal Ultrasound Classifier",
    page_icon="🔬",
    layout="centered"
)

st.title("🔬 Fetal Ultrasound Plane Classifier")
st.markdown("Upload a fetal ultrasound image to classify the anatomical plane.")

uploaded = st.file_uploader("Upload Image", type=["png","jpg","jpeg"])

if uploaded:
    img   = Image.open(uploaded).convert("RGB")
    model = load_model()

    col1, col2 = st.columns(2)

    with col1:
        st.image(img, caption="Uploaded Image", use_column_width=True)

    tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probs = torch.softmax(model(tensor), 1).squeeze().numpy()

    pred = probs.argmax()

    with col2:
        st.subheader("Prediction")
        st.success(f"**{CLASSES[pred]}**")
        st.metric("Confidence", f"{probs[pred]*100:.1f}%")

        st.subheader("All Class Probabilities")
        for cls, p in zip(CLASSES, probs):
            st.progress(float(p), text=f"{cls}: {p*100:.1f}%")

    fig, ax = plt.subplots(figsize=(6, 3))

    colors = ["green" if i == pred else "steelblue"
              for i in range(len(CLASSES))]

    ax.barh(CLASSES, probs, color=colors)
    ax.set_xlabel("Confidence")
    ax.set_title("Class Probabilities")

    st.pyplot(fig)
