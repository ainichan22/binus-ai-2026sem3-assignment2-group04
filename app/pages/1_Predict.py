"""Predict page — main feature.

Pipeline: input picker (upload / sample / camera) → preprocess → model.predict
→ Top-3 + Grad-CAM overlay. All displayed strings go through `t()`.
"""
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import numpy as np
import streamlit as st
from PIL import Image

from i18n import t, language_selector
from utils.preprocess import (
    pil_to_array, preprocess_for_model, preprocess_batch, get_class_names,
)
from utils.model_loader import load_transfer_model
from utils.gradcam import grad_cam, overlay_heatmap


st.set_page_config(page_title="Predict — CIFAR-10", page_icon="🖼️")
language_selector()

st.title(t("nav.predict"))


SAMPLES_DIR = APP_DIR / "samples"


def _list_samples():
    return sorted(p for p in SAMPLES_DIR.glob("*.png"))


def _read_input():
    """Sidebar input picker — returns a PIL.Image or None."""
    method_options = {
        "upload": t("predict.upload"),
        "sample": t("predict.sample"),
        "camera": t("predict.camera"),
    }
    chosen_label = st.sidebar.radio(t("predict.input_method"), list(method_options.values()))
    method = next(k for k, v in method_options.items() if v == chosen_label)

    if method == "upload":
        f = st.sidebar.file_uploader(t("predict.upload_label"), type=["jpg", "jpeg", "png"])
        return Image.open(f) if f else None

    if method == "sample":
        sample_files = _list_samples()
        if not sample_files:
            st.sidebar.warning(t("predict.no_samples"))
            return None
        names = [p.stem for p in sample_files]
        chosen_name = st.sidebar.selectbox(t("predict.sample_label"), names)
        chosen_path = next(p for p, n in zip(sample_files, names) if n == chosen_name)
        return Image.open(chosen_path)

    if method == "camera":
        captured = st.sidebar.camera_input("📷")
        return Image.open(captured) if captured else None

    return None


img = _read_input()

if img is None:
    st.info(t("predict.no_input"))
    st.stop()


# Preprocess + predict
arr = pil_to_array(img)

with st.spinner(t("predict.processing")):
    model = load_transfer_model()
    proba = model.predict(preprocess_batch(arr), verbose=0)[0]


# Display: image left, top-3 right
col1, col2 = st.columns([1, 1])
with col1:
    st.image(arr, caption=f"{arr.shape[1]}×{arr.shape[0]}", width="stretch")

with col2:
    st.subheader(t("predict.top3_header"))
    lang = st.session_state.get("lang", "en")
    class_names = get_class_names(lang)
    top3_idx = np.argsort(proba)[-3:][::-1]
    for rank, idx in enumerate(top3_idx, 1):
        st.markdown(f"**{rank}. {class_names[idx]}**")
        st.progress(float(proba[idx]), text=f"{t('predict.confidence')}: {proba[idx]:.1%}")


# Grad-CAM overlay for the top-1 class
st.subheader(t("predict.gradcam_header"))
top1_idx = int(top3_idx[0])
preprocessed = preprocess_for_model(arr)
heatmap = grad_cam(model, preprocessed, top1_idx)
overlay = overlay_heatmap(arr, heatmap, alpha=0.45)
st.image(overlay, caption=f"Grad-CAM → {class_names[top1_idx]}", width="stretch")


st.caption(t("predict.disclaimer"))
