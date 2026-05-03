"""Metrics page — training curves, per-class report, confusion matrix.

History plots are loaded from ../models/*_history.pkl. Per-class numbers
are hard-coded from the trained notebooks' classification_report outputs
(no need to re-run inference at app startup)."""
import sys
from pathlib import Path
import pickle

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from i18n import t, language_selector


st.set_page_config(page_title="Metrics — CIFAR-10", page_icon="📊")
language_selector()

st.title(t("metrics.title"))


# Pre-computed numbers from the trained notebooks (see notebooks/01_baseline_EN
# Section 9 and notebooks/03_transfer_learning_EN Section 8 outputs).
SUMMARY = {
    "baseline": {"test_acc": 0.8735, "test_loss": 0.5291, "params": "~570 K", "epochs": 50},
    "transfer": {"test_acc": 0.8993, "test_loss": 0.2971, "params": "~1.52 M", "epochs": 26},
}

PER_CLASS = {
    "baseline": [
        ("airplane",   0.8887, 0.8700, 0.8792),
        ("automobile", 0.9392, 0.9270, 0.9331),
        ("bird",       0.8701, 0.8040, 0.8358),
        ("cat",        0.8129, 0.7040, 0.7546),
        ("deer",       0.8419, 0.8840, 0.8624),
        ("dog",        0.8380, 0.8020, 0.8196),
        ("frog",       0.8442, 0.9480, 0.8931),
        ("horse",      0.9051, 0.9160, 0.9105),
        ("ship",       0.9447, 0.9220, 0.9332),
        ("truck",      0.8508, 0.9580, 0.9012),
    ],
    "transfer": [
        ("airplane",   0.9191, 0.8980, 0.9084),
        ("automobile", 0.9429, 0.9570, 0.9499),
        ("bird",       0.9331, 0.8650, 0.8978),
        ("cat",        0.8111, 0.7860, 0.7984),
        ("deer",       0.8602, 0.9110, 0.8849),
        ("dog",        0.8769, 0.8190, 0.8469),
        ("frog",       0.9124, 0.9370, 0.9245),
        ("horse",      0.9228, 0.9090, 0.9159),
        ("ship",       0.9258, 0.9480, 0.9368),
        ("truck",      0.8892, 0.9630, 0.9246),
    ],
}

HISTORY_PATHS = {
    "baseline": APP_DIR.parent / "models" / "baseline_v1_history.pkl",
    "transfer": APP_DIR.parent / "models" / "transfer_mobilenet_v1_history.pkl",
}

CONFUSION_FIGS = {
    "baseline": APP_DIR.parent / "report" / "figures" / "confusion-matrix.png",
}


# Model picker
options = {
    "Baseline (VGG-style, from scratch)":     "baseline",
    "Transfer (MobileNetV2 fine-tuned)":      "transfer",
}
choice_label = st.selectbox(t("metrics.model_picker"), list(options.keys()), index=1)
choice = options[choice_label]


# Summary cards
st.subheader(t("metrics.summary_header"))
s = SUMMARY[choice]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Test accuracy", f"{s['test_acc']:.4f}")
c2.metric("Test loss",     f"{s['test_loss']:.4f}")
c3.metric("Params",        s["params"])
c4.metric("Epochs",        s["epochs"])


# Side-by-side comparison table (always shows both models)
st.subheader(t("metrics.compare_header"))
compare = pd.DataFrame({
    "Model": ["baseline_v1", "transfer_mobilenet_v1"],
    "Test accuracy": [SUMMARY["baseline"]["test_acc"], SUMMARY["transfer"]["test_acc"]],
    "Trainable params": [SUMMARY["baseline"]["params"], SUMMARY["transfer"]["params"]],
    "Epochs": [SUMMARY["baseline"]["epochs"], SUMMARY["transfer"]["epochs"]],
})
st.dataframe(compare, use_container_width=True, hide_index=True)


# Training curves (loaded from history pickle)
st.subheader(t("metrics.training_curves_header"))
hpath = HISTORY_PATHS[choice]
if hpath.exists():
    with open(hpath, "rb") as f:
        h = pickle.load(f)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(h.get("accuracy", []),     label="Train",      linewidth=2)
    axes[0].plot(h.get("val_accuracy", []), label="Validation", linewidth=2)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy"); axes[0].set_title("Accuracy")
    axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(h.get("loss", []),     label="Train",      linewidth=2)
    axes[1].plot(h.get("val_loss", []), label="Validation", linewidth=2)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss"); axes[1].set_title("Loss")
    axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
else:
    st.warning(t("metrics.history_missing", path=str(hpath)))


# Per-class report
st.subheader(t("metrics.classification_report_header"))
df = pd.DataFrame(PER_CLASS[choice], columns=["Class", "Precision", "Recall", "F1"])
st.dataframe(df, use_container_width=True, hide_index=True)


# Confusion matrix (static figure for baseline; transfer has it inside the notebook)
st.subheader(t("metrics.confusion_matrix_header"))
fig_path = CONFUSION_FIGS.get(choice)
if fig_path and fig_path.exists():
    st.image(str(fig_path))
else:
    st.info("See the trained notebook for the full confusion matrix figure.")
