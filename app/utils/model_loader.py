"""Cached model loader. `@st.cache_resource` keeps the model in memory
for the lifetime of the Streamlit session — critical because each .keras
load on CPU takes 1–3 seconds and we don't want that on every interaction."""
from pathlib import Path

import streamlit as st
import tensorflow as tf


# Search paths in priority order. HF Spaces deployment puts models at
# the repo root; local dev keeps them under ../models/.
_SEARCH_PATHS = [
    Path("models"),
    Path("../models"),
    Path(__file__).resolve().parents[2] / "models",
]


def _resolve(filename: str) -> Path:
    for base in _SEARCH_PATHS:
        candidate = base / filename
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        f"{filename} not found in any of: {[str(p) for p in _SEARCH_PATHS]}"
    )


@st.cache_resource(show_spinner="Loading transfer-learning model — first request only")
def load_transfer_model() -> tf.keras.Model:
    """Load the MobileNetV2 transfer-learning model. Cached for session."""
    return tf.keras.models.load_model(_resolve("transfer_mobilenet_v1.keras"))


@st.cache_resource(show_spinner=False)
def load_baseline_model() -> tf.keras.Model:
    """Load the from-scratch baseline (used for the metrics page)."""
    return tf.keras.models.load_model(_resolve("baseline_v1.keras"))


def model_path(filename: str) -> Path:
    """Resolve the on-disk path to a known model artifact (for display)."""
    return _resolve(filename)
