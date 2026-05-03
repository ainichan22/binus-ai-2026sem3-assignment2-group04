"""CIFAR-10 Image Classifier — Streamlit app entry point.

Pages are auto-discovered from ./pages/. The sidebar shows:
  - language selector at the top (state shared across pages via st.session_state)
  - native page nav rendered by Streamlit
"""
import sys
from pathlib import Path

# Allow `from utils.*` and `from i18n.*` from inside pages/
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import streamlit as st
from i18n import t, language_selector


st.set_page_config(
    page_title="CIFAR-10 Classifier",
    page_icon="🖼️",
    layout="centered",
    initial_sidebar_state="expanded",
)

language_selector()

st.title(t("app.title"))
st.caption(t("app.subtitle"))

st.markdown(f"#### 👋 {t('app.tagline')}")

st.markdown("---")
st.markdown(f"**{t('app.pick_a_page')}:**")
st.markdown(f"- **{t('nav.predict')}** — {t('app.predict_desc')}")
st.markdown(f"- **{t('nav.metrics')}** — {t('app.metrics_desc')}")
st.markdown(f"- **{t('nav.about')}** — {t('app.about_desc')}")
