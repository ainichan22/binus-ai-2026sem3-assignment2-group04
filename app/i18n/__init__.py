"""Internationalization helper.

Usage:
    from i18n import t, language_selector
    language_selector()             # render the sidebar picker once per page
    st.title(t("app.title"))        # all displayed strings go through t()
"""
import json
import os
import streamlit as st

_BASE = os.path.dirname(__file__)
_CACHE = {}

LANGUAGES = {
    "English": "en",
    "Bahasa Indonesia": "id",
}


def _load(lang: str) -> dict:
    if lang not in _CACHE:
        with open(os.path.join(_BASE, f"{lang}.json"), encoding="utf-8") as f:
            _CACHE[lang] = json.load(f)
    return _CACHE[lang]


def t(key: str, **kwargs) -> str:
    """Translate `key` into the active language's string. Missing keys
    fall back to the key itself so untranslated keys are immediately
    visible during development."""
    lang = st.session_state.get("lang", "en")
    text = _load(lang).get(key, key)
    return text.format(**kwargs) if kwargs else text


def language_selector():
    """Render the language picker in the sidebar. Call once per page."""
    current = st.session_state.get("lang", "en")
    current_label = next(
        label for label, code in LANGUAGES.items() if code == current
    )
    label = st.sidebar.selectbox(
        "🌐 Language / Bahasa",
        list(LANGUAGES.keys()),
        index=list(LANGUAGES.keys()).index(current_label),
    )
    st.session_state.lang = LANGUAGES[label]
