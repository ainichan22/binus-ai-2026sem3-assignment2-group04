"""About page — model info, team, references."""
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import streamlit as st
from i18n import t, language_selector


st.set_page_config(page_title="About — CIFAR-10", page_icon="ℹ️")
language_selector()

st.title(t("about.title"))


st.subheader(t("about.architecture"))
st.markdown("""
This app uses a **MobileNetV2 transfer-learning model** fine-tuned on CIFAR-10:

```
Input (96, 96, 3)
  → MobileNetV2 backbone (ImageNet pretrained, ~2.26 M params)
  → GlobalAveragePooling2D
  → Dropout(0.3)
  → Dense(10, softmax)
```

Two-stage training: head-only with the backbone frozen (Stage 1, lr=1e-3, 6 epochs),
then fine-tuning the top 30 layers (Stage 2, lr=1e-5, 20 epochs).
Final test accuracy: **0.8993**, vs. **0.8735** for the from-scratch VGG-style baseline.
""")


st.subheader(t("about.team"))
st.markdown(t("about.team_tbd"))


st.subheader(t("about.references"))
st.markdown("""
1. Krizhevsky, A. (2009). *Learning Multiple Layers of Features from Tiny Images.*
2. Simonyan, K., & Zisserman, A. (2014). *Very Deep Convolutional Networks for Large-Scale Image Recognition* (VGGNet).
3. Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L.-C. (2018). *MobileNetV2: Inverted Residuals and Linear Bottlenecks.*
4. Srivastava, N., et al. (2014). *Dropout: A Simple Way to Prevent Neural Networks from Overfitting.*
5. Ioffe, S., & Szegedy, C. (2015). *Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift.*
6. Li, L., Jamieson, K., DeSalvo, G., Rostamizadeh, A., & Talwalkar, A. (2017). *Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization.*
7. Selvaraju, R. R., et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization.*
""")


st.subheader(t("about.github"))
st.markdown(
    "- [GitHub Repository](https://github.com/ainichan22/binus-ai-2026sem3-assignment2-group04)\n"
    "- [Project Tasks (TASKS.md)](https://github.com/ainichan22/binus-ai-2026sem3-assignment2-group04/blob/main/TASKS.md)"
)
