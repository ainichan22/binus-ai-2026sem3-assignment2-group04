"""Image preprocessing pipeline that mirrors the transfer-learning training run.

CRITICAL: this must stay in lockstep with `notebooks/03_transfer_learning_EN.ipynb` §3.
Any divergence (wrong order of resize vs preprocess_input, wrong target size,
wrong pixel range) silently degrades accuracy at inference.
"""
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Must match training: see notebooks/03_transfer_learning_EN.ipynb cell 3
IMG_SIZE = 96


def center_crop_square(img: np.ndarray) -> np.ndarray:
    """Crop a (H, W, 3) array to its largest centered square. CIFAR-10
    training images are square; web photos rarely are. Cropping before
    resize avoids the aspect-ratio distortion that would otherwise
    confuse the classifier."""
    h, w = img.shape[:2]
    s = min(h, w)
    y, x = (h - s) // 2, (w - s) // 2
    return img[y:y + s, x:x + s]


def pil_to_array(img: Image.Image) -> np.ndarray:
    """PIL.Image → (H, W, 3) uint8, forced to RGB (drops alpha)."""
    return np.array(img.convert("RGB"))


def preprocess_for_model(img: np.ndarray) -> np.ndarray:
    """Apply the full transfer-learning preprocessing pipeline.

    Steps (must match training):
        1. center-crop to square (avoid distortion in resize)
        2. cast to float32
        3. resize to 96×96 (bilinear default)
        4. mobilenet_v2.preprocess_input → maps [0, 255] to [-1, 1]

    Returns a (96, 96, 3) float32 array in [-1, 1].
    """
    img = center_crop_square(img)
    img = tf.cast(img, tf.float32)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = preprocess_input(img)
    return img.numpy()


def preprocess_batch(img: np.ndarray) -> np.ndarray:
    """Wrap a single preprocessed image into a 1-element batch ready
    for `model.predict()` or `model(inputs)`."""
    return np.expand_dims(preprocess_for_model(img), axis=0)


CLASS_NAMES_EN = ["airplane", "automobile", "bird", "cat", "deer",
                  "dog", "frog", "horse", "ship", "truck"]

CLASS_NAMES_ID = ["pesawat", "mobil", "burung", "kucing", "rusa",
                  "anjing", "katak", "kuda", "kapal", "truk"]


def get_class_names(lang: str = "en") -> list:
    return CLASS_NAMES_ID if lang == "id" else CLASS_NAMES_EN
