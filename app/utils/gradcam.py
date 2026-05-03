"""Grad-CAM (Selvaraju et al. 2017) — manual implementation that doesn't
depend on tf-keras-vis. For MobileNetV2 the standard target layer is
`Conv_1` (the final 1×1 conv before the global pooling)."""
import numpy as np
import tensorflow as tf


# Default target layer for MobileNetV2 (last conv before GAP)
DEFAULT_LAYER = "Conv_1"


def _resolve_target_layer(model: tf.keras.Model, layer_name: str = DEFAULT_LAYER):
    """The transfer-learning model wraps MobileNetV2 as a sub-model, so
    we search both top-level layers and the immediate sub-models."""
    for layer in model.layers:
        if layer.name == layer_name:
            return layer
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            try:
                return layer.get_layer(layer_name)
            except (ValueError, KeyError):
                continue
    raise ValueError(f"Could not find layer named {layer_name!r} in model")


def grad_cam(
    model: tf.keras.Model,
    image: np.ndarray,
    class_idx: int,
    layer_name: str = DEFAULT_LAYER,
) -> np.ndarray:
    """Compute a Grad-CAM heatmap.

    Args:
        model:     the trained Keras model.
        image:     preprocessed (H, W, 3) float32 array (already in [-1, 1]).
        class_idx: index of the class whose evidence we want to visualize.
        layer_name: target conv layer (default Conv_1 for MobileNetV2).

    Returns:
        Heatmap as (h, w) numpy array in [0, 1] at the target layer's
        feature map resolution. Caller should resize to the original
        image size for display.
    """
    if image.ndim == 3:
        image = np.expand_dims(image, 0)

    target_layer = _resolve_target_layer(model, layer_name)

    # Build a model whose input is the wrapper's input and whose outputs
    # are (target_layer.output, model.output). The sub-model is rebuilt
    # each call — small overhead but no caching pitfalls.
    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[target_layer.output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(image, training=False)
        loss = preds[:, class_idx]

    grads = tape.gradient(loss, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))    # mean over batch, h, w

    conv_out = conv_out[0]
    heatmap = conv_out @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val
    return heatmap.numpy()


def overlay_heatmap(
    rgb_image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap_name: str = "jet",
) -> np.ndarray:
    """Overlay a Grad-CAM heatmap on an RGB image. Returns a uint8
    (H, W, 3) array, with the heatmap resized to match the input image."""
    import matplotlib.pyplot as plt

    h, w = rgb_image.shape[:2]
    heatmap_resized = tf.image.resize(
        heatmap[..., np.newaxis], (h, w), method="bilinear"
    ).numpy().squeeze()

    cmap = plt.get_cmap(colormap_name)
    colored = (cmap(heatmap_resized)[..., :3] * 255).astype(np.uint8)

    overlay = (alpha * colored + (1 - alpha) * rgb_image).astype(np.uint8)
    return overlay
