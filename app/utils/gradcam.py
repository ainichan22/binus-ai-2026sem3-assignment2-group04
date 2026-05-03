"""Grad-CAM (Selvaraju et al. 2017) — manual implementation that doesn't
depend on tf-keras-vis. For MobileNetV2 the standard target layer is
`Conv_1` (the final 1×1 conv before the global pooling)."""
import numpy as np
import streamlit as st
import tensorflow as tf


# Default target layer for MobileNetV2 (last conv before GAP)
DEFAULT_LAYER = "Conv_1"


def _find_target(model: tf.keras.Model, layer_name: str):
    """Locate the target layer. Returns (target_layer, submodel_or_None).
    If the target is at the outer model's top level, submodel is None;
    if it's inside a nested sub-model (e.g. the MobileNetV2 backbone),
    submodel is that wrapper."""
    for layer in model.layers:
        if layer.name == layer_name:
            return layer, None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            try:
                return layer.get_layer(layer_name), layer
            except (ValueError, KeyError):
                continue
    raise ValueError(f"Could not find layer named {layer_name!r} in model")


@st.cache_resource(show_spinner=False)
def _build_grad_fn(_model: tf.keras.Model, layer_name: str):
    """Build a callable that, given an input batch, returns
    (target_conv_output, predictions). Cached per session.

    Keras 3 refuses to build a Functional `tf.keras.Model(inputs, outputs)`
    when one of the outputs lives inside a nested sub-model — there's no
    direct symbolic path from the outer input to that intermediate
    tensor. Workaround: build an inner Functional model that runs from
    the sub-model's input to (target output, sub-model output), then
    finish the forward pass by manually applying the outer model's head
    layers.

    The leading underscore on `_model` tells Streamlit to skip hashing
    the model arg; identity is stable because the caller comes from
    another @st.cache_resource loader."""
    target_layer, submodel = _find_target(_model, layer_name)

    if submodel is None:
        # Simple case: target is at the outer-model top level.
        grad_model = tf.keras.Model(
            inputs=_model.inputs,
            outputs=[target_layer.output, _model.output],
        )

        def forward(x):
            return grad_model(x, training=False)
        return forward

    # Hard case: target is inside a sub-model. Build inner extractor
    # then walk the outer head manually.
    inner = tf.keras.Model(
        inputs=submodel.inputs,
        outputs=[target_layer.output, submodel.output],
    )

    head_layers = []
    seen_submodel = False
    for layer in _model.layers:
        if layer is submodel:
            seen_submodel = True
            continue
        if seen_submodel and not isinstance(layer, tf.keras.layers.InputLayer):
            head_layers.append(layer)

    def forward(x):
        conv_out, sub_out = inner(x, training=False)
        h = sub_out
        for layer in head_layers:
            h = layer(h, training=False)
        return conv_out, h
    return forward


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

    forward = _build_grad_fn(model, layer_name)

    with tf.GradientTape() as tape:
        conv_out, preds = forward(image)
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
