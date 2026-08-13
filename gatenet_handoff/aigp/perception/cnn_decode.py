"""Turn GateNet's raw heads into gate corners, in numpy alone.

This is the deployment-side half of the network. It lives in `aigp` rather than
in `training/` because the pilot has to run it and the pilot has no PyTorch:
on the Jetson the model is a TensorRT engine, and the only thing that comes
back is three float arrays.

It must agree with `training.model.decode` exactly. A decode that drifts by one
cell between training and deployment is a 4 px corner error, which is a large
fraction of a gate aperture at range -- and nothing would report it, because
each side is self-consistent. `tests/test_cnn_decode.py` pins the two together.
"""

import numpy as np

STRIDE = 4
CORNER_NAMES = ("TL", "TR", "BR", "BL")


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def decode_heatmaps(hm_logits, offsets, image_size, stride=STRIDE):
    """(N,4,h,w) logits + (N,8,h,w) offsets -> corners, scores.

    Returns corners as (N,4,2) in NORMALISED image coordinates, in the
    TL, TR, BR, BL order that `geometry.pnp` expects, and (N,4) peak scores.

    The sigmoid is applied only to the winning cell, never to the whole map:
    it is monotonic, so it cannot change which cell wins, and skipping it over
    3,600 cells x 4 channels is free latency on an embedded target running at
    30 Hz.
    """
    hm_logits = np.asarray(hm_logits, dtype=np.float32)
    offsets = np.asarray(offsets, dtype=np.float32)
    if hm_logits.ndim != 4 or offsets.ndim != 4:
        raise ValueError("expected NCHW heatmap and offset tensors")
    n, c, h, w = hm_logits.shape

    flat = hm_logits.reshape(n, c, -1)
    idx = flat.argmax(axis=2)
    score = _sigmoid(np.take_along_axis(flat, idx[..., None], axis=2)[..., 0])

    cy, cx = np.divmod(idx, w)
    off = offsets.reshape(n, c, 2, -1)
    gather = np.broadcast_to(idx[:, :, None, None], (n, c, 2, 1))
    d = np.take_along_axis(off, gather, axis=3)[..., 0]

    px = (cx + d[..., 0]) * stride
    py = (cy + d[..., 1]) * stride
    corners = np.stack([px / image_size[0], py / image_size[1]], axis=-1)
    return corners.astype(np.float32), score.astype(np.float32)


def corners_to_pixels(corners_norm, image_size):
    """Normalised corners -> pixel corners at `image_size`, float32 for PnP."""
    return (np.asarray(corners_norm, dtype=np.float32)
            * np.asarray(image_size, dtype=np.float32))


def preprocess(image_bgr, image_size):
    """Recorded BGR frame -> (1,3,H,W) float32 RGB in [0,1].

    The mean/std normalisation is NOT here: it is a layer inside the exported
    graph, so it travels with the model and deployment cannot disagree with
    training about it.
    """
    import cv2
    if (image_bgr.shape[1], image_bgr.shape[0]) != tuple(image_size):
        image_bgr = cv2.resize(image_bgr, tuple(image_size),
                               interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    chw = np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32)
    return (chw / 255.0)[None]
