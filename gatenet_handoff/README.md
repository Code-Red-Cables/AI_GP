# GateNet — gate-corner detector

A 1.95 M-parameter CNN that finds the four corners of a race gate's inner
aperture. Trained for the AI Grand Prix VQ2 simulator.

Source, training code and evaluation:
https://github.com/Etienne-Sasenarine/AI-Grand-Prix

## What is in this bundle

| file | |
| --- | --- |
| `gatenet.onnx` | the model — use this unless you need to fine-tune |
| `best.pt` | PyTorch checkpoint, if you want to train further |
| `cnn_decode.py` | turns the raw outputs into corners; pure numpy, no dependencies beyond numpy/cv2 |

## Read this before using it

**It is specific to the AI-GP simulator.** One track, one lighting condition,
one renderer. It was distilled from a classical CV detector that only works on
that gate's colour signature. Expect it to fail on real cameras, other
simulators, or differently-coloured gates. It is not a general gate detector.

**It predicts corners, not a pose.** Feed the corners to `cv2.solvePnP` with
`SOLVEPNP_IPPE_SQUARE` and the known inner-square size (1.5 m in AI-GP).

## Interface

```
input   image   (1, 3, 180, 320)  float32, RGB, [0, 1], NCHW
output  heatmap (1, 4, 45, 80)    per-corner logits, stride 4
        offset  (1, 8, 45, 80)    sub-cell dx,dy per corner
        conf    (1,)              unreliable, see below
```

Mean/std normalisation is **inside the graph** — hand it raw RGB in [0, 1] and
do not normalise yourself.

Corners come back in the order `TL, TR, BR, BL`, as **normalised** coordinates,
so multiply by your own image size — not by 320x180 — to get pixels.

## Minimal usage

```python
import cv2, numpy as np, onnxruntime as ort
from cnn_decode import decode_heatmaps, preprocess

sess = ort.InferenceSession("gatenet.onnx", providers=["CPUExecutionProvider"])
names = [o.name for o in sess.get_outputs()]

frame = cv2.imread("frame.jpg")                    # BGR, any size
hm, off, conf = sess.run(names, {"image": preprocess(frame, (320, 180))})

corners, score = decode_heatmaps(hm, off, (320, 180))
h, w = frame.shape[:2]
corners_px = corners[0] * np.array([w, h])         # (4,2) in YOUR pixels

if score[0].min() >= 0.80:                         # all four corners confident
    ...  # solvePnP with corners_px
```

**Gate on `score.min() >= 0.80`, not on `conf`.** The confidence head collapsed
during training (the split is 77% positives and plain BCE followed the majority
class) and fires on almost everything. The per-corner peak scores do all the
real discrimination. Measured on held-out flights: 0.30 detects 89.9% of gates
but fires on 22.2% of gate-free frames; 0.80 gives 48.6% and 3.3%.

If only two or three corners clear the threshold, the aperture is clipped by
the frame edge — the visible corners still give a usable bearing even though
PnP does not.

## Measured performance

Held-out **sessions** (never trained on), 4,049 raw frames:

| | |
| --- | --- |
| Corner error | 0.88 px median at 320x180 (0.60 px at 4-8 m) |
| PnP range agreement with the classical teacher | 0.209 m median |
| Usable pose fixes vs that teacher | +23%, at equal temporal coherence |
| Inference | 16.8 ms CPU (14-core) · 0.80 ms TensorRT FP16 on a T4 |

**If you build a TensorRT engine, use FP16.** INT8 was 7% faster but its p99
corner error was 78 px — about one frame in a hundred landed a corner on a
different object. Its *median* drift was a harmless 0.073 px, so that only
shows up if you measure the tail.

## Licence

MIT, same as the repository.
