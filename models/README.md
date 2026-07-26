# Custom gate detector weights

Place the trained one-class Ultralytics model here:

```text
models/gate_detector.pt
```

The model must contain a detection class named `gate`. Generic COCO weights
are useful as a training starting point, but they are intentionally rejected
for live inference because they do not define the custom racing-gate class.

Dataset layout:

```text
datasets/gates/
  images/train/
  images/val/
  labels/train/
  labels/val/
```

Create those empty folders with:

```powershell
.\.venv\Scripts\python.exe tools\train_gate_yolo.py --init-dataset
```

Each label is standard YOLO detection format:

```text
0 center_x center_y width height
```

Coordinates are normalized to `[0, 1]`. Label each physical gate separately,
including gates that overlap in the same image. The repository's existing
`frames/` images may be used as source material, but they are not currently
labeled and cannot train YOLO by themselves.

Train and install the resulting best weights:

```powershell
.\.venv\Scripts\python.exe tools\train_gate_yolo.py
```

The script copies `best.pt` to `models/gate_detector.pt`.
