# Custom gate model weights

The runtime uses the four-keypoint pose model:

```text
models/gate_pose.pt
```

Place the Roboflow YOLOv8 keypoint export at:

```text
datasets/AI_GP.v1i.yolov8/
```

Its four points are outer-gate corners in TL, TR, BL, BR order. They feed
`vision/yolo_pnp.py` / `vision/dual_gate_pnp.py` for the PnP range and bearing
fixes that correct the EKF. Ensure `data.yaml` uses dataset-local paths and
correct horizontal flip semantics:

```yaml
train: train/images
val: valid/images
kpt_shape: [4, 3]
flip_idx: [1, 0, 3, 2]
names: ['gate']
```

Train locally and install `models/gate_pose.pt`:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe tools\train_gate_pose.py
```

The model must contain a class named `gate`. Generic COCO weights are useful as
a training starting point, but they are rejected for live inference because they
do not define the custom racing-gate class.

`models/gate_detector.pt` (bounding-box weights) is optional: the box detector
in `vision/yolo_gate_detector.py` is retained only as the shared box/HSV
scaffolding used by the pose detector, and `GATE_DETECTOR_BACKEND=yolo_hybrid`
is a fallback for detector debugging. It cannot drive the PnP path on its own
because it produces no corners.
