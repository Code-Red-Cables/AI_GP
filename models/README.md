# Custom gate model weights

The runtime uses the eight-keypoint pose model:

```text
models/gate_pose.pt
```

Place the Roboflow YOLOv8 keypoint export at:

```text
datasets/AIGP_8keypoints.v1i.yolov8/
```

Its eight points label both rings of the gate, each clockwise from top-left:

| ids | ring | real size |
|---|---|---|
| 0-3 | outer square | 2.7 m |
| 4-7 | flyable opening | 1.5 m |

They feed `vision/yolo_pnp.py` / `vision/dual_gate_pnp.py` for the PnP range and
bearing fixes that correct the EKF. Unlike the old four-corner path these are
trusted by keypoint id rather than re-ordered geometrically, which is what lets
a gate that is partly out of frame still solve: any four good points produce a
pose, and points the model did not see (reported as `0 0 0`) are left out.

Ensure `data.yaml` uses dataset-local paths and correct horizontal flip
semantics:

```yaml
train: train/images
val: valid/images
kpt_shape: [8, 3]
flip_idx: [1, 0, 3, 2, 5, 4, 7, 6]
names: ['gate']
```

Roboflow gets three of those wrong on export and they must be fixed by hand. It
writes `../train/images`, which resolves outside the dataset; it names the class
after the project rather than `gate`; and it writes `flip_idx` as the identity
`[0, 1, ...]`, which silently teaches mirrored corner identities because
training runs with `fliplr: 0.5`. `tools/train_gate_pose.py` fails preflight on
the last two rather than letting a quietly wrong model get trained.

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
