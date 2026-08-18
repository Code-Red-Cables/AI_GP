# Gate and policy weights

## Live inference (pose)

The client default is the unstretched eight-keypoint pose model:

```text
models/ROBOFLOW_RETRAIN.pt
```

Set `YOLO_POSE_MODEL_PATH` (or the env of the same name) to point at another
checkpoint after you copy it in. Earlier files still in this folder:

| File | Role |
|---|---|
| `ROBOFLOW_RETRAIN.pt` | Default live pose (YOLO11s, unstretched, 8 kpts) |
| `ROBOFLOW_gatepose.pt` | Older stretched-nano train — do not use |
| `gate_pose.pt` / `gate_pose_v2.pt` | Local Ultralytics installs from earlier datasets |
| `gate_pose_v5.pt` | Local v5 train (pass `--yolo models/gate_pose_v5.pt` to fly it) |
| `policy_seed_17.pt` | Live policy flyer (`H=64`, `chunk=5`, `bins=21`, `--context`) |
| `policy.pt` | Trainer default output name — not automatically the flyer |

Roboflow hosted weights keep the project class `AIGP-8keypoints` plus a dummy
class `0`. `vision/yolo_gate_detector.py` `resolve_gate_class_ids` accepts
`gate` and that project name, and ignores the dummy.

Do not stretch frames to 640×640 for pose. Letterbox.

## Dataset

Current export:

```text
datasets/AIGP_8keypoints.v5i.yolov8/
```

Eight points label both rings, each clockwise from top-left:

| ids | ring | real size |
|---|---|---|
| 0–3 | outer square | 2.7 m |
| 4–7 | flyable opening | 1.5 m |

They feed `vision/yolo_pnp.py` / `vision/dual_gate_pnp.py` for PnP on the
classical path. The **policy** uses the same keypoints as normalised image
features (`race_obs.py`); it does not consume PnP.

Keypoints are trusted by id, not re-ordered geometrically. Any four good
points can still solve a pose. Unseen points are `0 0 0` in the label and
`-1` in the policy observation.

`data.yaml` must use dataset-local paths and correct horizontal-flip
semantics:

```yaml
train: train/images
val: valid/images
kpt_shape: [8, 3]
flip_idx: [1, 0, 3, 2, 5, 4, 7, 6]
names: ['gate']
```

Roboflow export typically gets three of those wrong: `../train/images`
(resolves outside the dataset), the project title as the class name, and
identity `flip_idx`, which silently teaches mirrored corner ids when
`fliplr` is on. `tools/train_gate_pose.py` fails preflight on a broken yaml
rather than training a quietly wrong model. Dummy class-`0` objects with
all-zero keypoints should be dropped from the labels (those images become
empty negatives).

## Train pose

The v5 export already contains augmented copies. Do **not** add a second
round of geometric augs on top:

```powershell
.\winvenv\Scripts\python.exe tools\train_gate_pose.py --no-augment --name gate_pose_v5 --output models\gate_pose_v5.pt
```

Without `--no-augment` the trainer applies race-style augs (Ultralytics
defaults leave `degrees` at 0): ±90° roll, wide scale/translate, shear,
perspective, copy-paste, multi-scale. HSV colour jitter stays off. `flipud`
stays off so top/bottom corner ids stay intact.

`--output` copies `runs/pose/<name>/weights/best.pt` to the path you pass.
Do not point `YOLO_POSE_MODEL_PATH` at a new file until that copy exists.

The live class must be `gate` or `AIGP-8keypoints`. Generic COCO weights
are a training start only; they are rejected for live inference.

`models/gate_detector.pt` (boxes only) is optional scaffolding.
`GATE_DETECTOR_BACKEND=yolo_hybrid` is a detector-debug fallback. It cannot
drive PnP or the policy on its own.

## Train policy

See [`docs/HG_DAGGER.md`](../docs/HG_DAGGER.md). The flyer that matches the
current architecture:

```powershell
.\winvenv\Scripts\python.exe tools\train_policy.py --glob "logs/seed/telem_*.csv" --history 64 --chunk 5 --bins 21 --context --balance-gates --epochs 150 --out models\policy_seed_17.pt
```
