# floorplan_gen v3 — graph-conditioned floor-plan generator + 3D viewer (ResPlan)

You give: **boundary + rooms + which rooms connect** (JSON). You get: floor plan PNG + interactive 3D page with
**interior walls, doors, windows and an entrance**.

> Continuing with an AI assistant? Point it to **`PROJECT_STATE.md`** and **`logs/WORKLOG.md`** first.

## Architecture in one minute
```
JSON (boundary, rooms [+relative sizes], connections)
  → Stage 1  BoxNet     graph transformer, AdaLN size conditioning, K noise hypotheses (winner-takes-all training) → one box per room
  → Stage 2  RefineNet  U-Net: boxes → per-room map (edge-weighted CE + anisotropic TV, soft-box robustness, flip/rotate TTA)
  → Stage 3  layout.py  (no learning) restore dropped rooms → Manhattan polygons (strict 90° corners) → walls = room borders,
                        doors = graph edges (+auto-repair), windows, entrance, labels → PNG + 3D page
  → generate.py builds K candidates, scores them (hand weights, or weights learned by calibrate_score.py) and returns the best
```
## Setup (Windows, RTX 5050)
```
python -m venv venv && venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128     # RTX 50-series needs CUDA 12.8+ build
pip install -r requirements.txt
python -c "import torch;print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
Folders: `...\PCFPG_V2\ResPlan\` (ResPlan.pkl + split.json) next to `...\PCFPG_V2\floorplan_gen\`
(else add `--data-dir "D:\path\ResPlan"` to `prepare_data.py`).

## Run
```
python prepare_data.py                 # 1) only needed once (cache4 is unchanged since v3.1)
python train_box.py --epochs 200 --d 256 --heads 8 --layers 6        # 2a) Stage 1 — RETRAIN (architecture changed in v3.2)
python train_refine.py --epochs 15 --lr 5e-4 --init checkpoints\refine_best.pt --boxnet checkpoints\box_best.pt   # 2b) fine-tune Stage 2
python calibrate_score.py --n 300      # 3) learn best-of-K scoring weights (+ honest random/hand/learned/oracle comparison)
python evaluate.py                     # 4) metrics -> logs/eval_test.json
python generate.py --samples-dir samples
python generate.py --input samples/rect_2bhk.json --n-samples 3      # 3 different layouts for one input
# any training command: add --resume to continue after a stop.   Stage 2 from scratch: python train_refine.py --epochs 40
```
Quick check: `python prepare_data.py --limit 500`, `python train_box.py --epochs 3`, `python train_refine.py --epochs 1 --base 16`.
Colab/Kaggle: `colab_kaggle_train.ipynb`, then copy `box_best.pt` + `refine_best.pt` into `checkpoints/`.
**Compatibility (v3.2):** `refine_best.pt` and `data/cache4_*` still work. The old `box_best.pt` does NOT (AdaLN layers) → retrain Stage 1 (minutes). v2 pipeline is in `legacy_v2/`.

## Input JSON
```json
{"name": "my_flat",
 "boundary": [[0,0],[12,0],[12,9],[0,9]],
 "front_door": [6, 0],
 "rooms": [{"id":"living","type":"living"}, {"id":"kitchen","type":"kitchen"}, {"id":"bed1","type":"bedroom"},
           {"id":"bath1","type":"bathroom"}, {"id":"balcony","type":"balcony"}],
 "connections": [["living","kitchen"], ["living","bed1"], ["bed1","bath1"], ["living","balcony"]],
 "entrance": "living"}
```
- **types**: living, bedroom, bathroom, kitchen, balcony, storage, stair (max 16 rooms).
- **sizes** (optional): `{"id":"bed1","type":"bedroom","area":16}` — numbers are RELATIVE weights between rooms (any unit); rooms without
  `area` get typical sizes. `"use_area": false` ignores sizes.
- **connections**: `[a, b]` = door between a and b; `[a, b, "adjacent"]` = share a wall without a door.
- **shortcuts**: `"rooms": {"living":1,"bedroom":3,"bathroom":2,"kitchen":1}` (ids become bedroom_1…) or a list of type names;
  if `connections` is omitted a typical graph is built (living = hub, bathrooms attach to bedrooms). `entrance` defaults to the living room.
- `boundary` is any polygon in any units (rescaled). `front_door` snaps to the nearest wall.

## Output
`outputs/<name>.png` (labelled plan) · `<name>.json` · `<name>_3d.html` (browser; drag = rotate, wheel = zoom, Shift+drag = pan).
Console line per plan: `rooms 7/7  doors 6/6 (+1 auto)  windows 8  connected True  best of 8` = rooms present, requested doors cut (+ doors added automatically so no room is isolated), windows, all rooms reachable.
3D page: room name labels (toggle button), colour legend, true floor colours.

## Metrics (`evaluate.py`)
`box_iou_{true,default,no}_sizes` (true vs no sizes = does size conditioning work; default carries no extra information) ·
`box_iou_oracle_best_of_8` + `sample_diversity_L1` (do noise samples differ?) · `room_iou_gt_box` / `room_iou_e2e` ·
per selection rule (k=1, best_of_8): `doors_placed`, `all_rooms_connected`, `plans_with_all_rooms`, `straightness` (real plans 0.98), `rectangularity` (real plans 0.92).

## Tuning
Stage 1: `--d 256 --heads 8 --layers 6 --epochs 200`, `--adj 12` (stronger "connected rooms must touch"). Stage 2: `--base 48`, `--epochs 60`. `generate.py --k 16`.
More ideas: `PROJECT_STATE.md` §7.

## Files
`common.py` · `graphdata.py` (rooms/graph extraction) · `prepare_data.py` · `models.py` (BoxNet, RefineNet) · `train_box.py` ·
`train_refine.py` · `layout.py` (walls/doors/windows + export) · `generate.py` · `evaluate.py` · `viewer_template.html` ·
`samples/` · `colab_kaggle_train.ipynb` · `legacy_v2/` · `logs/` · `PROJECT_STATE.md` · `CLAUDE.md`
