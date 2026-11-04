# PCFPG v3 — Parameter-Conditioned Floor Plan Generator

A graph-conditioned floor-plan generation system built on the **ResPlan** dataset. Provide a building boundary, a set of rooms, optional relative room sizes, and room-to-room connections. PCFPG generates a 2D floor plan and an interactive 3D browser view with walls, doors, windows, and an entrance.

## Contents

- [What it does](#what-it-does)
- [How it works](#how-it-works)
- [Repository structure](#repository-structure)
- [Getting started](#getting-started)
- [Run a trained model](#run-a-trained-model)
- [Train the models](#train-the-models)
- [Input JSON format](#input-json-format)
- [Generated output](#generated-output)
- [Evaluation](#evaluation)
- [Troubleshooting](#troubleshooting)
- [Dataset and citation](#dataset-and-citation)

## What it does

PCFPG turns a room graph and a building boundary into a complete floor-plan layout.

- **Graph-conditioned generation:** uses room types and room connections as structural conditions.
- **Relative room-size conditioning:** optional room-area weights influence the generated room proportions.
- **Multiple candidates:** generates and ranks multiple layout hypotheses, then selects the best candidate.
- **Geometric reconstruction:** converts predicted room boxes into Manhattan-style room polygons with 90-degree corners.
- **Architectural elements:** reconstructs walls, places doors from graph connections, adds windows, and places the entrance.
- **2D and 3D outputs:** exports a labelled PNG and an interactive HTML viewer.

## How it works

```text
Input JSON
(boundary + rooms + optional sizes + connections)
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 1 — BoxNet                                             │
│ Graph Transformer + AdaLN size conditioning                  │
│ Generates K candidate room-box layouts from noise hypotheses │
│ Trained with winner-takes-all supervision                     │
└─────────────────────────────────────────────────────────────┘
                  │ room boxes
                  ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 2 — RefineNet                                          │
│ U-Net refines room boxes into a per-room segmentation map    │
│ Edge-weighted cross-entropy + anisotropic TV                 │
│ Soft-box robustness and flip/rotation test-time augmentation │
└─────────────────────────────────────────────────────────────┘
                  │ refined room map
                  ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 3 — Geometric reconstruction (layout.py)               │
│ Restore dropped rooms → Manhattan polygons → walls           │
│ Doors from graph edges (+ automatic connectivity repair)     │
│ Windows + entrance + labels                                  │
└─────────────────────────────────────────────────────────────┘
                  │
                  ▼
       Labelled PNG + JSON + interactive 3D HTML

       generate.py builds and scores K candidates
       using hand-tuned or calibrated scoring weights.
```

The third stage is deterministic post-processing; it does not use a learned model. Candidate selection is handled by `generate.py`, with scoring weights that can be calibrated using `calibrate_score.py`.

## Repository structure

The main project files are:

| File or folder | Purpose |
|---|---|
| `common.py` | Shared configuration, constants, and utilities |
| `graphdata.py` | Room and connectivity-graph extraction |
| `prepare_data.py` | Dataset preprocessing and cache generation |
| `models.py` | BoxNet and RefineNet definitions |
| `train_box.py` | Stage 1 training |
| `train_refine.py` | Stage 2 training and fine-tuning |
| `layout.py` | Room geometry, walls, doors, windows, entrance, and exports |
| `generate.py` | Inference, candidate generation, scoring, and output export |
| `calibrate_score.py` | Learns candidate-scoring weights |
| `evaluate.py` | Evaluation metrics and test-set evaluation |
| `checkpoints/` | Model checkpoints, when supplied |
| `samples/` | Example input JSON files |
| `outputs/` | Generated floor plans and viewer files |
| `logs/` | Training and evaluation logs and metric outputs |
| `viewer_template.html` | Template for the interactive 3D viewer |
| `colab_kaggle_train.ipynb` | Notebook for Colab/Kaggle training |
| `legacy_v2/` | Previous v2 pipeline |

## Getting started

These instructions assume **Windows 10/11** and a local NVIDIA GPU. The commands use PowerShell. You can also run the project on CPU, but training and generation may be significantly slower.

### 1. Prerequisites

Install the following:

- Git
- Python and `venv`
- An NVIDIA driver compatible with the CUDA-enabled PyTorch build, if using an NVIDIA GPU

The setup below uses the CUDA 12.8 PyTorch wheel, which is appropriate for the RTX 50-series setup used during development. For another GPU or a CPU-only machine, install the PyTorch build that matches your hardware from the [official PyTorch installation selector](https://pytorch.org/get-started/locally/).

### 2. Clone the project

```powershell
git clone https://github.com/nerfelitewar/PCFPG.git
cd PCFPG
```

**Access note:** this repository may be private. You must be signed in to GitHub with an account that has been granted access before cloning it.

### 3. Download the ResPlan dataset

PCFPG uses the **ResPlan** residential floor-plan dataset. Download it from the upstream project:

- [ResPlan dataset repository](https://github.com/m-agour/ResPlan)
- [ResPlan research paper](https://arxiv.org/abs/2508.14006)

The upstream repository contains `ResPlan.zip` and `split.json`. Keep the dataset outside the PCFPG Git repository; the recommended layout is to place both folders next to each other:

```text
workspace/
├── PCFPG/
│   ├── prepare_data.py
│   ├── train_box.py
│   └── ...
└── ResPlan/
    ├── ResPlan.pkl
    └── split.json
```

After cloning PCFPG, open PowerShell in the `PCFPG` directory and run the following. The first command moves one level up so `PCFPG/` and `ResPlan/` become sibling folders:

```powershell
cd ..
git clone https://github.com/m-agour/ResPlan.git ResPlan-source
New-Item -ItemType Directory -Force ResPlan | Out-Null
Expand-Archive -Path .\ResPlan-source\ResPlan.zip -DestinationPath .\ResPlan -Force
Copy-Item .\ResPlan-source\split.json .\ResPlan\split.json -Force
```

Check that the expected files exist:

```powershell
Test-Path .\ResPlan\ResPlan.pkl
Test-Path .\ResPlan\split.json
```

Both commands should return `True`. If the extracted archive places `ResPlan.pkl` inside an additional nested folder, move it into the top-level `ResPlan` folder so the paths match the layout above.

If your dataset is stored elsewhere, pass its path to preprocessing:

```powershell
python prepare_data.py --data-dir "D:\path\to\ResPlan"
```

### 4. Create the Python environment

From the `PCFPG` project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If PowerShell blocks environment activation, run this in the same terminal and activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 5. Install dependencies

For an NVIDIA RTX 50-series GPU:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Verify that PyTorch can access the GPU:

```powershell
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU mode')"
```

`CUDA available: True` means PyTorch can see the GPU. If it reports `False`, see [Troubleshooting](#troubleshooting).

## Run a trained model

This is the quickest way to try PCFPG without training the networks yourself.

### 1. Check the checkpoints

The inference pipeline needs compatible Stage 1 and Stage 2 checkpoints. Check whether these files are present:

```text
checkpoints/
├── box_best.pt
└── refine_best.pt
```

If the checkpoint files are not included in your copy of the repository, obtain the project checkpoints from the project maintainer or train the models using the next section. Do not use an old v2 or pre-v3.2 `box_best.pt` with the v3.2 architecture.

### 2. Generate a floor plan

From the project root, run an example input:

```powershell
python generate.py --input samples/rect_2bhk.json --n-samples 3
```

This requests three candidate layouts for the supplied input. To generate plans from the sample set, run:

```powershell
python generate.py --samples-dir samples
```

Use `python generate.py --help` to see the options supported by your checked-out version.

### 3. View the result

Open the generated `.png` to inspect the labelled 2D layout. Open the generated `_3d.html` file in a browser to interact with the 3D model:

- Drag to rotate.
- Use the mouse wheel to zoom.
- Hold `Shift` and drag to pan.
- Use the viewer controls to toggle room labels.

See [Generated output](#generated-output) for the output files and console status line.

## Train the models

Training is only necessary when you do not have compatible checkpoints, or when you want to retrain or fine-tune the models.

### 1. Prepare the dataset

Run this once after placing the dataset in the expected location:

```powershell
python prepare_data.py
```

The script creates the processed cache used by the training pipeline. The existing `cache4` format is unchanged from v3.1, so you normally do not need to regenerate it unless the dataset or preprocessing changes.

### 2. Train Stage 1 — BoxNet

The v3.2 architecture adds AdaLN layers, so Stage 1 must be retrained when upgrading from an older v3 checkpoint:

```powershell
python train_box.py --epochs 200 --d 256 --heads 8 --layers 6
```

### 3. Fine-tune Stage 2 — RefineNet

With the compatible Stage 1 checkpoint available, fine-tune Stage 2:

```powershell
python train_refine.py --epochs 15 --lr 5e-4 --init checkpoints\refine_best.pt --boxnet checkpoints\box_best.pt
```

To train Stage 2 from scratch instead:

```powershell
python train_refine.py --epochs 40
```

### 4. Calibrate candidate scoring

Learn scoring weights for selecting the best candidate from multiple generated layouts:

```powershell
python calibrate_score.py --n 300
```

The calibration command also compares random, hand-tuned, learned, and oracle selection strategies.

### 5. Evaluate the model

```powershell
python evaluate.py
```

Evaluation metrics are written to `logs/eval_test.json`.

### 6. Generate layouts

```powershell
python generate.py --samples-dir samples
```

### Resume training

Add `--resume` to a supported training command to continue after an interrupted run. For example:

```powershell
python train_box.py --epochs 200 --d 256 --heads 8 --layers 6 --resume
```

Check each script's `--help` output for its available arguments.

### Quick smoke test

To verify the preprocessing and training pipeline with a small run before starting a full experiment:

```powershell
python prepare_data.py --limit 500
python train_box.py --epochs 3
python train_refine.py --epochs 1 --base 16
```

This is a quick pipeline check, not a meaningful model training run or evaluation.

### Train using Google Colab or Kaggle

Use [`colab_kaggle_train.ipynb`](colab_kaggle_train.ipynb) to train in a hosted notebook environment. After training, copy the compatible `box_best.pt` and `refine_best.pt` files into the local `checkpoints/` directory.

## Input JSON format

A generation request describes the building boundary, rooms, room connections, and optionally the entrance and relative room sizes.

### Example

```json
{
  "name": "my_flat",
  "boundary": [[0, 0], [12, 0], [12, 9], [0, 9]],
  "front_door": [6, 0],
  "rooms": [
    {"id": "living", "type": "living"},
    {"id": "kitchen", "type": "kitchen"},
    {"id": "bed1", "type": "bedroom"},
    {"id": "bath1", "type": "bathroom"},
    {"id": "balcony", "type": "balcony"}
  ],
  "connections": [
    ["living", "kitchen"],
    ["living", "bed1"],
    ["bed1", "bath1"],
    ["living", "balcony"]
  ],
  "entrance": "living"
}
```

### Fields

| Field | Required | Description |
|---|---|---|
| `name` | No | Name used when creating output files |
| `boundary` | Yes | Polygon coordinates describing the building boundary; coordinates can use any consistent units because the boundary is rescaled |
| `rooms` | Yes | Room IDs and semantic room types |
| `connections` | No | Room-to-room relationships; if omitted, the generator builds a typical graph |
| `front_door` | No | Point on or near the exterior boundary; it is snapped to the nearest wall |
| `entrance` | No | Room connected to the building entrance; defaults to the living room when available |
| `use_area` | No | Set to `false` to ignore relative room-size conditioning |

### Supported room types

`living`, `bedroom`, `bathroom`, `kitchen`, `balcony`, `storage`, and `stair`.

The current input limit is **16 rooms**.

### Relative room sizes

The optional `area` value is a relative weight, not a strict real-world area constraint. Rooms with larger values are encouraged to receive more space. Rooms without an `area` value use typical room-size defaults.

```json
{"id": "bed1", "type": "bedroom", "area": 16}
```

To ignore size conditioning, set:

```json
{"use_area": false}
```

### Connections

A connection describes the desired relationship between two room IDs:

```json
["living", "kitchen"]
```

This requests a door between the rooms. To request a shared wall without a door, use:

```json
["living", "kitchen", "adjacent"]
```

### Room-count shortcuts

Instead of listing every room object, you can use a room-count mapping. The generator expands the counts into room IDs such as `bedroom_1`, `bedroom_2`, and so on:

```json
"rooms": {
  "living": 1,
  "bedroom": 3,
  "bathroom": 2,
  "kitchen": 1
}
```

A list of room type names can also be used. If `connections` is omitted, a typical graph is generated, with the living room acting as a hub and bathrooms generally attached to bedrooms.

## Generated output

Generated files are saved under `outputs/`:

| Output | Description |
|---|---|
| `<name>.png` | Labelled 2D floor plan |
| `<name>.json` | Generated plan data |
| `<name>_3d.html` | Interactive 3D floor-plan viewer |

The 3D view includes room labels, a colour legend, and floor colours. It supports mouse drag to rotate, wheel to zoom, and `Shift` + drag to pan.

The generator prints a status line for each plan. For example:

```text
rooms 7/7  doors 6/6 (+1 auto)  windows 8  connected True  best of 8
```

This indicates that all 7 requested rooms were placed, all 6 requested doors were placed, one additional door was added automatically to repair connectivity, 8 windows were generated, and all rooms are reachable in the resulting layout. `best of 8` means the selected plan was chosen from 8 generated candidates.

## Evaluation

Run evaluation with:

```powershell
python evaluate.py
```

Results are saved to `logs/eval_test.json`. The evaluation includes:

| Metric | What it measures |
|---|---|
| `box_iou_true` | Box IoU using the true room-size conditioning |
| `box_iou_default` | Box IoU using the default size configuration |
| `box_iou_no_sizes` | Box IoU without size conditioning; useful for checking whether size conditioning helps |
| `box_iou_oracle_best_of_8` | Best candidate IoU when the ground-truth-best candidate is selected as an oracle upper bound |
| `sample_diversity_L1` | Diversity between noise-generated samples |
| `room_iou_gt_box` | Room IoU from ground-truth room boxes |
| `room_iou_e2e` | End-to-end room IoU |
| `doors_placed` | Door placement performance |
| `all_rooms_connected` | Whether every room is reachable in the generated plan |
| `plans_with_all_rooms` | Whether all requested rooms are present |
| `straightness` | How closely room geometry follows straight, axis-aligned edges |
| `rectangularity` | How rectangular the generated room polygons are |

Metrics are reported for the selection strategies and candidate counts supported by the evaluator, including `k=1` and `best_of_8`. The real-plan reference values noted during development are approximately **0.98 straightness** and **0.92 rectangularity**; these are reference statistics, not guaranteed PCFPG results.

## Tuning

The following are the main training and inference settings used during development:

| Component | Setting | Purpose |
|---|---|---|
| Stage 1 | `--d 256 --heads 8 --layers 6 --epochs 200` | BoxNet model capacity and training duration |
| Stage 1 | `--adj 12` | Stronger penalty/weight for connected rooms that should touch, as implemented by the training script |
| Stage 2 | `--base 48 --epochs 60` | RefineNet capacity and training duration |
| Generation | `--k 16` | Generate more hypotheses for candidate selection |


## Compatibility notes

- **v3.2:** `refine_best.pt` and `data/cache4_*` remain compatible with the current pipeline.
- The older `box_best.pt` checkpoint is **not compatible** with the v3.2 AdaLN architecture. Retrain Stage 1 before using the v3.2 pipeline.
- The previous v2 implementation is retained in `legacy_v2/`.

## Troubleshooting

### `torch.cuda.is_available()` returns `False`

Check that the NVIDIA driver is installed and working, and that the installed PyTorch build matches your CUDA setup. Reinstall the appropriate PyTorch build using the [official installation selector](https://pytorch.org/get-started/locally/), then rerun the verification command.

### Dataset file not found

Confirm that the dataset directory contains both `ResPlan.pkl` and `split.json`. The default expected location is a sibling `ResPlan/` directory next to the `PCFPG/` project directory. Otherwise, pass the dataset location using `--data-dir` when running `prepare_data.py`.

### Checkpoint missing

Inference requires compatible BoxNet and RefineNet checkpoints. Obtain the project checkpoints from the maintainer or train them using the instructions above. A checkpoint directory existing by itself does not mean the required weight files are present.

### Old BoxNet checkpoint fails to load

The v3.2 BoxNet architecture includes AdaLN layers. Retrain Stage 1 with the current code and use the resulting `checkpoints/box_best.pt`.

### The 3D viewer does not open

Open the generated `_3d.html` file directly in a modern browser. Check that the file exists in `outputs/` and that the generation command completed without an export error.

## Dataset and citation

PCFPG uses **ResPlan: A Large-Scale Vector-Graph Dataset of 17,000 Residential Floor Plans**.

- **Dataset and source code:** [github.com/m-agour/ResPlan](https://github.com/m-agour/ResPlan)
- **Research paper:** [arXiv:2508.14006](https://arxiv.org/abs/2508.14006)
- **Dataset licence:** the upstream repository identifies the dataset files as [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). Review the upstream licence, provenance, and takedown policy before redistributing the data.

Please cite and attribute the ResPlan dataset according to the upstream repository and paper. The dataset is maintained by its original authors; PCFPG is a separate project that uses the dataset.

