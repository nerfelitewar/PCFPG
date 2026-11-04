# PROJECT_STATE.md — living hand-off file (READ THIS FIRST)

> **For any Claude (or human) continuing this project:** read this whole file, then `logs/WORKLOG.md`
> (newest entries at the bottom). Do not redo finished work. After every work session, **update
> section 6 (Status) and append to `logs/WORKLOG.md`.** This file is the single source of truth.

## 1. Goal (from the user)
Deep-learning project on the **ResPlan** dataset: generate a residential floor plan from a user input,
then show it as a simple **3D web view** (walls extruded, door openings left open).
User hardware: **RTX 5050 laptop/desktop GPU** (Blackwell, 8 GB). Can also train on **Colab/Kaggle**
and copy `best.pt` back. User wants things *easy*, crisp explanations, no fluff.
User paths (Windows): `D:\MY CODING WORKSPACE\PCFPG_V2\ResPlan\` holds `ResPlan.pkl` + `split.json`;
project lives in `D:\MY CODING WORKSPACE\PCFPG_V2\floorplan_gen\`.

## 2. Pipeline (v3, two-stage + geometric post-processing)  [v2 pixel-U-Net is in legacy_v2/ — user found its interior rooms/walls/doors bad]
```
ResPlan.pkl --prepare_data.py (graphdata.py)--> data/cache3_{train,val,test}.npz + data/stats.json
   cond  (2,128,128)  boundary mask (= plan['inner']) + front-door mask
   ntype (16,) -1 pad; room type 0..6 = living,bedroom,bathroom,kitchen,balcony,storage,stair; rooms shuffled per plan (seeded)
   entr  (16,) entrance-room flag (room nearest to front_door)      edge (16,16) 0 none /1 adjacent /2 door
   sig   (16,) Weisfeiler-Lehman colour: rooms the model cannot tell apart (used for permutation-invariant box matching)
   boxes (16,4) x0,y0,x1,y1 in [0,1]    inst (128,128) dense instance map: 0 outside, k+1 = room k (wall gaps filled by nearest room)
--train_box.py--> checkpoints/box_best.pt     BoxNet: CNN(boundary + coord channels)->16x16 tokens; room tokens (type, entrance, noise z);
       4 layers of [self-attn with learned per-head bias per edge type | cross-attn to boundary | FFN]; deep supervision;
       loss = 5*L1 + 2*(1-GIoU) after Hungarian matching inside WL-equivalent groups; aug = random rot90/flip (boxes transformed too)
--train_refine.py--> checkpoints/refine_best.pt   RefineNet: U-Net, in = boundary+door+16 slot-box masks+7 type masks (25 ch),
       out = 17 logits (outside + room slot k) => instance segmentation; trained on GT boxes + random jitter; EMA, bf16, CE loss
--generate.py--> BoxNet -> boxes -> RefineNet -> logits bilinearly upsampled to 256 -> layout.py:
       clean (largest blob per room, fill gaps) -> walls = borders between instances (3px dilated; balcony|outside left open)
       -> doors = for every graph door-edge, cut ~16px opening in the shared wall -> front door on exterior wall near the given point
       -> windows on exterior walls of living/bedroom/bathroom/kitchen -> label grid (CLASSES ids) -> PNG + JSON + Three.js html
```
**v3.2 additions (after user's 2nd real run; see §6):**
- BoxNet: AdaLN-Zero conditioning (per-room c = MLP[type, entrance, size, known]) in all 3 sub-layers; multi-hypothesis training (`--hypotheses 4`, winner-takes-all
  over noise samples, so z really yields different layouts => best-of-K has something to pick from); GT-anchored tiling loss (`--tile`); optional pairwise overlap
  (`--overlap`, default 0: MEASURED non-informative — shuffled wrong layouts have LOWER overlap (0.014) than GT (0.054)); curriculum (size-noise sigma 0->0.25 over first 30%,
  p(known) 0.9->0.6, early-layer deep-supervision weight 0.8->0.2, aux losses fade in over first 20%). Old box_best.pt incompatible.
- RefineNet (train_refine.py): edge-weighted CE (`--edge-w`), anisotropic TV (`--tv`, diagonal edge costs ~1.94x straight), soft box edges (`--soft`). Same architecture => refine_best.pt reusable.
- layout.py: `manhattanize_instances` (Douglas-Peucker -> strictly orthogonal polygons, coords clustered window ~1px) default; border-straightening as fallback; `restore_missing`
  (dropped rooms re-inserted from predicted boxes); `straightness`; `layout_features` + `FEATS`/`HAND_W` scoring features.
- generate.py: TTA (4 flips/rotations, exact inverse unit-tested), per-candidate features, score weights from `checkpoints/score_weights.json` if present.
- calibrate_score.py: within-plan ridge regression of candidate features -> TRUE room IoU; reports random / hand / learned / oracle selection on held-out plans.
- evaluate.py: true/default/no-size box IoU, oracle best-of-K + sample diversity, straightness, rectangularity.
**v3.1 additions (after user's first real run):**
- Stage 1 gets per-room RELATIVE SIZE input (`share`, trained with noise + 30% masked; defaults = ResPlan type medians in generate.DEFAULT_SIZE).
- layout.py: per-border wall straightening (`straighten_instances`, 2 passes; a global-grid snap was tried and rejected: loses ~9% IoU even on clean GT),
  `ensure_connected` (adds the most sensible door so no room is isolated; PREF table), room labels (`room_labels`), stats (rectangularity, doors_auto, all_connected).
- generate.py: best-of-K candidates (`--k`, default 8) scored by `score()` (doors placed, rooms present, rectangularity, connectivity).
- train_refine.py: `--init` + `--boxnet` fine-tunes Stage 2 on Stage 1's own boxes (slots re-matched via `align_perm`/`relabel_inst`, unit-tested).
- viewer: room name sprites (toggle), legend, unlit true colours, transparent outside. cache is now `cache4_*` (has `share`).
Input JSON has rooms (+ optional `area` weights) + connections (see README). z-noise in BoxNet => `--n-samples K` gives K different layouts.

## 3. Verified facts about the dataset (inspected on the real files — don't re-derive)
- `ResPlan.pkl` = python `list` of 17000 `dict`s. Keys: balcony, bathroom, bedroom, door, garden, inner, parking,
  pool, wall, window, front_door, id, kitchen, land, net_area, area, neighbor, living, wall_depth
  (+ storage in ~10%, stair in ~4%; some plans lack window/kitchen/balcony/front_door).
- Values are shapely `Polygon`/`MultiPolygon`. `front_door`, `kitchen`, `living` are often a single `Polygon`.
- Coordinates are roughly in **[-12, 268]** (plan ≈ 0..256). We map this range onto 128 px (`common.LO/HI/K`).
- `split.json` keys: train 13053 / val 1632 / test 1632 / augmented 683. Entries are plan **`id` values**
  (NOT list indices) — `prepare_data.py` maps through `{p["id"]: p}`.
- Coordinates of `front_door` sit just outside `inner` (on the wall line).
- **Plans are normalised: the longer side of `inner` always spans exactly 118 px (5..123) and is centred on the other axis** -> generate.py scales input boundaries the same way.
- Graph stats match the ResPlan paper (~6.4 adjacency-only + ~6.3 door edges per plan, ~8.2 rooms/plan, 99.6% have <=16 rooms).
- Per-type-mean box baseline IoU = 0.078.
- Image y axis points down (kept as-is; 3D viewer maps raster y -> world z).
- The README says only geometry is in the pickle (no graph). We do NOT use graphs yet.

## 4. File map
| file | role |
|---|---|
| `common.py` | constants (classes, palette, ROOM_TYPES, M=16 slots, OUT=256), rasterization helpers |
| `graphdata.py` | plan -> rooms, connectivity graph, WL signatures, dense instance map, boxes |
| `prepare_data.py` | pickle -> `data/cache3_*.npz` + `stats.json` (`--limit N`, `--data-dir`) |
| `models.py` | `BoxNet`, `RefineNet`, `render_slots`, `dihedral` (aug), `align_gt`, `box_loss`, `adjacency_loss` |
| `train_box.py` / `train_refine.py` | Stage 1 / Stage 2 training (resume, csv logs, `--max-minutes`, `--ckpt-dir`) |
| `calibrate_score.py` | learn best-of-K scoring weights from data -> checkpoints/score_weights.json |
| `layout.py` | Manhattan polygons / restore dropped rooms / straighten borders, walls/doors(+repair)/windows/front door from instance map + graph; labels; PNG/JSON/3D export |
| `generate.py` | JSON spec parser (`parse_spec`, default graph), `Gen` pipeline class, CLI |
| `evaluate.py` | metrics -> `logs/eval_test.json` |
| `viewer_template.html` | Three.js r128 viewer (size-aware, lintels over doors) |
| `samples/*.json` | 4 inputs (explicit graph, counts shortcut, list shortcut, 3BHK+storage) |
| `colab_kaggle_train.ipynb` | two-stage training on Colab/Kaggle |
| `legacy_v2/` | old pixel-U-Net pipeline (superseded; its best.pt is incompatible) |
| `logs/` | `WORKLOG.md`, `box_log.csv`, `refine_log.csv`, `eval_test.json`, `generate_log.jsonl` |

## 5. How to run (user's PC, Windows, inside `floorplan_gen`)
```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # RTX 50-series needs CUDA 12.8+ build
pip install -r requirements.txt
python prepare_data.py
python train_box.py --epochs 200 --d 256 --heads 8 --layers 6      # minutes; --resume to continue (v3.2: retrain, old box_best.pt incompatible)
python train_refine.py --epochs 40        # long one (skip if refine_best.pt already exists); --resume to continue
copy checkpoints\refine_best.pt checkpoints\refine_gt_boxes.pt
python train_refine.py --epochs 15 --lr 5e-4 --init checkpoints\refine_gt_boxes.pt --boxnet checkpoints\box_best.pt
python calibrate_score.py --n 300
python evaluate.py
python generate.py --samples-dir samples
```
Old `box_best.pt` / `cache3_*` are incompatible with v3.1 (size inputs) -> rerun prepare_data + train_box. Old `refine_best.pt` still valid.

## 6. STATUS  (update me!)
Legend: [x] done+tested, [~] written, partially tested, [ ] todo
USER RUN 2 (v3.1, RTX 5050, evaluate.py N=100): box_iou_sizes 0.6043 | box_iou_nosizes 0.6075 | room_iou_gt_box 0.922 | room_iou_e2e 0.644 |
  k=1: doors_placed 0.9764, all_connected 0.95, rooms_present 0.9942, plans_with_all_rooms 0.95, rectangularity 0.8831 | best_of_8: doors 0.9867, connected 0.95, rect 0.8932
USER-REPORTED ISSUES: sizes not helping; Stage-1 ceiling; stair-step/curved walls; best-of-8 ineffective.
DIAGNOSIS (measured on ground truth in sandbox): (1) evaluate.py v3.1 fed DEFAULT type-median sizes = a function of room type = zero new information, so
  sizes==no-sizes is EXPECTED; the real test is true-vs-no sizes (now reported). (2) GT rectangularity is 0.922 (p10 0.67): 0.89 is near ceiling -> use `straightness` (GT 0.982).
  (3) GT boxes overlap heavily (mean excess occupancy 0.25) -> raw repulsion is wrong; anchored tiling hinge is 0 at GT and rises monotonically with box noise.
  (4) best-of-8 weak because (a) deterministic regression loss gives z no role -> near-identical candidates (hypothesis; on a tiny model diversity was 6% of error; v3.2 logs oracle best-of-4 & diversity)
  and (b) the 5% plans with a missing room fail in every candidate -> fixed by restore_missing.
- [x] Sandbox-verified (ground truth + synthetic wobble, CPU): Manhattan snapping straightness 0.70-0.86 -> 0.97-0.98 and IoU +0.9..+1.4pt on wobbly maps (clean GT 0.969->0.962);
      restore_missing + Manhattan: all rooms present/connected on 6/6 plans with a deleted room; TTA inverse exact; box matching/relabel unit-tested; all new scripts run end-to-end on tiny models
      (train_box with hypotheses+curriculum, train_refine with new losses + --boxnet, calibrate_score, evaluate, generate).
- [~] NOT verified at scale: gains from AdaLN / multi-hypothesis / tiling loss / edge+TV loss / learned scoring (need user's GPU run). Each has a switch for ablation:
      `--hypotheses 1`, `--tile 0`, `--tv 0 --edge-w 0 --soft 0`, `--no-tta`, delete score_weights.json.
- [ ] **USER: retrain Stage 1, fine-tune Stage 2, run calibrate_score.py + evaluate.py; paste eval_test.json + logs/calibration.json here.**
- [ ] Targets: true_sizes >> no_sizes (else raise AdaLN capacity / sizes weight); oracle_best_of_8 clearly > random_sample (diversity works); straightness >= 0.97; plans_with_all_rooms ~1.0; room_iou_e2e > 0.70.

## 7. Next steps / ideas (priority order)
0. (superseded by §6 targets)  Targets after v3.1 run: box_iou_sizes > 0.65, room_iou_e2e > 0.75, best_of_8 doors_placed > 0.97, all_rooms_connected ~1.0.
1. If `box_iou` is low (<~0.4) or `doors_placed` < ~90%: train Stage 1 longer/bigger (`--epochs 250 --d 256 --layers 6`), raise `--adj`,
   add overlap penalty between boxes, or have Stage 1 predict in an autoregressive / diffusion way for sharper multi-modality.
2. Fine-tune RefineNet on BoxNet's *own* predictions (currently trained on GT boxes + jitter) -> robustness.
3. Door repair: if a graph door-edge's rooms are not adjacent, nudge/grow boxes or re-sample z (`--n-samples`, pick best by stats).
4. Optional room areas as input (user-specified sizes); entrance placement search; window rules per orientation.
5. FastAPI + browser form (draw boundary, drag room graph) -> live 3D.
6. Furniture / door swings / roof slab in 3D; export OBJ/GLB.

## 8. Known risks / gotchas
- Stage-1 quality at full scale is UNPROVEN (only tiny CPU runs). Stage 2 + layout are proven to work well given good boxes.
- `--resume` needs the same `--epochs` (OneCycle length); scripts force the saved value with a warning.
- Input boundaries are rescaled so the LONGER side = 118 px and centred (matches ResPlan). Very elongated/odd shapes are out-of-distribution.
- Connection ids: `generate.parse_spec` accepts room ids or unique prefixes ("bedroom_1", "bed"). Max 16 rooms.
- `torch.compile` not used. Windows: no DataLoader workers needed (data lives on GPU).
- Viewer needs internet once (three.js from cdnjs).
