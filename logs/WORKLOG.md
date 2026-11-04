# WORKLOG (append-only, newest at bottom). Format: date — what was done — result — next.

## 2026-10-15 — Session 1 (Claude, claude.ai web)
- Discussed options: no official pretrained ResPlan generator found. (BOOMI Stage-A 7B on HF is text->spec only.)
  Chosen: train own conditional U-Net (boundary mask -> room label map), then Three.js 3D view.
- Downloaded real ResPlan.pkl in sandbox; inspected structure (see PROJECT_STATE §3).
- Wrote: common.py, prepare_data.py, model.py, train.py (v1), generate.py, postprocess.py, viewer_template.html, samples/, README.
- Tested: prepare_data --limit 300 OK (3 s); rasterized plan viewed OK; postprocess export OK; viewer JS passes `node --check`;
  build_input() for 4 samples OK. NOT tested: any torch code (sandbox pip install torch failed: no disk space).
- Delivered v1 zip to user.

## 2026-10-15 — Session 1 (cont.) — v2 upgrade requested by user
- User asked: end-to-end? RTX 5050 vs Colab/Kaggle? logs + README + handoff so work survives token exhaustion.
- train.py v2: resumable (last.pt), EMA weights in best.pt, bf16 autocast, channels_last, cudnn/TF32, whole dataset on GPU,
  CE+Dice loss, grad clip, early stopping, `--max-minutes`, `--ckpt-dir/--log-dir`, CSV logging.
- Added evaluate.py (metrics + room-count match), colab_kaggle_train.ipynb, PROJECT_STATE.md, CLAUDE.md, this log.
- Status: files compile (py_compile). Smoke test result recorded below if it was possible.
- Smoke test (sandbox, CPU, torch 2.14): prepare_data --limit 256 -> train 2 epochs base16 (val mIoU 0.114, junk as expected)
  -> resume test (found+fixed bug: --epochs mismatch on resume crashed OneCycle; now forced to saved value) -> evaluate.py OK
  -> generate.py on 4 samples OK. GPU path untested.
- Delivered v2 zip. NEXT (user): run full training on RTX 5050 or Colab, report val mIoU + count_match_rate; then tune (PROJECT_STATE §7).

## 2026-10-15 — Session 1 (cont.) — v3 REDESIGN after user feedback
- User feedback on v2: boundary + 3D great, but interior rooms poor, no interior walls, no doors/windows. Asked for room-connectivity input + new architecture.
- Diagnosis: pixel classifier (CE) regresses to the mean (blurry rooms) and 2-3 px walls/doors/windows vanish at 128 px.
- v3 design: (1) graph in the JSON, (2) Stage 1 BoxNet (graph transformer, Graph2Plan-style), (3) Stage 2 RefineNet instance U-Net,
  (4) walls/doors/windows DERIVED geometrically (layout.py) at 256 px. v2 moved to legacy_v2/.
- Found: ResPlan footprints always span 118 px on the long axis (fixed generate.py scaling; v2 used ~105 px).
- Tests (CPU, tiny): see PROJECT_STATE §6. Notable: layout builder placed 100% of graph doors on ground-truth rooms;
  RefineNet 0.70 room IoU after 1 epoch with real boxes; BoxNet box IoU 0.10->0.175 in 16 tiny epochs (baseline 0.078).
  e2e on weak tiny models: room_iou_e2e 0.08, doors_placed 0.59 (rooms not adjacent) -> added adjacency loss to Stage 1.
- Delivered v3 zip. NEXT (user): prepare_data -> train_box -> train_refine -> evaluate -> generate; report eval_test.json.

## 2026-10-15 — Session 1 (cont.) — v3.1 after user's first real run
- User trained v3 fully. evaluate.py: box_iou 0.5456, room_iou_gt_box 0.9172, room_iou_e2e 0.6043, doors_placed 0.9437, rooms_present 0.9978,
  plans_with_all_rooms 0.98, avg_windows 8.46. Screenshots: boundary fine; interior walls wobbly/curved; no 3D labels; bath2 had no door; room proportions meh.
- Diagnosis: Stage 1 box quality is the bottleneck (gt-box 0.92 vs e2e 0.60); wobble = U-Net region borders; missing door = rooms not adjacent.
- Tried global grid snapping of borders: REJECTED (clean-GT IoU 0.969 -> 0.88). Adopted per-border straightening (0.969 -> 0.964, visibly straight).
- Added: relative room sizes (Stage 1 input), best-of-K candidates, door auto-repair (connectivity guarantee), room labels in 3D, `--boxnet` fine-tune of Stage 2,
  --heads option, evaluate.py v2 (k=1 vs best-of-K, connectivity, rectangularity). cache -> cache4.
- Sandbox verification limited to tiny CPU models (see PROJECT_STATE §6). NEXT (user): rerun pipeline per README, report eval_test.json.

## 2026-10-15 — Session 1 (cont.) — v3.2
- User run 2 metrics + 4 issues (see PROJECT_STATE §6). Measured GT facts: rectangularity 0.922, GT straightness 0.982, GT box overlap mass 0.25, anchored tiling hinge = 0 at GT.
- Findings: sizes "failure" = evaluation artifact (default sizes carry no info); pairwise overlap loss non-informative (kept optional, default off); global-grid and cluster-window>=2px Manhattan variants lost IoU
  -> final: Manhattan polygons with ~1px cluster window (IoU 0.962 clean / better than raw on wobble, straightness ~0.98).
- Implemented: AdaLN BoxNet, multi-hypothesis WTA, tiling loss, curriculum, edge-weighted CE + TV + soft boxes (Stage 2), Manhattan + restore_missing, TTA, features + learned scoring (calibrate_score.py), evaluate v3.2.
- Tested on tiny CPU models only. NEXT (user): retrain Stage 1 -> fine-tune Stage 2 -> calibrate_score -> evaluate; report eval_test.json + calibration.json.

## 2026-10-17 — Session 1 (cont.) — v3.2

UI: app.py and boundary_editor/index.html added. They only orchestrate generate.Gen.candidates, parse_spec, build_input, layout.export. Run with streamlit run app.py. Tested headlessly with AppTest, not in a browser.

UI v2: room palette, dropdown connections, tooltips on all controls, quality report, Model & training tab. app.py and boundary_editor/index.html only; the pipeline is unchanged.

UI v3: drag-and-drop room graph editor, 32 room presets mapped onto the 7 trained types, balcony railing/deck/plants in the 3D viewer, gauge and plan-overlay metrics. Files: app.py, room_graph/, boundary_editor/, viewer_template.html. The pipeline is unchanged.