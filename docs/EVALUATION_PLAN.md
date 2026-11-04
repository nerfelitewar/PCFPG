# PCFPG — Evaluation Framework (IEEE-style)

Scope: derived from the repository as inspected (`evaluate.py`, `bench.py`, `calibrate_score.py`, `layout.py` FEATS, `generate.py`, training scripts). Nothing below is a measured result; "Status" = what the code does today. Links/DOIs were written without live verification in this session — open each before citing.

## 0. What exists today (verified from code) and what is wrong with it
| Item | Finding |
|---|---|
| Reported now | box IoU (true / default / no sizes), room IoU (real boxes, end-to-end), doors_placed, all_rooms_connected, plans_with_all_rooms, straightness, rectangularity, avg_windows, auto_doors_per_plan; k=1 vs best-of-K; learned vs hand selection (`calibrate_score.py`); `bench.py` adds per-plan values, bootstrap 95% CI, mean-box baseline, sec/plan |
| Missing | GED/graph consistency, adjacency satisfaction (non-door), area/size conformance in eval output (`size_err` exists only as a ranking feature), validity rate, diversity, novelty, FID/KID, memory/FLOPs/params, multi-seed training, significance tests, ablation table, any non-trivial baseline, human study |
| Flaw 1 | **Metrics satisfied by construction.** `restore_missing` re-inserts dropped rooms, `ensure_connected` adds doors, `manhattanize_instances` forces right angles. So room-count, connectivity, straightness, rectangularity ≈ 1.0 say little about learning. Report **pre-repair** rates (before restore/repair) and post-repair separately. |
| Flaw 2 | **One-to-many problem.** IoU against the single ground-truth layout penalises valid alternatives; it measures *fidelity to the architect's choice*, not quality. Keep it as a secondary reference metric, add constraint-satisfaction + distribution metrics. |
| Flaw 3 | **Room types are inputs**, so room-type accuracy is trivial. Only presence/count after generation (pre-repair) is informative. |
| Flaw 4 | One training run, one split: no variance across seeds. `--seed` exists in training scripts but results are not aggregated. |
| Check | Confirm best-of-K selection at test time uses only the learned score (no ground truth); `calibrate_score.py` fits on val, evaluation must use test only. |

## 1. Why accuracy / F1 are not primary
Generation is one-to-many: many different layouts satisfy the same boundary + graph. Pixel accuracy is dominated by wall/background imbalance; F1 on classes presumes a single correct answer. Use P/R/F1 only for **constraint satisfaction** (e.g. requested vs realised adjacencies), never as a generation-quality score.

## 2. Metric catalogue
Tier: **E** essential, **O** optional. Notation: $P_r,G_r$ predicted/true pixel sets of room $r$; $R$ rooms; $N$ plans.

### A. Fidelity to reference (secondary)
| Metric | Formula | Measures / interpretation | Implementation (inputs) | Limits | Tier · Status |
|---|---|---|---|---|---|
| Box IoU (Stage 1) | $\frac1R\sum_r \frac{|B_r\cap B^*_r|}{|B_r\cup B^*_r|}$ | placement accuracy; 1 = identical | `models.box_loss` (pred, GT boxes, valid mask) | one reference only | E · ✅ |
| Room IoU (e2e) | $\frac1N\sum_n\frac1{R_n}\sum_r\frac{|P_r\cap G_r|}{|P_r\cup G_r|}$ | final-region overlap, rooms matched via WL signature + Hungarian | `bench.py room_iou_pp` | same | E · ✅ |
| Oracle best-of-K IoU | $\max_{k\le K}\mathrm{IoU}(\ell_k)$ | upper bound of selection | calibrate_score | uses GT → label "oracle" | O · ✅ |

### B. Validity and architecture
| Metric | Formula | Measures | Implementation | Limits | Tier · Status |
|---|---|---|---|---|---|
| Validity rate | $\frac1N\sum_n \mathbb 1[\text{all checks pass}]$ | checks: all rooms present, each room one connected component, valid polygons (shapely), area ≥ $a_{min}$, every room reachable from entrance through doors | new `metrics.validity()` on label+inst maps | thresholds are design choices; report per-check | E · ❌ (partial: present/connected) |
| Pre-repair rates | same checks **before** `restore_missing`/`ensure_connected` | what the networks achieve without rule-based rescue | add flags in `generate.py` | — | E · ❌ |
| Boundary violation | $\frac{|\cup_r P_r\setminus \Omega|}{|\Omega|}$, coverage $\frac{|\cup_r P_r\cap\Omega|}{|\Omega|}$ | rooms outside / uncovered footprint $\Omega$ | mask ops | ≈0 / ≈1 by construction (Stage 2 tiles outline) → verify, then report once | E · ❌ |
| Room overlap | $\frac{\sum_{i<j}|P_i\cap P_j|}{|\cup P|}$ | overlap | instance map is single-label ⇒ 0 by construction; report **box** overlap (Stage 1) | trivial for Stage 2 | O · partial (feature `box_overlap`) |
| Accessibility | $\frac1N\sum_n \mathbb 1[\text{graph of door edges connected}]$ | circulation | door graph BFS from entrance | repaired doors inflate it → pre/post | E · ✅ post-repair only |

### C. Parameter conditioning (core claim)
| Metric | Formula | Measures | Implementation | Limits | Tier · Status |
|---|---|---|---|---|---|
| Room count / type match | exact: $\mathbb 1[\text{multiset}(\hat T)=\text{multiset}(T)]$; per type P/R/F1 | requested rooms realised (pre-repair) | label→room list | types given as input | E · partial |
| Adjacency satisfaction | $\mathrm{Adj\text{-}R}=\frac{|\{(i,j)\in E_{req}: \text{shared border}\ge\tau\}|}{|E_{req}|}$ ; $\mathrm{Adj\text{-}P}=\frac{|E_{real}\cap E_{req}|}{|E_{real}|}$ | requested links physically touch; unrequested ones are not forced | border pixel count between instances, τ≈4 px | extra adjacency may be legitimate → report P as informational | E · ❌ |
| Door satisfaction | $\frac{\#\text{requested door edges with a door}}{\#E^{door}_{req}}$ | doors honoured (pre-repair) | existing `doors_placed` | auto-doors inflate | E · ✅ |
| Area conformance | $\mathrm{MAE}_{share}=\frac1R\sum_r|\hat s_r-s_r|$, $\hat s_r=|P_r|/\sum|P|$; Spearman ρ of sizes | requested relative sizes obeyed | instance areas vs input `share` | only relative sizes are conditioned | E · ❌ (feature only) |
| Entrance satisfaction | $\mathbb 1[\text{entrance door lies on requested room}]$ | entrance conditioning | label map | — | O · ❌ |
| Dimension conformance | $|\hat w-w|/w$ for rooms with given extents | only if the product supports explicit dimensions | — | **not supported by current inputs** | — |

### D. Graph consistency
$d_{GED}=\frac{|E_{req}\,\triangle\,E_{real}|}{|E_{req}\cup E_{real}|}$ with node identity known (room ids) — exact and cheap. Classic GED (Sanfeliu & Fu) is needed only when comparing with methods that use unlabeled nodes; our identity-based version is **not comparable** to published GED numbers. Tier E · ❌.

### E. Image-level quality (only where justified)
| Metric | Formula | Notes | Tier · Status |
|---|---|---|---|
| FID | $\|\mu_r-\mu_g\|^2+\mathrm{Tr}(\Sigma_r+\Sigma_g-2(\Sigma_r\Sigma_g)^{1/2})$ | Inception features are not trained on plans; biased for small n (test split ≈ 1.6k) → use with KID / unbiased FID; needed for comparability because ResPlan proposes it | O (E if comparing with published) · ❌ |
| KID | unbiased MMD² with cubic polynomial kernel | preferable for small n | O · ❌ |
| SSIM | $\frac{(2\mu_x\mu_y+c_1)(2\sigma_{xy}+c_2)}{(\mu_x^2+\mu_y^2+c_1)(\sigma_x^2+\sigma_y^2+c_2)}$ | meaningful only vs a single reference; same one-to-many issue; low value | O (not recommended) · ❌ |
| Precision/Recall, Density/Coverage | k-NN manifold in a feature space | needs domain feature extractor (e.g. our own layout descriptor) | O · ❌ |
Protocol if used: identical rasterisation (same palette, size, wall rendering) for every method; generated vs held-out real plans **for the same conditions**; report n.

### F. Diversity and novelty
- **Intra-condition diversity** (K layouts for one input): $\mathrm{Div}=\frac{2}{K(K-1)}\sum_{a<b}\big(1-\mathrm{IoU}_{match}(\ell_a,\ell_b)\big)$. E · ❌ (candidates exist in `Gen`).
- **Novelty / memorisation:** $\mathrm{NN\text{-}IoU}(\ell)=\max_{t\in\text{train}}\mathrm{IoU}(\ell,t)$ over training plans with the same room-type multiset; report mean and the share > 0.9 (near-copies). E · ❌.
- Trade-off curve: quality (room IoU, validity) vs diversity as K and sampling noise vary.

### G. Efficiency
Params; FLOPs (fvcore/thop); latency p50/p95 per plan for K=1, K=8 with/without TTA (warm-up excluded, `torch.cuda.synchronize`); throughput plans/s; peak memory `torch.cuda.max_memory_allocated`; hardware string. E · partial (`sec_per_plan` in bench.py).

### H. Statistical reliability
≥3 **training** seeds × ≥5 **sampling** seeds; per-seed test mean → mean ± std across training seeds; per-plan bootstrap 95% CI (paired across methods, 2000 resamples); paired Wilcoxon signed-rank on per-plan differences with Holm correction; effect size Vargha–Delaney A12 or Cliff's δ; evaluate on the **entire** `split.json` test set (no subsampling; n ≈ 1,632 gives CI half-width ≈ $1.96\,s/\sqrt n$); publish split hash, seeds, commit. E · partial (CI exists).

### I. Human evaluation (optional but required for any "realistic/usable" claim)
Blinded, randomised; 60–100 conditions × {ours, retrieval baseline, ground truth, ≥1 published baseline}; ≥3 raters (architects/students); 5-point Likert on functionality, circulation, realism, usability + pairwise preference (Bradley–Terry); inter-rater reliability ICC(2,k) or Krippendorff α; report mean + CI; ethics/consent statement. O · ❌.

## 3. Baselines
| # | Baseline | Fair? | Status |
|---|---|---|---|
| B0 | Mean box per room type | yes, lower bound | ✅ `bench.py` |
| B1 | **Nearest-neighbour retrieval**: training plan with same type multiset and closest boundary, warped to boundary (Graph2Plan-style retrieval) | yes, cheap, strong sanity check | ❌ proposed |
| B2 | Our ablations (below) | yes | ❌ |
| B3 | Graph2Plan, House-GAN++, HouseDiffusion retrained on ResPlan train split, same inputs, same test split, same metrics | only fair if retrained; their published numbers (other dataset/metrics) must not be compared | ❌ large effort, do last |
Ablations: −AdaLN size conditioning; −multi-hypothesis; −tiling loss; −TV/edge-weighting; −RefineNet fine-tune on predicted boxes; −TTA; −Manhattanisation; K ∈ {1,2,4,8,16}; hand vs learned vs random selection; boxes-only (no Stage 2).

## 4. Prioritised checklist
**P0 (needed for submission):** full-test-set run with CIs · pre/post-repair separation · validity · adjacency/area/entrance conformance · graph Jaccard-GED · diversity + novelty · 3 training seeds · ablation table · B0+B1 · efficiency table · paired tests.
**P1:** FID/KID (comparability) · human study · one retrained published baseline (B3).
**P2:** P/R-D/C, SSIM.

## 5. Experiment plan
E1 main results (3×5 seeds, full test) → E2 ablations → E3 robustness (rooms 3–12, rectangle/L/T boundaries, sizes known/unknown) → E4 K-sweep quality/diversity → E5 efficiency → E6 baselines B0/B1 (then B3) → E7 human study → E8 write-up with checklist from `bench` tab.

## 6. Implementation roadmap
1. `metrics.py` (new): `validity`, `adjacency_sat`, `area_err`, `entrance_ok`, `graph_jaccard`, `diversity`, `novelty_nn`; unit tests on ground-truth plans (GT should score ≈1 on constraints).
2. `generate.py`: expose pre-repair outputs (flags around `restore_missing`, `ensure_connected`).
3. `bench.py`: call `metrics.py`, add `--seeds`, `--ckpt-dirs` (multi-seed), write per-plan arrays.
4. `stats.py`: paired bootstrap, Wilcoxon + Holm, A12.
5. `baselines.py`: B1 retrieval.
6. `efficiency.py`: params/FLOPs/latency/memory.
7. `fid_kid.py` (optional): fixed rasteriser + torch-fidelity/clean-fid style computation.
8. Benchmark tab: read `bench.json`; keep the "claims supported / not supported" table.

## 7. Claims — current support
Supported by code: pipeline produces walls/doors/windows/3D; beats mean-box baseline (measure with CI). **Not supported:** better than any published method; state of the art; architecturally valid/liveable; generalises beyond ResPlan; ResPlan paper (arXiv:2508.14006) reports no baseline numbers in the version read, so no published ResPlan leaderboard exists to cite.

## 8. References (check each link before citing)
1. M. Abouagour, E. Garyfallidis, *ResPlan*, arXiv:2508.14006, 2025. https://arxiv.org/abs/2508.14006
2. R. Hu et al., *Graph2Plan*, ACM TOG 39(4), 2020. doi:10.1145/3386569.3392391
3. N. Nauata et al., *House-GAN*, ECCV 2020. arXiv:2003.06988
4. N. Nauata et al., *House-GAN++*, CVPR 2021. arXiv:2103.02574
5. M. A. Shabani et al., *HouseDiffusion*, CVPR 2023. arXiv:2211.13287
6. W. Wu et al., *Data-driven interior plan generation (RPLAN)*, ACM TOG 38(6), 2019. doi:10.1145/3355089.3356556
7. M. Heusel et al., *GANs trained by a two time-scale update rule (FID)*, NeurIPS 2017. arXiv:1706.08500
8. M. Bińkowski et al., *Demystifying MMD GANs (KID)*, ICLR 2018. arXiv:1801.01401
9. M. J. Chong, D. Forsyth, *Effectively unbiased FID*, CVPR 2020. arXiv:1911.07023
10. Z. Wang et al., *Image quality assessment: from error visibility to structural similarity*, IEEE TIP 13(4), 2004. doi:10.1109/TIP.2003.819861
11. T. Kynkäänniemi et al., *Improved precision and recall metric*, NeurIPS 2019. arXiv:1904.06991
12. M. F. Naeem et al., *Reliable fidelity and diversity metrics (density/coverage)*, ICML 2020. arXiv:2002.09797
13. A. Sanfeliu, K.-S. Fu, *A distance measure between attributed relational graphs*, IEEE Trans. SMC 13(3), 1983. doi:10.1109/TSMC.1983.6313167
14. H. Rezatofighi et al., *Generalized IoU*, CVPR 2019. doi:10.1109/CVPR.2019.00075
15. H. W. Kuhn, *The Hungarian method*, Nav. Res. Logist. Q. 2, 1955. doi:10.1002/nav.3800020109
16. B. Efron, *Bootstrap methods*, Ann. Statist. 7(1), 1979. doi:10.1214/aos/1176344552
17. F. Wilcoxon, *Individual comparisons by ranking methods*, Biometrics Bull. 1(6), 1945. doi:10.2307/3001968
18. J. Demšar, *Statistical comparisons of classifiers over multiple data sets*, JMLR 7, 2006. https://jmlr.org/papers/v7/demsar06a.html
19. S. Holm, *A simple sequentially rejective multiple test procedure*, Scand. J. Stat. 6(2), 1979.
20. A. Vargha, H. Delaney, *A critique and improvement of the CL common language effect size*, J. Educ. Behav. Stat. 25(2), 2000. doi:10.3102/10769986025002101
21. P. Henderson et al., *Deep reinforcement learning that matters*, AAAI 2018. arXiv:1709.06560 (multi-seed reporting practice)
22. A. Borji, *Pros and cons of GAN evaluation measures*, CVIU 179, 2019. arXiv:1802.03446
23. P. E. Shrout, J. L. Fleiss, *Intraclass correlations*, Psychol. Bull. 86(2), 1979. doi:10.1037/0033-2909.86.2.420
24. R. A. Bradley, M. E. Terry, *Rank analysis of incomplete block designs*, Biometrika 39, 1952. doi:10.2307/2334029