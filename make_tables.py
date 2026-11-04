"""Turns logs/*.json into LaTeX tables for the paper: paper/results_tables.tex + paper/results_numbers.tex. Run after bench_full.py / baselines.py / ablate.py."""
import json, glob, numpy as np
from common import ROOT
import metrics as mt
L, P = ROOT / "logs", ROOT / "paper"; P.mkdir(exist_ok=True); rng = np.random.RandomState(0)
ld = lambda p: json.loads(p.read_text()) if p.exists() else None
B, BL = ld(L / "bench_full_main.json"), ld(L / "baselines.json")
assert B, "run: python bench_full.py --n 1632 --k 8 --seeds 3 --tag main --novelty --latency"
nn = lambda x: isinstance(x, (int, float)) and x == x
def c(d, nd=3):
    if not d or not nn(d["mean"]): return "--"
    return f"{d['mean']:.{nd}f} {{\\scriptsize[{d['ci95'][0]:.{nd}f}, {d['ci95'][1]:.{nd}f}]}}"
S = lambda scn, tag, k: B["scenarios"].get(scn, {}).get(tag, {}).get(k)
GT = lambda k: (BL or {}).get("GT", {}).get(k)
def table(cap, lab, head, rows, cols="l" + "c" * 3):
    return ("\\begin{table}[t]\n\\caption{%s}\n\\label{%s}\n\\centering\\scriptsize\n\\resizebox{\\columnwidth}{!}{%%\n\\begin{tabular}{%s}\n\\toprule\n%s \\\\\n\\midrule\n%s\\\\\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n"
            % (cap, lab, cols, " & ".join(head), " \\\\\n".join(" & ".join(r) for r in rows)))
meta = f"$n={B['n']}$ test plans, $K={B['k']}$, {B['seeds']} sampling seed(s); mean and 95\\% bootstrap CI over plans."
out = []
rows = [("Room IoU (matched) $\\uparrow$", "room_iou", None), ("Stage-1 box IoU $\\uparrow$", "box_iou", None), ("Requested doors realised $\\uparrow$", "door_recall", "door_recall"), ("Door-graph distance $\\downarrow$", "door_dist", "door_dist"),
        ("Adjacency recall $\\uparrow$", "adj_recall", "adj_recall"), ("Adjacency precision $\\uparrow$", "adj_precision", "adj_precision"), ("Valid, after repair $\\uparrow$", "valid_post", "valid_post"),
        ("Valid, before repair $\\uparrow$", "valid_pre", "valid_pre"), ("Repair-free rate $\\uparrow$", "repair_free", "repair_free"), ("Straightness $\\uparrow$", "straight", "straight"),
        ("Footprint coverage $\\uparrow$", "coverage", "coverage"), ("Outside footprint $\\downarrow$", "outside", "outside")]
out.append(table("Main results, default room sizes. GT = ground-truth layouts through the same builder and metrics (reference ceiling). " + meta, "tab:main", ["Metric", "Best of $K$", "$K{=}1$", "GT ref."],
                 [[l, c(S("default", "bestK", k)), c(S("default", "k1", k)), c(GT(g)) if g else "--"] for l, k, g in rows]))
if "true" in B["scenarios"]:
    out.append(table("Parameter conditioning: requested sizes = the real plan's relative sizes (true) versus type-median sizes (default); best of $K$. " + meta, "tab:cond", ["Metric", "True sizes", "Default sizes"],
                     [[l, c(S("true", "bestK", k)), c(S("default", "bestK", k))] for l, k in (("Size MAE (area fraction) $\\downarrow$", "size_mae"), ("Size relative error $\\downarrow$", "size_rel"), ("Size Spearman $\\rho$ $\\uparrow$", "size_rho"), ("Room IoU $\\uparrow$", "room_iou"), ("Adjacency recall $\\uparrow$", "adj_recall"))], "lcc"))
num = {}
if BL:
    pm = {}; pr = ld(L / "baselines.json"); A = np.load(L / "bench_full_main_perplan.npz"); Bp = np.load(L / "baselines_perplan.npz"); rows = []
    for l, k, bk in (("Room IoU $\\uparrow$", "room_iou", "room_iou"), ("Adjacency recall $\\uparrow$", "adj_recall", "adj_recall"), ("Size MAE $\\downarrow$", "size_mae", "size_mae")):
        rows.append([l, c(BL["B1"].get(bk)), c(S("true", "k1", k)), c(S("true", "bestK", k))])
    rows.append(["Stage-1 box IoU (B0 vs.\\ ours) $\\uparrow$", c(BL["B0"]["box_iou"]), c(S("true", "k1", "box_iou")), c(S("true", "bestK", "box_iou"))])
    cmp = []
    for k in ("room_iou", "adj_recall", "size_mae"):
        if f"B1/{k}" in Bp.files and Bp[f"B1/{k}"].shape == A[f"true/bestK/{k}"].shape: cmp.append((k, mt.paired(A[f"true/bestK/{k}"], Bp[f"B1/{k}"], rng)))
    hp = mt.holm([d["p"] for _, d in cmp]) if cmp else []
    for (k, d), p in zip(cmp, hp): rows.append([f"Paired diff. vs.\\ B1, {k.replace('_', ' ')}", f"{d['diff']:+.3f} {{\\scriptsize[{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]}}", f"Holm $p={p:.1e}$", f"$A_{{12}}={d['a12']:.2f}$"])
    out.append(table("Comparison with baselines on the same split (true-size scenario). B0: mean box per room type. B1: retrieval of the nearest training plan with the same room set (graph ignored, true sizes given). Paired Wilcoxon with Holm correction; $A_{12}$ = Vargha--Delaney effect size. " + meta,
                     "tab:base", ["Metric", "Baseline", "Ours $K{=}1$", "Ours best of $K$"], rows))
    num["resBaseIoU"] = f"{BL['B1']['room_iou']['mean']:.3f}"
d = S("default", "bestK", "room_iou")
rows = [["Diversity (mean pairwise $1-$IoU of the $K$ candidates)", c(S("default", "bestK", "diversity"))], ["Novelty: nearest-neighbour type-map IoU to training plans", c(S("default", "bestK", "novelty_nn"))]]
E = B.get("efficiency")
if E:
    rows += [["Parameters (BoxNet / RefineNet), millions", f"{E['params_boxnet'] / 1e6:.2f} / {E['params_refinenet'] / 1e6:.2f}"]]
    rows += [[f"Latency per plan, $K={kk}$ (p50 / p95), s", f"{E[f'latency_k{kk}_p50_s']:.2f} / {E[f'latency_k{kk}_p95_s']:.2f}"] for kk in (1, B["k"])]
    if "peak_gpu_mem_mb" in E: rows.append(["Peak GPU memory, MB (" + E.get("gpu", "GPU").replace("_", "\\_") + ")", f"{E['peak_gpu_mem_mb']:.0f}"])
out.append(table("Diversity, novelty and efficiency. " + meta, "tab:eff", ["Quantity", "Value"], rows, "lc"))
ab = [(f.split("bench_full_")[1][:-5], ld(__import__("pathlib").Path(f))) for f in sorted(glob.glob(str(L / "bench_full_abl_*.json")))]
if ab:
    Am = np.load(L / "bench_full_main_perplan.npz"); rows = [["Full model", c(S("default", "bestK", "room_iou")), c(S("default", "k1", "room_iou")), "--"]]
    for tag, j in ab:
        dd = j["scenarios"]["default"]; pz = np.load(L / f"bench_full_{tag}_perplan.npz"); pp = mt.paired(Am["default/bestK/room_iou"], pz["default/bestK/room_iou"], rng) if Am["default/bestK/room_iou"].shape == pz["default/bestK/room_iou"].shape else None
        rows.append([tag.replace("abl_", "").replace("_", " "), c(dd["bestK"]["room_iou"]), c(dd["k1"]["room_iou"]), f"{pp['diff']:+.3f} ($p={pp['p']:.1e}$)" if pp else "--"])
    out.append(table("Ablations (room IoU, default sizes). Last column: full minus variant, paired Wilcoxon. " + meta, "tab:abl", ["Variant", "Best of $K$", "$K{=}1$", "Full $-$ variant"], rows))
txt = "\n".join(out)
for lab, msg in (("tab:main", "main results missing"), ("tab:cond", "conditioning scenario (true sizes) not run"), ("tab:base", "baselines not run: python baselines.py"), ("tab:eff", "efficiency missing"), ("tab:abl", "ablations not run: python ablate.py --run")):
    if "\\label{%s}" % lab not in txt: txt += "\n\\todo{%s}\\label{%s}\n" % (msg, lab)
(P / "results_tables.tex").write_text(txt)
num.update(resRoomIoU=f"{d['mean']:.3f}", resRoomIoULo=f"{d['ci95'][0]:.3f}", resRoomIoUHi=f"{d['ci95'][1]:.3f}", resN=str(B["n"]), resK=str(B["k"]))
(P / "results_numbers.tex").write_text("".join("\\renewcommand{\\%s}{%s}\n" % kv for kv in num.items()))
print("wrote paper/results_tables.tex and paper/results_numbers.tex")