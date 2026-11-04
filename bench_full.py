"""Full evaluation: quality, conditioning, validity before/after repair, diversity, novelty, efficiency, bootstrap CIs, multiple sampling seeds.
  python bench_full.py --n 1632 --k 8 --seeds 3 --tag main --novelty --latency
Scenarios: 'default' = type-median room sizes (what a user gets), 'true' = the real plan's relative sizes are requested (tests parameter conditioning).
Outputs logs/bench_full_<tag>.json (+ per-plan arrays in logs/bench_full_<tag>_perplan.npz)."""
import argparse, json, time, warnings, numpy as np, torch, cv2
from common import ROOT, OUT, DEFAULT_SIZE
from generate import Gen
from models import box_loss, align_gt, group_sig
import metrics as mt
warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=200); ap.add_argument("--k", type=int, default=8); ap.add_argument("--seeds", type=int, default=1)
ap.add_argument("--tag", default="main"); ap.add_argument("--split", default="test"); ap.add_argument("--no-tta", action="store_true"); ap.add_argument("--novelty", action="store_true"); ap.add_argument("--latency", action="store_true")
ap.add_argument("--scenarios", default="default,true"); ap.add_argument("--box-ckpt", default=str(ROOT / "checkpoints" / "box_best.pt")); ap.add_argument("--ref-ckpt", default=str(ROOT / "checkpoints" / "refine_best.pt")); a = ap.parse_args()
g = Gen(a.box_ckpt, a.ref_ckpt); dev = g.dev; rng = np.random.RandomState(0); z = np.load(ROOT / "data" / f"cache4_{a.split}.npz"); n = min(a.n, len(z["cond"]))
nov = mt.Novelty(ROOT / "data" / "cache4_train.npz") if a.novelty else None; tg = lambda x, dt: torch.as_tensor(x, dtype=dt, device=dev)[None]


def one(i, scn, seed):
    nt = z["ntype"][i]; m = int((nt >= 0).sum()); true = z["share"][i].astype(np.float64); dflt = np.array([DEFAULT_SIZE[int(t)] if t >= 0 else 0.0 for t in nt]); dflt /= dflt.sum()
    share = true if scn == "true" else dflt; sig, edge = z["sig"][i], z["edge"][i]
    cs = g.candidates(z["cond"][i].astype(np.float32), nt, z["entr"][i].astype(np.float32), edge, None, share, True, seed=seed * 100003 + i, k=a.k, tta=not a.no_tta)
    gt = cv2.resize(z["inst"][i].astype(np.uint8), (OUT, OUT), interpolation=cv2.INTER_NEAREST); inner = cv2.resize(z["cond"][i][0].astype(np.uint8), (OUT, OUT), interpolation=cv2.INTER_NEAREST) > 0
    v = tg(nt >= 0, torch.bool); gs = group_sig(tg(sig, torch.long), tg(share, torch.float32), torch.ones(1, dtype=torch.bool, device=dev)); best = max(cs, key=lambda c: c["score"]); out = {}
    for tag, c in (("k1", cs[0]), ("bestK", best)):
        r = {}; st = c["stats"]; pb = tg(c["boxes"], torch.float32)
        r["box_iou"] = float(box_loss(pb, align_gt(pb, tg(z["boxes"][i], torch.float32), gs), v, False)[1][0]); r["room_iou"] = mt.matched_iou(c["inst"], gt, m, sig)
        ad = mt.adjacency(c["inst"], edge, m); r["adj_recall"] = ad["sat"] / ad["req"] if ad["req"] else np.nan; r["adj_precision"] = ad["sat"] / ad["real"] if ad["real"] else np.nan
        r["door_dist"], r["door_recall"] = mt.door_graph(c["label"], c["inst"], edge, m); r["size_mae"], r["size_rel"], r["size_rho"] = mt.size_error(c["inst"], share, m)
        r.update(mt.validity(c["inst"], c["raw"], st, m)); r.update(mt.boundary(c["inst"], inner)); r["straight"] = st["straightness"]; r["rect"] = st["rectangularity"]; r["auto_doors"] = st["doors_auto"] / m
        if tag == "bestK":
            r["diversity"] = mt.diversity([x["inst"] for x in cs], m, sig); r["novelty_nn"] = nov.query(c["inst"], nt, m) if nov else np.nan
        out[tag] = r
    return out


R = {}; t0 = time.time(); scns = a.scenarios.split(",")
for scn in scns:
    acc = {}
    for seed in range(a.seeds):
        for i in range(n):
            for tag, r in one(i, scn, seed).items():
                for k_, v_ in r.items(): acc.setdefault((tag, k_), np.full((a.seeds, n), np.nan))[seed, i] = v_
            if (i + 1) % 25 == 0: print(f"[{scn} seed {seed}] {i + 1}/{n}  {time.time() - t0:.0f}s", flush=True)
    R[scn] = acc
res = {"tag": a.tag, "n": n, "k": a.k, "seeds": a.seeds, "tta": not a.no_tta, "device": dev, "split": a.split, "scenarios": {}, "protocol": "ResPlan split.json; matched (Hungarian within Weisfeiler-Lehman groups) room IoU; percentile bootstrap 95% CI over plans (2000); per-plan values averaged over sampling seeds"}
per = {}
for scn, acc in R.items():
    res["scenarios"][scn] = {}
    for (tag, k_), A in acc.items():
        pm = np.nanmean(A, 0); d = mt.boot_ci(pm, rng); sm = np.nanmean(A, 1); d["seed_means"] = [float(x) for x in sm]; d["seed_std"] = float(sm.std(ddof=1)) if len(sm) > 1 else None
        res["scenarios"][scn].setdefault(tag, {})[k_] = d; per[f"{scn}/{tag}/{k_}"] = pm
if a.latency:
    sub = min(30, n); eff = {"params_boxnet": sum(p.numel() for p in g.box.parameters()), "params_refinenet": sum(p.numel() for p in g.ref.parameters())}
    for kk in (1, a.k):
        ts = []
        for i in range(-2, sub):
            ii = max(i, 0); nt = z["ntype"][ii]; sh = z["share"][ii]
            if dev == "cuda": torch.cuda.synchronize()
            t1 = time.time(); g.candidates(z["cond"][ii].astype(np.float32), nt, z["entr"][ii].astype(np.float32), z["edge"][ii], None, sh, True, seed=max(i, 0), k=kk, tta=not a.no_tta)
            if dev == "cuda": torch.cuda.synchronize()
            if i >= 0: ts.append(time.time() - t1)
        eff[f"latency_k{kk}_p50_s"] = float(np.percentile(ts, 50)); eff[f"latency_k{kk}_p95_s"] = float(np.percentile(ts, 95))
    if dev == "cuda": eff["peak_gpu_mem_mb"] = torch.cuda.max_memory_allocated() / 2 ** 20; eff["gpu"] = torch.cuda.get_device_name(0)
    res["efficiency"] = eff
(ROOT / "logs").mkdir(exist_ok=True); (ROOT / "logs" / f"bench_full_{a.tag}.json").write_text(json.dumps(res, indent=1, default=float)); np.savez(ROOT / "logs" / f"bench_full_{a.tag}_perplan.npz", **per)
for scn in res["scenarios"]:
    for tag in ("k1", "bestK"):
        print(scn, tag, {k_: round(v_["mean"], 3) for k_, v_ in res["scenarios"][scn][tag].items() if k_ in ("room_iou", "box_iou", "adj_recall", "door_recall", "valid_pre", "valid_post", "size_mae", "diversity")})
print("wrote logs/bench_full_%s.json" % a.tag)