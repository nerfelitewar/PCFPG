"""Baselines on the SAME test split: B0 mean box per room type (no learning); B1 retrieval (Graph2Plan-style retrieval stage): nearest training plan with the same room-type multiset
and most similar footprint, rooms assigned by type and size rank, clipped to the test footprint. B1 ignores the connectivity graph and sees the true relative sizes.
  python baselines.py --n 1632"""
import argparse, json, cv2, numpy as np, torch
from common import ROOT, OUT
from layout import fill_nearest, build_layout
import metrics as mt
from models import box_loss


class Retriever:
    def __init__(s):
        s.tr = np.load(ROOT / "data" / "cache4_train.npz"); s.nt, s.sh, s.inst = s.tr["ntype"], s.tr["share"], s.tr["inst"]; s.foot = s.tr["cond"][:, 0, ::2, ::2] > 0
        s.cnt = np.stack([(s.nt == t).sum(1) for t in range(7)], 1); s.key = {}
        for i, r in enumerate(s.nt): s.key.setdefault(tuple(sorted(int(x) for x in r[r >= 0])), []).append(i)

    def __call__(s, cond0, nt, share):
        m = int((nt >= 0).sum()); idx = s.key.get(tuple(sorted(int(x) for x in nt[:m])))
        if not idx: c = np.array([(nt == t).sum() for t in range(7)]); d = np.abs(s.cnt - c).sum(1); idx = np.where(d == d.min())[0]
        f = cond0[::2, ::2] > 0; F = s.foot[idx]; j = idx[int((((F & f).sum((1, 2))) / np.maximum((F | f).sum((1, 2)), 1)).argmax())]; new = np.zeros_like(s.inst[j])
        for t in set(int(x) for x in nt[:m]):
            ti = sorted([k for k in range(m) if nt[k] == t], key=lambda k: -share[k]); si = sorted([k for k in range(16) if s.nt[j][k] == t], key=lambda k: -s.sh[j][k])
            for a_, b_ in zip(ti, si): new[s.inst[j] == b_ + 1] = a_ + 1
        inner = cond0 > 0; new[~inner] = 0; return fill_nearest(new, inner)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=200); ap.add_argument("--split", default="test"); a = ap.parse_args()
    z = np.load(ROOT / "data" / f"cache4_{a.split}.npz"); n = min(a.n, len(z["cond"])); rng = np.random.RandomState(0); tr = np.load(ROOT / "data" / "cache4_train.npz")
    mean = np.stack([tr["boxes"][tr["ntype"] == t].mean(0) for t in range(7)]); nt_ = z["ntype"][:n]
    b0 = box_loss(torch.tensor(mean[np.clip(nt_, 0, None)]), torch.tensor(z["boxes"][:n]), torch.tensor(nt_ >= 0), False)[1].numpy()
    ret = Retriever(); P = {k: [] for k in ("room_iou", "adj_recall", "adj_precision", "size_mae", "coverage", "outside")}
    for i in range(n):
        nt = z["ntype"][i]; m = int((nt >= 0).sum()); inst = ret(z["cond"][i][0], nt, z["share"][i]); gt = z["inst"][i].astype(np.uint8)
        P["room_iou"].append(mt.matched_iou(inst, gt, m, z["sig"][i])); ad = mt.adjacency(inst, z["edge"][i], m, tau=2)
        P["adj_recall"].append(ad["sat"] / ad["req"] if ad["req"] else np.nan); P["adj_precision"].append(ad["sat"] / ad["real"] if ad["real"] else np.nan)
        P["size_mae"].append(mt.size_error(inst, z["share"][i], m)[0]); b = mt.boundary(inst, z["cond"][i][0] > 0); P["coverage"].append(b["coverage"]); P["outside"].append(b["outside"])
    G = {}      # GT reference: the real layouts pushed through the SAME builder and metrics (ceiling for constraint metrics)
    for i in range(n):
        nt = z["ntype"][i]; m = int((nt >= 0).sum()); ins = cv2.resize(z["inst"][i].astype(np.uint8), (OUT, OUT), interpolation=cv2.INTER_NEAREST); inn = cv2.resize(z["cond"][i][0].astype(np.uint8), (OUT, OUT), interpolation=cv2.INTER_NEAREST) > 0
        lab, ins2, st = build_layout(ins.copy(), list(nt[:m]), z["edge"][i][:m, :m], inn, None, None, boxes=z["boxes"][i][:m]); r = {}; ad = mt.adjacency(ins2, z["edge"][i], m)
        r["adj_recall"] = ad["sat"] / ad["req"] if ad["req"] else np.nan; r["adj_precision"] = ad["sat"] / ad["real"] if ad["real"] else np.nan; r["door_dist"], r["door_recall"] = mt.door_graph(lab, ins2, z["edge"][i], m)
        r["size_mae"] = mt.size_error(ins2, z["share"][i], m)[0]; r.update(mt.validity(ins2, ins, st, m)); r.update(mt.boundary(ins2, inn)); r["straight"] = st["straightness"]
        for k_, v_ in r.items(): G.setdefault(k_, []).append(v_)
    out = {"n": n, "GT": {k: mt.boot_ci(v, rng) for k, v in G.items()}, "B0": {"box_iou": mt.boot_ci(b0, rng)}, "B1": {k: mt.boot_ci(v, rng) for k, v in P.items()}, "note": "B1: retrieval by room-type multiset + footprint IoU, rooms assigned by type/size rank, clipped to footprint; ignores graph; uses true relative sizes"}
    (ROOT / "logs").mkdir(exist_ok=True); (ROOT / "logs" / "baselines.json").write_text(json.dumps(out, indent=1))
    np.savez(ROOT / "logs" / "baselines_perplan.npz", **{"B0/box_iou": b0, **{f"B1/{k}": np.array(v) for k, v in P.items()}, **{f"GT/{k}": np.array(v) for k, v in G.items()}})
    print("B0 box IoU", round(out["B0"]["box_iou"]["mean"], 3), "| B1 room IoU", round(out["B1"]["room_iou"]["mean"], 3), "| wrote logs/baselines.json")