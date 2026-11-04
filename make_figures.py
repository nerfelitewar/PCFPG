"""Qualitative figures for the paper -> paper/fig_qualitative.png (random, NOT cherry-picked test plans: input | ground truth | retrieval B1 | ours) and paper/fig_candidates.png (K candidates of one input).
  python make_figures.py --rows 4 --k 8"""
import argparse, numpy as np, cv2
from common import ROOT, PALETTE, OUT, DEFAULT_SIZE
from generate import Gen
from baselines import Retriever
ap = argparse.ArgumentParser(); ap.add_argument("--rows", type=int, default=4); ap.add_argument("--k", type=int, default=8); ap.add_argument("--n", type=int, default=1632); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--box-ckpt", default=str(ROOT / "checkpoints" / "box_best.pt")); ap.add_argument("--ref-ckpt", default=str(ROOT / "checkpoints" / "refine_best.pt")); a = ap.parse_args()
z = np.load(ROOT / "data" / "cache4_test.npz"); n = min(a.n, len(z["cond"])); g = Gen(a.box_ckpt, a.ref_ckpt); ret = Retriever(); S = 220
idx = np.random.RandomState(a.seed).choice(n, min(a.rows, n), replace=False); (ROOT / "paper").mkdir(exist_ok=True)
rs = lambda m: cv2.resize(m, (S, S), interpolation=cv2.INTER_NEAREST); col = lambda lab: rs(PALETTE[lab])[..., ::-1]
def typ(inst, nt): lut = np.array([0] + [int(t) + 2 for t in nt if t >= 0] + [0] * 17, np.uint8); return lut[inst]
def tag(img, txt): img = img.copy(); cv2.putText(img, txt, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA); return img
rows = []
for i in idx:
    nt = z["ntype"][i]; m = int((nt >= 0).sum()); share = z["share"][i]; dflt = np.array([DEFAULT_SIZE[int(t)] if t >= 0 else 0.0 for t in nt]); dflt /= dflt.sum()
    cs = g.candidates(z["cond"][i].astype(np.float32), nt, z["entr"][i].astype(np.float32), z["edge"][i], None, share, True, seed=int(i), k=a.k); best = max(cs, key=lambda c: c["score"])
    inp = np.full((128, 128, 3), 255, np.uint8); inp[z["cond"][i][0] > 0] = (245, 215, 200); inp[z["cond"][i][1] > 0] = (3, 54, 166); inp = rs(inp); cv2.putText(inp, f"{m} rooms", (6, S - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    gt = col(typ(z["inst"][i], nt)); rt = col(typ(ret(z["cond"][i][0], nt, share), nt)); ours = cv2.resize(col(best["label"]), (S, S), interpolation=cv2.INTER_NEAREST)
    rows.append(np.hstack([tag(inp, "input"), tag(gt, "ground truth"), tag(rt, "retrieval (B1)"), tag(ours, "ours")]))
cv2.imwrite(str(ROOT / "paper" / "fig_qualitative.png"), np.vstack(rows))
i = int(idx[0]); nt = z["ntype"][i]; cs = g.candidates(z["cond"][i].astype(np.float32), nt, z["entr"][i].astype(np.float32), z["edge"][i], None, z["share"][i], True, seed=int(i), k=min(a.k, 6))
cv2.imwrite(str(ROOT / "paper" / "fig_candidates.png"), np.hstack([tag(cv2.resize(col(c["label"]), (S, S), interpolation=cv2.INTER_NEAREST), f"cand {j + 1} s={c['score']:.1f}") for j, c in enumerate(cs)]))
print("wrote paper/fig_qualitative.png and paper/fig_candidates.png")