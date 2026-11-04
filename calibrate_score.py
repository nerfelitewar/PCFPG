"""Learn the best-of-K scoring weights FROM DATA.
For validation plans: generate K candidates, measure each candidate's TRUE room IoU against the ground truth, then fit (within-plan ridge
regression) which layout features predict that IoU. Writes checkpoints/score_weights.json (generate.py/evaluate.py pick it up automatically)
and logs/calibration.json with the honest comparison  random pick  vs  hand-made score  vs  learned score  vs  oracle.
  python calibrate_score.py --n 300 --k 8"""
import argparse, json, numpy as np, torch, cv2
from common import ROOT, M, DEFAULT_SIZE
from generate import Gen
from layout import FEATS, HAND_W
from models import align_perm, relabel_inst, group_sig

ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=300); ap.add_argument("--k", type=int, default=8)
ap.add_argument("--ridge", type=float, default=1.0)
ap.add_argument("--box-ckpt", default=str(ROOT / "checkpoints" / "box_best.pt")); ap.add_argument("--ref-ckpt", default=str(ROOT / "checkpoints" / "refine_best.pt"))
a = ap.parse_args(); g = Gen(a.box_ckpt, a.ref_ckpt); dev = g.dev
z = np.load(ROOT / "data" / "cache4_val.npz"); n_pl = min(a.n, len(z["cond"])); X, Y, P = [], [], []
for i in range(n_pl):
    n = int((z["ntype"][i] >= 0).sum()); nt = z["ntype"][i]
    sh = np.zeros(M, np.float32); w = np.array([DEFAULT_SIZE[int(t)] for t in nt[:n]], np.float32); sh[:n] = w / w.sum()      # sizes a user gets by default
    cands = g.candidates(z["cond"][i].astype(np.float32), nt, z["entr"][i], z["edge"][i], None, sh, True, seed=i, k=a.k)
    gtb = torch.from_numpy(z["boxes"][i])[None].to(dev); gi = torch.from_numpy(z["inst"][i])[None].long().to(dev)
    sig = group_sig(torch.from_numpy(z["sig"][i])[None].to(dev), torch.from_numpy(sh)[None].to(dev), torch.ones(1, dtype=torch.bool, device=dev))
    for c in cands:
        pb = torch.zeros(1, M, 4, device=dev); pb[0] = torch.from_numpy(c["boxes"]).to(dev)
        gt = cv2.resize(relabel_inst(gi, align_perm(pb, gtb, sig))[0].cpu().numpy().astype(np.uint8), (256, 256), interpolation=cv2.INTER_NEAREST)
        ious = [((c["inst"] == k) & (gt == k)).sum() / max(1, ((c["inst"] == k) | (gt == k)).sum()) for k in range(1, n + 1)]
        X.append([c["feats"][f] for f in FEATS]); Y.append(float(np.mean(ious))); P.append(i)
X, Y, P = np.array(X), np.array(Y), np.array(P)


def demean(A):
    A = A.copy()
    for p in np.unique(P): A[P == p] -= A[P == p].mean(0)
    return A


def fit(mask):
    Xc, Yc = demean(X)[mask], demean(Y[:, None])[mask, 0]; sd = Xc.std(0) + 1e-6
    ws = np.linalg.solve((Xc / sd).T @ (Xc / sd) + a.ridge * np.eye(len(FEATS)), (Xc / sd).T @ Yc)
    return ws / sd


def pick(w, mask_plans):
    r = {"random": [], "hand score": [], "learned score": [], "oracle": []}; hw = np.array([HAND_W[f] for f in FEATS])
    for p in mask_plans:
        m = P == p
        r["random"].append(Y[m].mean()); r["hand score"].append(Y[m][np.argmax(X[m] @ hw)])
        r["learned score"].append(Y[m][np.argmax(X[m] @ w)]); r["oracle"].append(Y[m].max())
    return {k: round(float(np.mean(v)), 4) for k, v in r.items()}


plans = np.unique(P); cut = int(0.7 * len(plans)); tr_p, te_p = plans[:cut], plans[cut:]
w_tr = fit(np.isin(P, tr_p)); held = pick(w_tr, te_p) if len(te_p) else {}
w_all = fit(np.ones(len(P), bool))
out = {"plans": int(len(plans)), "k": a.k, "held_out_room_iou_by_selection_rule": held, "weights": dict(zip(FEATS, map(float, w_all)))}
(ROOT / "logs").mkdir(exist_ok=True); (ROOT / "logs" / "calibration.json").write_text(json.dumps(out, indent=1))
sp = {"w": out["weights"]}; (g_dir := __import__("pathlib").Path(a.ref_ckpt).parent).mkdir(parents=True, exist_ok=True); (g_dir / "score_weights.json").write_text(json.dumps(sp, indent=1))
print(json.dumps(out, indent=1)); print("saved", g_dir / "score_weights.json")
