"""PCFPG evaluation metrics (see EVALUATION_PLAN.md). Instance maps: (H,W) uint8, ids 1..n (0 = outside). Pure numpy/scipy/cv2."""
import numpy as np, cv2
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr, wilcoxon
from layout import DOOR


def iou_matrix(a, b, n):
    A = [a == i + 1 for i in range(n)]; B = [b == j + 1 for j in range(n)]; I = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            u = (A[i] | B[j]).sum(); I[i, j] = (A[i] & B[j]).sum() / u if u else 0.0
    return I


def matched_iou(a, b, n, sig=None):
    """Mean per-room IoU. Rooms with the same Weisfeiler-Lehman signature (indistinguishable to the model) may be swapped: Hungarian inside each group."""
    I = iou_matrix(a, b, n); perm = np.arange(n)
    if sig is not None:
        sg = np.asarray(sig)[:n]
        for v in np.unique(sg):
            idx = np.where(sg == v)[0]
            if len(idx) > 1: r, c = linear_sum_assignment(-I[np.ix_(idx, idx)]); perm[idx[r]] = idx[c]
    return float(np.mean(I[np.arange(n), perm]))


def border_len(inst, n):
    k = np.ones((3, 3), np.uint8); D = [cv2.dilate((inst == i + 1).astype(np.uint8), k) > 0 for i in range(n)]; B = np.zeros((n, n), int)
    for i in range(n):
        for j in range(n):
            if i != j: B[i, j] = int((D[i] & (inst == j + 1)).sum())
    return B


def adjacency(inst, edge, n, tau=4):
    """Requested links (edge>0) that physically touch (shared border >= tau px) and realised touching pairs."""
    B = border_len(inst, n); real = np.maximum(B, B.T) >= tau; req = np.asarray(edge)[:n, :n] > 0; iu = np.triu_indices(n, 1); r, p = req[iu], real[iu]
    return dict(req=int(r.sum()), sat=int((r & p).sum()), real=int(p.sum()))


def realized_doors(label, inst, n, r=3):
    nc, cc = cv2.connectedComponents((label == DOOR).astype(np.uint8)); pairs = set()
    for c in range(1, nc):
        ys, xs = np.where(cc == c); win = inst[max(ys.min() - r, 0):ys.max() + r + 1, max(xs.min() - r, 0):xs.max() + r + 1]
        ids, cnt = np.unique(win, return_counts=True); top = [int(i) for i in ids[np.argsort(-cnt)] if 0 < i <= n][:2]
        if len(top) == 2: pairs.add((min(top) - 1, max(top) - 1))
    return pairs


def door_graph(label, inst, edge, n):
    """(Jaccard distance between requested and realised door edges, recall of requested door edges). Node identity is known, so GED reduces to this."""
    e = np.asarray(edge)[:n, :n]; req = {(i, j) for i in range(n) for j in range(i + 1, n) if e[i, j] == 2}; real = realized_doors(label, inst, n); u = req | real
    return (1 - len(req & real) / len(u) if u else 0.0), (len(req & real) / len(req) if req else np.nan)


def size_error(inst, want, n):
    """MAE of area fractions, mean relative error, Spearman rho between realised and requested room sizes."""
    a = np.array([(inst == i + 1).sum() for i in range(n)], float); got = a / max(a.sum(), 1); w = np.asarray(want[:n], float); w = w / w.sum()
    return float(np.abs(got - w).mean()), float(np.mean(np.abs(got - w) / np.maximum(w, 1e-6))), (float(spearmanr(got, w)[0]) if n > 2 else np.nan)


def validity(inst, raw, st, n, min_px=100):
    """valid_post: all rooms present, each one connected blob, all rooms reachable (after repairs).
       valid_pre: networks+geometry only: raw rooms present, every requested door placed, no auto-added doors."""
    pres = lambda m: all((m == i + 1).sum() >= min_px for i in range(n))
    single = all(cv2.connectedComponents((inst == i + 1).astype(np.uint8))[0] == 2 for i in range(n)); raw_ok = pres(raw); req_ok = st["doors_placed"] >= st["doors_requested"]
    return dict(valid_post=float(pres(inst) and single and bool(st["all_connected"])), valid_pre=float(raw_ok and req_ok and st["doors_auto"] == 0), raw_rooms_present=float(raw_ok), repair_free=float(st["doors_auto"] == 0 and raw_ok))


def boundary(inst, inner):
    occ = inst > 0; s = max(1, inner.sum()); return dict(outside=float((occ & ~inner).sum() / s), coverage=float((occ & inner).sum() / s))


def diversity(insts, n, sig):
    """Mean pairwise (1 - matched room IoU) between the K candidates of one input (0 = identical layouts)."""
    if len(insts) < 2: return np.nan
    d = [i[::2, ::2] for i in insts]; return float(np.mean([1 - matched_iou(d[a], d[b], n, sig) for a in range(len(d)) for b in range(a + 1, len(d))]))


class Novelty:
    """Nearest-neighbour type-map IoU against training plans with the same room-type multiset (64x64). Boundary differences lower it, so read as an upper-bound check for copying."""
    def __init__(self, train_npz):
        z = np.load(train_npz); nt = z["ntype"]; N = len(nt); L = np.full((N, 17), 255, np.uint8); L[:, 1:] = np.where(nt >= 0, nt, 255)
        self.tm = np.take_along_axis(L, z["inst"][:, ::2, ::2].reshape(N, -1).astype(int), 1).reshape(N, 64, 64); self.key = {}
        for i, r in enumerate(nt): self.key.setdefault(tuple(sorted(int(x) for x in r[r >= 0])), []).append(i)
    def query(self, inst, ntype, n):
        idx = self.key.get(tuple(sorted(int(x) for x in ntype[:n]))); 
        if not idx: return np.nan
        L = np.full(17, 255, np.uint8); L[1:n + 1] = ntype[:n]; q = L[inst[::4, ::4].astype(int)]; T = self.tm[idx]; s = 0
        ts = sorted(set(int(x) for x in ntype[:n]))
        for t in ts: A = q == t; B = T == t; s = s + (A & B).sum((1, 2)) / np.maximum((A | B).sum((1, 2)), 1)
        return float((s / len(ts)).max())


def boot_ci(x, rng, B=2000):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) == 0: return dict(mean=np.nan, sd=np.nan, ci95=[np.nan, np.nan], n=0)
    bs = [x[rng.randint(0, len(x), len(x))].mean() for _ in range(B)]
    return dict(mean=float(x.mean()), sd=float(x.std(ddof=1)) if len(x) > 1 else 0.0, ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], n=int(len(x)))


def paired(a, b, rng, B=2000):
    """Paired comparison a vs b (per-plan arrays): mean difference + bootstrap CI, Wilcoxon signed-rank p, Vargha-Delaney A12."""
    a, b = np.asarray(a, float), np.asarray(b, float); ok = ~(np.isnan(a) | np.isnan(b)); a, b = a[ok], b[ok]; d = a - b
    bs = [d[rng.randint(0, len(d), len(d))].mean() for _ in range(B)]
    try: p = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
    except ValueError: p = np.nan
    a12 = float((np.sum(a[:, None] > b[None, :]) + 0.5 * np.sum(a[:, None] == b[None, :])) / (len(a) * len(b)))
    return dict(diff=float(d.mean()), ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], p=p, a12=a12, n=int(len(d)))


def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); out = np.empty_like(ps); m = len(ps); run = 0.0
    for r, i in enumerate(o): run = max(run, (m - r) * ps[i]); out[i] = min(1.0, run)
    return out