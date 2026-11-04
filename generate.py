"""Step 3: generate floor plans.  Input JSON = boundary + rooms + connectivity graph (see samples/).
  python generate.py --samples-dir samples            python generate.py --input samples/rect_2bhk.json --n-samples 3
  python generate.py --test-index 5                   (a real held-out plan; add --gt-boxes to test only RefineNet)"""
import argparse, json, numpy as np, torch, torch.nn.functional as F, cv2
from pathlib import Path
from shapely.geometry import Polygon, Point
from shapely.ops import nearest_points
from shapely import affinity
from common import ROOT, S, OUT, M, T, ROOM_TYPES, LO, K, fill_geom, EDGE_ADJ, EDGE_DOOR, DEFAULT_SIZE
from models import BoxNet, RefineNet, refine_input, dihedral
from layout import build_layout, export, layout_features, FEATS, HAND_W


ALIAS = {"living": 0, "living room": 0, "hall": 0, "lounge": 0, "bedroom": 1, "bed": 1, "master": 1, "bathroom": 2, "bath": 2,
         "toilet": 2, "kitchen": 3, "balcony": 4, "storage": 5, "store": 5, "stair": 6, "stairs": 6}


def default_connections(ids, types):
    """Typical Indian-apartment graph: living is the hub; bathrooms attach to bedrooms; extra balconies attach to bedrooms."""
    idx = lambda t: [i for i, x in enumerate(types) if x == t]
    living, beds, baths, kit, balc = idx(0), idx(1), idx(2), idx(3), idx(4)
    hub = living[0] if living else (beds[0] if beds else 0); E = []
    for i in range(len(ids)):
        if i != hub and types[i] in (1, 3, 5, 6) and not (types[i] == 1 and not living and i == hub): E.append((hub, i))
    for k, b in enumerate(baths): E.append((beds[k] if k < len(beds) else hub, b))
    for k, b in enumerate(balc): E.append((hub if k == 0 else (beds[min(k - 1, len(beds) - 1)] if beds else hub), b))
    return E


def parse_spec(cfg):
    r = cfg.get("rooms"); ids, types = [], []
    sizes = []
    def add(rid, t, area=None):
        ids.append(rid); types.append(ALIAS[t.lower()]); sizes.append(area)
    if isinstance(r, dict):
        for t, cnt in r.items():
            for i in range(int(cnt)): add(t if cnt == 1 else f"{t}_{i+1}", t)
    elif isinstance(r, list):
        for it in r:
            if isinstance(it, str): add(it, it.rstrip("0123456789_ ") or it)
            else: add(it["id"], it["type"], it.get("area"))
    else:   # legacy counts
        add("living", "living")
        for key, t in (("bedrooms", "bedroom"), ("bathrooms", "bathroom"), ("kitchens", "kitchen"), ("balconies", "balcony")):
            c = int(cfg.get(key, 1 if key != "balconies" else 0))
            for i in range(c): add(t if c == 1 else f"{t}_{i+1}", t)
    n = len(ids)
    if not 2 <= n <= M: raise ValueError(f"need 2..{M} rooms, got {n}")
    pos = {rid: i for i, rid in enumerate(ids)}
    def ref(x):
        if x in pos: return pos[x]
        m = [i for i, rid in enumerate(ids) if rid.startswith(str(x))]
        if len(m) >= 1: return m[0]
        raise ValueError(f"unknown room '{x}' in connections")
    E = np.zeros((M, M), np.uint8); conns = cfg.get("connections")
    pairs = [(ref(c[0]), ref(c[1]), (c[2] if len(c) > 2 else "door")) for c in conns] if conns else [(a, b, "door") for a, b in default_connections(ids, types)]
    for a, b, kind in pairs:
        if a != b: E[a, b] = E[b, a] = EDGE_DOOR if kind == "door" else EDGE_ADJ
    ent = ref(cfg["entrance"]) if cfg.get("entrance") else next((i for i, t in enumerate(types) if t == 0), 0)
    w = np.array([float(sz) if sz is not None else DEFAULT_SIZE[t] for sz, t in zip(sizes, types)], np.float32)
    share = np.zeros(M, np.float32); share[:n] = w / w.sum()                 # relative room sizes (user "area" = relative weight)
    return ids, types, E, ent, share, bool(cfg.get("use_area", True))


def build_input(cfg):
    """Boundary polygon -> (cond[2,S,S], front-door point in OUT px). Scaled/centred exactly like ResPlan (long side = 118 px)."""
    poly0 = Polygon(cfg["boundary"]); minx, miny, maxx, maxy = poly0.bounds
    s = 118.0 / K / max(maxx - minx, maxy - miny)
    def tf(g):
        g = affinity.translate(g, -(minx + maxx) / 2, -(miny + maxy) / 2); g = affinity.scale(g, s, s, origin=(0, 0))
        return affinity.translate(g, 64 / K + LO, 64 / K + LO)
    poly = tf(poly0); door = np.zeros((S, S), np.uint8); fpt = None
    if cfg.get("front_door") is not None:
        c = nearest_points(poly.exterior, tf(Point(*cfg["front_door"])))[0]; h = 2.5 / K
        door = fill_geom(Polygon([(c.x - h, c.y - h), (c.x + h, c.y - h), (c.x + h, c.y + h), (c.x - h, c.y + h)]))
        fpt = ((c.x - LO) * K * OUT / S, (c.y - LO) * K * OUT / S)
    return np.stack([fill_geom(poly), door]).astype(np.float32), fpt


def undo_dihedral(x, k, flip):
    """Inverse of models.dihedral for tensors (B,C,H,W) (dihedral = flip, then rotate k times)."""
    x = torch.rot90(x, -(k % 4), (2, 3))
    return x.flip(3) if flip else x


TTA = [(0, False), (0, True), (2, False), (2, True)]


class Gen:
    def __init__(a, box_ckpt, ref_ckpt, weights=None):
        a.dev = "cuda" if torch.cuda.is_available() else "cpu"
        bk = torch.load(box_ckpt, map_location=a.dev); rk = torch.load(ref_ckpt, map_location=a.dev)
        a.box = BoxNet(bk.get("d", 192), bk.get("heads", 6), bk.get("layers", 4)).to(a.dev); a.box.load_state_dict(bk["model"]); a.box.eval()
        a.ref = RefineNet(rk.get("base", 32)).to(a.dev); a.ref.load_state_dict(rk["model"]); a.ref.eval()
        wp = Path(weights) if weights else Path(ref_ckpt).parent / "score_weights.json"      # learned by calibrate_score.py
        a.w = np.array([HAND_W[f] for f in FEATS]); a.b = 0.0; a.learned = False
        if wp.exists():
            d = json.loads(wp.read_text()); a.w = np.array([d["w"][f] for f in FEATS]); a.learned = True

    def score(a, feats): return float(sum(a.w[i] * feats[f] for i, f in enumerate(FEATS)))

    @torch.no_grad()
    def refine(a, c, boxes, nt, tta=True):
        """RefineNet logits; test-time augmentation averages 4 flipped/rotated passes (boxes transformed with the image)."""
        outs = []
        for k, fl in (TTA if tta else TTA[:1]):
            ct, bt, _ = dihedral(c, boxes, None, k, fl)
            outs.append(undo_dihedral(a.ref(refine_input(ct, bt, nt)).float(), k, fl))
        return torch.stack(outs).mean(0)

    @torch.no_grad()
    def candidates(a, cond, ntype, entr, edge, front_pt, share=None, known=False, seed=0, k=8, gt_boxes=None, tta=True):
        """k candidate layouts (different noise z) -> list of dicts with label/inst/stats/features/score/boxes."""
        torch.manual_seed(seed); k = 1 if gt_boxes is not None else k
        t = lambda x, dt: torch.as_tensor(x, dtype=dt, device=a.dev)[None].expand(k, *np.shape(x)).contiguous()
        c, nt, en, ed = t(cond, torch.float32), t(ntype, torch.long), t(entr, torch.float32), t(edge, torch.long)
        n = int((nt[0] >= 0).sum())
        if gt_boxes is not None: boxes = t(gt_boxes, torch.float32)
        else:
            sh = t(share, torch.float32) if share is not None else None
            boxes = a.box(c, nt, en, ed, sh, torch.full((k,), bool(known and share is not None), device=a.dev))
        logits = a.refine(c, boxes, nt, tta); logits[:, n + 1:] = -1e4
        insts = F.interpolate(logits, size=OUT, mode="bilinear", align_corners=False).argmax(1).cpu().numpy().astype(np.uint8)
        inner = cv2.resize(cond[0].astype(np.uint8), (OUT, OUT), interpolation=cv2.INTER_NEAREST) > 0
        ent = int(np.argmax(entr)) if entr.sum() > 0 else None
        target = np.asarray(share[:n]) if (share is not None and known) else np.array([DEFAULT_SIZE[int(x)] for x in ntype[:n]])
        out = []; bx = boxes.cpu().numpy()
        for j in range(k):
            label, inst, st = build_layout(insts[j], list(ntype[:n]), edge[:n, :n], inner, front_pt, ent, boxes=bx[j, :n])
            f = layout_features(inst, bx[j, :n], list(ntype[:n]), target, st)
            out.append(dict(label=label, inst=inst, stats=st, feats=f, score=a.score(f), boxes=bx[j], raw=insts[j]))
        return out

    def __call__(a, cond, ntype, entr, edge, front_pt, share=None, known=False, seed=0, k=8, gt_boxes=None, tta=True):
        cs = a.candidates(cond, ntype, entr, edge, front_pt, share, known, seed, k, gt_boxes, tta); b = max(cs, key=lambda c: c["score"])
        b["stats"].update(candidates=len(cs), score=round(b["score"], 3), learned_score=a.learned)
        return b["label"], b["inst"], b["stats"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input"); ap.add_argument("--samples-dir"); ap.add_argument("--test-index", type=int)
    ap.add_argument("--box-ckpt", default=str(ROOT / "checkpoints" / "box_best.pt"))
    ap.add_argument("--ref-ckpt", default=str(ROOT / "checkpoints" / "refine_best.pt"))
    ap.add_argument("--k", type=int, default=8, help="candidates per layout (best one is kept)")
    ap.add_argument("--n-samples", type=int, default=1, help="different layouts for the same input"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gt-boxes", action="store_true"); ap.add_argument("--no-tta", action="store_true"); a = ap.parse_args()
    gen = Gen(a.box_ckpt, a.ref_ckpt); logf = open(ROOT / "logs" / "generate_log.jsonl", "a")

    def run(name, cond, ntype, entr, edge, fpt, names, gtb=None, share=None, known=False):
        for j in range(a.n_samples):
            label, inst, st = gen(cond, ntype, entr, edge, fpt, share, known, a.seed + j, a.k, gtb, not a.no_tta); nm = name if a.n_samples == 1 else f"{name}_s{j}"
            export(label, inst, names, nm, st, types=[t for t in ntype if t >= 0]); logf.write(json.dumps({"name": nm, **st}) + "\n")
            print(f"[{nm}] rooms {st['rooms_present']}/{st['rooms']}  doors {st['doors_placed']}/{st['doors_requested']} (+{st['doors_auto']} auto)  windows {st['windows']}  connected {st['all_connected']}  straight {st['straightness']}  best of {st['candidates']}  -> outputs/{nm}_3d.html")

    if a.test_index is not None:
        z = np.load(ROOT / "data" / "cache4_test.npz"); i = a.test_index; n = int((z["ntype"][i] >= 0).sum())
        fd = np.where(z["cond"][i][1] > 0); fpt = (fd[1].mean() * OUT / S, fd[0].mean() * OUT / S) if len(fd[0]) else None
        run(f"test_{i}", z["cond"][i].astype(np.float32), z["ntype"][i], z["entr"][i], z["edge"][i], fpt,
            [ROOM_TYPES[t] for t in z["ntype"][i][:n]], z["boxes"][i] if a.gt_boxes else None, z["share"][i], False)
    else:
        for f in (sorted((ROOT / a.samples_dir).glob("*.json")) if a.samples_dir else [ROOT / a.input]):
            cfg = json.loads(f.read_text()); ids, types, E, ent, share, use_area = parse_spec(cfg); cond, fpt = build_input(cfg)
            nt = np.full(M, -1, np.int64); nt[:len(ids)] = types; en = np.zeros(M, np.float32); en[ent] = 1
            run(cfg.get("name", f.stem), cond, nt, en, E, fpt, ids, None, share, use_area)