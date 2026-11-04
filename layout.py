"""Stage 3 (no learning): room-instance map + connectivity graph -> walls, doors, windows, front door -> 3D export.
Everything structural is DERIVED from the room layout, so interior walls always exist and every graph door is cut."""
import json
import numpy as np, cv2
from common import S, OUT, M, T, ROOM_TYPES, PALETTE, CLASSES, ROOT, EDGE_DOOR

WALL, DOOR, WINDOW, FRONT = 1, 9, 10, 11
WIN_TYPES = {0, 1, 2, 3}                      # living, bedroom, bathroom, kitchen get windows
K5 = np.ones((5, 5), np.uint8)
dil = lambda m, k=K5: cv2.dilate(m.astype(np.uint8), k) > 0


def fill_nearest(inst, region):
    todo = region & (inst == 0)
    if todo.any() and (inst > 0).any():
        _, lab = cv2.distanceTransformWithLabels((inst == 0).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
        lut = np.zeros(lab.max() + 1, inst.dtype); lut[lab[inst > 0]] = inst[inst > 0]
        inst[todo] = lut[lab[todo]]
    return inst


def clean_instances(inst, ntype, inner):
    """inst: (OUT,OUT) ids 0..n. Keep rooms inside the footprint (balconies may stick out), one blob per room, no holes."""
    n = len(ntype); allowed = dil(inner, np.ones((3, 3), np.uint8)); inst = inst.copy()
    for k in range(1, n + 1):
        if ntype[k - 1] != 4: inst[(inst == k) & ~allowed] = 0
    for k in range(1, n + 1):
        m = (inst == k).astype(np.uint8); c, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=4)
        if c > 2:
            big = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA])); inst[(m > 0) & (lab != big)] = 0
    return fill_nearest(inst, allowed)



def _straighten_dir(inst, ntype, band=10, min_len=14, max_cross=16):
    """Straighten every VERTICAL border between two regions: each border gets its own straight line (median position)."""
    H, W = inst.shape; n = len(ntype); k3 = np.ones((3, 3), np.uint8)
    ids = list(range(1, n + 1)); pairs = [(i, j) for i in ids for j in ids if i < j] + [(i, 0) for i in ids if ntype[i - 1] == 4]
    for i, j in pairs:
        mi, mj = inst == i, (inst == j) if j else (inst == 0)
        if not mi.any() or not mj.any(): continue
        bi = mi & (cv2.dilate(mj.astype(np.uint8), k3) > 0); bj = mj & (cv2.dilate(mi.astype(np.uint8), k3) > 0)
        border = bi | bj
        if border.sum() < min_len: continue
        c, lab, st, _ = cv2.connectedComponentsWithStats(border.astype(np.uint8), connectivity=8)
        for ci in range(1, c):
            x, y, w, h, area = st[ci]
            if h < min_len or w > max_cross or h < 1.5 * w: continue            # keep only (near-)vertical borders
            comp = lab == ci; xi = np.where(comp & bi)[1]; xj = np.where(comp & bj)[1]
            if len(xi) == 0 or len(xj) == 0: continue
            left, right = (i, j) if np.median(xi) < np.median(xj) else (j, i)
            split = 0.5 * (np.median(xi) + np.median(xj)) + 0.5
            x0, x1 = max(0, int(split - band)), min(W, int(split + band) + 1)
            seg = inst[y:y + h, x0:x1]; cols = np.arange(x0, x1)[None, :] + 0.0 + np.zeros((h, 1))
            take = (seg == i) | (seg == j)
            seg[take & (cols < split)] = left; seg[take & (cols >= split)] = right
    return inst


def straighten_instances(inst, ntype):
    """Local wall straightening in both directions (rooms keep their own offsets, T-junctions stay correct)."""
    out = inst.copy()
    for _ in range(2):                                    # two passes remove the leftover jogs
        out = _straighten_dir(out, ntype)
        out = np.ascontiguousarray(_straighten_dir(np.ascontiguousarray(out.T), ntype).T)
    return out



# ----------------------------- Manhattan vectorisation -----------------------------
def straightness(inst, min_run=8):
    """Share of interior room borders that lie on straight axis-aligned runs >= min_run px (real ResPlan: ~0.98)."""
    tot = long_ = 0
    for a in (inst, inst.T):
        e = (a[:, :-1] != a[:, 1:]) & (a[:, :-1] > 0) & (a[:, 1:] > 0)
        for x in range(e.shape[1]):
            if not e[:, x].any(): continue
            d = np.diff(np.concatenate([[0], e[:, x].astype(np.int8), [0]])); L = np.where(d == -1)[0] - np.where(d == 1)[0]
            tot += L.sum(); long_ += L[L >= min_run].sum()
    return float(long_ / max(1, tot))


def manhattan_polygon(mask, eps=2.0, min_seg=6):
    """Binary mask -> strictly orthogonal polygon as a cyclic list of segments [orient 'H'|'V', coord, length].
    Douglas-Peucker (eps) -> classify every edge as H/V by its dominant axis -> merge same-orientation neighbours (length-weighted
    coordinate) -> drop segments shorter than min_seg (neighbours merge) -> H and V strictly alternate => only 90/270 deg corners."""
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs: return None
    ap = cv2.approxPolyDP(max(cs, key=cv2.contourArea), eps, True)[:, 0, :].astype(np.float64)
    if len(ap) < 4: return None
    segs = []
    for i in range(len(ap)):
        p, q = ap[i], ap[(i + 1) % len(ap)]; dx, dy = q[0] - p[0], q[1] - p[1]; L = abs(dx) + abs(dy)
        if L == 0: continue
        segs.append(["H", (p[1] + q[1]) / 2, abs(dx)] if abs(dx) >= abs(dy) else ["V", (p[0] + q[0]) / 2, abs(dy)])
    def merge(sg):                                             # merge cyclic runs of equal orientation
        out = []
        for o, c, L in sg:
            if out and out[-1][0] == o: out[-1][1] = (out[-1][1] * out[-1][2] + c * L) / max(out[-1][2] + L, 1e-9); out[-1][2] += L
            else: out.append([o, c, L])
        while len(out) > 1 and out[0][0] == out[-1][0]:
            a, b = out.pop(), out[0]; b[1] = (b[1] * b[2] + a[1] * a[2]) / max(a[2] + b[2], 1e-9); b[2] += a[2]
        return out
    segs = merge(segs)
    while len(segs) > 4:
        k = min(range(len(segs)), key=lambda i: segs[i][2])
        if segs[k][2] >= min_seg: break
        segs.pop(k); segs = segs[k:] + segs[:k]; segs = merge(segs)         # neighbours of a removed segment share an orientation
    return segs if len(segs) >= 4 and len(segs) % 2 == 0 else None


def _cluster_1d(vals, tol):
    vals = sorted(set(round(v, 3) for v in vals)); mp = {}; i = 0
    while i < len(vals):
        j = i
        while j + 1 < len(vals) and vals[j + 1] - vals[i] <= 2 * tol: j += 1
        m = float(np.mean(vals[i:j + 1]))
        for v in vals[i:j + 1]: mp[v] = m
        i = j + 1
    return mp


def manhattanize_instances(inst, ntype, tol=0.5, eps=1.5, min_seg=4):
    """Per-room orthogonal polygons; H/V line coordinates of ALL rooms are clustered (tol px) so neighbouring rooms share
    exactly the same wall line -> no gaps/overlaps, no stair-steps. Rooms whose polygon fails keep their original pixels."""
    n = len(ntype); segs = {}; ys, xs = [], []
    for k in range(1, n + 1):
        m = (inst == k).astype(np.uint8)
        if m.sum() < 30: continue
        c, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=4)
        if c > 2: m = (lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
        sg = manhattan_polygon(m, eps, min_seg)
        if sg is None: continue
        segs[k] = sg; ys += [c for o, c, _ in sg if o == "H"]; xs += [c for o, c, _ in sg if o == "V"]
    my, mx = _cluster_1d(ys, tol), _cluster_1d(xs, tol)
    out = np.zeros_like(inst); area = {k: int((inst == k).sum()) for k in segs}
    for k in sorted(segs, key=lambda k: -area[k]):                      # big rooms first, small rooms on top
        sg = [[o, (my if o == "H" else mx)[round(c, 3)], L] for o, c, L in segs[k]]; pts = []
        for i in range(len(sg)):
            (o1, c1, _), (o2, c2, _) = sg[i], sg[(i + 1) % len(sg)]
            pts.append((c2, c1) if o1 == "H" else (c1, c2))              # (x, y) corner between segment i and i+1
        cv2.fillPoly(out, [np.round(np.array(pts)).astype(np.int32).reshape(-1, 1, 2)], k)
    for k in range(1, n + 1):
        if k not in segs: out[(inst == k) & (out == 0)] = k              # fallback: keep original pixels
    return out


def restore_missing(inst, boxes, ntype, inner, shrink=0.7):
    """A room the network dropped (tiny storage/bathroom) is re-inserted from its predicted box (shrunk) so it always exists."""
    inst = inst.copy(); H = inst.shape[0]; allowed = dil(inner, np.ones((3, 3), np.uint8))
    for k in range(1, len(ntype) + 1):
        if (inst == k).sum() >= 20: continue
        x0, y0, x1, y1 = boxes[k - 1] * H; cx, cy, w, h = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) * shrink, (y1 - y0) * shrink
        reg = np.zeros((H, H), bool); reg[max(0, int(cy - h / 2)):int(cy + h / 2) + 1, max(0, int(cx - w / 2)):int(cx + w / 2) + 1] = True
        if ntype[k - 1] != 4: reg &= allowed
        if reg.sum() >= 20: inst[reg] = k
    return inst


def wall_mask(inst, ntype, thick=3):
    H, W = inst.shape; isb = np.zeros(M + 2, bool)
    for k, t in enumerate(ntype): isb[k + 1] = (t == 4)          # balcony | outside gets no wall (open)
    b = np.zeros((H, W), bool)
    for dy, dx in ((0, 1), (1, 0)):
        a, c = inst[:H - dy, :W - dx], inst[dy:, dx:]
        d = (a != c) & ~((isb[a] & (c == 0)) | (isb[c] & (a == 0)))
        b[:H - dy, :W - dx] |= d; b[dy:, dx:] |= d
    return cv2.dilate(b.astype(np.uint8), np.ones((thick, thick), np.uint8)) > 0


def _cut(comp, half, pos=None):
    """Pixels of `comp` within +-half of a position along the component's long axis (default: its middle)."""
    ys, xs = np.where(comp); w = xs.max() - xs.min() + 1; h = ys.max() - ys.min() + 1
    Y, X = np.mgrid[:comp.shape[0], :comp.shape[1]]
    if w >= h:
        c = (xs.min() + xs.max()) // 2 if pos is None else pos; return comp & (np.abs(X - c) <= half), max(w, h)
    c = (ys.min() + ys.max()) // 2 if pos is None else pos; return comp & (np.abs(Y - c) <= half), max(w, h)


def _largest(mask):
    c, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return None if c < 2 else (lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA])))


def _door_between(walls, rooms, i, j, door):
    comp = _largest(walls & rooms[i] & rooms[j] & ~door)
    if comp is None: return None, 0
    _, L = _cut(comp, 8)
    if L < 10: return None, L
    sel, _ = _cut(comp, max(3, min(8, (L - 4) // 2)))
    return sel, L


def carve_doors(walls, inst, edges, n):
    """One opening for every door-edge of the graph (if the two rooms really share a wall)."""
    door = np.zeros_like(walls); req = 0; placed = []; rooms = {k: dil(inst == k + 1) for k in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            if edges[i][j] != EDGE_DOOR: continue
            req += 1; sel, _ = _door_between(walls, rooms, i, j, door)
            if sel is not None: door |= sel; placed.append((i, j))
    return door, req, placed, rooms


PREF = {(0, 1): 3, (0, 2): 2, (0, 3): 3, (0, 4): 3, (0, 5): 2, (0, 6): 2, (0, 0): 1, (1, 2): 3, (1, 4): 2.5, (1, 1): 0.6, (1, 3): 0.5,
        (1, 5): 0.8, (1, 6): 0.8, (2, 3): 0.15, (2, 2): 0.3, (2, 4): 0.1, (2, 5): 0.3, (2, 6): 0.2, (3, 4): 0.6, (3, 5): 1.5,
        (3, 6): 0.3, (4, 4): 0.3, (4, 5): 0.2, (4, 6): 0.2, (5, 6): 0.3, (5, 5): 0.3, (6, 6): 0.2}


def ensure_connected(walls, door, rooms, ntype, placed):
    """Every room must be reachable: if requested doors could not be cut (rooms not touching), add the most sensible door
    between touching rooms (bathroom->bedroom/living, kitchen->living, ...) until all rooms form ONE connected group."""
    n = len(ntype); par = list(range(n))
    def find(x):
        while par[x] != x: par[x] = par[par[x]]; x = par[x]
        return x
    for i, j in placed: par[find(i)] = find(j)
    cands = []
    for i in range(n):
        for j in range(i + 1, n):
            comp = _largest(walls & rooms[i] & rooms[j])
            if comp is None: continue
            _, L = _cut(comp, 8)
            if L >= 10: cands.append((PREF.get(tuple(sorted((int(ntype[i]), int(ntype[j])))), 0.4) * min(L, 40), i, j))
    added = 0
    for sc, i, j in sorted(cands, reverse=True):
        if find(i) == find(j): continue
        sel, _ = _door_between(walls, rooms, i, j, door)
        if sel is not None: door |= sel; par[find(i)] = find(j); added += 1
    return added, len({find(i) for i in range(n)}) == 1


def carve_front(walls, inst, door, pt, entrance):
    ext = walls & dil(inst == 0) & ~door
    if not ext.any(): return np.zeros_like(walls)
    ys, xs = np.where(ext)
    if pt is None and entrance is not None and entrance >= 0:
        e = ext & dil(inst == entrance + 1)
        if e.any(): ys, xs = np.where(e); pt = (xs.mean(), ys.mean())
    if pt is None: pt = (xs.mean(), ys.mean())
    i = int(np.argmin((xs - pt[0]) ** 2 + (ys - pt[1]) ** 2)); cx, cy = xs[i], ys[i]
    Y, X = np.mgrid[:inst.shape[0], :inst.shape[1]]
    return ext & (np.abs(X - cx) <= 9) & (np.abs(Y - cy) <= 9)


def add_windows(walls, inst, ntype):
    out = dil(inst == 0); win = np.zeros_like(walls); count = 0
    for k, t in enumerate(ntype):
        if t not in WIN_TYPES: continue
        ext = walls & dil(inst == k + 1) & out & ~win
        c, lab, st, _ = cv2.connectedComponentsWithStats(ext.astype(np.uint8), connectivity=8)
        for ci in np.argsort(-st[1:, cv2.CC_STAT_AREA])[:2] + 1:
            comp = lab == ci; ys, xs = np.where(comp); L = max(xs.max() - xs.min(), ys.max() - ys.min()) + 1
            if L < 18: continue
            ww = int(min(0.5 * L, 30) if t != 2 else min(0.4 * L, 14))
            sel, _ = _cut(comp, ww // 2)
            if sel.any(): win |= sel; count += 1
    return win, count


def build_layout(inst, ntype, edges, inner, front_pt=None, entrance=None, snap=True, boxes=None):
    """inst (OUT,OUT) uint8 ids; ntype list[n]; edges (n,n); inner (OUT,OUT) bool; front_pt (x,y) in OUT px; boxes (n,4) predicted
    boxes in [0,1] (used to restore dropped rooms).  -> label (OUT,OUT) uint8 [CLASSES ids], inst (cleaned), stats"""
    n = len(ntype); inst = clean_instances(inst.astype(np.uint8), ntype, inner); snapped = "none"
    if boxes is not None: inst = restore_missing(inst, boxes, ntype, inner)
    if snap:                                           # Manhattan polygons first, local border straightening as fallback
        for name, fn in (("manhattan", manhattanize_instances), ("border", straighten_instances)):
            cand = clean_instances(fn(inst, ntype), ntype, inner)
            if len(set(np.unique(cand)) - {0}) == n: inst, snapped = cand, name; break
    walls = wall_mask(inst, ntype)
    door, req, placed, rooms = carve_doors(walls, inst, edges, n)
    auto, connected = ensure_connected(walls, door, rooms, ntype, placed)
    front = carve_front(walls, inst, door, front_pt, entrance)
    win, nwin = add_windows(walls & ~door & ~front, inst, ntype)
    cls = np.array([0] + [int(t) + 2 for t in ntype] + [0] * (M + 1 - n), np.uint8)
    label = cls[inst]
    label[walls] = WALL; label[door] = DOOR; label[win & walls] = WINDOW; label[front] = FRONT
    fill = []
    for k in range(1, n + 1):
        ys, xs = np.where(inst == k)
        if len(xs): fill.append(len(xs) / ((xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1)))
    return label, inst, dict(rooms=n, rooms_present=len(set(np.unique(inst)) - {0}), doors_requested=req, doors_placed=len(placed),
                             doors_auto=auto, all_connected=bool(connected), windows=nwin, front_door=bool(front.any()),
                             rectangularity=round(float(np.mean(fill)) if fill else 0.0, 3), straightness=round(straightness(inst), 3), snapped=snapped)


FEATS = ["door_ratio", "present", "connected", "straight", "box_consistency", "box_overlap", "aspect_bad", "size_err", "auto_doors", "rectangularity"]
HAND_W = dict(door_ratio=4.0, present=3.0, connected=1.0, straight=2.0, box_consistency=1.5, box_overlap=-2.0, aspect_bad=-1.5,
              size_err=-1.5, auto_doors=-0.5, rectangularity=1.0)


def layout_features(inst, boxes, ntype, target, st):
    """Quality features of one candidate (all roughly in [0,1]).  boxes (n,4) predicted boxes, target (n,) wanted relative sizes."""
    n = len(ntype); H = inst.shape[0]; occ = np.zeros((64, 64), np.float32); cons = []; asp = []; area = []
    for k in range(1, n + 1):
        x0, y0, x1, y1 = boxes[k - 1]; r = np.zeros((H, H), bool)
        r[int(y0 * H):int(y1 * H) + 1, int(x0 * H):int(x1 * H) + 1] = True
        m = inst == k; area.append(m.sum()); cons.append((m & r).sum() / max(1, (m | r).sum()))
        occ[int(y0 * 64):int(y1 * 64) + 1, int(x0 * 64):int(x1 * 64) + 1] += 1
        ys, xs = np.where(m)
        if len(xs) and ntype[k - 1] in (0, 1, 3): asp.append(float(max(np.ptp(xs) + 1, np.ptp(ys) + 1) / max(1, min(np.ptp(xs) + 1, np.ptp(ys) + 1))) > 4.0)
    area = np.maximum(np.array(area, np.float64), 1.0); got = area / area.sum(); want = np.asarray(target, np.float64) / np.sum(target)
    return dict(door_ratio=st["doors_placed"] / st["doors_requested"] if st["doors_requested"] else 1.0, present=st["rooms_present"] / st["rooms"],
                connected=float(st["all_connected"]), straight=st["straightness"], box_consistency=float(np.mean(cons)),
                box_overlap=float(np.maximum(occ - 1.5, 0).mean()), aspect_bad=float(np.mean(asp)) if asp else 0.0,
                size_err=float(np.clip(np.abs(np.log(got / want)).mean(), 0, 2) / 2), auto_doors=st["doors_auto"] / n, rectangularity=st["rectangularity"])


def rects(mask):
    """Greedy merge of a boolean mask into rectangles [x, y, w, h]."""
    active, out = {}, []; H, W = mask.shape
    for y in range(H + 1):
        runs = set()
        if y < H:
            row, x = mask[y], 0
            while x < W:
                if row[x]:
                    x0 = x
                    while x < W and row[x]: x += 1
                    runs.add((x0, x))
                else: x += 1
        for key in list(active):
            if key not in runs: y0 = active.pop(key); out.append([key[0], y0, key[1] - key[0], y - y0])
        for key in runs: active.setdefault(key, y)
    return out


def room_labels(inst, names, types=None):
    """One label per room, placed at its most interior point (works for L-shaped rooms)."""
    out = []
    for k, nm in enumerate(names):
        m = (inst == k + 1).astype(np.uint8)
        if not m.any(): continue
        dt = cv2.distanceTransform(np.pad(m, 1), cv2.DIST_L2, 3)[1:-1, 1:-1]; y, x = np.unravel_index(int(dt.argmax()), dt.shape)
        out.append({"name": nm, "type": ROOM_TYPES[int(types[k])] if types is not None else "", "x": int(x), "y": int(y)})
    return out


def export(label, inst, names, name, stats=None, out_dir=None, types=None):
    out_dir = out_dir or ROOT / "outputs"; out_dir.mkdir(exist_ok=True); N = label.shape[0]
    plan = {"name": name, "size": N, "classes": CLASSES, "palette": PALETTE.tolist(), "grid": label.tolist(),
            "labels": room_labels(inst, names, types), "walls": rects(label == WALL), "windows": rects(label == WINDOW), "doors": rects((label == DOOR) | (label == FRONT)),
            "stats": stats or {}}
    (out_dir / f"{name}.json").write_text(json.dumps(plan))
    (out_dir / f"{name}_3d.html").write_text((ROOT / "viewer_template.html").read_text(encoding="utf-8")
                                             .replace("__PLAN_JSON__", json.dumps(plan)), encoding="utf-8")
    sc = 3; img = cv2.resize(PALETTE[label], None, fx=sc, fy=sc, interpolation=cv2.INTER_NEAREST)
    for k, nm in enumerate(names):
        ys, xs = np.where(inst == k + 1)
        if len(xs): cv2.putText(img, nm[:10], (int(xs.mean() * sc) - 5 * min(len(nm), 10), int(ys.mean() * sc)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_dir / f"{name}.png"), img[..., ::-1])
    return out_dir
