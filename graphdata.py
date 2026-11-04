"""ResPlan vector plan -> room nodes + connectivity graph + dense instance map (all numpy)."""
import numpy as np, cv2
from common import S, T, M, ROOM_TYPES, parts, fill_geom, EDGE_ADJ, EDGE_DOOR


def get_rooms(plan, seed=0):
    """List of (type_idx, Polygon). Order is shuffled (seeded) so slot index leaks nothing about position."""
    rooms = []
    for t, key in enumerate(ROOM_TYPES):
        g = plan.get(key)
        if g is None and key == "balcony":
            g = plan.get("balacony")
        rooms += [(t, q) for q in parts(g) if q.area >= 4]
    np.random.RandomState(seed % (2 ** 31)).shuffle(rooms)
    return rooms


def build_graph(plan, rooms):
    """-> edge (n,n) uint8 {0 none, 1 adjacent, 2 door}, entrance index (-1 if unknown)."""
    n = len(rooms); polys = [r[1] for r in rooms]
    E = np.zeros((n, n), np.uint8)
    thr = max(float(plan.get("wall_depth", 4.5)) * 1.5, 5.0)     # rooms are separated by one wall thickness
    for i in range(n):
        for j in range(i + 1, n):
            if polys[i].distance(polys[j]) <= thr:
                E[i, j] = E[j, i] = EDGE_ADJ
    for d in parts(plan.get("door")):
        near = sorted((d.distance(p), k) for k, p in enumerate(polys))
        near = [k for dist, k in near if dist <= thr][:2]
        if len(near) == 2:
            E[near[0], near[1]] = E[near[1], near[0]] = EDGE_DOOR
    ent = -1
    fd = plan.get("front_door")
    if fd is not None and not fd.is_empty and n:
        ent = int(np.argmin([fd.distance(p) for p in polys]))
    return E, ent


def wl_signature(types, E, ent, rounds=3):
    """Weisfeiler-Lehman colours: nodes with equal colour are indistinguishable by the model (symmetric rooms)."""
    n = len(types)
    sig = [(int(types[i]), int(i == ent)) for i in range(n)]
    for _ in range(rounds):
        new = [(sig[i], tuple(sorted((int(E[i, j]), sig[j]) for j in range(n) if E[i, j]))) for i in range(n)]
        ids = {}
        sig = [ids.setdefault(s, len(ids)) for s in new]
    return np.array(sig, np.int64)


def instance_map(plan, rooms):
    """Dense (S,S) uint8 map: 0 outside, k+1 = room k. Wall gaps inside the footprint are filled with the nearest room."""
    inst = np.zeros((S, S), np.uint8)
    for k, (_, p) in enumerate(rooms):
        inst[fill_geom(p) > 0] = k + 1
    inner = fill_geom(plan["inner"]) > 0
    todo = inner & (inst == 0)
    if todo.any() and (inst > 0).any():
        _, lab = cv2.distanceTransformWithLabels((inst == 0).astype(np.uint8), cv2.DIST_L2, 5,
                                                 labelType=cv2.DIST_LABEL_PIXEL)
        lut = np.zeros(lab.max() + 1, np.uint8)
        lut[lab[inst > 0]] = inst[inst > 0]
        inst[todo] = lut[lab[todo]]
    return inst


def plan_to_sample(plan):
    """-> dict of arrays, or None if the plan is unusable."""
    rooms = get_rooms(plan, int(plan["id"]))
    n = len(rooms)
    if n < 2 or n > M:
        return None
    E, ent = build_graph(plan, rooms)
    types = np.array([r[0] for r in rooms], np.int64)
    inst = instance_map(plan, rooms)
    boxes = np.zeros((M, 4), np.float32)
    for k, (_, p) in enumerate(rooms):
        x0, y0, x1, y1 = p.bounds
        from common import LO, K
        boxes[k] = np.clip([(x0 - LO) * K, (y0 - LO) * K, (x1 - LO) * K, (y1 - LO) * K], 0, S) / S
    ntype = np.full(M, -1, np.int64); ntype[:n] = types
    edge = np.zeros((M, M), np.uint8); edge[:n, :n] = E
    entr = np.zeros(M, np.float32)
    if ent >= 0: entr[ent] = 1
    sig = np.full(M, -1, np.int64); sig[:n] = wl_signature(types, E, ent)
    cond = np.stack([fill_geom(plan["inner"]), fill_geom(plan.get("front_door"))]).astype(np.uint8)
    inner_area = max(plan["inner"].area, 1.0)
    areas = np.array([r[1].area for r in rooms], np.float32)
    share = np.zeros(M, np.float32); share[:n] = areas / areas.sum()          # relative room size (sums to 1)
    return dict(cond=cond, inst=inst, ntype=ntype, edge=edge, entr=entr, sig=sig, boxes=boxes, share=share,
                areas=areas / inner_area)
