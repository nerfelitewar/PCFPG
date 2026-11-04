"""Shared constants + rasterization (ResPlan vector plan -> 128x128 tensors)."""
from pathlib import Path
import numpy as np
import cv2
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ROOT.parent / "ResPlan"      # D:\MY CODING WORKSPACE\PCFPG_V2\ResPlan

S = 128                       # canvas size in pixels
LO, HI = -12.0, 268.0         # ResPlan coordinate range mapped onto the canvas
K = S / (HI - LO)             # pixels per ResPlan unit

CLASSES = ["outside", "wall", "living", "bedroom", "bathroom", "kitchen",
           "balcony", "storage", "stair", "door", "window", "front_door"]
N_CLS = len(CLASSES)
# (plan key, class id) in paint order (later paints on top)
PAINT = [("wall", 1), ("living", 2), ("bedroom", 3), ("bathroom", 4), ("kitchen", 5),
         ("balcony", 6), ("storage", 7), ("stair", 8), ("door", 9), ("window", 10),
         ("front_door", 11)]
COUNT_KEYS = ["bedroom", "bathroom", "kitchen", "balcony"]   # conditioning vector
N_COND = 2 + len(COUNT_KEYS)  # boundary + front door + 4 count planes

PALETTE = np.array([  # RGB
    (255, 255, 255), (60, 60, 60), (217, 217, 217), (102, 194, 165), (252, 141, 98),
    (141, 160, 203), (179, 179, 179), (255, 140, 105), (158, 154, 200),
    (231, 138, 195), (166, 216, 84), (166, 54, 3)], dtype=np.uint8)


def parts(g):
    """Yield Polygon parts of any shapely geometry (or None)."""
    if g is None or g.is_empty:
        return
    if isinstance(g, Polygon):
        yield g
    elif isinstance(g, (MultiPolygon, GeometryCollection)):
        for x in g.geoms:
            yield from parts(x)


def _px(coords):
    return np.round((np.asarray(coords, dtype=np.float64) - LO) * K).astype(np.int32).reshape(-1, 1, 2)


def fill_geom(geom, px_fn=_px, size=S):
    """Rasterize a shapely geometry (holes respected) to a uint8 0/1 mask."""
    m = np.zeros((size, size), np.uint8)
    for p in parts(geom):
        cv2.fillPoly(m, [px_fn(p.exterior.coords)], 1)
        for h in p.interiors:
            cv2.fillPoly(m, [px_fn(h.coords)], 0)
    return m


def plan_to_arrays(plan):
    """-> cond (2,S,S) uint8, counts (4,) float32, target (S,S) uint8"""
    target = np.zeros((S, S), np.uint8)
    for key, cid in PAINT:
        g = plan.get(key)
        if g is None and key == "balcony":
            g = plan.get("balacony")
        target[fill_geom(g) > 0] = cid
    cond = np.stack([fill_geom(plan["inner"]), fill_geom(plan.get("front_door"))]).astype(np.uint8)
    counts = np.array([min(len(list(parts(plan.get(k)))), 4) / 4.0 for k in COUNT_KEYS], np.float32)
    return cond, counts, target


def colorize(label, scale=4):
    img = PALETTE[label]
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    return img[..., ::-1]  # BGR for cv2.imwrite


# ======================= v3 (graph-conditioned, two-stage) =======================
ROOM_TYPES = ["living", "bedroom", "bathroom", "kitchen", "balcony", "storage", "stair"]
T = len(ROOM_TYPES)          # 7 room types; room type t has class id t+2 in CLASSES
M = 16                       # max rooms per plan (slots); covers 99.6% of ResPlan
OUT = 256                    # final layout resolution (walls/doors/windows are built here)
EDGE_NONE, EDGE_ADJ, EDGE_DOOR = 0, 1, 2

# typical share of the footprint per room type (ResPlan medians) — used when the user gives no room sizes
DEFAULT_SIZE = {0: 0.365, 1: 0.152, 2: 0.040, 3: 0.088, 4: 0.044, 5: 0.023, 6: 0.031}
