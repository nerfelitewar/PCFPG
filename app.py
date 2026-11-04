"""Floor-plan Studio v3 — interactive Streamlit front-end for floorplan_gen.
Orchestration only: it builds the same JSON spec you would write by hand and calls the existing pipeline (generate.py / layout.py).
Run:  streamlit run app.py"""
import json, sys, time
from pathlib import Path
import cv2, numpy as np, pandas as pd, plotly.graph_objects as go, streamlit as st
import streamlit.components.v1 as components
from shapely.geometry import Polygon

ROOT = Path(__file__).resolve().parent; sys.path.insert(0, str(ROOT))
from common import ROOM_TYPES, PALETTE, M, DEFAULT_SIZE
from generate import Gen, parse_spec, build_input, default_connections, ALIAS
from layout import export, FEATS, PREF

st.set_page_config(page_title="Floor-plan Studio", page_icon="🏠", layout="wide")
st.markdown("""<style>.block-container{padding-top:1rem;max-width:1560px}
.hero{padding:22px 28px;border-radius:20px;background:linear-gradient(120deg,#2f6fed 0%,#7a5cf5 58%,#e0559c 125%);color:#fff;margin-bottom:12px;box-shadow:0 8px 24px rgba(47,111,237,.25)}
.hero h1{margin:0;font-size:31px;color:#fff}.hero p{margin:5px 0 0;opacity:.93;font-size:15px}
.steps{display:flex;gap:8px;margin-top:12px;flex-wrap:wrap}.steps span{background:rgba(255,255,255,.2);border-radius:99px;padding:3px 13px;font-size:13px}
div[data-testid="stMetric"]{background:rgba(128,128,128,.08);border-radius:14px;padding:8px 14px}
.pill{display:inline-block;padding:2px 10px;border-radius:99px;font-size:12px;background:rgba(47,111,237,.14);margin-right:6px}
.card{border:1px solid rgba(128,128,128,.25);border-radius:16px;padding:14px 18px;background:rgba(128,128,128,.05)}
.verdict{border-radius:18px;padding:16px 22px;background:linear-gradient(90deg,rgba(47,111,237,.16),rgba(26,157,85,.14));margin:6px 0 10px}
.verdict h3{margin:0 0 2px 0}.stars{font-size:23px;color:#f5a623;letter-spacing:2px}.badge{display:inline-block;padding:1px 9px;border-radius:99px;font-size:12px;background:rgba(128,128,128,.16);margin:2px 3px 2px 0}
.ok{color:#1a9d55}.bad{color:#d33}.warn{color:#d98a00}</style>""", unsafe_allow_html=True)
ss = st.session_state
_graph_editor = components.declare_component("room_graph", path=str(ROOT / "room_graph"))
_editor = components.declare_component("boundary_editor", path=str(ROOT / "boundary_editor"))

# ------------------------------ rooms library ------------------------------
# Exactly the 7 room classes annotated in ResPlan, i.e. what the networks were trained on. Nothing else is offered.
G7 = "ResPlan room types"
PRESETS = [("living", "Living room", "🛋️", "living", 1.0, G7), ("bedroom", "Bedroom", "🛏️", "bedroom", 1.0, G7), ("bathroom", "Bathroom", "🚿", "bathroom", 1.0, G7),
           ("kitchen", "Kitchen", "🍳", "kitchen", 1.0, G7), ("balcony", "Balcony", "🌿", "balcony", 1.0, G7), ("storage", "Storage", "📦", "storage", 1.0, G7),
           ("stair", "Staircase", "🪜", "stair", 1.0, G7)]
PRESET = {k: dict(key=k, label=l, icon=i, base=b, mult=m, group=g) for k, l, i, b, m, g in PRESETS}
PRESET_LIST = list(PRESET.values())
DEFAULT_PRESET = {t: t for t in ROOM_TYPES}
TEMPLATES = {"Studio": ["living", "kitchen", "bathroom"], "1 BHK": ["living", "kitchen", "bedroom", "bathroom"],
             "2 BHK": ["living", "kitchen", "bedroom", "bedroom", "bathroom", "bathroom", "balcony"],
             "3 BHK": ["living", "kitchen"] + ["bedroom"] * 3 + ["bathroom"] * 3 + ["balcony"] * 2,
             "3 BHK + storage": ["living", "kitchen"] + ["bedroom"] * 3 + ["bathroom"] * 2 + ["balcony", "storage"],
             "4 BHK": ["living", "kitchen"] + ["bedroom"] * 4 + ["bathroom"] * 3 + ["balcony"] * 2 + ["storage"]}
EXAMPLES = {"Studio": ("Rectangle", 7, 5.5), "1 BHK": ("Rectangle", 8, 6), "2 BHK": ("Rectangle", 12, 9), "3 BHK": ("L-shape", 14, 12),
            "3 BHK + storage": ("Rectangle", 14, 10), "4 BHK": ("Rectangle", 18, 13)}
CATS = ["Overall", "Connectivity", "Geometry", "Completeness", "Room sizes", "Model agreement"]
CAT_HELP = {"Overall": "Weighted blend of the five checks (30% connectivity, 25% geometry, 20% completeness, 15% sizes, 10% model agreement).",
            "Connectivity": "Were the requested doors cut, and can you walk from every room to every other room?",
            "Geometry": "Are walls straight and axis-aligned, and are rooms reasonably rectangular? (real plans: ~98% straight)",
            "Completeness": "Are all requested rooms present, and none squeezed into a thin sliver?",
            "Room sizes": "How close are the generated room areas to the sizes you asked for?",
            "Model agreement": "Do the final rooms follow the boxes the first neural network proposed? Low = the two networks disagree."}
FEAT_NAMES = {"door_ratio": "Requested doors cut", "present": "All rooms present", "connected": "All rooms reachable", "straight": "Straight walls",
              "box_consistency": "Agrees with Stage-1 boxes", "box_overlap": "Stacked boxes (penalty)", "aspect_bad": "Too-thin rooms (penalty)",
              "size_err": "Size error (penalty)", "auto_doors": "Auto-added doors (penalty)", "rectangularity": "Rectangular rooms"}
TIDX = lambda t: ALIAS[t]
HEX = lambda t: "rgb(%d,%d,%d)" % tuple(PALETTE[ROOM_TYPES.index(t) + 2])
COLORS = {t: HEX(t) for t in ROOM_TYPES}


def tip(label, help): st.markdown(label, help=help)


_PC = [0]


def wide(fn, *a, **k):                      # works on old (use_container_width) and new (width="stretch") Streamlit
    if getattr(fn, "__name__", "") == "plotly_chart" and "key" not in k: k["key"] = f"pc{_PC[0]}"; _PC[0] += 1      # unique id per chart
    try: return fn(*a, width="stretch", **k)
    except Exception: return fn(*a, use_container_width=True, **k)


def embed_html(html, height):               # st.components.v1.html is being removed -> st.iframe (new) / components.html (old)
    if hasattr(st, "iframe"): st.iframe(html, height=height)
    else: components.html(html, height=height)


# ------------------------------ graph state ------------------------------
def shape_pts(name, w, h):
    r = lambda P: [[round(x * 2) / 2, round(y * 2) / 2] for x, y in P]
    return r({"Rectangle": [[0, 0], [w, 0], [w, h], [0, h]], "L-shape": [[0, 0], [w, 0], [w, .5 * h], [.55 * w, .5 * h], [.55 * w, h], [0, h]],
              "T-shape": [[0, 0], [w, 0], [w, .45 * h], [.7 * w, .45 * h], [.7 * w, h], [.3 * w, h], [.3 * w, .45 * h], [0, .45 * h]],
              "U-shape": [[0, 0], [w, 0], [w, h], [.7 * w, h], [.7 * w, .4 * h], [.3 * w, .4 * h], [.3 * w, h], [0, h]]}[name])


def make_nodes(keys, mults=None):
    nodes, used = [], set()
    for i, k in enumerate(keys):
        p = PRESET[k]; nm = p["label"]; j = 2
        while nm.lower() in used: nm = f"{p['label']} {j}"; j += 1
        used.add(nm.lower()); nodes.append(dict(id=i + 1, name=nm, preset=k, type=p["base"], mult=(mults[i] if mults else p["mult"]), x=0, y=0))
    return arrange(nodes)


def arrange(nodes):
    n = max(len(nodes), 1)
    for i, nd in enumerate(nodes):
        a = 2 * np.pi * i / n - np.pi / 2; nd["x"] = int(430 + 330 * np.cos(a) - 89); nd["y"] = int(300 + 215 * np.sin(a) - 64)
    return nodes


def auto_edges(nodes):
    types = [TIDX(n["type"]) for n in nodes]; E = {tuple(sorted(p)) for p in default_connections([n["name"] for n in nodes], types)}
    hub = next((i for i, t in enumerate(types) if t == 0), 0)
    for i, t in enumerate(types):
        if t == 0 and i != hub: E.add(tuple(sorted((hub, i))))
    for i in range(len(nodes)):                                    # nobody left alone: connect to the most sensible room
        if not any(i in e for e in E) and len(nodes) > 1:
            j = max((j for j in range(len(nodes)) if j != i), key=lambda j: PREF.get(tuple(sorted((types[i], types[j]))), 0.4)); E.add(tuple(sorted((i, j))))
    return [dict(a=nodes[a]["id"], b=nodes[b]["id"], kind="door") for a, b in sorted(E)]


def entrance_of(nodes):
    return next((n["id"] for n in nodes if n["type"] == "living"), nodes[0]["id"] if nodes else None)


def set_graph(nodes, edges=None, entrance=None):
    ss.graph = {"nodes": nodes, "edges": auto_edges(nodes) if edges is None else edges, "entrance": entrance if entrance is not None else entrance_of(nodes)}; ss.gver += 1


def apply_example(name):
    shape, w, h = EXAMPLES[name]; ss.pts = shape_pts(shape, w, h); ss.door = [ss.pts[0][0] + w / 2, ss.pts[0][1]]; ss.ver += 1
    set_graph(make_nodes(TEMPLATES[name])); ss.res = None; ss.name = name.lower().replace(" ", "_").replace("&", "and").replace("+", "plus")


def apply_template():
    set_graph(make_nodes(TEMPLATES[ss.get("tpl_pick", "2 BHK")])); ss.res = None


def apply_rooms(name): set_graph(make_nodes(TEMPLATES[name])); ss.res = None          # rooms only - the drawn boundary is kept


TYP_M2 = {"living": 20, "bedroom": 12, "bathroom": 4.5, "kitchen": 8, "balcony": 5, "storage": 3, "stair": 8}      # rule of thumb (m²), NOT ResPlan statistics


def fit_msg():
    P = ss.pts; area = Polygon(P).area if len(P) >= 3 and Polygon(P).is_valid else 0.0; need = sum(TYP_M2[n["type"]] * n["mult"] for n in ss.graph["nodes"])
    if area <= 0: return "warn", "Draw a valid boundary first."
    r = area / max(need, 1e-9); t = f"Your boundary is **{area:.0f} m²**; these rooms typically need about **{need:.0f} m²** (rule of thumb)."
    return ("warn", t + " ⚠ Rooms will be tight.") if r < 0.8 else (("info", t + " Rooms will come out generous.") if r > 1.8 else ("ok", t + " ✓ Good fit."))


def cb_auto(): g = ss.graph; set_graph(g["nodes"], None, g["entrance"])


def cb_clear(): g = ss.graph; set_graph(g["nodes"], [], g["entrance"])


def cb_arrange(): g = ss.graph; set_graph(arrange(g["nodes"]), g["edges"], g["entrance"])


def load_cfg(cfg):
    ids, types, E, ent, share, use_area = parse_spec(cfg)
    ss.pts = [list(map(float, p)) for p in cfg["boundary"]]; ss.door = list(cfg["front_door"]) if cfg.get("front_door") else None; ss.ver += 1
    areas = {r["id"]: r.get("area") for r in cfg["rooms"] if isinstance(r, dict)} if isinstance(cfg.get("rooms"), list) else {}
    keys = [DEFAULT_PRESET[ROOM_TYPES[t]] for t in types]; mults = [float(np.clip(areas[i] / DEFAULT_SIZE[t], 0.3, 2.5)) if areas.get(i) else 1.0 for i, t in zip(ids, types)]
    nodes = make_nodes(keys, mults)
    for nd, nm in zip(nodes, ids): nd["name"] = nm
    edges = [dict(a=nodes[i]["id"], b=nodes[j]["id"], kind="door" if E[i, j] == 2 else "adjacent") for i in range(len(ids)) for j in range(i + 1, len(ids)) if E[i, j]]
    set_graph(nodes, edges, nodes[ent]["id"]); ss.name = cfg.get("name", "my_plan"); ss.res = None


def init():
    if "pts" in ss: return
    ss.update(ver=0, gver=0, res=None, name="my_plan", run_req=False, pts=[], door=None, graph={"nodes": [], "edges": [], "entrance": None}); apply_example("2 BHK")


def cb_reset():
    for k in list(ss.keys()): del ss[k]


def build_cfg(use_sizes):
    g = ss.graph; nm = {n["id"]: n["name"].strip() for n in g["nodes"]}; R = []
    for n in g["nodes"]:
        d = {"id": nm[n["id"]], "type": n["type"]}
        if abs(n["mult"] - 1.0) > 1e-6: d["area"] = round(DEFAULT_SIZE[TIDX(n["type"])] * n["mult"], 4)     # same unit as the built-in typical sizes
        R.append(d)
    C = [[nm[e["a"]], nm[e["b"]], e["kind"]] for e in g["edges"] if e["a"] in nm and e["b"] in nm]
    return {"name": ss.name, "boundary": ss.pts, "front_door": ss.door, "rooms": R, "connections": C, "entrance": nm.get(g["entrance"], ""), "use_area": bool(use_sizes)}


def validate(cfg):
    out = []; P = cfg["boundary"]; poly = Polygon(P) if len(P) >= 3 else None; okp = poly is not None and poly.is_valid and poly.area > 8
    out.append((okp, f"Boundary: {len(P)} corners, {poly.area:.1f} m²" if okp else "Boundary needs ≥3 corners and must not cross itself (step ①)"))
    out.append((cfg["front_door"] is not None, "Entrance is placed on the boundary" if cfg["front_door"] is not None else "Place the entrance on the boundary (step ①, '🚪 Place entrance')"))
    ids = [r["id"] for r in cfg["rooms"]]; uniq = all(ids) and len({i.lower() for i in ids}) == len(ids)
    out.append((2 <= len(ids) <= M, f"{len(ids)} rooms (allowed 2–{M})"))
    out.append((uniq, "Every room has a unique, non-empty name" if uniq else "Room names must be unique and not empty (step ②)"))
    out.append((cfg["entrance"] in ids and cfg["entrance"] != "", "Entrance room selected" if cfg["entrance"] in ids else "Mark one room as the entrance (step ②)"))
    return out


@st.cache_resource(show_spinner="Loading models…")
def load_gen(box, ref, stamp): return Gen(box, ref)


@st.cache_data(show_spinner=False)
def ckpt_meta(path, mtime):
    import torch
    d = torch.load(path, map_location="cpu"); d.pop("model", None); return {k: v for k, v in d.items() if isinstance(v, (int, float, str))}


def thumb(label, size=250): return cv2.resize(PALETTE[label], (size, size), interpolation=cv2.INTER_NEAREST)


def touching(inst, i, j): return int((cv2.dilate((inst == i + 1).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0)[inst == j + 1].sum()) > 12


# ------------------------------ generation + scoring ------------------------------
def cat_scores(c):
    f = c["feats"]; conn = 100 * (0.6 * f["door_ratio"] + 0.4 * f["connected"]); geom = 100 * (0.7 * f["straight"] + 0.3 * min(1, f["rectangularity"] / 0.92))
    comp = 100 * (0.7 * f["present"] + 0.3 * (1 - f["aspect_bad"])); size = 100 * (1 - f["size_err"]); agree = 100 * f["box_consistency"]
    return dict(zip(CATS, [0.30 * conn + 0.25 * geom + 0.20 * comp + 0.15 * size + 0.10 * agree, conn, geom, comp, size, agree]))


def grade(v): return "A" if v >= 90 else "B" if v >= 80 else "C" if v >= 70 else "D" if v >= 60 else "E"
def stars(v): n = int(round(v / 20)); return "★" * n + "☆" * (5 - n)


def run_generation(cfg, gen, k, seed, tta):
    ids, types, E, ent, share, use_area = parse_spec(cfg); cond, fpt = build_input(cfg); n = len(ids)
    nt = np.full(M, -1, np.int64); nt[:n] = types; en = np.zeros(M, np.float32); en[ent] = 1
    t0 = time.time(); cs = gen.candidates(cond, nt, en, E, fpt, share, use_area, seed=seed, k=k, tta=tta); secs = time.time() - t0
    P = np.array(cfg["boundary"]); maxdim = float(max(np.ptp(P[:, 0]), np.ptp(P[:, 1]))); m2px = (maxdim / 236.0) ** 2
    target = share[:n] if use_area else np.array([DEFAULT_SIZE[int(t)] for t in types]); target = target / target.sum()
    for i, c in enumerate(cs):
        c["cid"] = i + 1; a = np.bincount(c["inst"].ravel(), minlength=M + 2)[1:n + 1] * m2px; c["areas"] = a; c["got"] = a / max(a.sum(), 1e-9); c["cats"] = cat_scores(c)
        f, s = c["feats"], c["stats"]
        c["row"] = {"Cand": c["cid"], "Score": round(c["score"], 3), "Doors": f"{s['doors_placed']}/{s['doors_requested']}", "Auto doors": s["doors_auto"], "Connected": "✓" if s["all_connected"] else "✗",
                    "Rooms": f"{s['rooms_present']}/{s['rooms']}", "Straightness": s["straightness"], "Rectangularity": s["rectangularity"], "Windows": s["windows"],
                    "Size error": round(f["size_err"], 3), "Thin rooms": round(f["aspect_bad"], 3), "Box agreement": round(f["box_consistency"], 3), **{k_: round(v, 1) for k_, v in c["cats"].items()}}
    B = np.stack([c["boxes"][:n] for c in cs]); div = float(np.mean([np.abs(B[i] - B[j]).sum(-1).mean() for i in range(len(cs)) for j in range(i + 1, len(cs))])) if len(cs) > 1 else None
    cs.sort(key=lambda c: -c["score"]); run_dir = ROOT / "outputs" / "ui" / time.strftime("%Y%m%d_%H%M%S"); run_dir.mkdir(parents=True, exist_ok=True)
    return dict(cands=cs, ids=ids, types=list(types), n=n, cfg=cfg, target=target, secs=secs, dir=run_dir, learned=gen.learned, w=gen.w, diversity=div, k=k, tta=tta, exp={})


def exported(res, i):
    if i not in res["exp"]:
        c = res["cands"][i]; nm = f"cand_{c['cid']}"
        export(c["label"], c["inst"], res["ids"], nm, c["stats"], out_dir=res["dir"], types=res["types"]); d = res["dir"]
        res["exp"][i] = ((d / f"{nm}.png").read_bytes(), (d / f"{nm}_3d.html").read_text(encoding="utf-8"), (d / f"{nm}.json").read_text())
    return res["exp"][i]


def gauge(v, big=False):
    col = "#2e9e5b" if v >= 85 else ("#f0ad4e" if v >= 65 else "#d9534f")
    f = go.Figure(go.Indicator(mode="gauge+number", value=float(v), number=dict(font=dict(size=40 if big else 32)),
                               gauge=dict(axis=dict(range=[0, 100], visible=False), bar=dict(color=col, thickness=0.8), bgcolor="rgba(128,128,128,.18)", borderwidth=0)))
    f.update_layout(height=140, margin=dict(l=8, r=8, t=8, b=0), paper_bgcolor="rgba(0,0,0,0)"); return f


def plan_fig(res, c, overlay):
    ids, n, inst = res["ids"], res["n"], c["inst"]; fig = go.Figure(go.Image(z=PALETTE[c["label"]], hoverinfo="skip"))
    cen = []
    for k in range(n): ys, xs = np.where(inst == k + 1); cen.append((float(xs.mean()), float(ys.mean())) if len(xs) else (np.nan, np.nan))
    if overlay:
        for ia, ib, kind in res["cfg"]["connections"]:
            a, b = ids.index(ia), ids.index(ib); ok = touching(inst, a, b)
            fig.add_trace(go.Scatter(x=[cen[a][0], cen[b][0]], y=[cen[a][1], cen[b][1]], mode="lines", showlegend=False, hoverinfo="text",
                                     text=f"{ia} ⟷ {ib}: {'rooms touch ✓' if ok else 'rooms do NOT touch ✗'}", line=dict(color="#1a9d55" if ok else "#d9534f", width=3.5 if ok else 3, dash="solid" if ok else "dash")))
    for k in range(n):
        if np.isnan(cen[k][0]): continue
        fig.add_annotation(x=cen[k][0], y=cen[k][1] - (4 if overlay else 0), text=f"<b>{ids[k]}</b>", showarrow=False, font=dict(size=11, color="#1d2433"), bgcolor="rgba(255,255,255,.72)", borderpad=2)
        if overlay:
            d = c["got"][k] / max(res["target"][k], 1e-9) - 1; colr = "#1a9d55" if abs(d) < .15 else ("#d98a00" if abs(d) < .35 else "#d33")
            fig.add_annotation(x=cen[k][0], y=cen[k][1] + 9, text=f"{'▲' if d >= 0 else '▼'} {abs(d) * 100:.0f}% · {c['areas'][k]:.1f} m²", showarrow=False, font=dict(size=10, color=colr), bgcolor="rgba(255,255,255,.8)", borderpad=1)
    fig.update_xaxes(visible=False); fig.update_yaxes(visible=False); fig.update_layout(height=640, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)")
    return fig



# ------------------------------ end-to-end walkthrough (animated) ------------------------------
import base64


def _b64(img):
    ok, buf = cv2.imencode(".png", img[..., ::-1]); return "data:image/png;base64," + base64.b64encode(buf).decode()


def frames(res, c):
    """Real intermediate pictures of the selected layout, one per pipeline stage."""
    ids, n, S = res["ids"], res["n"], 384; cfg = res["cfg"]; cond, _ = build_input(cfg); pal = PALETTE; out = []
    up = lambda m: cv2.resize(m, (S, S), interpolation=cv2.INTER_NEAREST)
    inner = np.full((128, 128, 3), 245, np.uint8); inner[cond[0] > 0] = (200, 215, 245); inner[cond[1] > 0] = (166, 54, 3)
    out.append(("1 · Your input", "The outline you drew (blue) and the entrance (brown), rescaled to the 128×128 canvas the networks use. Tensor: boundary mask + entrance mask = 2×128×128.", _b64(up(inner))))
    g = np.full((S, S, 3), 248, np.uint8); r = 135; cen = [(int(S / 2 + r * np.cos(2 * np.pi * k / n - np.pi / 2)), int(S / 2 + r * np.sin(2 * np.pi * k / n - np.pi / 2))) for k in range(n)]
    for cn in cfg["connections"]:
        a, b = cen[ids.index(cn[0])], cen[ids.index(cn[1])]; cv2.line(g, a, b, (90, 90, 90), 3 if (cn[2] if len(cn) > 2 else "door") == "door" else 1, cv2.LINE_AA)
    for k in range(n):
        cv2.circle(g, cen[k], 17, tuple(int(v) for v in pal[res["types"][k] + 2]), -1, cv2.LINE_AA); cv2.circle(g, cen[k], 17, (60, 60, 60), 2, cv2.LINE_AA)
        cv2.putText(g, ids[k][:12], (cen[k][0] - 28, cen[k][1] - 23), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (30, 30, 30), 1, cv2.LINE_AA)
    out.append(("2 · Room graph", "Your rooms become nodes (colour = ResPlan room type) and your doors become edges. Tensors: room types, entrance flag, relative sizes, 16×16 edge matrix (none / adjacent / door).", _b64(g)))
    bx = up(inner).copy()
    for k in range(n):
        x0, y0, x1, y1 = (c["boxes"][k] * S).astype(int); col = tuple(int(v) for v in pal[res["types"][k] + 2]); ov = bx.copy(); cv2.rectangle(ov, (x0, y0), (x1, y1), col, -1); bx = cv2.addWeighted(ov, 0.45, bx, 0.55, 0)
        cv2.rectangle(bx, (x0, y0), (x1, y1), (50, 50, 50), 2); cv2.putText(bx, ids[k][:10], (x0 + 4, y0 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (20, 20, 20), 1, cv2.LINE_AA)
    out.append(("3 · Stage 1 — BoxNet", "A graph transformer reads the outline and the graph and proposes one rectangle per room (x0, y0, x1, y1). Output: 16×4 numbers. Several random tries give several different arrangements.", _b64(bx)))
    rooms = pal[np.where(c["inst"] > 0, np.array([0] + [t + 2 for t in res["types"]] + [0] * (M + 1 - n))[c["inst"]], 0)]; out.append(("4 · Stage 2 — RefineNet", "A U-Net turns the overlapping boxes into clean room shapes that tile the outline exactly. Output: 17×128×128 scores (outside + one per room) → one room per pixel.", _b64(up(rooms))))
    lab = c["label"].copy(); lab[np.isin(lab, [9, 10, 11])] = 1; out.append(("5 · Builder — straighten + walls", "No AI here: borders are redrawn as strict right-angle polygons and walls are placed on every border between two rooms.", _b64(up(pal[lab]))))
    out.append(("6 · Builder — doors, windows, entrance", "Doors are cut where you linked rooms (pink), windows go on outside walls (green), the entrance is marked (brown). Many tries are scored and the best is kept.", _b64(up(pal[c["label"]]))))
    out.append(("7 · 3D model", "The final layout is extruded to 3D in the browser: walls, window glass, balcony railing and deck. Open the 3D tab to explore it.", _b64(up(pal[c["label"]]))))
    return out


PIPE = r"""<!DOCTYPE html><html><head><meta charset="utf-8"><style>
:root{--ink:#1d2433;--mut:#6b7280;--line:#d5dcea;--bg:#f6f8fc;--ac:#2f6fed}@media(prefers-color-scheme:dark){:root{--ink:#e6eaf2;--mut:#9aa4b8;--line:#2d3650;--bg:#141a28;--ac:#6ea0ff}}
body{margin:0;font-family:system-ui,"Segoe UI",sans-serif;color:var(--ink);background:transparent}h3{margin:6px 0 8px;font-size:17px}.sec{border:1px solid var(--line);border-radius:16px;padding:12px 16px;margin-bottom:12px;background:var(--bg)}
button{border:1px solid var(--line);background:transparent;color:var(--ink);border-radius:9px;padding:5px 12px;cursor:pointer;font-size:13px}button:hover{border-color:var(--ac)}
#td{min-height:42px;font-size:13.5px;margin-top:4px}#pl{display:flex;gap:18px;align-items:flex-start;flex-wrap:wrap}#im{width:380px;max-width:100%;border-radius:12px;border:1px solid var(--line);transition:opacity .35s}
#tx{flex:1;min-width:260px}#tt{font-size:19px;font-weight:700;margin-bottom:6px}#cp{font-size:14px;line-height:1.5}.dots{display:flex;gap:6px;margin-top:12px}.dots i{width:30px;height:6px;border-radius:4px;background:var(--line);cursor:pointer}.dots i.on{background:var(--ac)}</style></head><body>
<div class="sec"><h3>① Training pipeline — done once, before you use the app</h3><svg id="sv" viewBox="0 0 1000 150" width="100%"></svg><div id="td"></div></div>
<div class="sec"><h3>② Inference — what happens to YOUR input when you press Generate</h3><div id="pl"><img id="im"><div id="tx"><div id="tt"></div><div id="cp"></div><div class="dots" id="dots"></div>
<div style="margin-top:12px"><button id="bp">⏮</button> <button id="bg">⏸ Pause</button> <button id="bn">⏭</button></div></div></div></div>
<script>
const FR=__FRAMES__, TR=[["📦","ResPlan.pkl","17,000 vector plans (shapely polygons)","The dataset: every room, wall, door and window of 17,000 real apartments as polygons."],
["🔲","Rasterise + graph","graphdata.py","Each plan becomes 128×128 maps, one box per room, a room-type list and a connectivity graph (adjacent / door edges)."],
["🗄️","Cache","cache4_*.npz","Train / val / test arrays ready for the GPU (the whole set fits in GPU memory)."],
["🧠","Train BoxNet","L1 + GIoU + adjacency + tiling","Graph transformer learns to place one box per room. Winner-takes-all over 4 noise samples makes tries diverse."],
["🧩","Train RefineNet","edge-weighted CE + TV","U-Net learns to fill the outline with clean rooms; then fine-tuned on BoxNet's own (imperfect) boxes."],
["🎯","Calibrate scorer","calibrate_score.py","Learns which layout properties predict real quality, so the best of K tries is chosen sensibly."],
["✅","Checkpoints","box_best.pt · refine_best.pt","The trained models this app loads. Nothing is trained while you use the app."]];
const sv=document.getElementById("sv"),NS="http://www.w3.org/2000/svg",bw=122,gap=24;let act=0,play=true,tm=0;
TR.forEach((t,i)=>{const x=i*(bw+gap)+6,g=document.createElementNS(NS,"g");g.setAttribute("id","b"+i);g.style.cursor="pointer";g.innerHTML=`<rect x="${x}" y="22" width="${bw}" height="104" rx="14" fill="var(--bg)" stroke="var(--line)" stroke-width="2"/><text x="${x+bw/2}" y="56" text-anchor="middle" font-size="26">${t[0]}</text><text x="${x+bw/2}" y="82" text-anchor="middle" font-size="12.5" font-weight="700" fill="currentColor">${t[1]}</text><text x="${x+bw/2}" y="100" text-anchor="middle" font-size="9.5" fill="var(--mut)">${t[2].length>24?t[2].slice(0,23)+"…":t[2]}</text>`;g.onclick=()=>{act=i;draw()};sv.appendChild(g);
 if(i<TR.length-1){const l=document.createElementNS(NS,"line");l.setAttribute("x1",x+bw);l.setAttribute("x2",x+bw+gap);l.setAttribute("y1",74);l.setAttribute("y2",74);l.setAttribute("stroke","var(--mut)");l.setAttribute("stroke-width",2);sv.appendChild(l);}});
const pk=[];for(let i=0;i<TR.length-1;i++){const c=document.createElementNS(NS,"circle");c.setAttribute("r",5);c.setAttribute("fill","var(--ac)");c.setAttribute("cy",74);sv.appendChild(c);pk.push(c)}
function draw(){TR.forEach((t,i)=>{const r=document.querySelector("#b"+i+" rect");r.setAttribute("stroke",i===act?"var(--ac)":"var(--line)");r.setAttribute("stroke-width",i===act?4:2)});document.getElementById("td").innerHTML="<b>"+TR[act][1]+"</b> — "+TR[act][3]}
function loop(t){pk.forEach((c,i)=>{const p=((t/900+i*.18)%1);c.setAttribute("cx",(i+1)*(bw+gap)-gap+6+p*gap)});requestAnimationFrame(loop)}requestAnimationFrame(loop);
setInterval(()=>{if(play){act=(act+1)%TR.length;draw()}},2200);draw();
let fi=0;const im=document.getElementById("im"),dots=document.getElementById("dots");
FR.forEach((f,i)=>{const d=document.createElement("i");d.onclick=()=>{fi=i;show()};dots.appendChild(d)});
function show(){if(!FR.length){document.getElementById("tt").textContent="Generate a plan first";document.getElementById("cp").textContent="Press Generate floor plans — then this player shows how YOUR rooms flow through the networks, with the real pictures of every stage.";im.style.display="none";return}
 im.style.opacity=0;setTimeout(()=>{im.src=FR[fi][2];im.style.opacity=1},150);document.getElementById("tt").textContent=FR[fi][0];document.getElementById("cp").textContent=FR[fi][1];[...dots.children].forEach((d,i)=>d.className=i===fi?"on":"")}
document.getElementById("bp").onclick=()=>{fi=(fi+FR.length-1)%FR.length;show()};document.getElementById("bn").onclick=()=>{fi=(fi+1)%FR.length;show()};
document.getElementById("bg").onclick=e=>{play=!play;e.target.textContent=play?"⏸ Pause":"▶ Play"};setInterval(()=>{if(play&&FR.length){fi=(fi+1)%FR.length;show()}},3200);show();
</script></body></html>"""


def pipeline_html(fr): return PIPE.replace("__FRAMES__", json.dumps([[t, c, im] for t, c, im in fr]))


def line_fig(path, rename, title, ycol_hint, lower_better=False):
    d = pd.read_csv(path); cols = [c for c in d.columns if ycol_hint(c)]; f = go.Figure()
    for c in cols:
        f.add_trace(go.Scatter(x=d["epoch"], y=d[c], name=rename.get(c, c), mode="lines", line=dict(width=2.6), hovertemplate="epoch %{x}: %{y:.3f}<extra>" + rename.get(c, c) + "</extra>"))
        j = int(d[c].idxmin() if lower_better else d[c].idxmax()); f.add_trace(go.Scatter(x=[d["epoch"][j]], y=[d[c][j]], mode="markers", marker=dict(size=11, symbol="star", line=dict(width=1, color="#222")), showlegend=False, hoverinfo="skip"))
    f.update_layout(title=dict(text=title, font=dict(size=16)), height=340, margin=dict(l=10, r=10, t=50, b=10), xaxis_title="training epoch", hovermode="x unified", legend=dict(orientation="h", y=-0.28),
                    yaxis_title="loss (lower is better ↓)" if lower_better else "overlap with true layout (higher is better ↑)")
    return f

# ------------------------------ sidebar ------------------------------
init()
with st.sidebar:
    st.header("⚙️ Settings")
    box = st.text_input("Stage 1 weights", str(ROOT / "checkpoints" / "box_best.pt"), help="Trained 'BoxNet' file (places one box per room). Created by train_box.py.")
    ref = st.text_input("Stage 2 weights", str(ROOT / "checkpoints" / "refine_best.pt"), help="Trained 'RefineNet' file (turns boxes into clean rooms). Created by train_refine.py.")
    files_ok = Path(box).exists() and Path(ref).exists()
    st.markdown(f"<span class='pill'>{'✓ weights found' if files_ok else '✗ weights missing'}</span>", unsafe_allow_html=True)
    K = st.slider("How many layouts to try", 1, 32, 8, help="The AI draws this many different layouts for your input and keeps the best-scoring one. More = better chance of a great plan, a little slower. You can still inspect all of them afterwards.")
    seed = st.number_input("Random seed", 0, 99999, 0, help="The AI uses randomness to create different layouts. Same seed + same input = same result. Change it to get a fresh set of layouts.")
    tta = st.checkbox("Smooth borders (test-time augmentation)", True, help="Runs the second network 4 times on flipped/rotated copies and averages them. Gives cleaner room borders. Turn off only if you want maximum speed.")
    use_sizes = st.checkbox("Follow my room sizes", True, help="ON: the size sliders on the room cards guide the layout (e.g. 'Master bedroom ×1.4'). OFF: the AI picks all sizes freely from typical values.")
    st.button("🚀 Generate floor plans", type="primary", key="gen_side", on_click=lambda: ss.__setitem__("run_req", True), help="Same as the button in step ③. The checklist in step ③ must be green.")
    st.caption("Boundary units are **metres**. Everything here builds the same JSON spec you could write by hand (see step ③).")
    st.button("↺ Reset everything", on_click=cb_reset, help="Clears the boundary, rooms, connections and results and starts again from the 2 BHK example.")

st.markdown("""<div class="hero"><h1>🏠 Floor-plan Studio</h1><p>Sketch the outline, drop in rooms, wire them together — the AI designs the plan and shows it in 2D and 3D.</p>
<div class="steps"><span>① Draw boundary</span><span>② Drop & connect rooms</span><span>③ Generate</span><span>④ Compare & export</span></div></div>""", unsafe_allow_html=True)
r1 = st.columns([1.6] + [1] * len(TEMPLATES))
r1[0].markdown("**🏠 Rooms for MY boundary**", help="Replaces ONLY rooms and connections. The boundary you drew stays as it is.")
for col, nm_ in zip(r1[1:], TEMPLATES): col.button(nm_, key=f"rm_{nm_}", on_click=apply_rooms, args=(nm_,), help=f"{nm_}: {len(TEMPLATES[nm_])} rooms with typical doors, placed on YOUR current boundary.")
qs = st.columns([1.6] + [1] * len(EXAMPLES))
qs[0].markdown("**⚡ Full example (boundary + rooms)**", help="Loads a ready-made outline AND rooms. Your current boundary is replaced.")
for col, nm_ in zip(qs[1:], EXAMPLES): col.button(nm_, key=f"ex_{nm_}", on_click=apply_example, args=(nm_,), help=f"Loads an example: {nm_} ({len(TEMPLATES[nm_])} rooms, {EXAMPLES[nm_][0].lower()} {EXAMPLES[nm_][1]}×{EXAMPLES[nm_][2]} m).")
t1, t2, t3, t4 = st.tabs(["① Boundary", "② Rooms & connections", "③ Review & generate", "🧠 Model & training"])

# ------------------------------ ① boundary ------------------------------
with t1:
    c1, c2 = st.columns([3.2, 1.4])
    with c2:
        tip("**Start from a shape**", "Pick a basic outline, set its size, press the button. Or ignore this and draw your own on the canvas.")
        shape = st.selectbox("Shape", ["Rectangle", "L-shape", "T-shape", "U-shape"], label_visibility="collapsed", help="Basic outline to start from.")
        w = st.slider("Width (m)", 6.0, 24.0, 12.0, 0.5, help="Overall width of the shape, in metres."); h = st.slider("Depth (m)", 5.0, 18.0, 9.0, 0.5, help="Overall depth (top to bottom) of the shape, in metres.")
        if st.button("Use this shape", help="Replaces the current boundary on the canvas with this shape."):
            ss.pts = shape_pts(shape, w, h); ss.door = [ss.pts[0][0] + w / 2, ss.pts[0][1]]; ss.ver += 1; ss.res = None; st.rerun()
        with st.expander("Import / edit as numbers"):
            samples = sorted((ROOT / "samples").glob("*.json")); pick = st.selectbox("Sample JSON", ["—"] + [p.name for p in samples], help="Load one of the example JSON files from the samples folder.")
            if pick != "—" and st.button("Load sample", help="Replaces boundary, rooms and connections with the sample."): load_cfg(json.loads((ROOT / "samples" / pick).read_text())); st.rerun()
            up = st.file_uploader("Upload your JSON spec", type="json", help="A JSON file in the same format the command-line tool uses (boundary, rooms, connections).")
            if up is not None and st.button("Load uploaded file"):
                try: load_cfg(json.load(up)); st.rerun()
                except Exception as e: st.error(f"Could not load: {e}")
            tip("**Corner coordinates**", "Corner coordinates in metres, in order around the outline. x grows to the right, y grows downwards. Add or delete rows with the table controls.")
            df = st.data_editor(pd.DataFrame(ss.pts, columns=["x (m)", "y (m)"]), num_rows="dynamic", key=f"vt{ss.ver}", hide_index=True)
            dx, dy = st.columns(2); ex = dx.number_input("Entrance x", value=float(ss.door[0]) if ss.door else 0.0, help="Entrance position (m). It snaps to the nearest wall.")
            ey = dy.number_input("Entrance y", value=float(ss.door[1]) if ss.door else 0.0, help="Entrance position (m). It snaps to the nearest wall.")
            if st.button("Apply numbers", help="Uses the corner table and entrance numbers above."): ss.pts = df.dropna().values.tolist(); ss.door = [ex, ey]; ss.ver += 1; st.rerun()
    with c1:
        tip("**Draw the outer boundary**", "Click to add corners, drag a corner to move it, right-click a corner to delete it, click on an edge to insert a corner. Then press '🚪 Place entrance' and click the wall where the front door should be.")
        ex_w = max(24, int(np.ceil(max(p[0] for p in ss.pts))) + 4) if ss.pts else 24; ex_h = max(16, int(np.ceil(max(p[1] for p in ss.pts))) + 3) if ss.pts else 16
        val = _editor(points=ss.pts, door=ss.door, extent=[ex_w, ex_h], ver=ss.ver, key=f"be{ss.ver}", default=None)
        if val and val.get("points") is not None: ss.pts, ss.door = val["points"], val.get("door")
    if len(ss.pts) >= 3:
        pg = Polygon(ss.pts); a, b, c, d = st.columns(4)
        a.metric("Floor area", f"{pg.area:.1f} m²", help="Area inside the outline you drew."); b.metric("Perimeter", f"{pg.length:.1f} m", help="Total length of the outer walls.")
        c.metric("Bounding box", f"{np.ptp([p[0] for p in ss.pts]):.1f} × {np.ptp([p[1] for p in ss.pts]):.1f} m", help="Width × depth of the smallest rectangle around the outline.")
        d.metric("Valid shape", "yes ✓" if pg.is_valid else "self-crossing ✗", help="The outline must not cross itself.")

# ------------------------------ ② rooms & connections (node editor) ------------------------------
with t2:
    tip("**Build your floor plan as a graph**", "Each card is a room. Drag rooms from the library on the left onto the canvas, drag the blue ● of one room onto another to connect them, and use the size slider on each card. The picture does not have to look like the final plan — only WHO connects to WHOM matters.")
    b1, b2, b3, b4, b5 = st.columns([1.5, 0.8, 1.3, 1.1, 1])
    b1.selectbox("Start from a template", list(TEMPLATES), index=2, key="tpl_pick", label_visibility="collapsed", help="Pick a ready-made set of rooms. Press 'Load' to replace what is on the canvas.")
    b2.button("Load", on_click=apply_template, help="Replaces all rooms and connections with the chosen template.")
    b3.button("✨ Auto-connect", on_click=cb_auto, help="Builds a typical layout: living room is the hub; kitchen, bedrooms and other rooms attach to it; each bathroom attaches to a bedroom. Replaces all current connections.")
    b4.button("Clear links", on_click=cb_clear, help="Removes every connection so you can wire the rooms yourself.")
    b5.button("↻ Arrange", on_click=cb_arrange, help="Tidies the cards into a circle. Only changes this picture, not the floor plan.")
    val = _graph_editor(graph=ss.graph, presets=PRESET_LIST, colors=COLORS, ver=ss.gver, height=680, key=f"rg{ss.gver}", default=None)
    if val and val.get("nodes") is not None: ss.graph = {"nodes": val["nodes"], "edges": val["edges"], "entrance": val.get("entrance")}
    g = ss.graph; st.caption(f"**{len(g['nodes'])} rooms · {len(g['edges'])} connections** · each preset maps to one of the 7 types the AI was trained on (hover a library item to see which). "
                             "Solid link 🚪 = door · dashed ▭ = wall only.")
    _k, _m = fit_msg(); {"ok": st.success, "warn": st.warning, "info": st.info}[_k](_m)

# ------------------------------ ③ review & generate ------------------------------
with t3:
    ss.name = st.text_input("Plan name", ss.name, help="Used for the downloaded file names.")
    cfg = build_cfg(use_sizes); checks = validate(cfg); ready = all(ok for ok, _ in checks) and files_ok
    l, r = st.columns([1, 1.3])
    with l:
        tip("**Checklist**", "Everything must be green before you can generate. Red items tell you where to fix it.")
        for ok, msg in checks: st.markdown(f"<span class='{'ok' if ok else 'bad'}'>{'✓' if ok else '✗'}</span> {msg}", unsafe_allow_html=True)
        nodes_ = ss.graph["nodes"]; isl = [x["name"] for x in nodes_ if len(nodes_) > 1 and not any(x["id"] in (e["a"], e["b"]) for e in ss.graph["edges"])]
        st.markdown(f"<span class='{'warn' if isl else 'ok'}'>{'⚠' if isl else '✓'}</span> " + (f"Not connected yet: {', '.join(isl)} (link them in step ②, or the AI will add a door itself)" if isl else "Every room has at least one connection"), unsafe_allow_html=True)
        if not files_ok: st.warning("Model weights not found — set the paths in the sidebar.")
        st.button("🚀 Generate floor plans", type="primary", key="generate", disabled=not ready, on_click=lambda: ss.__setitem__("run_req", True), help="Runs the two neural networks and the wall/door builder, then shows the best layout and all alternatives below.")
    with r:
        with st.expander("Technical: the JSON this page sends to the AI", expanded=False):
            st.caption("This is exactly the format of the .json files used by the command-line tool."); st.json(cfg, expanded=False)

# ------------------------------ run generation ------------------------------
if ss.run_req:
    ss.run_req = False
    if not ready: st.error("Some checklist items in step ③ are red — fix them first.")
    else:
        try:
            gen = load_gen(box, ref, (Path(box).stat().st_mtime, Path(ref).stat().st_mtime, (Path(ref).parent / "score_weights.json").exists()))
            with st.spinner(f"Generating {K} layouts…"): ss.res = run_generation(cfg, gen, K, int(seed), tta)
            ss["sel_box"], ss["cmpA"], ss["cmpB"] = 0, 0, min(1, K - 1); st.toast("✅ Done — scroll down to the results", icon="🏠")
        except Exception as e: st.error(f"Generation failed: {e}"); st.exception(e)

# ------------------------------ 🧠 model & training ------------------------------
with t4:
    tip("**The whole system, end to end**", "Top: what was done once to train the networks. Bottom: what happens to your input. The bottom player uses real pictures from the layout you are inspecting (generate one first). Click a training box for details.")
    _r = ss.res; _i = min(ss.get("sel_box", 0), len(_r["cands"]) - 1) if _r else 0
    embed_html(pipeline_html(frames(_r, _r["cands"][_i]) if _r else []), 800)
    tip("**Loaded model files**", "Information stored inside the checkpoint files: which training epoch was kept and how well it scored on validation data.")
    mc = st.columns(2)
    for col, nm_, path in ((mc[0], "Stage 1 · BoxNet", box), (mc[1], "Stage 2 · RefineNet", ref)):
        with col:
            if Path(path).exists():
                meta = ckpt_meta(path, Path(path).stat().st_mtime); st.markdown(f"**{nm_}** — `{Path(path).name}` ({Path(path).stat().st_size / 1e6:.0f} MB)")
                k1, k2, k3 = st.columns(3); k1.metric("Best epoch", meta.get("epoch", "?"), help="Training epoch whose weights were saved as 'best'.")
                k2.metric("Validation score", f"{meta.get('val_iou', float('nan')):.3f}", help="Mean overlap (IoU) with the true layout on unseen validation plans, 0–1. Higher is better.")
                k3.metric("Network size", f"{meta.get('d', meta.get('base', '?'))}", help="Network width (BoxNet: embedding size, RefineNet: base channels).")
            else: st.info(f"{nm_}: file not found.")
    ev = ROOT / "logs" / "eval_test.json"
    if ev.exists():
        E_ = json.loads(ev.read_text()); tip(f"**Test-set results** ({E_.get('test_plans', '?')} unseen plans)", "Measured by evaluate.py on plans the networks never saw in training. IoU = overlap with the true layout (1.0 = perfect).")
        e = st.columns(4)
        e[0].metric("Box placement (Stage 1)", f"{E_.get('box_iou_true_sizes', E_.get('box_iou_sizes', float('nan'))):.2f}", help="How well Stage 1 puts each room rectangle where the real architect put it (IoU). 1.0 = perfect.")
        e[1].metric("Final rooms (end-to-end)", f"{E_.get('room_iou_e2e', float('nan')):.2f}", help="Overlap of the final generated rooms with the real rooms, after both networks.")
        e[2].metric("Upper bound (Stage 2)", f"{E_.get('room_iou_gt_box', float('nan')):.2f}", help="What Stage 2 achieves if Stage 1 were perfect. The gap to 'end-to-end' is the room for improvement in Stage 1.")
        bk = [k for k in E_ if k.startswith("best_of")]
        if bk: e[3].metric("Plans with every room", f"{E_[bk[0]].get('plans_with_all_rooms', float('nan')) * 100:.0f}%", help="Share of test plans where all requested rooms appear in the result (best-of-K).")
        if "k=1" in E_ and bk:
            labs = ["Requested doors cut", "Rooms reachable", "Straight walls", "Plans with all rooms"]; keys = ["doors_placed", "all_rooms_connected", "straightness", "plans_with_all_rooms"]
            y1 = [E_["k=1"].get(k_, 0) * 100 for k_ in keys]; y2 = [E_[bk[0]].get(k_, 0) * 100 for k_ in keys]
            fg = go.Figure([go.Bar(name="1 try", x=labs, y=y1, marker_color="#9aa7bd", text=[f"{v:.0f}%" for v in y1], textposition="outside"), go.Bar(name=f"best of {bk[0].split('_')[-1]} tries", x=labs, y=y2, marker_color="#2f6fed", text=[f"{v:.0f}%" for v in y2], textposition="outside")])
            fg.update_layout(barmode="group", height=330, margin=dict(t=50, b=10), title=dict(text="Does trying several layouts help? (test set, higher is better)", font=dict(size=16)), yaxis=dict(range=[0, 115], title="% of test plans / rooms"), legend=dict(orientation="h", y=-0.2))
            wide(st.plotly_chart, fg)
    else: st.info("No evaluation yet. Run `python evaluate.py` to see test-set results here.")
    cur = []
    for fn, ttl, rn in (("box_log.csv", "Stage 1 · BoxNet", {"val_iou_true_sizes": "exact room sizes given", "val_iou_default_sizes": "typical sizes", "val_iou_no_sizes": "no sizes given", "val_oracle_best4": "best of 4 tries", "val_iou_sizes": "with sizes", "val_iou_nosizes": "no sizes"}),
                        ("refine_log.csv", "Stage 2 · RefineNet", {"val_roomIoU": "with real boxes (upper bound)", "val_roomIoU_jitter": "with noisy / predicted boxes", "val_acc": "pixel accuracy"})):
        p = ROOT / "logs" / fn
        if p.exists(): cur.append((p, ttl, rn))
    if cur:
        tip("**Training curves**", "One panel per network. Top: validation overlap (★ = best epoch, the weights that were saved). Bottom: training loss. Curves that flatten mean the network has learned what it can.")
        for col, (p, ttl, rn) in zip(st.columns(len(cur)), cur):
            with col:
                wide(st.plotly_chart, line_fig(p, rn, f"{ttl} — validation quality", lambda c: "iou" in c.lower() or c.lower().startswith("val_oracle")))
                if "loss" in pd.read_csv(p).columns: wide(st.plotly_chart, line_fig(p, {"loss": "training loss"}, f"{ttl} — training loss", lambda c: c == "loss", True))
    cal = ROOT / "logs" / "calibration.json"
    if cal.exists():
        sel_ = json.loads(cal.read_text()).get("held_out_room_iou_by_selection_rule", {})
        if sel_:
            f = go.Figure(go.Bar(x=list(sel_), y=list(sel_.values()), marker_color=["#9aa7bd", "#f0ad4e", "#2f6fed", "#1a9d55"][:len(sel_)], text=[f"{v:.3f}" for v in sel_.values()], textposition="outside"))
            f.update_layout(height=300, margin=dict(t=50, b=10), title=dict(text="How good is 'pick the best of K'? (room overlap, higher is better)", font=dict(size=16)), yaxis=dict(title="room IoU", range=[min(sel_.values()) * 0.9, max(sel_.values()) * 1.05]))
            tip("**Layout selection quality**", "On unseen plans: a random layout vs the one chosen by the hand-made score, the learned score, and the best possible pick (oracle). The closer 'learned' is to 'oracle', the better the selection."); wide(st.plotly_chart, f)
    with st.expander("📐 The maths behind it"):
        st.latex(r"\mathcal{L}_{\text{BoxNet}}=\sum_{j}\Big[\,5\,\lVert b-b^{*}\rVert_1+2\,(1-\mathrm{GIoU}(b,b^{*}))\,\Big]^{[6]}+8\,\mathcal{L}_{\text{adj}}+\mathcal{L}_{\text{tile}},\qquad \mathcal{L}=0.9\min_{h}\mathcal{L}^{(h)}+0.1\,\overline{\mathcal{L}^{(h)}}\ ^{[8]}")
        st.latex(r"\mathcal{L}_{\text{RefineNet}}=\frac{\sum_{p}w_p\,\mathrm{CE}(\hat y_p,y_p)}{\sum_p w_p}+3\,\mathrm{TV}_{\text{aniso}},\quad \mathrm{TV}_{\text{aniso}}=\overline{|P_{i+1,j}-P_{i,j}|}+\overline{|P_{i,j+1}-P_{i,j}|}\ ^{[13]}")
        st.latex(r"\text{score}(\ell)=\sum_k \lambda_k f_k(\ell),\qquad \ell^{\star}=\arg\max_{\ell\in\{\ell_1,\dots,\ell_K\}}\text{score}(\ell)")
        st.caption("L_adj: connected rooms must touch · L_tile: rooms must cover the outline without piling up · min over h: only the best of the noise samples is trained (winner-takes-all) · TV: favours straight, axis-parallel room borders · λ_k: learned by calibrate_score.py.")

# ------------------------------ results dashboard ------------------------------
res = ss.res
if res:
    cs = res["cands"]; best = cs[0]; df = pd.DataFrame([c["row"] for c in cs]); df.insert(0, "Rank", range(1, len(df) + 1)); st.divider(); ov = best["cats"]["Overall"]
    st.markdown(f"<div class='verdict'><h3>🏆 Best layout: #{best['cid']} &nbsp; <span class='stars'>{stars(ov)}</span> &nbsp; grade {grade(ov)} · {ov:.0f}/100</h3>"
                f"<span style='opacity:.8'>{len(cs)} layouts tried in {res['secs']:.1f}s · ranked with {'learned' if res['learned'] else 'hand-made'} weights</span></div>", unsafe_allow_html=True)
    i = st.selectbox("Layout to inspect", range(len(cs)), key="sel_box", help="Layouts are sorted best-first. Pick any to see it in 2D, 3D and in the gauges.",
                     format_func=lambda i: f"Rank {i + 1}  ·  #{cs[i]['cid']}  ·  {stars(cs[i]['cats']['Overall'])}  ({cs[i]['cats']['Overall']:.0f})" + ("   ★ best" if i == 0 else ""))
    c = cs[i]; s = c["stats"]; f_ = c["feats"]; cats = c["cats"]
    sent = {"Overall": f"{stars(cats['Overall'])} grade {grade(cats['Overall'])}", "Connectivity": f"{s['doors_placed']}/{s['doors_requested']} doors" + (f" +{s['doors_auto']} auto" if s["doors_auto"] else "") + (", all reachable" if s["all_connected"] else ", ⚠ cut-off rooms"),
            "Geometry": f"{s['straightness'] * 100:.0f}% straight walls", "Completeness": f"{s['rooms_present']}/{s['rooms']} rooms present",
            "Room sizes": f"~{int(round(100 * (np.exp(2 * f_['size_err']) - 1)))}% off requested"}
    gcols = st.columns([1.25, 1, 1, 1, 1])
    for col, name in zip(gcols, ["Overall", "Connectivity", "Geometry", "Completeness", "Room sizes"]):
        with col:
            st.markdown(f"**{name}**", help=CAT_HELP[name]); wide(st.plotly_chart, gauge(cats[name], name == "Overall"), config={"displayModeBar": False}); st.caption(sent[name])
    v2, v3, vl, vu, vx = st.tabs(["🖼 Plan", "🧊 3D view", "🏅 Leaderboard", "🔬 Details", "📦 Export"])
    png, html, js = exported(res, i)
    with v2:
        a, b = st.columns([2.3, 1])
        with a:
            ov_on = st.toggle("Show plan health (door links + size badges)", True, help="Draws the checks directly on the plan: green line = requested connection where the rooms touch, red dashed = rooms that do not touch. Badge under each room = how much bigger (▲) or smaller (▼) it is than you asked, with its area in m².")
            wide(st.plotly_chart, plan_fig(res, c, ov_on), config={"displayModeBar": False})
        with b:
            tip("**How to read it**", "Walls are black, doors pink, windows green, brown = entrance. With plan health on: 🟢 within 15% of the size you asked, 🟠 within 35%, 🔴 further off.")
            tip("**Rooms**", "Estimated real area of each room in square metres (assuming the boundary was drawn in metres) and its share of the floor area compared with what you asked for.")
            wide(st.dataframe, pd.DataFrame({"room": res["ids"], "area m²": np.round(c["areas"], 1), "got %": np.round(100 * c["got"], 1), "asked %": np.round(100 * res["target"], 1)}), hide_index=True)
            tip("**Connections**", "✓ = the two rooms share a wall in this layout, so a door can be cut. ✗ = they do not touch (the system may have added another door instead).")
            rows = [{"link": f"{ia} ⟷ {ib}", "type": "door" if k == "door" else "wall", "touch": "✓" if touching(c["inst"], res["ids"].index(ia), res["ids"].index(ib)) else "✗"} for ia, ib, k in res["cfg"]["connections"]]
            wide(st.dataframe, pd.DataFrame(rows, columns=["link", "type", "touch"]), hide_index=True)
    with v3:
        embed_html(html, 740); st.caption("Drag = rotate · wheel = zoom · Shift+drag = pan · 'Labels' button shows/hides room names · balconies get a glass railing, wooden deck, plants and a small table · needs internet once for the 3D library.")
    with vl:
        tip("**Leaderboard**", "All layouts sorted best-first with a star rating (5 stars = 90+/100). Press Open to inspect one in all the tabs, or compare any two side by side below.")
        cols = st.columns(4)
        for j, cj in enumerate(cs[:8]):
            with cols[j % 4]:
                st.markdown(f"<div class='card'><b>#{cj['cid']}</b> · rank {j + 1}{' 🏆' if j == 0 else ''}<br><span class='stars' style='font-size:18px'>{stars(cj['cats']['Overall'])}</span> <b>{cj['cats']['Overall']:.0f}</b></div>", unsafe_allow_html=True)
                wide(st.image, thumb(cj["label"]))
                st.markdown(f"<span class='badge'>🚪 {cj['row']['Doors']}</span><span class='badge'>📐 {cj['stats']['straightness'] * 100:.0f}% straight</span><span class='badge'>{'✓ reachable' if cj['stats']['all_connected'] else '✗ cut-off'}</span>", unsafe_allow_html=True)
                st.button("Open", key=f"insp{j}", on_click=lambda j=j: ss.__setitem__("sel_box", j), help="Open this layout in the Plan, 3D and Details tabs.")
        x, y = st.columns(2)
        ca = x.selectbox("Left layout", range(len(cs)), key="cmpA", format_func=lambda i: f"rank {i + 1} (#{cs[i]['cid']})", help="First layout for the side-by-side view.")
        cb = y.selectbox("Right layout", range(len(cs)), key="cmpB", format_func=lambda i: f"rank {i + 1} (#{cs[i]['cid']})", help="Second layout for the side-by-side view.")
        x.image(exported(res, ca)[0]); y.image(exported(res, cb)[0])
    with vu:
        tip("**What happened inside the pipeline**", "Numbers for each stage of this run, so you can see where a good or bad result comes from.")
        u1, u2, u3 = st.columns(3)
        with u1:
            st.markdown("**① BoxNet (Stage 1)**"); st.metric("Layouts proposed", len(cs), help="Number of different box arrangements the first network produced (one per random seed).")
            st.metric("Variety between layouts", "n/a" if res["diversity"] is None else f"{res['diversity']:.3f}", help="Average difference between the proposed box arrangements. 0 = all identical (trying several then cannot help). Higher = more different options.")
            st.metric("Stacked boxes", f"{np.mean([x['feats']['box_overlap'] for x in cs]):.3f}", help="How much the proposed boxes pile on top of each other (lower is better).")
        with u2:
            st.markdown("**② RefineNet (Stage 2)**"); st.metric("Follows Stage-1 boxes", f"{np.mean([x['feats']['box_consistency'] for x in cs]) * 100:.0f}%", help="How closely the final rooms follow the boxes from Stage 1, averaged over all layouts. Low means the networks disagree.")
            st.metric("Border smoothing", "on" if res["tta"] else "off", help="Test-time augmentation: 4 flipped/rotated passes averaged.")
        with u3:
            st.markdown("**③ Builder (no AI)**"); mth = pd.Series([x["stats"]["snapped"] for x in cs]).value_counts()
            st.metric("Wall straightening", ", ".join(f"{k} ×{v}" for k, v in mth.items()), help="'manhattan' = rooms redrawn as strict right-angle polygons (best). 'border' = fallback. 'none' = not applied.")
            st.metric("Doors added automatically", f"{np.mean([x['stats']['doors_auto'] for x in cs]):.1f} per layout", help="Doors the builder had to add because a requested connection could not be realised. Lower is better.")
        tip("**What the ranking rewards**", "The ranking score is a weighted sum of these properties. Green bars add to the score, red bars subtract. Learned weights come from calibrate_score.py (fitted to real quality on your data); otherwise they are hand-made.")
        wv = [float(res["w"][j]) for j in range(len(FEATS))]
        fw = go.Figure(go.Bar(x=wv, y=[FEAT_NAMES[k] for k in FEATS], orientation="h", marker_color=["#2e9e5b" if v >= 0 else "#d9534f" for v in wv]))
        fw.update_layout(height=320, margin=dict(t=10, b=10), yaxis=dict(autorange="reversed")); wide(st.plotly_chart, fw)
        with st.expander("All numbers (table)"): wide(st.dataframe, df, hide_index=True)
    with vx:
        tip("**Download**", "Save this layout. The JSON spec can be re-used with the command-line tool.")
        a, b, c3, d = st.columns(4); nm = res["cfg"]["name"]
        a.download_button("⬇ Spec (JSON)", json.dumps(res["cfg"], indent=1), f"{nm}_spec.json", help="The input spec for this run. Works with generate.py too.")
        b.download_button("⬇ Plan (PNG)", png, f"{nm}_{i + 1}.png", help="2D picture of the selected layout.")
        c3.download_button("⬇ 3D viewer (HTML)", html, f"{nm}_{i + 1}_3d.html", help="A standalone web page with the interactive 3D model. Open it in any browser.")
        d.download_button("⬇ Metrics (CSV)", df.to_csv(index=False), f"{nm}_metrics.csv", help="All layouts and their numbers in a spreadsheet-friendly file.")
        st.caption(f"All files of this run are also saved in `{res['dir']}`.")
else:
    st.info("Set up the boundary and rooms above, then press **🚀 Generate floor plans** (sidebar or step ③).")


# ------------------------------ references ------------------------------
st.divider(); st.subheader("📚 References & credits")
st.markdown("Design inspired by two-stage layout generation from a room graph (box proposal, then pixel refinement) in Graph2Plan $^{[2]}$; data from ResPlan $^{[1]}$. Superscripts in the maths above refer to this list.")
REFS = ["**[1]** *ResPlan: A Large-Scale Vector-Graph Dataset of 17,000 Residential Floor Plans.* arXiv:2508.14006, 2025 (dataset used for all training; code: github.com/m-agour/ResPlan).",
        "**[2]** R. Hu, Z. Huang, Y. Tang, O. van Kaick, H. Zhang, H. Huang. *Graph2Plan: Learning Floorplan Generation from Layout Graphs.* ACM Trans. Graph. 39(4), 2020.",
        "**[3]** O. Ronneberger, P. Fischer, T. Brox. *U-Net: Convolutional Networks for Biomedical Image Segmentation.* MICCAI, 2015 (RefineNet backbone).",
        "**[4]** A. Vaswani et al. *Attention Is All You Need.* NeurIPS, 2017 (BoxNet transformer).",
        "**[5]** W. Peebles, S. Xie. *Scalable Diffusion Models with Transformers.* ICCV, 2023 (adaptive layer-norm conditioning used for room sizes); E. Perez et al. *FiLM: Visual Reasoning with a General Conditioning Layer.* AAAI, 2018.",
        "**[6]** H. Rezatofighi et al. *Generalized Intersection over Union.* CVPR, 2019.",
        "**[7]** H. W. Kuhn. *The Hungarian method for the assignment problem.* Naval Research Logistics Quarterly 2, 1955; B. Weisfeiler, A. Leman, 1968 / N. Shervashidze et al., *Weisfeiler-Lehman Graph Kernels*, JMLR 2011 (matching of indistinguishable rooms).",
        "**[8]** A. Guzman-Rivera, D. Batra, P. Kohli. *Multiple Choice Learning: Learning to Produce Multiple Structured Outputs.* NeurIPS, 2012 (winner-takes-all training).",
        "**[9]** I. Loshchilov, F. Hutter. *Decoupled Weight Decay Regularization.* ICLR, 2019 (AdamW); L. N. Smith, N. Topin. *Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates.* 2019 (one-cycle schedule).",
        "**[10]** D. H. Douglas, T. K. Peucker. *Algorithms for the reduction of the number of points required to represent a digitized line.* Cartographica 10(2), 1973 (wall/room polygon simplification).",
        "**[11]** J. Wu, C. Fu, Y. Zheng, C.-L. Tai, J. Hang. *Data-driven interior plan generation for residential buildings* (RPLAN). ACM Trans. Graph. 38(6), 2019. — **[12]** N. Nauata et al. *House-GAN.* ECCV, 2020; M. A. Shabani, S. Hosseini, Y. Furukawa. *HouseDiffusion.* CVPR, 2023 (related work).",
        "**[13]** L. Rudin, S. Osher, E. Fatemi. *Nonlinear total variation based noise removal algorithms.* Physica D 60, 1992 (anisotropic TV regulariser).",
        "**Software:** PyTorch (Paszke et al., NeurIPS 2019), NumPy, SciPy, OpenCV, Shapely, Streamlit, Plotly, Three.js (r128, 3D viewer)."]
for r_ in REFS: st.markdown(r_)
with st.expander("BibTeX for the main references"):
    st.code("""@article{resplan2025, title={ResPlan: A Large-Scale Vector-Graph Dataset of 17,000 Residential Floor Plans}, journal={arXiv:2508.14006}, year={2025}}
@article{hu2020graph2plan, title={Graph2Plan: Learning Floorplan Generation from Layout Graphs}, author={Hu, Ruizhen and Huang, Zeyu and Tang, Yuhan and van Kaick, Oliver and Zhang, Hao and Huang, Hui}, journal={ACM Transactions on Graphics}, volume={39}, number={4}, year={2020}}
@inproceedings{ronneberger2015unet, title={U-Net: Convolutional Networks for Biomedical Image Segmentation}, author={Ronneberger, Olaf and Fischer, Philipp and Brox, Thomas}, booktitle={MICCAI}, year={2015}}
@inproceedings{vaswani2017attention, title={Attention Is All You Need}, author={Vaswani, Ashish and others}, booktitle={NeurIPS}, year={2017}}
@inproceedings{rezatofighi2019giou, title={Generalized Intersection over Union}, author={Rezatofighi, Hamid and others}, booktitle={CVPR}, year={2019}}""", language="bibtex")