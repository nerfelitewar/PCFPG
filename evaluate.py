"""Evaluate the full pipeline on the TEST split -> logs/eval_test.json
 Stage 1 (box IoU vs real boxes):  true_sizes (oracle sizes) | default_sizes (what a user without sizes gets) | no_sizes
     -> compare true vs no_sizes to see whether size conditioning works at all (default_sizes carries no info beyond the room types).
 oracle_best_of_8 / sample_diversity : do different noise samples give different layouts?  (diversity ~0 => best-of-K cannot help)
 Stage 2: room_iou_gt_box (upper bound), room_iou_e2e (Stage 1 -> Stage 2)
 Layout k=1 vs best_of_K: doors_placed, all_rooms_connected, plans_with_all_rooms, straightness (real ResPlan ~0.98), rectangularity (real ~0.92)"""
import argparse, json, numpy as np, torch
from common import ROOT, M, DEFAULT_SIZE
from generate import Gen
from models import refine_input, align_gt, box_loss, group_sig

ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=100); ap.add_argument("--k", type=int, default=8); ap.add_argument("--no-tta", action="store_true")
ap.add_argument("--box-ckpt", default=str(ROOT / "checkpoints" / "box_best.pt")); ap.add_argument("--ref-ckpt", default=str(ROOT / "checkpoints" / "refine_best.pt"))
a = ap.parse_args(); g = Gen(a.box_ckpt, a.ref_ckpt); dev = g.dev; tta = not a.no_tta
z = np.load(ROOT / "data" / "cache4_test.npz"); n = min(a.n, len(z["cond"]))
T_ = lambda k, dt: torch.from_numpy(z[k][:n]).to(dev).to(dt)
cond, nt, en, ed, sg, bx, inst, true_sh = T_("cond", torch.float32), T_("ntype", torch.long), T_("entr", torch.float32), T_("edge", torch.long), T_("sig", torch.long), T_("boxes", torch.float32), T_("inst", torch.long), T_("share", torch.float32)
v = nt >= 0
dw = torch.tensor([DEFAULT_SIZE[t] for t in range(7)], device=dev); def_sh = dw[nt.clamp(min=0)] * v; def_sh = def_sh / def_sh.sum(1, keepdim=True)


def room_iou(p, gt, ntb):
    r = []
    for k in range(1, M + 1):
        pk, gk = p == k, gt == k; vv = ntb[:, k - 1] >= 0
        r.append(((pk & gk).sum((1, 2)).float() / (pk | gk).sum((1, 2)).clamp(min=1))[vv])
    return torch.cat(r).mean().item()


res = {}
with torch.no_grad():
    for tag, sh, known in (("true_sizes", true_sh, True), ("default_sizes", def_sh, True), ("no_sizes", def_sh, False)):
        torch.manual_seed(0); kn = torch.full((n,), known, device=dev); pb = g.box(cond, nt, en, ed, sh, kn)
        res["box_iou_" + tag] = box_loss(pb, align_gt(pb, bx, group_sig(sg, sh, kn)), v)[1]
        if tag == "default_sizes": pb_e2e = pb
    kn = torch.ones(n, dtype=torch.bool, device=dev); gs = group_sig(sg, def_sh, kn); samples = []; ious = []
    for s in range(a.k):                                                                # K noise samples per plan
        torch.manual_seed(100 + s); p = g.box(cond, nt, en, ed, def_sh, kn); samples.append(p)
        ious.append(box_loss(p, align_gt(p, bx, gs), v, False)[1])
    ious = torch.stack(ious)
    res[f"box_iou_oracle_best_of_{a.k}"] = ious.max(0).values.mean().item(); res["box_iou_random_sample"] = ious.mean().item()
    res["sample_diversity_L1"] = ((samples[0] - samples[1]).abs().sum(-1) * v).sum().item() / v.sum().item() if len(samples) > 1 else 0.0
    for tag, b in (("gt_box", bx), ("e2e", pb_e2e)):
        r = []
        for i in range(0, n, 64):
            s = slice(i, i + 64); r.append(room_iou(g.refine(cond[s], b[s], nt[s], tta).argmax(1), inst[s], nt[s]) * len(nt[s]))
        res["room_iou_" + tag] = sum(r) / n


def layout_stats(k):
    S = dict(req=0, placed=0, auto=0, conn=0, rooms=0, present=0, full=0, rect=0.0, win=0, straight=0.0)
    for i in range(n):
        sh = def_sh[i].cpu().numpy(); _, _, s = g(z["cond"][i].astype(np.float32), z["ntype"][i], z["entr"][i], z["edge"][i], None, sh, True, seed=i, k=k, tta=tta)
        S["req"] += s["doors_requested"]; S["placed"] += s["doors_placed"]; S["auto"] += s["doors_auto"]; S["conn"] += s["all_connected"]
        S["rooms"] += s["rooms"]; S["present"] += s["rooms_present"]; S["full"] += int(s["rooms_present"] == s["rooms"]); S["rect"] += s["rectangularity"]
        S["win"] += s["windows"]; S["straight"] += s["straightness"]
    return {"doors_placed": round(S["placed"] / max(1, S["req"]), 4), "auto_doors_per_plan": round(S["auto"] / n, 2), "all_rooms_connected": round(S["conn"] / n, 4),
            "rooms_present": round(S["present"] / S["rooms"], 4), "plans_with_all_rooms": round(S["full"] / n, 4), "straightness": round(S["straight"] / n, 4),
            "rectangularity": round(S["rect"] / n, 4), "avg_windows": round(S["win"] / n, 2)}


out = {"test_plans": n, "learned_score_weights": g.learned, "tta": tta, **{k: round(float(x), 4) for k, x in res.items()},
       "k=1": layout_stats(1), f"best_of_{a.k}": layout_stats(a.k)}
print(json.dumps(out, indent=1)); (ROOT / "logs").mkdir(exist_ok=True); (ROOT / "logs" / "eval_test.json").write_text(json.dumps(out, indent=1))