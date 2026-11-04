"""Step 2a: train BoxNet (graph + boundary + optional sizes -> room boxes).  Minutes on a GPU.
  python train_box.py --epochs 200 --d 256 --heads 8 --layers 6          (continue after a stop: add --resume)
v3.2 changes: AdaLN size conditioning, multi-hypothesis (winner-takes-all) training so the noise z really produces different
layouts, GT-anchored tiling loss, curriculum for size noise / p(known) / deep-supervision weights, 3 validation modes."""
import argparse, json, time
from pathlib import Path
import numpy as np, torch
from common import ROOT, DEFAULT_SIZE
from models import (BoxNet, dihedral, align_gt, box_loss, adjacency_loss, pairwise_overlap_loss, tiling_loss, noisy_share, group_sig)

ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=200); ap.add_argument("--batch", type=int, default=128)
ap.add_argument("--lr", type=float, default=6e-4); ap.add_argument("--d", type=int, default=192)
ap.add_argument("--heads", type=int, default=6, help="d must be divisible by heads (d=256 -> --heads 8)")
ap.add_argument("--layers", type=int, default=4); ap.add_argument("--patience", type=int, default=40)
ap.add_argument("--adj", type=float, default=8.0, help="weight: connected rooms must touch")
ap.add_argument("--tile", type=float, default=1.0, help="weight: GT-anchored tiling loss (coverage / containment / stacking)")
ap.add_argument("--overlap", type=float, default=0.0, help="weight: pairwise box repulsion (measured non-informative on ResPlan -> off)")
ap.add_argument("--hypotheses", type=int, default=4, help="N noise samples per input; only the best one gets the full loss (multiple-choice learning)")
ap.add_argument("--sigma-max", type=float, default=0.25, help="final size-noise sigma (ramps 0 -> max over the first 30%% of epochs)")
ap.add_argument("--max-minutes", type=float, default=0); ap.add_argument("--resume", action="store_true")
ap.add_argument("--ckpt-dir", default=str(ROOT / "checkpoints")); ap.add_argument("--log-dir", default=str(ROOT / "logs"))
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
assert a.d % a.heads == 0, "d must be divisible by heads"
CK, LG = Path(a.ckpt_dir), Path(a.log_dir); CK.mkdir(parents=True, exist_ok=True); LG.mkdir(parents=True, exist_ok=True)
torch.manual_seed(a.seed); np.random.seed(a.seed)
dev = "cuda" if torch.cuda.is_available() else "cpu"; print("device:", dev, torch.cuda.get_device_name(0) if dev == "cuda" else "")


def load(name):
    z = np.load(ROOT / "data" / f"cache4_{name}.npz")
    t = lambda k, dt: torch.from_numpy(z[k]).to(dev).to(dt)
    return dict(cond=t("cond", torch.uint8), nt=t("ntype", torch.long), en=t("entr", torch.float32), ed=t("edge", torch.uint8),
                sig=t("sig", torch.long), bx=t("boxes", torch.float32), sh=t("share", torch.float32))


tr, va = load("train"), load("val"); N = len(tr["cond"]); print("train", N, "val", len(va["cond"]))
model = BoxNet(a.d, a.heads, a.layers).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
spe = N // a.batch; sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * spe, pct_start=0.1)
DW = torch.tensor([DEFAULT_SIZE[t] for t in range(7)], device=dev)
rep = lambda x, n: x.repeat(n, *([1] * (x.dim() - 1)))                       # (B,..) -> (n*B,..)


def schedules(ep):
    p = (ep - 1) / max(1, a.epochs - 1)
    return dict(sigma=a.sigma_max * min(1.0, p / 0.3),        # size noise: clean sizes first, then robustify
                p_known=0.9 - 0.3 * p,                        # how often sizes are given: 90% -> 60%
                w_early=0.8 - 0.6 * p,                        # deep supervision: strong early guidance, then focus on last layer
                lam=min(1.0, p / 0.2))                        # aux layout losses fade in once boxes are roughly right


@torch.no_grad()
def evaluate(mode, nsamp=1):
    """mode: 'true' exact sizes | 'default' type-median sizes (what a user without sizes gets) | 'none'. nsamp>1 -> oracle best-of-n IoU."""
    model.eval(); tot = 0.0; cnt = 0
    for i in range(0, len(va["cond"]), 256):
        s = slice(i, i + 256); nt = va["nt"][s]; v = nt >= 0; B = len(nt); cond = va["cond"][s].float()
        if mode == "default": sh = DW[nt.clamp(min=0)] * v; sh = sh / sh.sum(1, keepdim=True)
        else: sh = va["sh"][s]
        kn = torch.full((B,), mode != "none", device=dev); gs = group_sig(va["sig"][s], sh, kn); best = torch.zeros(B, device=dev)
        for _ in range(nsamp):
            p = model(cond, nt, va["en"][s], va["ed"][s], sh, kn)
            best = torch.max(best, box_loss(p, align_gt(p, va["bx"][s], gs), v, reduce=False)[1])
        tot += best.sum().item(); cnt += B
    return tot / cnt


start, best_score, bad = 1, 0.0, 0
if a.resume and (CK / "box_last.pt").exists():
    s = torch.load(CK / "box_last.pt", map_location=dev)
    model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"]); sched.load_state_dict(s["sched"])
    start, best_score, bad = s["epoch"] + 1, s["best"], s["bad"]
    if s["epochs"] != a.epochs: print("WARNING: --epochs forced to", s["epochs"]); a.epochs = s["epochs"]
    print("resumed from epoch", s["epoch"])
csvp = LG / "box_log.csv"
if start == 1 or not csvp.exists(): csvp.write_text("epoch,loss,val_iou_true_sizes,val_iou_default_sizes,val_iou_no_sizes,val_oracle_best4,lr,seconds\n")
T0 = time.time()
for ep in range(start, a.epochs + 1):
    model.train(); perm = torch.randperm(N, device=dev); t0 = time.time(); run = 0.0; S = schedules(ep); Nh = a.hypotheses
    for i in range(spe):
        ix = perm[i * a.batch:(i + 1) * a.batch]
        cond, bx, _ = dihedral(tr["cond"][ix].float(), tr["bx"][ix], None, np.random.randint(4), np.random.rand() < 0.5)
        nt, en, ed, sg = tr["nt"][ix], tr["en"][ix], tr["ed"][ix], tr["sig"][ix]; v = nt >= 0
        kn = torch.rand(len(ix), device=dev) < S["p_known"]; sh = noisy_share(tr["sh"][ix], v, S["sigma"])
        R = lambda x: rep(x, Nh)                                                   # N noise hypotheses per plan
        outs = model(R(cond), R(nt), R(en), R(ed), R(sh), R(kn), return_all=True)
        gt = align_gt(outs[-1], R(bx), R(group_sig(sg, sh, kn)))
        per = 0.0
        for j, o in enumerate(outs):                                                # deep supervision, per-plan losses (N*B,)
            w = 1.0 if j == len(outs) - 1 else S["w_early"]
            per = per + w * (box_loss(o, gt, R(v), False)[0] + a.adj * adjacency_loss(o, R(ed), R(v), False))
        if S["lam"] > 0 and (a.tile > 0 or a.overlap > 0):
            aux = 0.0
            if a.tile > 0: aux = aux + a.tile * tiling_loss(outs[-1], R(nt), R(cond[:, 0]), ref=gt, reduce=False)
            if a.overlap > 0: aux = aux + a.overlap * pairwise_overlap_loss(outs[-1], R(v), reduce=False)
            per = per + S["lam"] * aux
        per = per.view(Nh, -1)                                                      # winner-takes-all over hypotheses
        loss = 0.9 * per.min(0).values.mean() + 0.1 * per.mean() if Nh > 1 else per.mean()
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); run += loss.item()
    i_t, i_d, i_n = evaluate("true"), evaluate("default"), evaluate("none"); i_o = evaluate("default", 4); lr = opt.param_groups[0]["lr"]
    score = (i_t + i_d + i_n) / 3
    print(f"epoch {ep:3d}/{a.epochs}  loss {run/spe:.4f}  val IoU: true-sizes {i_t:.4f} default-sizes {i_d:.4f} no-sizes {i_n:.4f} | oracle best-of-4 {i_o:.4f}  ({time.time()-t0:.1f}s)", flush=True)
    with open(csvp, "a") as f: f.write(f"{ep},{run/spe:.5f},{i_t:.5f},{i_d:.5f},{i_n:.5f},{i_o:.5f},{lr:.6f},{time.time()-t0:.1f}\n")
    if score > best_score:
        best_score, bad = score, 0
        torch.save({"model": model.state_dict(), "d": a.d, "heads": a.heads, "layers": a.layers, "val_iou": score, "epoch": ep}, CK / "box_best.pt")
    else: bad += 1
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "epoch": ep,
                "best": best_score, "bad": bad, "epochs": a.epochs}, CK / "box_last.pt")
    if bad >= a.patience: print("early stop"); break
    if a.max_minutes and (time.time() - T0) / 60 > a.max_minutes: print("time budget reached; use --resume"); break
(LG / "box_last_run.json").write_text(json.dumps({"best_val_iou_mean3": best_score, "finished": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1))
print("best mean val box IoU:", round(best_score, 4), "->", CK / "box_best.pt")
