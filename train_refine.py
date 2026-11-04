"""Step 2b: train RefineNet (room boxes + boundary -> per-room segmentation). GPU recommended.
  python train_refine.py --epochs 40          (continue after a stop: add --resume)"""
import argparse, copy, json, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from common import ROOT, S, M
from models import RefineNet, BoxNet, dihedral, refine_input, align_perm, relabel_inst, noisy_share, group_sig, edge_weights, aniso_tv

ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--batch", type=int, default=48)
ap.add_argument("--lr", type=float, default=2e-3); ap.add_argument("--base", type=int, default=32)
ap.add_argument("--jitter", type=float, default=0.04, help="box noise (fraction of canvas) so it tolerates BoxNet errors")
ap.add_argument("--ema", type=float, default=0.999); ap.add_argument("--patience", type=int, default=12)
ap.add_argument("--max-minutes", type=float, default=0); ap.add_argument("--resume", action="store_true")
ap.add_argument("--ckpt-dir", default=str(ROOT / "checkpoints")); ap.add_argument("--log-dir", default=str(ROOT / "logs"))
ap.add_argument("--edge-w", type=float, default=2.0, help="extra CE weight on pixels next to room borders (sharper borders)")
ap.add_argument("--tv", type=float, default=3.0, help="weight of anisotropic TV on class probabilities (straight, axis-parallel borders)")
ap.add_argument("--soft", type=float, default=0.02, help="max box-edge softness used in 50%% of batches (robustness to imprecise boxes)")
ap.add_argument("--init", default="", help="start from these weights, e.g. checkpoints/refine_best.pt (fine-tuning)")
ap.add_argument("--boxnet", default="", help="box_best.pt: also train on BoxNet's OWN predicted boxes (closes the gap to real boxes)")
ap.add_argument("--pred-frac", type=float, default=0.6, help="share of batches that use predicted boxes (when --boxnet is given)")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
CK, LG = Path(a.ckpt_dir), Path(a.log_dir); CK.mkdir(parents=True, exist_ok=True); LG.mkdir(parents=True, exist_ok=True)
torch.manual_seed(a.seed); np.random.seed(a.seed)
dev = "cuda" if torch.cuda.is_available() else "cpu"; amp = dev == "cuda"
if amp: torch.backends.cudnn.benchmark = True; torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
print("device:", dev, torch.cuda.get_device_name(0) if amp else "")


def load(name):
    z = np.load(ROOT / "data" / f"cache4_{name}.npz")
    t = lambda k, dt: torch.from_numpy(z[k]).to(dev).to(dt)
    return dict(cond=t("cond", torch.uint8), inst=t("inst", torch.uint8), nt=t("ntype", torch.long), bx=t("boxes", torch.float32),
                en=t("entr", torch.float32), ed=t("edge", torch.uint8), sig=t("sig", torch.long), sh=t("share", torch.float32))


tr, va = load("train"), load("val"); N = len(tr["cond"]); print("train", N, "val", len(va["cond"]))
model = RefineNet(a.base).to(dev).to(memory_format=torch.channels_last)
bnet = None
if a.boxnet:
    bk = torch.load(a.boxnet, map_location=dev); bnet = BoxNet(bk.get("d", 192), bk.get("heads", 6), bk.get("layers", 4)).to(dev)
    bnet.load_state_dict(bk["model"]); bnet.eval(); print("using BoxNet predictions from", a.boxnet)
if a.init:
    model.load_state_dict(torch.load(a.init, map_location=dev)["model"]); print("initialised from", a.init)
ema = copy.deepcopy(model).eval()
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
spe = N // a.batch; sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * spe, pct_start=0.15)


@torch.no_grad()
def ema_update(d):
    for pe, p in zip(ema.parameters(), model.parameters()): pe.mul_(d).add_(p.detach(), alpha=1 - d)
    for be, b in zip(ema.buffers(), model.buffers()): be.copy_(b)


def predicted(cond, nt, en, ed, sig, sh, bx, inst):
    """BoxNet boxes (+ GT instance map relabelled to the same slots)."""
    kn = torch.rand(len(nt), device=dev) < 0.5; s2 = noisy_share(sh, nt >= 0)
    pb = bnet(cond, nt, en, ed, s2, kn); return pb, relabel_inst(inst, align_perm(pb, bx, group_sig(sig, s2, kn)))


@torch.no_grad()
def evaluate(m, mode):
    """mode: 'gt' real boxes | 'jit' jittered boxes | 'pred' BoxNet boxes"""
    m.eval(); acc = tot = 0; ious = []
    for i in range(0, len(va["cond"]), 64):
        s = slice(i, i + 64); nt = va["nt"][s]; bx = va["bx"][s]; g = va["inst"][s].long(); c = va["cond"][s].float()
        if mode == "jit": bx = (bx + torch.randn_like(bx) * a.jitter).clamp(0, 1)
        if mode == "pred": bx, g = predicted(c, nt, va["en"][s], va["ed"][s], va["sig"][s], va["sh"][s], bx, g)
        x = refine_input(c, bx, nt).contiguous(memory_format=torch.channels_last)
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=amp): p = m(x).argmax(1)
        acc += (p == g).sum().item(); tot += g.numel()
        for k in range(1, M + 1):
            v = (nt[:, k - 1] >= 0); pk, gk = p == k, g == k
            ious.append(((pk & gk).sum((1, 2)).float() / (pk | gk).sum((1, 2)).clamp(min=1).float())[v])
    return acc / tot, torch.cat(ious).mean().item()


start, best, bad = 1, 0.0, 0
if a.resume and (CK / "refine_last.pt").exists():
    s = torch.load(CK / "refine_last.pt", map_location=dev)
    model.load_state_dict(s["model"]); ema.load_state_dict(s["ema"]); opt.load_state_dict(s["opt"]); sched.load_state_dict(s["sched"])
    start, best, bad = s["epoch"] + 1, s["best"], s["bad"]
    if s["epochs"] != a.epochs: print("WARNING: --epochs forced to", s["epochs"]); a.epochs = s["epochs"]
    print("resumed from epoch", s["epoch"])
csvp = LG / "refine_log.csv"
if start == 1 or not csvp.exists(): csvp.write_text("epoch,loss,val_acc,val_roomIoU,val_roomIoU_jitter,lr,seconds\n")
T0 = time.time()
for ep in range(start, a.epochs + 1):
    model.train(); perm = torch.randperm(N, device=dev); t0 = time.time(); run = 0.0
    for i in range(spe):
        ix = perm[i * a.batch:(i + 1) * a.batch]
        cond, bx, inst = dihedral(tr["cond"][ix].float(), tr["bx"][ix], tr["inst"][ix].long(), np.random.randint(4), np.random.rand() < 0.5)
        if bnet is not None and np.random.rand() < a.pred_frac:
            with torch.no_grad(): bx, inst = predicted(cond, tr["nt"][ix], tr["en"][ix], tr["ed"][ix], tr["sig"][ix], tr["sh"][ix], bx, inst)
        else:
            bx = (bx + torch.randn_like(bx) * a.jitter * torch.rand(len(bx), 1, 1, device=dev)).clamp(0, 1)   # varied noise level
        soft = float(np.random.uniform(0.004, a.soft)) if (a.soft > 0 and np.random.rand() < 0.5) else 0.0
        x = refine_input(cond, bx, tr["nt"][ix], soft).contiguous(memory_format=torch.channels_last)
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=amp): logits = model(x)
        ce = F.cross_entropy(logits.float(), inst, reduction="none"); w = edge_weights(inst, a.edge_w) if a.edge_w > 0 else torch.ones_like(ce)
        loss = (ce * w).sum() / w.sum() + (a.tv * aniso_tv(logits) if a.tv > 0 else 0.0)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0); opt.step(); sched.step()
        ema_update(min(a.ema, (1 + ep * spe) / (10 + ep * spe))); run += loss.item() if i % 10 == 0 else 0
    acc, iou = evaluate(ema, "gt"); _, iouj = evaluate(ema, "pred" if bnet is not None else "jit"); lr = opt.param_groups[0]["lr"]
    L = run / max(1, (spe + 9) // 10)
    print(f"epoch {ep:3d}/{a.epochs}  loss {L:.4f}  val_acc {acc:.4f}  roomIoU {iou:.4f}  roomIoU({'predicted boxes' if bnet is not None else 'jitter'}) {iouj:.4f}  ({time.time()-t0:.0f}s)", flush=True)
    with open(csvp, "a") as f: f.write(f"{ep},{L:.5f},{acc:.5f},{iou:.5f},{iouj:.5f},{lr:.6f},{time.time()-t0:.1f}\n")
    if iouj > best:
        best, bad = iouj, 0; torch.save({"model": ema.state_dict(), "base": a.base, "val_iou": iouj, "epoch": ep}, CK / "refine_best.pt")
    else: bad += 1
    torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                "epoch": ep, "best": best, "bad": bad, "epochs": a.epochs}, CK / "refine_last.pt")
    if bad >= a.patience: print("early stop"); break
    if a.max_minutes and (time.time() - T0) / 60 > a.max_minutes: print("time budget reached; use --resume"); break
(LG / "refine_last_run.json").write_text(json.dumps({"best_val_roomIoU_jitter": best, "finished": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1))
print("best room IoU:", round(best, 4), "->", CK / "refine_best.pt")
