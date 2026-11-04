"""Ablation runner. Trains BoxNet variants with ONE factor changed, then evaluates each with bench_full.py (same RefineNet, same test plans).
  python ablate.py --print            # show the commands
  python ablate.py --run --epochs 200 # run them (hours on one GPU). Use the SAME epochs as your main model.
Evaluation-only variants (no training): K=1 and no-TTA are already in the main table / run: python bench_full.py --tag abl_no_tta --no-tta ..."""
import argparse, subprocess, sys
ap = argparse.ArgumentParser(); ap.add_argument("--run", action="store_true"); ap.add_argument("--print", action="store_true"); ap.add_argument("--epochs", type=int, default=200)
ap.add_argument("--n", type=int, default=1632); ap.add_argument("--k", type=int, default=8); a = ap.parse_args(); py = sys.executable
V = {"no_multi_hypothesis": ["--hypotheses", "1"], "no_tiling_loss": ["--tile", "0"], "no_adjacency_loss": ["--adj", "0"]}
cmds = []
for name, fl in V.items():
    d = f"checkpoints/abl_{name}"; cmds.append([py, "train_box.py", "--epochs", str(a.epochs), "--ckpt-dir", d, "--log-dir", f"logs/abl_{name}"] + fl)
    cmds.append([py, "bench_full.py", "--n", str(a.n), "--k", str(a.k), "--tag", f"abl_{name}", "--box-ckpt", f"{d}/box_best.pt", "--scenarios", "default"])
cmds.append([py, "bench_full.py", "--n", str(a.n), "--k", str(a.k), "--tag", "abl_no_tta", "--no-tta", "--scenarios", "default"])
for c in cmds:
    print(" ".join(c))
    if a.run: subprocess.run(c, check=True)