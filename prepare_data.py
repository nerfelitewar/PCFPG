"""Step 1: ResPlan.pkl -> data/cache4_{train,val,test}.npz  (rooms, connectivity graph, boxes, instance maps)"""
import argparse, json, pickle, time
from pathlib import Path
import numpy as np
from common import DEFAULT_DATA_DIR, ROOT, ROOM_TYPES
from graphdata import plan_to_sample

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="folder with ResPlan.pkl and split.json")
ap.add_argument("--limit", type=int, default=0, help="max plans per split (0 = all)")
ap.add_argument("--use-aug", action="store_true", help="add the 683 'augmented' plans to train")
a = ap.parse_args()
d = Path(a.data_dir); t0 = time.time()
plans = pickle.load(open(d / "ResPlan.pkl", "rb"))
split = json.load(open(d / "split.json")); by_id = {p["id"]: p for p in plans}
print(f"loaded {len(plans)} plans in {time.time()-t0:.0f}s")
(ROOT / "data").mkdir(exist_ok=True)
areas = {t: [] for t in range(len(ROOM_TYPES))}
for name in ["train", "val", "test"]:
    ids = list(split[name]) + (list(split["augmented"]) if (a.use_aug and name == "train") else [])
    if a.limit: ids = ids[:a.limit]
    out, kept, skipped = {}, [], 0
    for i in ids:
        if i not in by_id: continue
        try: s = plan_to_sample(by_id[i])
        except Exception as e: print("skip", i, repr(e)[:80]); s = None
        if s is None: skipped += 1; continue
        for k, v in s.items(): out.setdefault(k, []).append(v)
        kept.append(i)
        if name == "train":
            nt = s["ntype"]
            for k in range(int((nt >= 0).sum())): areas[int(nt[k])].append(float(s["areas"][k]))
    arr = {k: np.stack(v) for k, v in out.items() if k != "areas"}
    np.savez_compressed(ROOT / "data" / f"cache4_{name}.npz", ids=np.array(kept), **arr)
    print(f"{name}: kept {len(kept)}, skipped {skipped} (>16 rooms or <2)")
json.dump({ROOM_TYPES[t]: float(np.median(v)) if v else 0.1 for t, v in areas.items()},
          open(ROOT / "data" / "stats.json", "w"), indent=1)
print("done in %.0fs" % (time.time() - t0))
