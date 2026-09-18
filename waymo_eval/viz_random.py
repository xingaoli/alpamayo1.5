#!/usr/bin/env python3
"""Render the SAME random (clip, t0) samples from two eval runs, side by side.

Both runs must share the eval_traj.py store format (preds.npz + per_frame.csv).
Samples are drawn once (seeded) from the common index — typically the two
stores have identical (clip_id, t0) lists, so every rendered pair matches
1:1 and lands under the same filename in each run's viz dir:

    waymo_eval/viz_outputs/<run-a-name>/<clip>_t<..>s.png
    waymo_eval/viz_outputs/<run-b-name>/<clip>_t<..>s.png
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from rds_dataset import RDSWaymoDataset
from viz_traj import render_sample, load_eval_store

_REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Render paired random samples from two eval runs.")
    p.add_argument("--run-a", required=True, metavar="DIR",
                   help="eval_traj output dir of model A (e.g. base model)")
    p.add_argument("--run-b", required=True, metavar="DIR",
                   help="eval_traj output dir of model B (e.g. sft model)")
    p.add_argument("-n", "--num", type=int, default=20,
                   help="number of random samples to render")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split", default="val")
    p.add_argument("--root", default=None)
    p.add_argument("--out-root", default=None,
                   help="default: waymo_eval/viz_outputs")
    return p.parse_args()


def main():
    args = parse_args()
    run_a, run_b = Path(args.run_a), Path(args.run_b)
    ids_a, t0s_a, preds_a, gts_a, metrics_a = load_eval_store(run_a)
    ids_b, t0s_b, preds_b, gts_b, metrics_b = load_eval_store(run_b)

    # pairing: sample indices valid in BOTH stores (identical order in practice)
    keys_a = list(zip(ids_a, t0s_a.tolist()))
    keys_b = list(zip(ids_b, t0s_b.tolist()))
    if keys_a == keys_b:
        common = list(range(len(keys_a)))
    else:
        kb = {k: i for i, k in enumerate(keys_b)}
        common = [i for i, k in enumerate(keys_a)
                  if k in kb and preds_b[kb[k]].shape == preds_a[i].shape]
        if not common:
            raise SystemExit("the two runs share no (clip_id, t0) samples")
    print(f"paired samples: {len(common)} common of "
          f"{len(keys_a)}/{len(keys_b)}")

    rng = np.random.default_rng(args.seed)
    pick = rng.choice(common, size=min(args.num, len(common)), replace=False)
    # render grouped by clip so the per-clip tar caches stay warm
    pick = sorted(pick, key=lambda i: (ids_a[i], int(t0s_a[i])))
    print(f"rendering {len(pick)} random sample(s), seed={args.seed}")

    out_root = Path(args.out_root) if args.out_root else (
        _REPO_ROOT / "waymo_eval" / "viz_outputs")
    out_a = out_root / run_a.name
    out_b = out_root / run_b.name
    out_a.mkdir(parents=True, exist_ok=True)
    out_b.mkdir(parents=True, exist_ok=True)

    dataset = RDSWaymoDataset(
        root=args.root, split=args.split,
        clip_ids=sorted({ids_a[i] for i in pick}), preload=False)
    clip = None
    for i in pick:
        cid, t0 = ids_a[i], int(t0s_a[i])
        if clip is None or clip.clip_id != cid:
            clip = dataset._get_clip(cid)
        for out_dir, ids, t0s, preds, gts, metrics, run in (
                (out_a, ids_a, t0s_a, preds_a, gts_a, metrics_a, run_a),
                (out_b, ids_b, t0s_b, preds_b, gts_b, metrics_b, run_b)):
            m = metrics.get((cid, t0), {})
            panel = "  ".join(f"{k}={float(m[k]):.2f}" for k in
                              ("mADE@1s", "mADE@3s", "mADE@5s", "mFDE@5s")
                              if k in m) or None
            render_sample(clip, t0, preds[i], gts[i], panel, out_dir,
                          panel_label=f"EVAL ({run.name})")
    print(f"\ndone: {len(pick)} paired sample(s)")
    print(f"  A -> {out_a}")
    print(f"  B -> {out_b}")


if __name__ == "__main__":
    main()
