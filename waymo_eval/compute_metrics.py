#!/usr/bin/env python3
"""Offline metric computation from stored eval predictions.

Reads the preds.npz written by eval_traj.py — preds [N,S,T,3], gts [N,T,3],
clip_ids, t0_us, all in the t0-local frame — and recomputes ADE/FDE/mADE/mFDE
at arbitrary horizons. No model, no GPU: run the eval once, then recompute
metrics as often as needed. (Box-based metrics like collision rate: later;
ego yaw comes from trajectory tangents, agents re-read from the rds set via
(clip_id, t0_us).)

Usage:
  python compute_metrics.py waymo_eval/eval_outputs/val_xxx
  python compute_metrics.py waymo_eval/eval_outputs/val_xxx --horizons 1,2,4
  python compute_metrics.py waymo_eval/eval_outputs/val_xxx --out-dir DIR
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, Path(__file__).resolve().parent)

import argparse
import csv
import json

import numpy as np

from eval_traj import frame_metrics, horizon_steps


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline metrics from preds.npz.")
    p.add_argument("eval_dir",
                   help="eval output dir containing preds.npz (or the npz path)")
    p.add_argument("--horizons", default="1,3,5",
                   help="comma-separated horizon seconds (plus full)")
    p.add_argument("--out-dir", default=None,
                   help="default: write per_frame.csv + summary.json back into "
                        "the eval dir (overwrites)")
    return p.parse_args()


def main():
    args = parse_args()
    src = Path(args.eval_dir)
    npz_path = src if src.suffix == ".npz" else src / "preds.npz"
    data = np.load(npz_path)
    preds, gts = data["preds"], data["gts"]
    clip_ids, t0s = data["clip_ids"], data["t0_us"]

    horizons = horizon_steps(
        [float(x) for x in args.horizons.split(",") if x.strip()],
        num_future_steps=gts.shape[1], time_step=0.1)

    rows = []
    for pred, gt, cid, t0 in zip(preds, gts, clip_ids, t0s):
        m = frame_metrics(pred, gt, horizons)
        rows.append({"clip_id": str(cid), "t0_us": int(t0),
                     "n_samples": int(pred.shape[0]), **m})

    out_dir = Path(args.out_dir) if args.out_dir else npz_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    metric_keys = [k for k in rows[0] if "@" in k]
    csv_path = out_dir / "per_frame.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def stats(vals: np.ndarray) -> dict:
        return {"mean": float(vals.mean()), "median": float(np.median(vals)),
                "p90": float(np.percentile(vals, 90))}

    # carry the original run config forward when recomputing in place
    config = {}
    orig = npz_path.parent / "summary.json"
    if out_dir == npz_path.parent and orig.exists():
        config = json.load(open(orig)).get("config", {})
    config["horizons"] = args.horizons

    summary = {
        "config": config,
        "n_clips": len({r["clip_id"] for r in rows}),
        "n_frames": len(rows),
        "source": str(npz_path),
        "metrics": {k: stats(np.asarray([r[k] for r in rows])) for k in metric_keys},
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== {summary['n_clips']} clips, {summary['n_frames']} frames ===")
    for k in metric_keys:
        s = summary["metrics"][k]
        print(f"{k:<12}{s['mean']:>10.3f}{s['median']:>10.3f}{s['p90']:>10.3f}")
    print(f"\nper-frame csv -> {csv_path}\nsummary       -> {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
