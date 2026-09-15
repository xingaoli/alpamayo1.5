#!/usr/bin/env python3
"""Quantitative trajectory evaluation of Alpamayo 1.5 on the rds_hq waymo set.

Metrics (per (clip, t0) sample, xy in the t0-local frame):
  ADE@Ks   mean L2 over the first K seconds (K in {1, 3, 5} + full horizon)
  FDE@Ks   L2 at the K-second waypoint
  mADE@Ks / mFDE@Ks   min over the sampled trajectories (best-of-N)

ADE/FDE are averaged over the N sampled trajectories, mADE/mFDE take the best
one; with ``--num-traj-samples 1`` both collapse to the plain metric.

Outputs (under ``--out-dir``):
  per_frame.csv    one row per (clip, t0)
  summary.json     aggregated stats (mean / median / p90) + run config
  preds.npz        all predictions + GT (for later visualization / collision
                   analysis without re-running the model)   [--no-dump to skip]

Visualization is intentionally NOT part of this script — see viz_traj.py.
"""

from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
# The model config's __repr__ builds a default config whose vlm path is
# CWD-relative ("ckpts/Qwen3-VL-8B-Instruct-config"); pin it absolutely so the
# scripts work from any cwd (and never hit the HF hub).
os.environ.setdefault(
    "ALPAMAYO_VLM_PROCESSOR_CKPT",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "ckpts", "Qwen3-VL-8B-Instruct-config"))

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from alpamayo1_5 import helper
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
from rds_dataset import RDSWaymoDataset

_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CKPT = os.environ.get(
    "ALPAMAYO_MODEL_CKPT", str(_REPO_ROOT / "ckpts" / "Alpamayo-1.5-10B"))

# horizon seconds -> number of 0.1s steps (full horizon appended at runtime)
DEFAULT_HORIZONS_S = (1.0, 3.0, 5.0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Quantitative waymo trajectory eval.")
    p.add_argument("--split", default="val", choices=["train", "val"])
    p.add_argument("--ckpt", default=DEFAULT_CKPT)
    p.add_argument("--root", default=None, help="rds_hq root (default: data/waymo_rds_hq)")
    p.add_argument("--clip-ids", default=None,
                   help="comma-separated clip ids to evaluate (default: all in split)")
    p.add_argument("--max-clips", type=int, default=None)
    p.add_argument("--clip-stride", type=int, default=1)
    p.add_argument("--num-segments", type=int, default=20,
                   help="max t0 segments per clip; span filter keeps t0 in "
                        "[2.0, 13.0]s -> 12 per 19.8s clip at the default 1s step")
    p.add_argument("--segment-start-us", type=int, default=2_000_000)
    p.add_argument("--segment-step-us", type=int, default=1_000_000)
    p.add_argument("--num-traj-samples", type=int, default=5)
    p.add_argument("--top-p", type=float, default=0.98)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--max-generation-length", type=int, default=256)
    p.add_argument("--horizons", default="1,3,5",
                   help="comma-separated horizon seconds for ADE/FDE (plus full)")
    p.add_argument("--out-dir", default=None,
                   help="default: waymo_eval/eval_outputs/<split>[_<tag>]")
    p.add_argument("--tag", default=None)
    p.add_argument("--no-dump", action="store_true",
                   help="do not save preds.npz")
    return p.parse_args()


def horizon_steps(horizons_s, num_future_steps: int, time_step: float) -> list[tuple[str, int]]:
    out = [(f"{h:g}s", int(round(h / time_step))) for h in horizons_s]
    out.append(("full", num_future_steps))
    return out


def frame_metrics(pred: np.ndarray, gt: np.ndarray,
                  horizons: list[tuple[str, int]]) -> dict:
    """``pred`` [S, T, 3], ``gt`` [T, 3] (t0-local). Returns flat metric dict."""
    n_cmp = min(pred.shape[1], gt.shape[0])
    d = np.linalg.norm(pred[:, :n_cmp, :2] - gt[None, :n_cmp, :2], axis=2)  # [S, n]
    out = {}
    for name, k in horizons:
        k = min(k, n_cmp)
        ade_s = d[:, :k].mean(axis=1)          # [S]
        fde_s = d[:, k - 1]                    # [S]
        out[f"ADE@{name}"] = float(ade_s.mean())
        out[f"mADE@{name}"] = float(ade_s.min())
        out[f"FDE@{name}"] = float(fde_s.mean())
        out[f"mFDE@{name}"] = float(fde_s.min())
    return out


def main():
    args = parse_args()
    horizons = horizon_steps(
        [float(x) for x in args.horizons.split(",") if x.strip()],
        num_future_steps=64, time_step=0.1)

    out_dir = Path(args.out_dir) if args.out_dir else (
        _REPO_ROOT / "waymo_eval" / "eval_outputs"
        / (args.split if args.tag is None else f"{args.split}_{args.tag}"))
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = RDSWaymoDataset(
        root=args.root, split=args.split,
        clip_ids=(args.clip_ids.split(",") if args.clip_ids else None),
        num_segments=args.num_segments,
        segment_start_us=args.segment_start_us,
        segment_step_us=args.segment_step_us,
        max_clips=args.max_clips, clip_stride=args.clip_stride)
    print(f"dataset: split={args.split} clips={len(dataset.clip_ids)} "
          f"samples={len(dataset)}")

    print(f"Loading model from {args.ckpt} ...")
    model = Alpamayo1_5.from_pretrained(args.ckpt, dtype=torch.bfloat16).to("cuda")
    model.eval()
    processor = helper.get_processor(model.tokenizer)

    rows: list[dict] = []
    preds_all, gts_all = [], []
    t_start = time.time()
    n_err = 0
    # live bar on a tty; when piped (tee/nohup) tqdm still logs one line per
    # refresh, so slow the refresh to every 30s to keep logs readable
    pbar = tqdm(range(len(dataset)), desc=f"eval[{args.split}]",
                unit="sample", disable=False,
                mininterval=0.1 if sys.stderr.isatty() else 30.0)
    k_ade, k_fde = f"mADE@{horizons[0][0]}", f"mFDE@{horizons[-1][0]}"
    sum_ade = sum_fde = 0.0
    for i in pbar:
        clip_id, t0_us = dataset.index[i]
        try:
            sample = dataset[i]
            messages = helper.create_message(
                frames=sample["image_frames"].flatten(0, 1),
                camera_indices=sample["camera_indices"])
            inputs = processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=False,
                continue_final_message=True, return_dict=True, return_tensors="pt")
            model_inputs = helper.to_device(
                {"tokenized_data": inputs,
                 "ego_history_xyz": sample["ego_history_xyz"],
                 "ego_history_rot": sample["ego_history_rot"]}, "cuda")

            torch.cuda.manual_seed_all(42)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred_xyz, _ = model.sample_trajectories_from_data_with_vlm_rollout(
                    data=model_inputs, top_p=args.top_p,
                    temperature=args.temperature,
                    num_traj_samples=args.num_traj_samples,
                    max_generation_length=args.max_generation_length)
            pred = pred_xyz[0, 0].float().cpu().numpy()          # [S, T, 3]
            gt = sample["ego_future_xyz"][0, 0].numpy()          # [T, 3]
        except Exception as exc:
            n_err += 1
            print(f"  [error] {clip_id} t0={t0_us}: {exc}")
            continue
        m = frame_metrics(pred, gt, horizons)
        rows.append({"clip_id": clip_id, "t0_us": t0_us,
                     "n_samples": int(pred.shape[0]), **m})
        sum_ade += m[k_ade]
        sum_fde += m[k_fde]
        if not args.no_dump:
            preds_all.append(pred.astype(np.float32))
            gts_all.append(gt.astype(np.float32))
        pbar.set_postfix({k_ade: f"{sum_ade / len(rows):.3f}",
                          k_fde: f"{sum_fde / len(rows):.3f}",
                          "err": n_err}, refresh=False)

    pbar.close()


    # ---------------- aggregate + write ----------------
    import csv
    metric_keys = [k for k in rows[0] if "@" in k]
    csv_path = out_dir / "per_frame.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def stats(vals: np.ndarray) -> dict:
        return {"mean": float(vals.mean()), "median": float(np.median(vals)),
                "p90": float(np.percentile(vals, 90))}

    summary = {
        "config": {k: getattr(args, k) for k in
                   ["split", "ckpt", "num_traj_samples", "top_p", "temperature",
                    "num_segments", "horizons"]},
        "n_clips": len({r["clip_id"] for r in rows}),
        "n_frames": len(rows),
        "n_errors": n_err,
        "wall_clock_s": time.time() - t_start,
        "metrics": {k: stats(np.asarray([r[k] for r in rows])) for k in metric_keys},
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    if not args.no_dump and preds_all:
        np.savez_compressed(
            out_dir / "preds.npz",
            preds=np.stack(preds_all), gts=np.stack(gts_all),
            clip_ids=np.asarray([r["clip_id"] for r in rows]),
            t0_us=np.asarray([r["t0_us"] for r in rows], dtype=np.int64))

    print(f"\n=== {args.split}: {summary['n_clips']} clips, "
          f"{summary['n_frames']} frames, {n_err} errors ===")
    hdr = f"{'metric':<12}" + "".join(f"{s:>10}" for s in ["mean", "median", "p90"])
    print(hdr)
    for k in metric_keys:
        s = summary["metrics"][k]
        print(f"{k:<12}{s['mean']:>10.3f}{s['median']:>10.3f}{s['p90']:>10.3f}")
    print(f"\nper-frame csv -> {csv_path}\nsummary       -> {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
