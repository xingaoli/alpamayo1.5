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
  preds.npz        preds [N,S,T,3] + gts [N,T,3] + clip_ids + t0_us (t0-local
                   frame) — the single source of truth for offline metric
                   recompute (compute_metrics.py; mADE etc. now, collision
                   rate later: ego yaw from trajectory tangents, agents
                   re-read from the rds set via (clip_id, t0_us))
                   [--no-dump to skip]

Visualization is intentionally NOT part of this script — see viz_traj.py.

Multi-GPU: launch via torchrun (one process per GPU), e.g.
  torchrun --standalone --nproc_per_node=4 waymo_eval/eval_traj.py ...
Samples are sharded by index across ranks; rank 0 gathers all results and
writes the outputs, which are identical to a single-GPU run (rows are
re-sorted by dataset index).
"""

from __future__ import annotations

import os
import sys
import time

# Direct single-GPU launch defaults to GPU 0; under torchrun (LOCAL_RANK set)
# device selection is handled per-rank, so do not pin CUDA_VISIBLE_DEVICES.
if "LOCAL_RANK" not in os.environ:
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


def create_traj_only_message(frames) -> list[dict]:
    """No-CoT prompt of the alpamayo-repo stage-2 models (e.g. Alpamayo-2B):
    user asks only for the trajectory, assistant is prefilled with
    <|traj_future_start|> so the model emits discrete traj tokens directly
    (mirrors alpamayo_r1 chat_template with components_prompt=[traj_future])."""
    hist = (f"<|traj_history_start|>{'<|traj_history|>' * 48}"
            "<|traj_history_end|>")
    return [
        {"role": "system", "content": [{"type": "text", "text":
            "You are a driving assistant that generates safe and accurate actions."}]},
        {"role": "user", "content":
            [{"type": "image", "image": f} for f in frames]
            + [{"type": "text", "text": f"{hist}output the future trajectory."}]},
        {"role": "assistant", "content": [
            {"type": "text", "text": "<|traj_future_start|>"}]},
    ]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Quantitative waymo trajectory eval.")
    p.add_argument("--split", default="val", choices=["train", "val"])
    p.add_argument("--ckpt", default=DEFAULT_CKPT)
    p.add_argument("--camera-mode", default="4cam", choices=["4cam", "2cam"],
                   help="camera views fed to the model: 4cam = the 1.5 "
                        "release slots (left/front/right/tele); 2cam = "
                        "front_wide + front_tele (alpamayo-repo 2B training)")
    p.add_argument("--prompt-format", default=None, choices=["cot", "traj_only"],
                   help="assistant prefill: 'cot' = <|cot_start|> after asking "
                        "for CoT (1.5 release models); 'traj_only' = "
                        "<|traj_future_start|> (alpamayo-repo stage-2 models). "
                        "Default: read 'prompt_format' from the checkpoint "
                        "config, else 'cot'.")
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
    p.add_argument("--batch-size", type=int, default=4,
                   help="samples per GPU forward (left-padded batch; batched "
                        "sampling consumes RNG differently, so results vary "
                        "slightly vs --batch-size 1)")
    p.add_argument("--num-workers", type=int, default=16,
                   help="DataLoader workers per rank: video decode + "
                        "tokenization run in workers, inference in the main "
                        "process")
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

def prepare_sample(dataset: RDSWaymoDataset, processor, i: int,
                   annotate_cameras: bool = True,
                   prompt_format: str = "cot") -> dict:
    """Build one sample's processor inputs (CPU). Raises on a bad sample.

    ``annotate_cameras``: prepend camera display names / frame numbers to
    each image (1.5 release models were trained with them; models converted
    from the alpamayo repo set include_camera_ids/include_frame_nums=false
    in their config and were trained on bare images).
    ``prompt_format``: "cot" (helper.create_message) or "traj_only"
    (create_traj_only_message, no camera annotations).
    """
    sample = dataset[i]
    if prompt_format == "traj_only":
        messages = create_traj_only_message(
            frames=sample["image_frames"].flatten(0, 1))
    else:
        messages = helper.create_message(
            frames=sample["image_frames"].flatten(0, 1),
            camera_indices=sample["camera_indices"] if annotate_cameras else None)
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False,
        continue_final_message=True, return_dict=True, return_tensors="pt")
    return {"sample": sample, "inputs": inputs}


def collate_batch(prepared: list[dict], pad_token_id: int, device: str) -> dict:
    """Left-pad per-sample processor outputs into one batched model input.

    Mirrors the training collate (``padding_side="left"``): batched generation
    requires left padding, and the model's expert mask propagates the
    attention-mask zeros over the padded prefix. Image-side tensors
    (pixel_values / image_grid_thw / ...) all carry one row per image and are
    concatenated along dim 0.
    """
    feats = [p["inputs"] for p in prepared]
    ids = [f["input_ids"].flatten() for f in feats]
    max_len = max(x.numel() for x in ids)
    input_ids = torch.full((len(ids), max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(ids), max_len), dtype=torch.long)
    for j, x in enumerate(ids):
        input_ids[j, max_len - x.numel():] = x
        attention_mask[j, max_len - x.numel():] = 1
    tokenized = {"input_ids": input_ids, "attention_mask": attention_mask}
    for k in feats[0]:
        if k in ("input_ids", "attention_mask"):
            continue
        tokenized[k] = torch.cat([f[k] for f in feats], dim=0)
    return helper.to_device(
        {"tokenized_data": tokenized,
         "ego_history_xyz": torch.cat([p["sample"]["ego_history_xyz"] for p in prepared], dim=0),
         "ego_history_rot": torch.cat([p["sample"]["ego_history_rot"] for p in prepared], dim=0)},
        device)


def infer_batch(model, prepared: list[dict], args, device: str, pad_token_id: int
                ) -> list[np.ndarray]:
    """Run the model on ``prepared``; returns one pred [S, T, 3] per sample."""
    model_inputs = collate_batch(prepared, pad_token_id, device)
    torch.cuda.manual_seed_all(42)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred_xyz, _ = model.sample_trajectories_from_data_with_vlm_rollout(
            data=model_inputs, top_p=args.top_p,
            temperature=args.temperature,
            num_traj_samples=args.num_traj_samples,
            max_generation_length=args.max_generation_length)
    return [pred_xyz[j, 0].float().cpu().numpy() for j in range(len(prepared))]

class _PreparedShardDataset(torch.utils.data.Dataset):
    """Shard of prepared samples for DataLoader workers (CPU-only work).

    Item ``j`` maps to global dataset index ``indices[j]``; build errors are
    returned as markers so one bad sample cannot kill a worker. Fork start
    method (Linux default) lets workers inherit the dataset/processor without
    pickling; workers never touch CUDA.
    """

    def __init__(self, dataset: RDSWaymoDataset, processor, indices: list[int],
                 annotate_cameras: bool = True, prompt_format: str = "cot"):
        self.dataset = dataset
        self.processor = processor
        self.indices = indices
        self.annotate_cameras = annotate_cameras
        self.prompt_format = prompt_format

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, j: int):
        i = self.indices[j]
        try:
            return i, prepare_sample(self.dataset, self.processor, i,
                                     self.annotate_cameras,
                                     self.prompt_format), None
        except Exception as exc:
            clip_id, t0_us = self.dataset.index[i]
            return i, None, f"{clip_id} t0={t0_us}: {exc}"


def _setup_distributed() -> tuple[int, int, str]:
    """Init torch.distributed when launched under torchrun.

    Returns (rank, world_size, device). A direct single-process launch
    (``python eval_traj.py``) initializes nothing and behaves as before.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return 0, 1, "cuda"
    torch.distributed.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return torch.distributed.get_rank(), world_size, f"cuda:{local_rank}"


def main():
    args = parse_args()
    rank, world_size, device = _setup_distributed()
    is_main = rank == 0
    horizons = horizon_steps(
        [float(x) for x in args.horizons.split(",") if x.strip()],
        num_future_steps=64, time_step=0.1)

    out_dir = Path(args.out_dir) if args.out_dir else (
        _REPO_ROOT / "waymo_eval" / "eval_outputs"
        / (args.split if args.tag is None else f"{args.split}_{args.tag}"))
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)

    dataset = RDSWaymoDataset(
        root=args.root, split=args.split,
        clip_ids=(args.clip_ids.split(",") if args.clip_ids else None),
        num_segments=args.num_segments,
        segment_start_us=args.segment_start_us,
        segment_step_us=args.segment_step_us,
        max_clips=args.max_clips, clip_stride=args.clip_stride,
        preload=False,  # lazy VideoCapture seek; preloading decodes whole clips
        camera_mode=args.camera_mode)
    if is_main:
        print(f"dataset: split={args.split} clips={len(dataset.clip_ids)} "
              f"samples={len(dataset)} world_size={world_size}")

    if is_main:
        print(f"Loading model from {args.ckpt} ...")
    model = Alpamayo1_5.from_pretrained(args.ckpt, dtype=torch.bfloat16).to(device)
    model.eval()
    annotate_cameras = bool(getattr(model.config, "include_camera_ids", False)
                            or getattr(model.config, "include_frame_nums", False))
    prompt_format = args.prompt_format or getattr(
        model.config, "prompt_format", None) or "cot"
    if is_main:
        print(f"prompt camera annotations (names/frame nums): {annotate_cameras}")
        print(f"prompt format: {prompt_format}; camera mode: {args.camera_mode}")
    processor = helper.get_processor(model.tokenizer)

    rows: list[dict] = []
    idxs: list[int] = []  # global dataset index per row -> stable output order
    preds_all, gts_all = [], []
    t_start = time.time()
    n_err = 0
    k_ade, k_fde = f"mADE@{horizons[0][0]}", f"mFDE@{horizons[-1][0]}"
    sum_ade = sum_fde = 0.0
    # contiguous shard: samples are ordered by clip, so each rank (and its
    # workers) streams a few clips instead of jumping across the whole split
    _per = -(-len(dataset) // world_size)
    shard = list(range(rank * _per, min((rank + 1) * _per, len(dataset))))
    bs = max(1, args.batch_size)
    pad_id = processor.tokenizer.pad_token_id
    loader = torch.utils.data.DataLoader(
        _PreparedShardDataset(dataset, processor, shard, annotate_cameras,
                              prompt_format),
        batch_size=bs, num_workers=args.num_workers, collate_fn=list)
    pbar = tqdm(total=len(shard), desc=f"eval[{args.split}]", unit="sample",
                disable=not is_main,
                mininterval=0.1 if sys.stderr.isatty() else 30.0)

    for batch in loader:
        prepared, metas = [], []  # metas[j] = (global_idx, clip_id, t0_us)
        for i, item, err in batch:
            if err is not None:
                n_err += 1
                print(f"  [error][r{rank}] {err}")
                continue
            clip_id, t0_us = dataset.index[i]
            metas.append((i, clip_id, t0_us))
            prepared.append(item)
        # (global_idx, clip_id, t0_us, gt, pred)
        results: list[tuple] = []
        if prepared:
            try:
                preds = infer_batch(model, prepared, args, device, pad_id)
                results = [(mi, cid, t0, p["sample"]["ego_future_xyz"][0, 0].numpy(), pr)
                           for (mi, cid, t0), p, pr in zip(metas, prepared, preds)]
            except Exception as exc:
                # a batch-level failure (e.g. OOM) must not kill the shard:
                # fall back to per-sample inference so only true per-sample
                # errors are lost
                print(f"  [warn][r{rank}] batch of {len(prepared)} failed "
                      f"({type(exc).__name__}: {exc}); retrying individually")
                for (mi, cid, t0), p in zip(metas, prepared):
                    try:
                        pr = infer_batch(model, [p], args, device, pad_id)[0]
                        results.append(
                            (mi, cid, t0,
                             p["sample"]["ego_future_xyz"][0, 0].numpy(), pr))
                    except Exception as exc2:
                        n_err += 1
                        print(f"  [error][r{rank}] {cid} t0={t0}: {exc2}")

        for i, clip_id, t0_us, gt, pred in results:
            m = frame_metrics(pred, gt, horizons)
            rows.append({"clip_id": clip_id, "t0_us": t0_us,
                         "n_samples": int(pred.shape[0]), **m})
            idxs.append(i)
            sum_ade += m[k_ade]
            sum_fde += m[k_fde]
            if not args.no_dump:
                preds_all.append(pred.astype(np.float32))
                gts_all.append(gt.astype(np.float32))

        pbar.update(len(batch))
        if rows:
            pbar.set_postfix({k_ade: f"{sum_ade / len(rows):.3f}",
                              k_fde: f"{sum_fde / len(rows):.3f}",
                              "err": n_err}, refresh=False)

    pbar.close()

    # ---------------- gather shards onto rank 0 ----------------
    if world_size > 1:
        payload = (idxs, rows, n_err, preds_all, gts_all)
        gathered = [None] * world_size if is_main else None
        torch.distributed.gather_object(payload, gathered, dst=0)
        if not is_main:
            torch.distributed.destroy_process_group()
            return
        idxs, rows, n_err_all, preds_all, gts_all = [], [], 0, [], []
        for g_idxs, g_rows, g_err, g_preds, g_gts in gathered:
            idxs += g_idxs
            rows += g_rows
            n_err_all += g_err
            preds_all += g_preds
            gts_all += g_gts
        n_err = n_err_all
        order = sorted(range(len(rows)), key=lambda j: idxs[j])
        rows = [rows[j] for j in order]
        preds_all = [preds_all[j] for j in order]
        gts_all = [gts_all[j] for j in order]

    if not rows:
        print(f"no samples evaluated ({n_err} errors)")
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        return


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
                    "num_segments", "horizons", "batch_size"]}
                  | {"world_size": world_size},
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
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
