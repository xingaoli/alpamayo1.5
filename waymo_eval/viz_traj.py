#!/usr/bin/env python3
"""Standalone visualization of Alpamayo 1.5 trajectory predictions on rds_hq.

Two decoupled modes:
  * offline (--from-eval DIR): render the predictions stored by eval_traj.py
    (preds.npz: preds/gts/clip_ids/t0_us) — no model, no GPU; the text panel
    shows the stored per-sample metrics from per_frame.csv.
  * online (default): load the model and infer fresh predictions for the
    chosen (clip, t0); the text panel shows the model's CoT.

Each rendered sample shows:
  * front-view overlay — predicted corridor(s) + GT corridor projected onto the
    pinhole_front frame at t0 (distance-annotated waypoints),
  * BEV panel — ego history / GT future / predicted trajectories,
  * text panel — CoT (online) or stored metrics (offline).

Usage:
  # offline, from eval outputs:
  python viz_traj.py --from-eval eval_outputs/val_smoke --clip-id <seg> \
      [--segment-idx 3 | --t0-us 1522... ] [--video]
  # online inference + render:
  python viz_traj.py --clip-id <seg> [--segment-idx 3 | --t0-us ... ]
  python viz_traj.py --clip-id <seg> --all-segments --video   # sweep + mp4
"""

from __future__ import annotations

import os
import sys
import textwrap

os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
# Pin the VLM processor path absolutely (the config default is CWD-relative
# and would fall through to the HF hub when run outside the repo root).
os.environ.setdefault(
    "ALPAMAYO_VLM_PROCESSOR_CKPT",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "ckpts", "Qwen3-VL-8B-Instruct-config"))

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from proj_utils import (
    project_ego_local_to_pixels,
    project_corridor,
    draw_polyline_on_image,
    draw_corridor_on_image,
)
from rds_dataset import FRONT_CAM, RDSWaymoClip, RDSWaymoDataset

_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CKPT = os.environ.get(
    "ALPAMAYO_MODEL_CKPT", str(_REPO_ROOT / "ckpts" / "Alpamayo-1.5-10B"))
DT = 0.1

# BEV panel view (t0-local vehicle frame): forward x, lateral y (left+).
# The axes box aspect is derived from these spans so the panel is filled.
BEV_FORWARD = (-10, 70)
BEV_LATERAL = (-25, 25)
FRONT_DISP_SCALE = 0.75   # front view displayed at 75% of its pixel size

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize waymo predictions (rds_hq).")
    p.add_argument("--clip-id", default=None, help="segment id (or use --clip-index)")
    p.add_argument("--split", default="val")
    p.add_argument("--clip-index", type=int, default=None,
                   help="index into the split's sorted clip list")
    p.add_argument("--t0-us", type=int, default=None, help="single absolute t0 (us)")
    p.add_argument("--segment-idx", type=int, default=None,
                   help="index into the 20-segment schedule")
    p.add_argument("--all-segments", action="store_true",
                   help="render every segment of the clip")
    p.add_argument("--from-eval", default=None, metavar="DIR",
                   help="render predictions stored by eval_traj.py in DIR "
                        "(preds.npz + per_frame.csv) instead of running the model")
    p.add_argument("--ckpt", default=DEFAULT_CKPT)
    p.add_argument("--root", default=None)
    p.add_argument("--num-traj-samples", type=int, default=1)
    p.add_argument("--top-p", type=float, default=0.98)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--out-dir", default=None,
                   help="default: waymo_eval/viz_outputs/<eval-run-name>/ "
                        "(offline) or .../online/<clip_id>/ (online)")
    p.add_argument("--keep-frames", action="store_true")
    p.add_argument("--video", action="store_true",
                   help="compose rendered PNGs into an mp4 (with --all-segments)")
    return p.parse_args()


def draw_bev_context(ax, clip: RDSWaymoClip, t0_us: int, t0_pose: np.ndarray,
                     view: tuple[float, float, float, float] | None = None):
    """Draw 3d map features + other agents on a t0-local BEV axis.

    Map polylines / object boxes come from the rds dataset in world frame;
    they are transformed into the same t0-local *vehicle* frame the
    trajectories live in, then rotated like the trajectories
    (``rotate_90cc``: plot-x = -y_local, plot-y = x_local).
    ``view`` = (x_min, x_max, y_min, y_max) local-frame culling box.
    """
    import matplotlib.patches as mpatches

    R, t = t0_pose[:3, :3], t0_pose[:3, 3]

    def to_local(pts_w: np.ndarray) -> np.ndarray:
        return (R.T @ (pts_w - t).T).T

    if view is None:
        # cull slightly beyond the BEV axes so lines reach the panel edge
        view = (BEV_FORWARD[0] - 3, BEV_FORWARD[1] + 3,
                BEV_LATERAL[0] - 3, BEV_LATERAL[1] + 3)
    x0, x1, y0, y1 = view
    def in_view(p: np.ndarray) -> np.ndarray:
        return ((p[:, 0] > x0) & (p[:, 0] < x1)
                & (p[:, 1] > y0) & (p[:, 1] < y1))

    def plot_polyline_runs(vertices_w: np.ndarray, style, **kw):
        """Plot only the in-view runs of a long world polyline (no jump
        lines across out-of-view gaps)."""
        p = to_local(vertices_w)
        m = in_view(p)
        if not m.any():
            return
        # extend each run by one out-of-view neighbour so lines reach the
        # axis edge instead of stopping mid-air
        m_ext = m.copy()
        m_ext[:-1] |= m[1:]
        m_ext[1:] |= m[:-1]
        runs, cur = [], []
        for k in range(len(p)):
            if m_ext[k]:
                cur.append(p[k, :2])
            elif cur:
                runs.append(cur)
                cur = []
        if cur:
            runs.append(cur)
        for run in runs:
            xy = rotate_90cc(np.asarray(run))
            ax.plot(xy[:, 0], xy[:, 1], **kw)

    # ---- map features (static, world frame) ----
    for kind, color, lw, style in (
            ("3d_road_boundaries", "#ff595e", 2.0, "-"),
            ("3d_lanelines", "#e8e8e8", 1.4, "-"),
            ("3d_lanes", "#ffd166", 1.4, (0, (5, 3)))):
        labels = clip.map_features(kind).get("labels", [])
        for lab in labels:
            verts = (lab.get("labelData", {}).get("shape3d", {})
                     .get("polyline3d", {}).get("vertices"))
            if not verts:
                continue
            plot_polyline_runs(np.asarray(verts, dtype=np.float64),
                               style, color=color, lw=lw, alpha=0.85, zorder=1)

    for lab in clip.map_features("3d_crosswalks").get("labels", []):
        verts = (lab.get("labelData", {}).get("shape3d", {})
                 .get("surface", {}).get("vertices"))
        if not verts:
            continue
        p = to_local(np.asarray(verts, dtype=np.float64))
        if not in_view(p).any():
            continue
        xy = rotate_90cc(p[:, :2])
        ax.fill(xy[:, 0], xy[:, 1], color="#4cc9f0", alpha=0.25, zorder=1)

    # ---- other agents at t0 (nearest keyframe) ----
    agent_colors = {"Car": "#ff9f1c", "Pedestrian": "#ef476f",
                    "Cyclist": "#b388eb"}
    for obj in clip.object_info_at(t0_us).values():
        T = np.asarray(obj["object_to_world"], dtype=np.float64)
        pos = to_local(T[:3, 3][None])[0]
        if not (x0 < pos[0] < x1 and y0 < pos[1] < y1):
            continue
        R_loc = R.T @ T[:3, :3]
        yaw = np.arctan2(R_loc[1, 0], R_loc[0, 0])
        hl, hw = (float(v) / 2 for v in obj["object_lwh"][:2])
        c, s = np.cos(yaw), np.sin(yaw)
        corners = pos[:2] + np.array(
            [[c * hl - s * hw, s * hl + c * hw],
             [-c * hl - s * hw, -s * hl + c * hw],
             [-c * hl + s * hw, -s * hl - c * hw],
             [c * hl + s * hw, s * hl - c * hw]])
        xy = rotate_90cc(corners)
        col = agent_colors.get(obj.get("object_type", ""), "#9aa0a6")
        ax.add_patch(mpatches.Polygon(
            xy, closed=True, facecolor=col, edgecolor=col,
            alpha=0.55, linewidth=1.5, zorder=2))


def rotate_90cc(xy: np.ndarray) -> np.ndarray:
    return np.stack([-xy[:, 1], xy[:, 0]], axis=1)


def infer_segment(clip: RDSWaymoClip, t0_us: int, model, processor, args):
    """Model inference for one (clip, t0) -> (pred [S,T,3], gt [T,3], cot text)."""
    import torch
    from alpamayo1_5 import helper

    sample = clip.build_sample(t0_us)
    annotate = bool(getattr(model.config, "include_camera_ids", False)
                    or getattr(model.config, "include_frame_nums", False))
    messages = helper.create_message(
        frames=sample["image_frames"].flatten(0, 1),
        camera_indices=sample["camera_indices"] if annotate else None)
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False,
        continue_final_message=True, return_dict=True, return_tensors="pt")
    model_inputs = helper.to_device(
        {"tokenized_data": inputs,
         "ego_history_xyz": sample["ego_history_xyz"],
         "ego_history_rot": sample["ego_history_rot"]}, "cuda")

    torch.cuda.manual_seed_all(42)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred_xyz, _, extra = model.sample_trajectories_from_data_with_vlm_rollout(
            data=model_inputs, top_p=args.top_p, temperature=args.temperature,
            num_traj_samples=args.num_traj_samples,
            max_generation_length=256, return_extra=True)
    pred = pred_xyz[0, 0].float().cpu().numpy()          # [S, T, 3]
    gt = sample["ego_future_xyz"][0, 0].numpy()          # [T, 3]

    cot_text = None
    cot = extra.get("cot") if isinstance(extra, dict) else None
    if cot is not None:
        cot_arr = np.asarray(cot)
        if cot_arr.size > 0:
            cot_text = str(cot_arr.reshape(-1)[0])
    return pred, gt, cot_text


def render_sample(clip: RDSWaymoClip, t0_us: int, pred: np.ndarray,
                  gt: np.ndarray, panel_text: str | None, out_dir: Path,
                  panel_label: str = "COT") -> Path:
    """Render the composite figure for one (clip, t0) with given pred/gt.

    ``panel_text`` fills the text panel (model CoT online / stored metrics
    offline). Returns the saved figure path.
    """
    hist = clip.build_sample(t0_us)["ego_history_xyz"][0, 0].numpy()   # [H, 3]

    # ---------------- front-view overlay ----------------
    t0_pose = clip.pose_at(t0_us)[0]
    front = clip.decode_frames(FRONT_CAM, np.asarray([clip.keyframe_index_at(t0_us)]))[0]
    frame_h, frame_w = front.shape[:2]
    intr = clip.intrinsics(FRONT_CAM, target_w=frame_w, target_h=frame_h)
    # pred/gt/hist are t0-local *vehicle*-frame; extrinsic maps vehicle -> cam
    ego_to_cam = clip.extrinsic(FRONT_CAM, t0_us)
    t0_xyz = t0_pose[:3, 3]

    SCALE = 2
    big = cv2.resize(front, (frame_w * SCALE, frame_h * SCALE),
                     interpolation=cv2.INTER_LANCZOS4)
    big = (big.astype(np.float32) * 0.6).astype(np.uint8)

    PRED_COLOR = (60, 140, 255)          # blue, primary prediction
    PRED_ALT_COLOR = (255, 0, 255)       # magenta, additional samples
    for s in range(pred.shape[0]):
        col = PRED_COLOR if s == 0 else PRED_ALT_COLOR
        lp, lv, rp, rv = project_corridor(pred[s], half_width=1.0,
                                          ego_to_cam=ego_to_cam, intrinsics=intr)
        big = draw_corridor_on_image(
            big, lp * SCALE, lv, rp * SCALE, rv, color=col, alpha=0.5)
        pix, valid = project_ego_local_to_pixels(pred[s], ego_to_cam, intr)
        d = np.linalg.norm(pred[s][:, :2], axis=1)
        big = draw_polyline_on_image(
            big, pix * SCALE, color=col, radius=5, thickness=4,
            label_every=8, labels=[f"{x:.0f}" for x in d])

    if len(gt) >= 2:
        gl, glv, gr, grv = project_corridor(gt, half_width=0.6,
                                            ego_to_cam=ego_to_cam, intrinsics=intr)
        big = draw_corridor_on_image(big, gl * SCALE, glv, gr * SCALE, grv,
                                     color=(0, 255, 0), alpha=0.45)
        gpix, _ = project_ego_local_to_pixels(gt, ego_to_cam, intr)
        gd = np.linalg.norm(gt[:, :2], axis=1)
        big = draw_polyline_on_image(
            big, gpix * SCALE, color=(0, 255, 0), radius=6, thickness=3,
            label_every=4, labels=[f"{d:.0f}" for d in gd])

    t0_s = (t0_us - int(clip.ts_us[0])) * 1e-6
    cv2.putText(big, "blue = pred   magenta = extra pred samples   "
                "green = GT future   labels = forward distance [m]",
                (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7 * SCALE, (255, 255, 255),
                2 * SCALE, cv2.LINE_AA)
    cv2.line(big, (int(intr["cx"] * SCALE) - 20, int(intr["cy"] * SCALE)),
             (int(intr["cx"] * SCALE) + 20, int(intr["cy"] * SCALE)),
             (200, 200, 200), 1)
    # ---------------- matplotlib composite ----------------
    dpi = 100
    img_h, img_w = big.shape[:2]
    img_w_in = img_w / dpi * FRONT_DISP_SCALE
    img_h_in = img_h / dpi * FRONT_DISP_SCALE
    title_h, cot_h, suptitle_h, margin = 0.90, 1.40, 0.90, 0.15
    left_h = img_h_in + title_h + cot_h
    right_h = img_h_in + title_h
    bev_aspect = ((BEV_LATERAL[1] - BEV_LATERAL[0])
                  / (BEV_FORWARD[1] - BEV_FORWARD[0]))
    right_w = right_h * bev_aspect
    gap = 0.80
    fig_w = margin + img_w_in + gap + right_w + margin
    fig_h = margin + suptitle_h + max(left_h, right_h) + margin

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    ax_img = fig.add_axes([margin / fig_w, (margin + cot_h) / fig_h,
                           img_w_in / fig_w, img_h_in / fig_h])
    ax_img.imshow(big)
    ax_img.set_title(f"Front view ({FRONT_CAM}) @ {t0_s:.1f}s", fontsize=36, pad=6)
    ax_img.axis("off")

    cot_ax = fig.add_axes([margin / fig_w, margin / fig_h,
                           img_w_in / fig_w, cot_h / fig_h])
    cot_ax.axis("off")
    txt = textwrap.fill(panel_text, width=65) if panel_text else "(no CoT)"
    cot_ax.text(0.0, 0.5, f"{panel_label}: {txt}", ha="left", va="center",
                family="monospace", fontsize=32, transform=cot_ax.transAxes)

    ax_trj = fig.add_axes([(margin + img_w_in + gap) / fig_w,
                           (margin + cot_h) / fig_h,
                           right_w / fig_w, img_h_in / fig_h])
    draw_bev_context(ax_trj, clip, t0_us, t0_pose)
    gt_rot = rotate_90cc(gt[:, :2])
    ax_trj.plot(gt_rot[:, 0], gt_rot[:, 1], "g-o", ms=3, label="GT future")
    for s in range(pred.shape[0]):
        style = "b-" if s == 0 else "m--"
        pr = rotate_90cc(pred[s, :, :2])
        ax_trj.plot(pr[:, 0], pr[:, 1], style, alpha=0.9,
                    label="pred" if s == 0 else None)
    hr = rotate_90cc(hist[:, :2])
    ax_trj.plot(hr[:, 0], hr[:, 1], "k.-", ms=3, alpha=0.5, label="history")
    ax_trj.scatter([0], [0], c="k", edgecolors="w", s=80, zorder=5,
                    label="t0 (ego)")
    ax_trj.set_aspect("equal")
    ax_trj.set_xlabel("y (left) [m]", fontsize=28)
    ax_trj.set_ylabel("x (forward) [m]", fontsize=28)
    ax_trj.set_xlim(-BEV_LATERAL[1], -BEV_LATERAL[0])
    ax_trj.set_ylim(*BEV_FORWARD)
    ax_trj.set_title("BEV trajectory", fontsize=36, pad=6)
    ax_trj.grid(True, alpha=0.3)
    import matplotlib.lines as mlines
    proxies = [
        mlines.Line2D([], [], color="#ff595e", lw=2, label="road bnd"),
        mlines.Line2D([], [], color="#ffd166", lw=1.4, ls=(0, (5, 3)),
                      label="lanes"),
        mlines.Line2D([], [], color="#ff9f1c", marker="s", ls="", ms=10,
                      label="agents"),
    ]
    handles, labels = ax_trj.get_legend_handles_labels()
    ax_trj.legend(handles + proxies, labels + [p.get_label() for p in proxies],
                  fontsize=20, loc="upper right", ncol=2)

    fig.suptitle(f"{clip.clip_id}  @  t0 = {t0_s:.1f}s", fontsize=36,
                 y=(fig_h - margin - suptitle_h / 2) / fig_h)
    fig_path = out_dir / f"{clip.clip_id}_t{t0_s:06.2f}s.png"
    fig.savefig(fig_path, dpi=dpi)
    plt.close(fig)
    print(f"  saved figure -> {fig_path}")
    return fig_path


def encode_video(png_files: list[Path], out_path: Path, fps: int = 2):
    """Compose rendered PNGs into an mp4 (PyAV x264).

    Odd figure sizes are cropped by one pixel — libx264/yuv420p requires
    even dimensions.
    """
    import av
    import PIL.Image
    container = av.open(str(out_path), mode="w")
    stream = None
    for f in png_files:
        arr = np.asarray(PIL.Image.open(f).convert("RGB"))
        arr = arr[:arr.shape[0] // 2 * 2, :arr.shape[1] // 2 * 2]
        if stream is None:
            stream = container.add_stream("libx264", rate=fps)
            stream.width, stream.height = arr.shape[1], arr.shape[0]
            stream.pix_fmt = "yuv420p"
        vf = av.VideoFrame.from_ndarray(arr, format="rgb24")
        for packet in stream.encode(vf):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    print(f"  composed video -> {out_path}")


def load_eval_store(eval_dir: Path):
    """Read eval_traj.py outputs: (clip_ids, t0s, preds, gts, metrics-by-key)."""
    npz = np.load(eval_dir / "preds.npz")
    clip_ids = [str(c) for c in npz["clip_ids"]]
    t0s = npz["t0_us"].astype(np.int64)
    preds, gts = npz["preds"], npz["gts"]        # [N,S,T,3], [N,T,3]
    metrics: dict[tuple[str, int], dict] = {}
    csv_path = eval_dir / "per_frame.csv"
    if csv_path.exists():
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                metrics[(row["clip_id"], int(row["t0_us"]))] = row
    return clip_ids, t0s, preds, gts, metrics


def main():
    args = parse_args()
    if args.clip_id is None and args.clip_index is None:
        raise SystemExit("specify --clip-id or --clip-index")

    dataset = RDSWaymoDataset(
        root=args.root, split=args.split,
        clip_ids=[args.clip_id] if args.clip_id else None, preload=False)
    if args.clip_index is not None:
        clip_id = dataset.clip_ids[args.clip_index]
        dataset = RDSWaymoDataset(root=args.root, split=args.split,
                                  clip_ids=[clip_id], preload=False)
    clip_id = dataset.clip_ids[0]
    clip = dataset._get_clip(clip_id)

    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
    elif args.from_eval:
        # group renders under the eval run they came from, like eval_outputs
        out_dir = _REPO_ROOT / "waymo_eval" / "viz_outputs" / Path(args.from_eval).name
    else:
        out_dir = _REPO_ROOT / "waymo_eval" / "viz_outputs" / "online" / clip_id
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[viz] out_dir={out_dir}")

    if args.from_eval:
        # ---- offline: render predictions stored by eval_traj.py (no model) ----
        eval_dir = Path(args.from_eval)
        clip_ids, t0s, preds, gts, metrics = load_eval_store(eval_dir)
        rows = [i for i, c in enumerate(clip_ids) if c == clip_id]
        if not rows:
            raise SystemExit(f"clip {clip_id} has no stored samples in "
                             f"{eval_dir / 'preds.npz'}")
        if args.t0_us is not None:
            rows = [i for i in rows if int(t0s[i]) == args.t0_us]
            if not rows:
                raise SystemExit(f"t0={args.t0_us} not stored for {clip_id}")
        elif args.segment_idx is not None:
            rows = [rows[args.segment_idx]]
        print(f"offline: {len(rows)} stored sample(s) of {clip_id} from {eval_dir}")

        for i in rows:
            t0 = int(t0s[i])
            m = metrics.get((clip_id, t0), {})
            panel = "  ".join(f"{k}={float(m[k]):.2f}" for k in
                              ("mADE@1s", "mADE@3s", "mADE@5s", "mFDE@5s")
                              if k in m) or None
            render_sample(clip, t0, preds[i], gts[i], panel, out_dir,
                          panel_label="EVAL")
    else:
        # ---- online: infer fresh predictions (with CoT) ----
        t0_list = clip.segment_t0s()
        if args.t0_us is not None:
            t0_list = [args.t0_us]
        elif args.segment_idx is not None:
            t0_list = [t0_list[args.segment_idx]]
        print(f"online: {len(t0_list)} segment(s) of {clip_id} to infer + render")

        import torch
        from alpamayo1_5 import helper
        from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
        print(f"Loading model from {args.ckpt} ...")
        model = Alpamayo1_5.from_pretrained(
            args.ckpt, dtype=torch.bfloat16).to("cuda")
        model.eval()
        processor = helper.get_processor(model.tokenizer)

        for t0 in t0_list:
            pred, gt, cot = infer_segment(clip, t0, model, processor, args)
            render_sample(clip, t0, pred, gt, cot, out_dir)

    if args.video:
        pngs = sorted(out_dir.glob(f"{clip_id}_t*.png"))
        encode_video(pngs, out_dir / f"{clip_id}.mp4")
        if not args.keep_frames:
            for f in pngs:
                f.unlink()
            print("  cleaned up per-segment PNG(s) (use --keep-frames to keep)")


if __name__ == "__main__":
    main()
