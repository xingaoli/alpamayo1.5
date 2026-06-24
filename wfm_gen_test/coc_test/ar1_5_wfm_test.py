#!/usr/bin/env python3
"""Test the Alpamayo 1.5 model on world-model-generated driving data.

This is the 1.5 update of ``wfm_gen_test/coc_test/ar1_5_wfm_test.py``. It is
modelled after the original but targets the in-tree ``alpamayo1_5`` package
(model checkpoint under ``ckpts/Alpamayo-1.5-10B``) and the moved dataset
root (``data/wfm_gen_data``). The model APIs are intentionally kept very
close to the previous version, so the rest of the script (data loading,
projection, visualisation) is mostly unchanged.

Adapts to the world-model output layout:

  * Cameras are plain ``.mp4`` files at 10 Hz (one clip per camera dir),
    instead of zipped PhysicalAI-AV shards with timestamp parquets.
  * Egomotion is a single ``labels/<clip>.json`` holding a list of 4x4 poses
    (one per frame), already normalized to the first frame -- no
    interpolation needed since the video and the poses share the exact 10 Hz
    frame grid.
  * There is no front ``tele`` camera, so we fill that slot with the ``wide``
    frames (the model still sees its trained 4-cam layout: cross_left, wide,
    cross_right, tele).

The script decodes the requested 4 image frames + 16 history ego steps, runs
Alpamayo 1.5 inference, prints the chain-of-thought reasoning and predicted
trajectory, and (optionally) overlays the prediction against the
ground-truth future poses that are available in the clip.

Protection: the test timestamp (``frame_idx``) must be >= 1.6 s, i.e.
``frame_idx >= 16``, so the 16-step history window (frames frame_idx-15 ..
frame_idx) is always fully contained.

Usage
-----
Run from the project root (so the relative ``ckpts/`` and ``data/`` paths
resolve). The package must be importable via ``PYTHONPATH=src``::

    cd /home/xingao/code/Alpamayo1.5
    export PYTHONPATH=src
    export CUDA_VISIBLE_DEVICES=0

Common invocations:

1. **Default** (single frame at t0 = 1.6 s, on every clip under
   ``<DATA_DIR>/camera/camera_front_wide_120fov``)::
    PYTHONPATH=src \
    python wfm_gen_test/coc_test/ar1_5_wfm_test.py

2. **Single frame at a chosen t0**::

    python wfm_gen_test/coc_test/ar1_5_wfm_test.py --frame-idx 30

3. **Sweep a range of frames** (saves one PNG per frame + a stitched MP4)::

    python wfm_gen_test/coc_test/ar1_5_wfm_test.py \
        --frame-start 20 --frame-end 40

4. **Sample multiple trajectories per frame** to visualise the model's
   stochastic spread (red = sample 0, magenta dashed = other samples)::

    python wfm_gen_test/coc_test/ar1_5_wfm_test.py \
        --frame-idx 30 --num-traj-samples 4

5. **Different clip / ckpt / output dir**::

    python wfm_gen_test/coc_test/ar1_5_wfm_test.py \
        --clip 71ae0d63-ba64-4517-b3f1-90d446abc67d_1869020100000_1869040100000_fire \
        --ckpt /abs/path/to/Alpamayo-1.5-10B \
        --out-dir /tmp/my_outputs

6. **All clips under the front-wide camera** (loop over every .mp4;
   each clip gets its own PNGs + stitched MP4)::

    python wfm_gen_test/coc_test/ar1_5_wfm_test.py

Environment variables (override the defaults; CLI flags take precedence):

  * ``CUDA_VISIBLE_DEVICES`` -- which GPU to use (default 0).
  * ``ALPAMAYO_MODEL_CKPT`` -- model checkpoint path (default
    ``<repo>/ckpts/Alpamayo-1.5-10B``).
  * ``WFM_GEN_DATA_DIR`` -- world-model data root (default
    ``<repo>/data/wfm_gen_data``).
  * ``ALPAMAYO_VLM_PROCESSOR_CKPT`` -- Qwen3-VL processor checkpoint
    (default ``<repo>/ckpts/Qwen3-VL-8B-Instruct-config``).

Outputs land in ``<out_dir>/<clip>_f<NNN>.png`` (composite front-view +
BEV + CoT) and ``<out_dir>/<clip>.mp4`` (stitched from the PNGs at 5 fps).
The per-frame PNGs are removed by default once the video has been written
successfully; pass ``--keep-frames`` to retain them.
"""

import os

# --- environment setup must happen before importing torch / the package ---
# The new package's __init__ does load_dotenv(); ALPAMAYO_MODEL_CKPT (and
# ALPAMAYO_DATA_DIR / ALPAMAYO_VLM_PROCESSOR_CKPT) are read from .env.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", "0"))

import argparse
import json
import textwrap
from pathlib import Path

import av  # PyAV (FFmpeg binding) -- used to encode stitched MP4s.
import cv2
import numpy as np
import scipy.spatial.transform as spt
import torch
import matplotlib

matplotlib.use("Agg")  # headless: save figures to disk instead of showing
import matplotlib.pyplot as plt

from alpamayo1_5 import helper
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5

from proj_utils import (
    load_extrinsics,
    load_intrinsics_for_clip,
    static_mount,
    project_ego_local_to_pixels,
    project_corridor,
    draw_polyline_on_image,
    draw_corridor_on_image,
    gradient_colors,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# ``_REPO_ROOT`` is the project root (this file lives at
# <repo>/wfm_gen_test/coc_test/ar1_5_wfm_test.py). We anchor DATA_DIR there so
# the script works regardless of the user's current working directory.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = Path(
    os.environ.get("WFM_GEN_DATA_DIR", _REPO_ROOT / "data" / "wfm_gen_data")
)
DEFAULT_CLIP = "71ae0d63-ba64-4517-b3f1-90d446abc67d_1869020100000_1869040100000_fire"
DEFAULT_CKPT = os.environ.get(
    "ALPAMAYO_MODEL_CKPT", str(_REPO_ROOT / "ckpts" / "Alpamayo-1.5-10B")
)
DEFAULT_OUT_DIR = _REPO_ROOT / "wfm_gen_test" / "coc_test" / "ar1_5_wfm_outputs"

NUM_HISTORY_STEPS = 16   # ego history fed to the model
NUM_IMAGE_FRAMES = 4     # image frames fed to the model (per camera)
DT = 0.1                 # 10 Hz frame rate
MIN_FRAME_IDX = 16       # protection: t0 >= 1.6 s -> history window fits

# 4-camera layout used by Alpamayo 1.5 (sorted by the trained camera index).
# `camera_front_tele_30fov` is absent in the world-model data -> filled with wide.
CAMERA_ORDER = [
    ("camera_cross_left_120fov", 0),
    ("camera_front_wide_120fov", 1),
    ("camera_cross_right_120fov", 2),
    ("camera_front_tele_30fov", 6),
]
WIDE_CAM_NAME = "camera_front_wide_120fov"


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------
def decode_video(path: Path) -> np.ndarray:
    """Decode an mp4 into an [N, H, W, 3] uint8 RGB array via OpenCV.

    mediapy/ffmpeg are unavailable in this env, so we fall back to cv2.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from: {path}")
    return np.stack(frames)  # [N, H, W, 3] uint8


def load_egomotion(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load the per-frame 4x4 pose list -> (xyz [N,3], rot [N,3,3]).

    Poses are already normalized to the first frame, so frame i <-> pose i.
    """
    with open(path) as f:
        mats = np.asarray(json.load(f), dtype=np.float64)  # [N, 4, 4]
    if mats.ndim != 3 or mats.shape[1:] != (4, 4):
        raise ValueError(f"Expected a list of 4x4 matrices, got shape {mats.shape}")
    xyz = mats[:, :3, 3]
    rot = mats[:, :3, :3]
    return xyz, rot


def load_cameras(clip: str) -> dict[str, np.ndarray]:
    """Decode every available camera video for the clip. Returns {name: [N,H,W,3]}."""
    cameras = {}
    cam_root = DATA_DIR / "camera"
    for cam_name, _ in CAMERA_ORDER:
        path = cam_root / cam_name / f"{clip}.mp4"
        if path.exists():
            cameras[cam_name] = decode_video(path)
            print(f"  loaded {cam_name}: {cameras[cam_name].shape}")
    return cameras


def build_local_history(
    xyz: np.ndarray, rot: np.ndarray, frame_idx: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Ego history (frames frame_idx-15 .. frame_idx) in the t0-local frame.

    The local frame is anchored at the LAST history step (t0 = current frame),
    exactly like ``load_physical_aiavdataset_local``.
    """
    t0_xyz = xyz[frame_idx].copy()
    t0_rot = rot[frame_idx].copy()
    t0_rot_inv = t0_rot.T  # rotation matrix inverse == transpose

    hist_idx = np.arange(frame_idx - (NUM_HISTORY_STEPS - 1), frame_idx + 1)
    hist_xyz = xyz[hist_idx]                                   # [16, 3]
    hist_rot = rot[hist_idx]                                   # [16, 3, 3]

    hist_xyz_local = (t0_rot_inv @ (hist_xyz - t0_xyz).T).T    # [16, 3]
    hist_rot_local = np.einsum("ij,njk->nik", t0_rot_inv, hist_rot)  # [16, 3, 3]
    return t0_xyz, t0_rot, t0_rot_inv, hist_xyz_local, hist_rot_local


def build_image_frames(
    cameras: dict[str, np.ndarray], frame_idx: int, n_frames: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build [num_cams, n_frames, 3, H, W] image tensor + camera indices, sorted."""
    img_idx = np.arange(frame_idx - (n_frames - 1), frame_idx + 1)  # most-recent n frames
    wide = cameras[WIDE_CAM_NAME]

    frames_list, cam_idx_list = [], []
    for cam_name, cam_idx in CAMERA_ORDER:
        if cam_name in cameras:
            src = cameras[cam_name]
        else:
            # Missing camera (e.g. front tele) -> fill with the wide stream.
            print(f"  [fill] {cam_name} missing -> using {WIDE_CAM_NAME} frames")
            src = wide
        ft = torch.from_numpy(src[img_idx])             # [n_frames, H, W, 3] uint8
        ft = ft.permute(0, 3, 1, 2).contiguous()        # [n_frames, 3, H, W]
        frames_list.append(ft)
        cam_idx_list.append(cam_idx)

    image_frames = torch.stack(frames_list, dim=0)       # [num_cams, n_frames, 3, H, W]
    camera_indices = torch.tensor(cam_idx_list, dtype=torch.int64)

    # Match the loader: sort cameras by their trained index.
    order = torch.argsort(camera_indices)
    image_frames = image_frames[order]
    camera_indices = camera_indices[order]
    return image_frames, camera_indices


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run Alpamayo 1.5 (CoC + trajectory) on world-model data."
    )
    p.add_argument(
        "--clip", default=None,
        help=(
            "clip id (filename stem). If not set, the script iterates over "
            "every .mp4 clip under <DATA_DIR>/camera/camera_front_wide_120fov."
        ),
    )
    p.add_argument(
        "--frame-idx", type=int, default=None,
        help=(
            f"single frame index (t0). Must be >= {MIN_FRAME_IDX} (1.6 s). "
            "Ignored when --frame-end is set; if given alone it produces "
            "a one-frame output. The default (no --frame-end, no "
            "--frame-idx) auto-sweeps the whole valid range."
        ),
    )
    p.add_argument(
        "--frame-start", type=int, default=None,
        help=f"start frame for a multi-frame sweep (>= {MIN_FRAME_IDX}). "
        "Use with --frame-end; overrides --frame-idx.",
    )
    p.add_argument(
        "--frame-end", type=int, default=None,
        help="end frame (inclusive) for a multi-frame sweep.",
    )
    p.add_argument(
        "--ckpt", default=DEFAULT_CKPT,
        help="model checkpoint path (defaults to ALPAMAYO_MODEL_CKPT or "
        "ckpts/Alpamayo-1.5-10B).",
    )
    p.add_argument("--num-traj-samples", type=int, default=1, help="# trajectory samples.")
    p.add_argument("--top-p", type=float, default=0.98)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument(
        "--out-dir", default=str(DEFAULT_OUT_DIR),
        help="where to save figures/metrics.",
    )
    p.add_argument(
        "--keep-frames", action="store_true",
        help="keep the per-frame PNG figures after the video is composed "
        "(default: PNGs are deleted once the video is written successfully).",
    )
    args = p.parse_args()

    # ``args.frames`` is filled in per-clip by _run_for_clip, because the
    # auto-sweep default needs the actual clip length. Stash ``None`` as a
    # sentinel so callers know it has not been resolved yet.
    args.frames = None
    return args


def rotate_90cc(xy: np.ndarray) -> np.ndarray:
    """Rotate (x, y) coords 90 deg CCW so forward (x) becomes up."""
    return np.stack([-xy[:, 1], xy[:, 0]], axis=1)


def run_frame(
    frame_idx: int,
    args: argparse.Namespace,
    cameras: dict[str, np.ndarray],
    xyz: np.ndarray,
    rot: np.ndarray,
    n_frames: int,
    model,
    processor,
    intr_wide: dict,
    ego_to_cam: np.ndarray,
    out_dir: Path,
) -> None:
    """Infer + render the composite figure and front-view overlay for one frame."""
    t0_s = frame_idx * DT
    print(f"\n{'=' * 70}\n  frame {frame_idx} ({t0_s:.1f} s)\n{'=' * 70}")

    # ---- build model inputs ---------------------------------------------
    t0_xyz, t0_rot, t0_rot_inv, hist_xyz_local, hist_rot_local = build_local_history(
        xyz, rot, frame_idx
    )
    ego_history_xyz = torch.from_numpy(hist_xyz_local).float()[None, None]
    ego_history_rot = torch.from_numpy(hist_rot_local).float()[None, None]
    image_frames, camera_indices = build_image_frames(
        cameras, frame_idx, NUM_IMAGE_FRAMES
    )

    # ---- build prompt ----------------------------------------------------
    # Alpamayo 1.5's helper.create_message takes a flattened (N, C, H, W) frame
    # stack plus the per-camera ``camera_indices`` so each image is annotated
    # with its display name and frame number (matching the training format).
    messages = helper.create_message(
        frames=image_frames.flatten(0, 1),
        camera_indices=camera_indices,
    )
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )
    model_inputs = helper.to_device(
        {
            "tokenized_data": inputs,
            "ego_history_xyz": ego_history_xyz,
            "ego_history_rot": ego_history_rot,
        },
        "cuda",
    )

    # ---- run inference ---------------------------------------------------
    torch.cuda.manual_seed_all(42)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
            data=model_inputs,
            top_p=args.top_p,
            temperature=args.temperature,
            num_traj_samples=args.num_traj_samples,
            max_generation_length=256,
            return_extra=True,
        )
    # pred_xyz: [B=1, num_traj_sets=1, num_traj_samples, T=64, 3]
    pred_xyz_np = pred_xyz[0, 0].float().cpu().numpy()  # [S, 64, 3]

    # ---- chain-of-thought ------------------------------------------------
    # Alpamayo 1.5 returns ``extra['cot']`` as a numpy array of shape
    # ``[B, num_traj_sets, num_traj_samples]`` of strings (see
    # ``alpamayo1_5.models.token_utils.extract_text_tokens`` +
    # ``Alpamayo1_5.sample_trajectories_from_data_with_vlm_rollout``).
    # Flatten and pick sample 0 to be shape-agnostic across num_traj_samples.
    cot_text = None
    cot = extra.get("cot") if isinstance(extra, dict) else None
    if cot is not None:
        cot_arr = np.asarray(cot)
        if cot_arr.size > 0:
            cot_text = str(cot_arr.reshape(-1)[0])
            if cot_text:
                print(f"=== Chain-of-Causation (frame {frame_idx}) ===\n{cot_text}")

    # ---- predicted trajectory summary -----------------------------------
    pred0 = pred_xyz_np[0]
    dist = float(np.linalg.norm(pred0[-1, :2]))
    print("=== Predicted trajectory (t0-local, sample 0) ===")
    print(f"  waypoints: {pred0.shape[0]} @ {DT:.1f}s -> {pred0.shape[0]*DT:.1f}s horizon")
    print(f"  start xy: ({pred0[0,0]:.2f}, {pred0[0,1]:.2f}) m")
    print(f"  end   xy: ({pred0[-1,0]:.2f}, {pred0[-1,1]:.2f}) m  (path ~{dist:.2f} m)")

    # ---- optional GT comparison (only frames that exist) -----------------
    first_fut = frame_idx + 1
    n_avail = max(0, n_frames - first_fut)
    n_cmp = min(n_avail, pred0.shape[0])
    gt_xy = None
    ade = fde = None
    gt_xyz_local = None
    if n_cmp > 0:
        fut_idx = np.arange(first_fut, first_fut + n_cmp)
        gt_xyz_local = (t0_rot_inv @ (xyz[fut_idx] - t0_xyz).T).T
        gt_xy = gt_xyz_local[:, :2]
        pred_xy = pred0[:n_cmp, :2]
        ade = float(np.mean(np.linalg.norm(pred_xy - gt_xy, axis=1)))
        fde = float(np.linalg.norm(pred_xy[-1] - gt_xy[-1]))
        print(f"  ADE: {ade:.3f} m   FDE: {fde:.3f} m  (over {n_cmp} steps)")

    # ---- project predictions + GT to front view --------------------------
    proj_per_sample = []
    for s in range(pred_xyz_np.shape[0]):
        pix, valid = project_ego_local_to_pixels(
            pred_xyz_np[s], ego_to_cam, intr_wide)
        proj_per_sample.append((pix, valid))
    gt_proj_pix = None
    gt_xyz_full = gt_xyz_local
    if gt_xyz_full is not None:
        gt_proj_pix, _ = project_ego_local_to_pixels(
            gt_xyz_full, ego_to_cam, intr_wide)

    front_img = cameras[WIDE_CAM_NAME][frame_idx]

    # ---- composite figure: front view + trajectory + CoT -----------------
    # ---- unified figure: front-view overlay + rotated BEV + CoT ----------
    frame_h, frame_w = front_img.shape[:2]
    SCALE = 2
    big = cv2.resize(front_img, (frame_w * SCALE, frame_h * SCALE),
                     interpolation=cv2.INTER_LANCZOS4)
    big = (big.astype(np.float32) * 0.6).astype(np.uint8)

    for s, (pix, valid) in enumerate(proj_per_sample):
        pred_local = pred_xyz_np[s]
        lp, lv, rp, rv = project_corridor(
            pred_local, half_width=1.0, ego_to_cam=ego_to_cam,
            intrinsics=intr_wide)
        big = draw_corridor_on_image(
            big, lp * SCALE, lv, rp * SCALE, rv,
            color=(0, 255, 255) if s == 0 else (255, 0, 255), alpha=0.5)
        d = np.linalg.norm(pred_local[:, :2], axis=1)
        cols = gradient_colors(len(pix), start=(255, 240, 60), end=(255, 30, 30))
        big = draw_polyline_on_image(
            big, pix * SCALE, colors=cols, radius=5, thickness=4,
            label_every=8, labels=[f"{x:.0f}" for x in d])

    if gt_proj_pix is not None and gt_xyz_full is not None and len(gt_xyz_full) >= 2:
        gl, glv, gr, grv = project_corridor(
            gt_xyz_full[:, :3], half_width=0.6, ego_to_cam=ego_to_cam,
            intrinsics=intr_wide)
        big = draw_corridor_on_image(
            big, gl * SCALE, glv, gr * SCALE, grv,
            color=(0, 255, 0), alpha=0.45)
        gt_d = np.linalg.norm(gt_xyz_full[:, :2], axis=1)
        big = draw_polyline_on_image(
            big, gt_proj_pix * SCALE, color=(0, 255, 0), radius=6, thickness=3,
            label_every=4, labels=[f"{d:.0f}" for d in gt_d])

    cv2.putText(big, "cyan corridor = AR1 pred   green = GT future "
                "(labels = forward distance [m]; line yellow->red = near->far)",
                (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7 * SCALE, (255, 255, 255),
                2 * SCALE, cv2.LINE_AA)
    cv2.line(big, (int(intr_wide["cx"] * SCALE) - 20,
                   int(intr_wide["cy"] * SCALE)),
             (int(intr_wide["cx"] * SCALE) + 20,
              int(intr_wide["cy"] * SCALE)), (200, 200, 200), 1)
    cv2.putText(big, f"t0 = {t0_s:.1f}s (frame {frame_idx})",
                (20, big.shape[0] - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7 * SCALE, (255, 255, 255),
                2 * SCALE, cv2.LINE_AA)

    # ---- matplotlib figure: overlay image + rotated BEV + CoT ------------
    # ---- precise layout: inches-based, no auto scaling -------------------
    dpi = 100
    img_h, img_w = big.shape[:2]
    img_w_in = img_w / dpi
    img_h_in = img_h / dpi

    title_h = 0.90   # inches for subplot title
    cot_h = 1.40     # inches for CoT
    suptitle_h = 0.90  # inches for overall suptitle
    margin = 0.15    # inches

    left_h = img_h_in + title_h + cot_h
    right_h = img_h_in + title_h  # BEV has its own title, no CoT
    bev_aspect = 2.5  # width = height / aspect (taller than wide)
    right_w = right_h / bev_aspect

    gap = 0.80  # inches between left image and BEV

    fig_w = margin + img_w_in + gap + right_w + margin
    fig_h = margin + suptitle_h + max(left_h, right_h) + margin

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)

    # Left: overlay image (title above, CoT below)
    ax_img = fig.add_axes([
        margin / fig_w,
        (margin + cot_h) / fig_h,
        img_w_in / fig_w,
        img_h_in / fig_h,
    ])
    ax_img.imshow(big)
    ax_img.set_title(f"Front view ({WIDE_CAM_NAME}) @ {t0_s:.1f}s", fontsize=36, pad=6)
    ax_img.axis("off")

    # CoT text below left image
    cot_ax = fig.add_axes([
        margin / fig_w,
        margin / fig_h,
        img_w_in / fig_w,
        cot_h / fig_h,
    ])
    cot_ax.axis("off")
    if cot_text:
        cot_wrapped = textwrap.fill(cot_text, width=65)
        cot_ax.text(0.0, 0.5, f"COT: {cot_wrapped}", ha="left", va="center",
                    family="monospace", fontsize=32, transform=cot_ax.transAxes)
    else:
        cot_ax.text(0.0, 0.5, "COT: (no CoT)", ha="left", va="center",
                    family="monospace", fontsize=32, transform=cot_ax.transAxes)

    # Right: BEV trajectory
    ax_trj = fig.add_axes([
        (margin + img_w_in + gap) / fig_w,
        (margin + cot_h) / fig_h,
        right_w / fig_w,
        img_h_in / fig_h,
    ])

    # Rotated BEV: forward (x) -> up
    if gt_xy is not None:
        gt_rot = rotate_90cc(gt_xy)
        ax_trj.plot(gt_rot[:, 0], gt_rot[:, 1], "g-o", ms=3, label="GT future")
    for s in range(pred_xyz_np.shape[0]):
        style = "r-" if s == 0 else "m--"
        pred_rot = rotate_90cc(pred_xyz_np[s, :, :2])
        ax_trj.plot(
            pred_rot[:, 0], pred_rot[:, 1], style,
            alpha=0.9, label="pred" if s == 0 else None,
        )
    hist_rot = rotate_90cc(hist_xyz_local[:, :2])
    ax_trj.plot(
        hist_rot[:, 0], hist_rot[:, 1], "k.-", ms=3,
        alpha=0.5, label="history",
    )
    ax_trj.scatter([0], [0], c="b", s=60, zorder=5, label="t0 (ego)")
    ax_trj.set_aspect("equal")
    ax_trj.set_xlabel("y (left) [m]", fontsize=28)
    ax_trj.set_ylabel("x (forward) [m]", fontsize=28)
    ax_trj.set_xlim(-15, 15)
    ax_trj.set_ylim(-5, 60)
    metric_str = f"  ADE={ade:.2f}m FDE={fde:.2f}m" if ade is not None else "  (no GT)"
    ax_trj.set_title(f"BEV trajectory{metric_str}", fontsize=36, pad=6)
    ax_trj.grid(True, alpha=0.3)
    ax_trj.legend(fontsize=28, loc="upper right")

    # Overall title
    fig.suptitle(f"{args.clip}  @  frame {frame_idx} ({t0_s:.1f}s)",
                 fontsize=36, y=(fig_h - margin - suptitle_h / 2) / fig_h)

    fig_path = out_dir / f"{args.clip}_f{frame_idx:03d}.png"
    fig.savefig(fig_path, dpi=dpi)
    plt.close(fig)
    print(f"  saved figure -> {fig_path}")


def _collect_clips(clip_arg) -> list:
    """Resolve the list of clip stems to process.

    ``clip_arg=None`` means "every .mp4 under
    DATA_DIR/camera/camera_front_wide_120fov``."
    """
    if clip_arg is not None:
        return [clip_arg]
    front_wide_dir = DATA_DIR / "camera" / WIDE_CAM_NAME
    if not front_wide_dir.exists():
        raise FileNotFoundError(
            "--clip not set and front-wide camera directory not found: "
            f"{front_wide_dir}"
        )
    clips = sorted(p.stem for p in front_wide_dir.glob("*.mp4"))
    if not clips:
        raise RuntimeError(f"No .mp4 clips found in {front_wide_dir}")
    print(
        f"--clip not specified; iterating over {len(clips)} clip(s) "
        f"under {front_wide_dir}"
    )
    return clips


def _encode_clip_video_pyx264(
    png_files: list,
    out_path: Path,
    fps: int = 5,
) -> None:
    """Encode a list of PNGs into an H.264 MP4 with PyAV (libx264).

    Why not ``cv2.VideoWriter``?
        OpenCV's ``mp4v``/``avc1`` backends do not always write correct
        per-sample durations, so some players report a longer duration
        in the progress bar than what the bitstream actually plays back
        (clip ends ~1 s early on a 4 s video). PyAV wraps FFmpeg's
        ``libx264`` encoder and writes well-formed, widely-compatible
        MP4s with proper 0.2 s per-frame timing.

    Notes:
        * ``x264-params keyint=1:min-keyint=1`` forces every frame to be
          a keyframe, which makes scrubbing / step-frame work without
          long GOP pauses. The cost is a slightly larger file, which is
          fine for a 1-fps-of-the-original sweep.
        * ``preset=ultrafast`` + ``tune=stillimage`` keep the encode
          fast for near-static figures.
    """
    container = av.open(str(out_path), mode="w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width = 0
    stream.height = 0
    stream.pix_fmt = "yuv420p"
    stream.options = {
        "preset": "ultrafast",
        "tune": "stillimage",
        "x264-params": "keyint=1:min-keyint=1",
    }
    try:
        for png in png_files:
            bgr = cv2.imread(str(png))
            if bgr is None:
                raise RuntimeError(f"failed to read {png}")
            if bgr.shape[1] != stream.width or bgr.shape[0] != stream.height:
                stream.width = bgr.shape[1]
                stream.height = bgr.shape[0]
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            frame.pts = None  # let libx264 assign monotonically
            for packet in stream.encode(frame):
                container.mux(packet)
        # Flush.
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()


def _compose_clip_video(clip: str, out_dir: Path, keep_frames: bool) -> None:
    """Stitch the per-frame PNGs of a single clip into a 5 fps MP4.

    Encoding is delegated to ``_encode_clip_video_pyx264`` (PyAV /
    libx264) because OpenCV's built-in ``mp4v`` / ``avc1`` backends
    write MP4s whose duration metadata disagrees with the actual
    playback time on some players. If encoding fails, the per-frame
    PNGs are kept so the user still has the figures.

    Per-frame PNGs are removed once the video is on disk unless
    ``keep_frames`` is set.
    """
    png_files = sorted(out_dir.glob(f"{clip}_f*.png"))
    if not png_files:
        return
    first = cv2.imread(str(png_files[0]))
    h, w = first.shape[:2]
    video_path = out_dir / f"{clip}.mp4"
    try:
        _encode_clip_video_pyx264(png_files, video_path, fps=5)
    except Exception as exc:  # noqa: BLE001 -- fall back to PNGs
        print(
            f"  warning: PyAV/libx264 encode failed for {clip}: {exc}; "
            f"keeping the per-frame PNGs."
        )
        return
    print(f"  saved video -> {video_path}")

    # Clean up per-frame PNGs once the video is on disk. The PNGs are
    # intermediate artifacts; keep disk tidy by default.
    if not keep_frames:
        removed = 0
        for p in png_files:
            try:
                p.unlink()
                removed += 1
            except OSError as e:
                print(f"  warning: failed to delete {p}: {e}")
        print(
            f"  cleaned up {removed}/{len(png_files)} per-frame PNG(s) "
            f"(use --keep-frames to retain them)"
        )


def _run_for_clip(
    clip: str,
    args,
    model,
    processor,
    out_dir: Path,
) -> None:
    """Run all requested frames for one clip.

    Loads the per-clip data, validates the frame range, projects the
    intrinsics, runs ``run_frame`` for every requested frame index, and
    finally stitches the per-frame PNGs into a 5 fps video.

    The model and processor are passed in (loaded once by ``main()``)
    so we do not pay the model-load cost for every clip.

    On any per-clip data error we log a warning and return without
    raising, so a single broken clip does not stop the loop.
    """
    # run_frame reads ``args.clip`` for the figure suptitle and the
    # output filename; shadow it with the current clip for this iteration.
    args.clip = clip
    print(f"\n{'=' * 70}\n=== clip {clip} ===\n{'=' * 70}")

    # ---- load data (once per clip) ----------------------------------
    print(f"Loading clip '{clip}' from {DATA_DIR}")
    try:
        cameras = load_cameras(clip)
    except Exception as exc:  # noqa: BLE001 -- keep the loop going
        print(f"  warning: failed to load cameras for {clip}: {exc}; skipping.")
        return
    if WIDE_CAM_NAME not in cameras:
        print(f"  warning: required camera {WIDE_CAM_NAME} not found in {clip}; skipping.")
        return

    try:
        xyz, rot = load_egomotion(DATA_DIR / "labels" / f"{clip}.json")
    except Exception as exc:  # noqa: BLE001 -- keep the loop going
        print(f"  warning: failed to load egomotion for {clip}: {exc}; skipping.")
        return
    n_frames = xyz.shape[0]
    for cam_name, vid in cameras.items():
        if vid.shape[0] != n_frames:
            print(
                f"  warning: {cam_name} has {vid.shape[0]} frames but egomotion has "
                f"{n_frames}; frame grid mismatch in {clip}; skipping."
            )
            return

    # ---- resolve the frame list for THIS clip ------------------------
    # Must have at least 1s (10 frames) of future data for GT comparison.
    max_allowed = n_frames - 10
    if args.frame_end is not None:
        # Explicit sweep: --frame-start (default MIN_FRAME_IDX) .. --frame-end.
        start = args.frame_start if args.frame_start is not None else MIN_FRAME_IDX
        frames = list(range(start, args.frame_end + 1))
    elif args.frame_idx is not None:
        # Explicit single frame.
        frames = [args.frame_idx]
    else:
        # Default (Plan A): auto-sweep from MIN_FRAME_IDX to the last
        # frame that still leaves >=1 s of GT future data, so the
        # stitched MP4 actually animates through the whole valid range.
        frames = list(range(MIN_FRAME_IDX, max_allowed + 1))

    # ---- validate & clamp the frame range ----------------------------
    frames = [f for f in frames if f <= max_allowed]
    if not frames:
        print(
            f"  warning: all requested frames exceed max allowed {max_allowed} "
            f"(need >=1s future data, {clip} has {n_frames} frames @ 10Hz); skipping."
        )
        return
    for f in frames:
        if f < MIN_FRAME_IDX:
            print(
                f"  warning: frame {f} < {MIN_FRAME_IDX} (t0 must be >= "
                f"{MIN_FRAME_IDX * DT:.1f} s so the {NUM_HISTORY_STEPS}-step history "
                f"window fits) for {clip}; skipping."
            )
            return

    print(
        f"  egomotion: {n_frames} frames @ {1/DT:.0f} Hz "
        f"({n_frames * DT:.1f} s)"
    )
    print(
        f"  frames to process: {frames[0]}..{frames[-1]} "
        f"({len(frames)} frame{'s' if len(frames) > 1 else ''})"
    )

    # ---- projection calibration (once per clip; frame size is constant) ---
    frame_h, frame_w = cameras[WIDE_CAM_NAME].shape[1:3]
    try:
        intr_wide = load_intrinsics_for_clip(
            DATA_DIR, clip, WIDE_CAM_NAME,
            target_w=frame_w, target_h=frame_h)
        ext_all = load_extrinsics(DATA_DIR, clip)
    except Exception as exc:  # noqa: BLE001 -- keep the loop going
        print(f"  warning: failed to load calibration for {clip}: {exc}; skipping.")
        return
    print(
        f"  intrinsics rescaled to {frame_w}x{frame_h} "
        f"(factor {intr_wide['scale']:.4f}); cx={intr_wide['cx']:.1f} "
        f"cy={intr_wide['cy']:.1f}"
    )
    ego_poses = np.stack([
        np.block([[r, t.reshape(3, 1)], [np.zeros((1, 3)), np.ones((1, 1))]])
        for r, t in zip(rot, xyz)
    ], axis=0)
    ego_to_cam = np.linalg.inv(static_mount(ego_poses, ext_all[WIDE_CAM_NAME]))

    # ---- per-frame loop ----------------------------------------------
    for frame_idx in frames:
        run_frame(
            frame_idx, args, cameras, xyz, rot, n_frames, model, processor,
            intr_wide, ego_to_cam, out_dir,
        )
    print(f"\nAll done for {clip}: {len(frames)} frame(s) -> {out_dir}")

    # ---- compose video (0.2s per frame = 5 fps) ----------------------
    _compose_clip_video(clip, out_dir, args.keep_frames)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve the clip list (--clip None -> iterate all front-wide clips).
    clips = _collect_clips(args.clip)

    # ---- load model (once) ----------------------------------------------
    print(f"Loading model from {args.ckpt} ...")
    model = Alpamayo1_5.from_pretrained(args.ckpt, dtype=torch.bfloat16).to("cuda")
    model.eval()
    processor = helper.get_processor(model.tokenizer)

    for i, clip in enumerate(clips, start=1):
        print(
            f"\n[coc] === [{i}/{len(clips)}] clip={clip} ==="
        )
        _run_for_clip(clip, args, model, processor, out_dir)


if __name__ == "__main__":
    main()
