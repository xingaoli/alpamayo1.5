"""VQA inference for Alpamayo 1.5 on locally-generated (wfm_gen_data) samples.

Replicates notebooks/inference_vqa.ipynb as a CLI:

  - Loads images by decoding mp4s under wfm_gen_data/camera/<cam>/<clip>.mp4
    with OpenCV (already available in the venv).
  - Reads ego poses from wfm_gen_data/labels/<clip>.json.
  - Builds the same dict schema returned by
    `alpamayo1_5.load_physical_aiavdataset.load_physical_aiavdataset`
    so the rest of the pipeline (`helper.create_vqa_message`,
    `processor.apply_chat_template`, `model.generate_text`) is unchanged.
  - Runs the chat template + autoregressive VLM generation and prints the
    answer.

Run with the existing venv (no install needed):

  source /home/xingao/code/Alpamayo/.venv/bin/activate
  export PYTHONPATH=/home/xingao/code/Alpamayo1.5/src

Common invocations:

1. **Default** (single front wide camera, default question, default
   t0=2.5 s, all clips under
   ``wfm_root/camera/camera_front_wide_120fov``)::
    PYTHONPATH=/home/xingao/code/Alpamayo1.5/src \
    python wfm_gen_test/vqa_test/vqa_inference.py \
    --question "What is happening on the road ahead? Describe the hazards in detail." \
    --out-dir wfm_gen_test/vqa_test/ar1_5_wfm_outputs/test1

2. **Custom question** on a chosen clip with an explicit t0 time::
    PYTHONPATH=/home/xingao/code/Alpamayo1.5/src \
    python wfm_gen_test/vqa_test/vqa_inference.py \
      --clip 71ae0d63-ba64-4517-b3f1-90d446abc67d_1869020100000_1869040100000_fire \
      --t0-seconds 3.0 \
      --question "What is happening on the road ahead? Describe the hazards in detail."

3. **Multi-camera input** (front wide + cross left + cross right)::

    python wfm_gen_test/vqa_test/vqa_inference.py \
      --cameras camera_front_wide_120fov \
                camera_cross_left_120fov \
                camera_cross_right_120fov \
      --num-frames 4

4. **Different t0 / data root / ckpt** (point at the in-tree data
   directory and a custom checkpoint)::

    python wfm_gen_test/vqa_test/vqa_inference.py \
      --wfm-root data/wfm_gen_data \
      --clip smoke_260616_001 \
      --t0-seconds 1.5 \
      --ckpt /abs/path/to/Alpamayo-1.5-10B

5. **Custom output directory** (writes
   ``/tmp/my_outputs/vqa_result_<clip>.png``)::

    python wfm_gen_test/vqa_test/vqa_inference.py \
      --out-dir /tmp/my_outputs

6. **Exact output path** (overrides ``--out-dir``)::

    python wfm_gen_test/vqa_test/vqa_inference.py \
      --save-image /tmp/qa_snapshot.png

7. **Skip the visualization PNG** (only print the answer)::

    python wfm_gen_test/vqa_test/vqa_inference.py --no-visualize

Environment variables (override the defaults; CLI flags take precedence):

  * ``CUDA_VISIBLE_DEVICES`` -- which GPU to use (default 0).
  * ``ALPAMAYO_MODEL_CKPT`` -- model checkpoint path (default
    ``<repo>/ckpts/Alpamayo-1.5-10B``).
  * ``WFM_GEN_DATA_DIR`` -- world-model data root (default
    ``/home/share_files/wfm_gen_data``).
  * ``ALPAMAYO_PROCESSOR_PATH`` -- Qwen3-VL processor checkpoint; auto-set
    to ``<repo>/ckpts/Qwen3-VL-2B-Instruct`` if that path exists.

Outputs
-------
The Q/A visualization PNG is written to
``wfm_gen_test/vqa_test/ar1_5_wfm_outputs/vqa_result_<clip>.png`` by
default (mirroring the ``coc_test`` layout). Override the directory
with ``--out-dir`` or specify an exact path with ``--save-image``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Resolve the local Qwen3-VL processor checkpoint relative to this file so
# the script works regardless of cwd. helper.py picks this up via the
# ALPAMAYO_PROCESSOR_PATH env var.
_REPO_ROOT = Path(__file__).resolve().parent
_LOCAL_PROCESSOR = _REPO_ROOT / "ckpts" / "Qwen3-VL-2B-Instruct"
if _LOCAL_PROCESSOR.exists():
    os.environ.setdefault("ALPAMAYO_PROCESSOR_PATH", str(_LOCAL_PROCESSOR))

# Default output directory for the Q/A visualization PNG, mirroring the
# ``coc_test`` layout (``wfm_gen_test/coc_test/ar1_5_wfm_outputs``).
DEFAULT_OUT_DIR = _REPO_ROOT / "ar1_5_wfm_outputs"

import cv2
import numpy as np
import scipy.spatial.transform as spt
import torch
from einops import rearrange

# Local package — import via src/ on PYTHONPATH; do NOT install.
from alpamayo1_5 import helper
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5


# Camera name -> index used by `helper.CAMERA_DISPLAY_NAMES` and the
# training-time prompt. Mirrors load_physical_aiavdataset.camera_name_to_index.
CAMERA_NAME_TO_INDEX = {
    "camera_cross_left_120fov": 0,
    "camera_front_wide_120fov": 1,
    "camera_cross_right_120fov": 2,
    "camera_rear_left_70fov": 3,
    "camera_rear_tele_30fov": 4,
    "camera_rear_right_70fov": 5,
    "camera_front_tele_30fov": 6,
}

# Default to the single front camera used by the original VQA notebook.
DEFAULT_CAMERAS = ["camera_front_wide_120fov"]


def decode_camera_mp4(
    mp4_path: Path,
    frame_indices: list[int],
) -> np.ndarray:
    """Decode specific frame indices from an mp4 with OpenCV.

    Args:
        mp4_path: Path to the mp4 file.
        frame_indices: 0-based frame indices in increasing order.

    Returns:
        uint8 array of shape ``(len(frame_indices), H, W, 3)`` in BGR.
        The returned frames are converted to RGB by swapping channels.
    """
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {mp4_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in frame_indices:
        if i < 0 or i >= total:
            cap.release()
            raise IndexError(
                f"Frame index {i} out of range (0..{total - 1}) in {mp4_path}"
            )

    frames: list[np.ndarray] = []
    # Seek frames in sorted order; reset between reads to be safe.
    last_pos = -1
    for i in frame_indices:
        if i != last_pos + 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError(f"Failed to read frame {i} from {mp4_path}")
        # cv2 returns BGR; the rest of the pipeline (Qwen processor, mediapy)
        # expects RGB.
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        last_pos = i
    cap.release()
    return np.stack(frames, axis=0)


def save_qa_visualization(
    image_frames: torch.Tensor,
    question: str,
    answer: str,
    out_path: Path,
) -> Path:
    """Save the current-moment front-view frame with Q/A text overlay.

    Renders a single image (the last frame of the first camera — i.e. t0)
    with the question drawn above and the answer drawn below, using
    OpenCV only (no GUI, no popup).

    Args:
        image_frames: ``(N_cameras, num_frames, 3, H, W)`` uint8 tensor (RGB).
        question: Question text (will be drawn above the image).
        answer: Answer text (will be drawn below the image).
        out_path: Destination PNG path.

    Returns:
        The path the image was written to.
    """
    # Take the last frame of the first camera (current moment, t0).
    frame = image_frames[0, -1].permute(1, 2, 0).contiguous().numpy()  # (H, W, 3) RGB
    h, w, _ = frame.shape

    font = cv2.FONT_HERSHEY_SIMPLEX
    # Scale font with image width so it stays readable at any resolution.
    font_scale = max(0.5, w / 1280.0)
    thickness = max(1, int(font_scale * 2))
    line_h = int(font_scale * 30) + 6  # vertical spacing per line

    def wrap_text(text: str, max_width_px: int) -> list[str]:
        """Greedy word-wrap text so each line fits within max_width_px."""
        words = text.split()
        lines: list[str] = []
        cur: list[str] = []
        for word in words:
            trial = " ".join(cur + [word])
            (tw, _), _ = cv2.getTextSize(trial, font, font_scale, thickness)
            if tw <= max_width_px or not cur:
                cur.append(word)
            else:
                lines.append(" ".join(cur))
                cur = [word]
        if cur:
            lines.append(" ".join(cur))
        return lines

    pad = 10
    q_lines = wrap_text(f"Q: {question}", w - 2 * pad)
    a_lines = wrap_text(f"A: {answer}", w - 2 * pad)
    q_block_h = line_h * len(q_lines) + 2 * pad
    a_block_h = line_h * len(a_lines) + 2 * pad

    # Top band: question (white text on black); bottom band: answer (same).
    top_band = np.zeros((q_block_h, w, 3), dtype=np.uint8)
    bot_band = np.zeros((a_block_h, w, 3), dtype=np.uint8)
    for i, line in enumerate(q_lines):
        cv2.putText(
            top_band, line,
            (pad, pad + (i + 1) * line_h - 6),
            font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA,
        )
    for i, line in enumerate(a_lines):
        cv2.putText(
            bot_band, line,
            (pad, pad + (i + 1) * line_h - 6),
            font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA,
        )

    canvas = np.concatenate([top_band, frame, bot_band], axis=0)
    cv2.imwrite(str(out_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    return out_path


def load_wfm_gen_data(
    wfm_root: Path,
    clip_name: str,
    camera_names: list[str],
    num_frames: int = 4,
    t0_index: int = -1,
) -> dict:
    """Build a `load_physical_aiavdataset`-shaped dict from wfm_gen_data.

    The wfm_gen_data layout (per clip):
        wfm_root/camera/<cam_name>/<clip_name>.mp4    (RGB, 10 Hz)
        wfm_root/labels/<clip_name>.json              (N_frames, 4, 4) ego poses

    Frame selection: pick `num_frames` consecutive frames ending at
    `t0_index` (default: the last frame in the video, which mirrors the
    notebook's `t0_us=5_100_000` choice of "use frames near the end of the
    clip"). Ego history / future are derived from the full pose list so
    downstream consumers get the same tensor shapes as the HF path.

    Returns:
        Dict with the same keys as `load_physical_aiavdataset`.
    """
    cam_dirs: list[Path] = []
    for cam in camera_names:
        cam_dirs.append(wfm_root / "camera" / cam / f"{clip_name}.mp4")
    labels_path = wfm_root / "labels" / f"{clip_name}.json"

    if not labels_path.exists():
        raise FileNotFoundError(f"Labels not found: {labels_path}")

    # Validate camera directories exist (wfm_gen_data usually only contains a
    # subset of the HF dataset's 7 cameras). This catches typos and missing
    # data early with a clear error message.
    available = sorted(
        d.name
        for d in (wfm_root / "camera").iterdir()
        if d.is_dir()
    )
    for cam, mp4_path in zip(camera_names, cam_dirs):
        if cam not in CAMERA_NAME_TO_INDEX:
            raise ValueError(
                f"Unknown camera {cam!r}. Valid names: "
                f"{sorted(CAMERA_NAME_TO_INDEX)}"
            )
        if not mp4_path.exists():
            raise FileNotFoundError(
                f"Camera video not found: {mp4_path}\n"
                f"Available cameras under {wfm_root / 'camera'}: {available}"
            )

    with open(labels_path) as f:
        poses_world = np.array(json.load(f), dtype=np.float64)  # (N, 4, 4)
    n_total = int(poses_world.shape[0])
    if n_total < 2:
        raise ValueError(f"Need >= 2 pose samples, got {n_total}")

    # Resolve t0 frame index (default: last frame in the video).
    if t0_index < 0:
        t0_index = n_total + t0_index  # e.g. -1 -> n_total - 1
    if t0_index < num_frames - 1 or t0_index >= n_total:
        raise ValueError(
            f"t0_index={t0_index} is out of range [num_frames-1, n_total) "
            f"= [{num_frames - 1}, {n_total})"
        )

    # Image frames: [t0 - (num_frames-1), ..., t0]
    image_frame_indices = list(range(t0_index - num_frames + 1, t0_index + 1))

    # Pose sampling: use the same `time_step` as the original loader
    # (10 Hz). For VQA, ego history / future are returned for schema
    # parity but not consumed by `create_vqa_message` / `generate_text`.
    time_step = 0.1
    num_history_steps = 16
    num_future_steps = 64
    history_offsets = np.arange(
        -(num_history_steps - 1) * time_step,
        time_step / 2,
        time_step,
    )
    future_offsets = np.arange(
        time_step,
        (num_future_steps + 0.5) * time_step,
        time_step,
    )

    def _idx_at_offset(off_s: float) -> int:
        # Round to nearest 10 Hz frame. Clamp to the clip boundary — ego
        # history/future are returned only for schema parity and are not
        # consumed by the VQA path, so clamping (repeating the boundary
        # pose) is harmless and avoids requiring the clip to span the full
        # 1.6 s history + 6.4 s future window.
        return int(round(t0_index + off_s / time_step))

    history_indices = [
        max(0, min(_idx_at_offset(off), n_total - 1)) for off in history_offsets
    ]
    future_indices = [
        max(0, min(_idx_at_offset(off), n_total - 1)) for off in future_offsets
    ]

    # Decode images.
    image_frames_list: list[torch.Tensor] = []
    camera_indices_list: list[int] = []
    for cam_name, mp4_path in zip(camera_names, cam_dirs):
        frames_np = decode_camera_mp4(mp4_path, image_frame_indices)
        frames_t = torch.from_numpy(frames_np)            # (T, H, W, 3)
        frames_t = rearrange(frames_t, "t h w c -> t c h w")
        image_frames_list.append(frames_t)
        camera_indices_list.append(CAMERA_NAME_TO_INDEX[cam_name])

    image_frames = torch.stack(image_frames_list, dim=0)            # (Nc, T, 3, H, W)
    camera_indices = torch.tensor(camera_indices_list, dtype=torch.int64)

    # Sort by camera index to match the HF loader's behavior.
    sort_order = torch.argsort(camera_indices)
    image_frames = image_frames[sort_order]
    camera_indices = camera_indices[sort_order]

    # Build ego history / future in the t0-local frame (matches
    # load_physical_aiavdataset's convention).
    t0_pose = poses_world[t0_index]
    t0_rot = spt.Rotation.from_matrix(t0_pose[:3, :3])
    t0_rot_inv = t0_rot.inv()
    t0_xyz = t0_pose[:3, 3]

    def _local_xyz(idx: int) -> np.ndarray:
        return t0_rot_inv.apply(poses_world[idx, :3, 3] - t0_xyz)

    def _local_rot(idx: int) -> np.ndarray:
        return (
            t0_rot_inv * spt.Rotation.from_matrix(poses_world[idx, :3, :3])
        ).as_matrix()

    ego_history_xyz = np.stack(
        [_local_xyz(i) for i in history_indices], axis=0
    ).astype(np.float32)
    ego_history_rot = np.stack(
        [_local_rot(i) for i in history_indices], axis=0
    ).astype(np.float32)
    ego_future_xyz = np.stack(
        [_local_xyz(i) for i in future_indices], axis=0
    ).astype(np.float32)
    ego_future_rot = np.stack(
        [_local_rot(i) for i in future_indices], axis=0
    ).astype(np.float32)

    add_batch = lambda x: torch.from_numpy(x).float().unsqueeze(0).unsqueeze(0)
    relative_timestamps = (
        torch.arange(num_frames, dtype=torch.float32) * time_step
    ).unsqueeze(0).expand(len(camera_names), -1).contiguous()
    absolute_timestamps = torch.tensor(image_frame_indices, dtype=torch.int64)
    absolute_timestamps = absolute_timestamps.unsqueeze(0).expand(
        len(camera_names), -1
    ).contiguous()

    return {
        "image_frames": image_frames,
        "camera_indices": camera_indices,
        "ego_history_xyz": add_batch(ego_history_xyz),
        "ego_history_rot": add_batch(ego_history_rot),
        "ego_future_xyz": add_batch(ego_future_xyz),
        "ego_future_rot": add_batch(ego_future_rot),
        "relative_timestamps": relative_timestamps,
        "absolute_timestamps": absolute_timestamps,
        "t0_us": -1,  # Not meaningful for wfm_gen_data; VQA path ignores it.
        "clip_id": clip_name,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Alpamayo 1.5 VQA on a wfm_gen_data sample.",
    )
    parser.add_argument(
        "--wfm-root",
        type=Path,
        default=Path("/home/share_files/wfm_gen_data"),
        help="Root directory of the locally-generated wfm_gen_data.",
    )
    parser.add_argument(
        "--clip",
        type=str,
        default=None,
        help=(
            "Clip name (basename of the mp4 and labels json, no extension). "
            "If not set, inference runs once on every .mp4 clip found under "
            "``wfm_root/camera/camera_front_wide_120fov``."
        ),
    )
    parser.add_argument(
        "--question",
        type=str,
        default="Describe the scene.",
        help="Question to ask the model.",
    )
    parser.add_argument(
        "--cameras",
        type=str,
        nargs="+",
        default=DEFAULT_CAMERAS,
        help=(
            "Camera directory names under wfm_root/camera to use. "
            "Defaults to the single front wide camera used by the notebook."
        ),
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=4,
        help="Number of temporal frames per camera (default 4).",
    )
    parser.add_argument(
        "--t0-seconds",
        type=float,
        default=2.5,
        help=(
            "Time (in seconds) of t0 in the source mp4 (default 2.5 s). "
            "Converted to a frame index assuming 10 Hz (the wfm_gen_data "
            "convention). Ignored when --t0-index is set explicitly. If "
            "the requested t0 is past the end of a short clip it is "
            "clamped to the latest valid frame."
        ),
    )
    parser.add_argument(
        "--t0-index",
        type=int,
        default=None,
        help=(
            "Frame index used as t0 in the source mp4. If not set, "
            "--t0-seconds is used (default 2.5 s → 25 frames at 10 Hz). "
            "Negative values count from the end (e.g. -1 = last frame)."
        ),
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="ckpts/Alpamayo-1.5-10B",
        help=(
            "Path or HF hub id of the model checkpoint. The local ckpts "
            "symlink resolves to /home/checkpoints/nvidia/Alpamayo-1.5-10B."
        ),
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.98,
        help="Nucleus sampling top-p.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="CUDA seed for reproducibility.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=(
            "Directory to write the Q/A visualization PNG into (created if "
            "missing). Defaults to "
            "``wfm_gen_test/vqa_test/ar1_5_wfm_outputs``, mirroring the "
            "``coc_test`` layout."
        ),
    )
    parser.add_argument(
        "--save-image",
        type=Path,
        default=None,
        help=(
            "Write the current-moment front-view frame with Q/A overlay to "
            "this exact PNG path (OpenCV, no popup). Overrides --out-dir. "
            "Use --no-visualize to skip."
        ),
    )
    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="Skip saving the visualization PNG.",
    )
    return parser.parse_args()


# Number of source-video frames per second; the wfm_gen_data mp4s are
# encoded at 10 Hz. Used to convert --t0-seconds -> frame index.
WFM_FPS = 10


def _probe_frame_count(mp4_path: Path) -> int:
    """Return the frame count of an mp4, or -1 on failure."""
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        cap.release()
        return -1
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def _resolve_t0_index(
    mp4_path: Path,
    requested_t0_index,
    requested_t0_seconds: float,
    num_frames: int,
) -> int:
    """Resolve the per-clip t0 frame index from CLI args.

    - Explicit ``--t0-index`` (positive or negative) takes precedence.
    - Otherwise ``--t0-seconds`` is converted to a frame index at 10 Hz
      and clamped to a valid range for the clip.
    """
    n_total = _probe_frame_count(mp4_path)
    if requested_t0_index is not None:
        idx = requested_t0_index
        if idx < 0 and n_total > 0:
            return n_total + idx
        return idx
    idx = int(round(requested_t0_seconds * WFM_FPS))
    if n_total <= 0:
        return idx
    safe = max(num_frames - 1, n_total - 1)
    if idx > safe:
        print(
            f"[vqa] WARNING: requested t0={idx} "
            f"({requested_t0_seconds:.2f} s) is past the end of "
            f"{mp4_path.name} ({n_total} frames @ {WFM_FPS} Hz); "
            f"clamping to t0={safe}."
        )
        idx = safe
    return idx


def _collect_clips(wfm_root: Path, clip_arg):
    """Return the list of clip names to run inference on.

    ``clip_arg=None`` means "iterate every .mp4 under
    wfm_root/camera/camera_front_wide_120fov``."
    """
    if clip_arg is not None:
        return [clip_arg]
    front_wide_dir = wfm_root / "camera" / "camera_front_wide_120fov"
    if not front_wide_dir.exists():
        raise FileNotFoundError(
            "--clip not set and front-wide camera directory not found: "
            f"{front_wide_dir}"
        )
    clips = sorted(p.stem for p in front_wide_dir.glob("*.mp4"))
    if not clips:
        raise RuntimeError(f"No .mp4 clips found in {front_wide_dir}")
    print(
        f"[vqa] --clip not specified; iterating over {len(clips)} clip(s) "
        f"under {front_wide_dir}"
    )
    return clips


def _run_inference_for_clip(
    args,
    model,
    processor,
    clip_name: str,
    t0_index: int,
) -> None:
    """Run VQA inference for a single clip; print + save the answer."""
    print(
        f"[vqa] Loading data from {args.wfm_root} "
        f"(clip={clip_name}, t0_index={t0_index})..."
    )
    data = load_wfm_gen_data(
        wfm_root=args.wfm_root,
        clip_name=clip_name,
        camera_names=args.cameras,
        num_frames=args.num_frames,
        t0_index=t0_index,
    )
    print(
        f"[vqa] image_frames: {tuple(data['image_frames'].shape)}, "
        f"camera_indices: {data['camera_indices'].tolist()}"
    )

    print(f"[vqa] Building chat message for question: {args.question!r}")
    messages = helper.create_vqa_message(
        data["image_frames"].flatten(0, 1),
        question=args.question,
        camera_indices=data["camera_indices"],
    )
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )
    print(f"[vqa] seq length: {tuple(inputs.input_ids.shape)}")

    model_inputs = {"tokenized_data": inputs}
    model_inputs = helper.to_device(model_inputs, "cuda")

    torch.cuda.manual_seed_all(args.seed)
    print("[vqa] Generating answer...")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        extra = model.generate_text(
            data=model_inputs,
            top_p=args.top_p,
            temperature=args.temperature,
            num_samples=1,
            max_generation_length=args.max_new_tokens,
        )

    answer = extra["answer"][0]
    print("\nQuestion:\n", args.question)
    print("Answer:\n", answer)

    if not args.no_visualize:
        if args.save_image is not None:
            out_path = args.save_image
        else:
            args.out_dir.mkdir(parents=True, exist_ok=True)
            out_path = args.out_dir / f"vqa_result_{clip_name}.png"
        save_qa_visualization(
            data["image_frames"], args.question, answer, out_path
        )
        print(f"[vqa] Saved Q/A visualization to {out_path}")


def main() -> None:
    args = parse_args()

    if not args.wfm_root.exists():
        raise FileNotFoundError(f"wfm_root does not exist: {args.wfm_root}")

    clips = _collect_clips(args.wfm_root, args.clip)

    print(f"[vqa] Loading model from {args.ckpt}...")
    model = Alpamayo1_5.from_pretrained(args.ckpt, dtype=torch.bfloat16).to("cuda")
    processor = helper.get_processor(model.tokenizer)

    for i, clip in enumerate(clips, start=1):
        print(f"\n[vqa] === [{i}/{len(clips)}] clip={clip} ===")
        mp4_path = (
            args.wfm_root / "camera" / "camera_front_wide_120fov" /
            f"{clip}.mp4"
        )
        t0_index = _resolve_t0_index(
            mp4_path,
            args.t0_index,
            args.t0_seconds,
            args.num_frames,
        )
        try:
            _run_inference_for_clip(
                args, model, processor, clip, t0_index
            )
        except Exception as exc:  # noqa: BLE001 -- keep the loop going
            print(
                f"[vqa] ERROR: inference failed for clip {clip!r}: {exc}"
            )
            continue


if __name__ == "__main__":
    main()
