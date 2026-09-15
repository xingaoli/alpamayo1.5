#!/usr/bin/env python3
"""rds_hq waymo dataloader.

Reads the pre-processed waymo dataset (``data/waymo_rds_hq`` ->
``/home/datasets/chengguang/waymo_rds_hq``) and emits samples in the exact
contract of ``load_physical_aiavdataset_local`` (the PAI loader used by the
alpamayo training repo and alpamayo1.5 inference), so the model side needs no
changes.

Layout (per split ``train`` / ``val``, per segment ``<seg>``):

  pinhole_front/<seg>.mp4 (etc.)    1920x1280 @10Hz camera videos (198 frames)
  pose/<seg>.tar                    {seg}.{idx:06d}.pose.{cam}.npy        4x4 world pose
  vehicle_pose/<seg>.tar            {seg}.{idx:06d}.vehicle_pose.npy      4x4 world pose
  timestamp/<seg>.tar               {seg}.{idx:06d}.timestamp_micros.txt  int (us)
  pinhole_intrinsic/<seg>.tar       {seg}.pinhole_intrinsic.{cam}.npy     [fx,fy,cx,cy,w,h]
  all_object_info/<seg>.tar         {seg}.{idx:06d}.all_object_info.json  (collision eval)
  3d_lanelines/ 3d_road_boundaries/ ... <seg>.tar                         (collision eval)

All stream tars hold data on a 30 Hz frame grid (idx 0..593); the real 10 Hz
keyframes sit at ``idx % 3 == 0`` and only those carry timestamps.  We build
the egomotion from the 198 keyframe ``vehicle_pose`` entries + their true
timestamps and interpolate onto the 0.1 s query grid (mirroring the PAI
``EgomotionState`` interpolator, which interpolates 100 Hz egomotion).

Camera mapping onto the trained 4-cam PAI layout
(cross_left=0, front_wide=1, cross_right=2, front_tele=6):
  pinhole_front_left -> 0, pinhole_front -> 1, pinhole_front_right -> 2,
  and the front_tele slot is filled with pinhole_front (waymo has no narrow
  FOV camera) — the same trick as the original waymo_eval script.

Ego reference frame is the *vehicle* pose (rds_hq ships real vehicle poses,
unlike the old extracted data where the front camera doubled as ego).
"""

from __future__ import annotations

import io
import os
import tarfile
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp

_REPO_ROOT = Path(__file__).resolve().parent.parent

# (rds camera stream name, model camera index) — must stay sorted by model idx.
CAMERA_SLOTS: tuple[tuple[str, int], ...] = (
    ("pinhole_front_left", 0),   # PAI cross_left slot
    ("pinhole_front", 1),        # PAI front_wide slot
    ("pinhole_front_right", 2),  # PAI cross_right slot
    ("pinhole_front", 6),        # PAI front_tele slot (filled from front)
)
FRONT_CAM = "pinhole_front"


def _read_tar_members(tar_path: Path, suffix_filter: str | None = None
                      ) -> dict[str, bytes]:
    """Read all members of a small metadata tar into memory ({name: bytes})."""
    out: dict[str, bytes] = {}
    with tarfile.open(tar_path, "r") as tf:
        for m in tf:
            if not m.isfile():
                continue
            if suffix_filter is not None and not m.name.endswith(suffix_filter):
                continue
            f = tf.extractfile(m)
            if f is not None:
                out[m.name] = f.read()
    return out


def _member_idx(name: str, seg: str) -> int:
    """``<seg>.000123.suffix`` -> 123."""
    stem = name.split("/")[-1]
    return int(stem[len(seg) + 1:].split(".")[0])


class RDSWaymoClip:
    """Lazy per-segment reader for one rds_hq clip."""

    def __init__(self, root: Path, split: str, clip_id: str, preload: bool = True):
        self.root = Path(root)
        self.split = split
        self.clip_id = clip_id
        self.preload = preload

        # --- timestamps (canonical 10Hz keyframe grid) ---
        ts_raw = _read_tar_members(
            self.root / split / "timestamp" / f"{clip_id}.tar",
            suffix_filter=".timestamp_micros.txt")
        if not ts_raw:
            raise RuntimeError(f"empty timestamp tar for {clip_id}")
        pairs = sorted((_member_idx(n, clip_id), int(content.decode().strip()))
                       for n, content in ts_raw.items())
        self.keyframe_idx = np.asarray([p[0] for p in pairs], dtype=np.int64)
        self.ts_us = np.asarray([p[1] for p in pairs], dtype=np.int64)
        if not np.all(np.diff(self.ts_us) > 0):
            raise RuntimeError(f"non-monotonic timestamps in {clip_id}")
        self.n_keyframes = len(self.ts_us)

        # --- vehicle poses at keyframes (4x4, world frame) ---
        vp_raw = _read_tar_members(
            self.root / split / "vehicle_pose" / f"{clip_id}.tar",
            suffix_filter=".vehicle_pose.npy")
        if not vp_raw:
            raise RuntimeError(f"empty vehicle_pose tar for {clip_id}")
        poses = np.zeros((self.n_keyframes, 4, 4), dtype=np.float64)
        poses[:, 3, 3] = 1.0
        n_filled = 0
        for n, content in vp_raw.items():
            idx = _member_idx(n, clip_id)
            i = np.searchsorted(self.keyframe_idx, idx)
            if i < self.n_keyframes and self.keyframe_idx[i] == idx:
                poses[i] = np.load(io.BytesIO(content))
                n_filled += 1
        if n_filled != self.n_keyframes:
            raise RuntimeError(
                f"{clip_id}: vehicle_pose tar covers {n_filled}/"
                f"{self.n_keyframes} keyframes")
        self.vehicle_poses = poses

        t_sec = self.ts_us.astype(np.float64) * 1e-6
        self._slerp = Slerp(t_sec, Rotation.from_matrix(poses[:, :3, :3]))

        # --- videos ---
        self._videos: dict[str, np.ndarray | cv2.VideoCapture] = {}
        if preload:
            for cam in {c for c, _ in CAMERA_SLOTS}:
                self._videos[cam] = self._decode_video(cam)

    # ------------------------------------------------------------------ videos
    def _video_path(self, cam: str) -> Path:
        return self.root / self.split / cam / f"{self.clip_id}.mp4"

    def _decode_video(self, cam: str) -> np.ndarray:
        cap = cv2.VideoCapture(str(self._video_path(cam)))
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video: {self._video_path(cam)}")
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        if not frames:
            raise RuntimeError(f"no frames decoded: {self._video_path(cam)}")
        return np.stack(frames)  # [T, H, W, 3] RGB

    def decode_frames(self, cam: str, frame_idx: np.ndarray) -> np.ndarray:
        """Decode keyframe indices -> [T, H, W, 3] RGB."""
        v = self._videos.get(cam)
        if isinstance(v, np.ndarray):
            if frame_idx.max() >= v.shape[0] or frame_idx.min() < 0:
                raise IndexError(
                    f"{self.clip_id}/{cam}: frame {frame_idx} outside "
                    f"[0, {v.shape[0] - 1}]")
            return v[frame_idx]
        cap = (v if isinstance(v, cv2.VideoCapture)
               else cv2.VideoCapture(str(self._video_path(cam))))
        self._videos[cam] = cap
        out = []
        for i in frame_idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"{self.clip_id}/{cam}: seek to {i} failed")
            out.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return np.stack(out)

    # ------------------------------------------------------------------ poses
    def pose_at(self, ts_us) -> np.ndarray:
        """Interpolated vehicle pose [N, 4, 4] at absolute timestamps (us)."""
        t = np.atleast_1d(np.asarray(ts_us, dtype=np.float64)) * 1e-6
        t0, t1 = self.ts_us[0] * 1e-6, self.ts_us[-1] * 1e-6
        if t.min() < t0 or t.max() > t1:
            raise ValueError(
                f"{self.clip_id}: query t=[{t.min():.3f},{t.max():.3f}]s outside "
                f"clip span [{t0:.3f},{t1:.3f}]s")
        xyz = np.stack(
            [np.interp(t, self.ts_us * 1e-6, self.vehicle_poses[:, k, 3])
             for k in range(3)], axis=1)
        rot = self._slerp(t).as_matrix()
        out = np.zeros((len(t), 4, 4), dtype=np.float64)
        out[:, :3, :3] = rot
        out[:, :3, 3] = xyz
        out[:, 3, 3] = 1.0
        return out

    def keyframe_index_at(self, ts_us: int) -> int:
        """Nearest keyframe index for an absolute timestamp."""
        return int(np.argmin(np.abs(self.ts_us - ts_us)))

    # ------------------------------------------------- calibration (viz/collision)
    @staticmethod
    def _calib_cam_name(cam: str) -> str:
        """Video dirs say ``pinhole_front``; tar members say ``front``."""
        return cam[len("pinhole_"):] if cam.startswith("pinhole_") else cam

    def intrinsics(self, cam: str, target_w: int | None = None,
                   target_h: int | None = None) -> dict:
        """Pinhole intrinsics dict (see proj_utils.build_pinhole_intrinsics)."""
        from proj_utils import build_pinhole_intrinsics
        short = self._calib_cam_name(cam)
        raw = _read_tar_members(
            self.root / self.split / "pinhole_intrinsic" / f"{self.clip_id}.tar")
        key = f"{self.clip_id}.pinhole_intrinsic.{short}.npy"
        if key not in raw:
            raise KeyError(f"{self.clip_id}: missing intrinsics for {cam}")
        v = np.load(io.BytesIO(raw[key]))
        return build_pinhole_intrinsics(v, target_w, target_h)

    def cam_pose_at(self, cam: str, ts_us: int) -> np.ndarray:
        """Camera world pose at the keyframe nearest ``ts_us`` (4x4)."""
        short = self._calib_cam_name(cam)
        idx = self.keyframe_idx[self.keyframe_index_at(ts_us)]
        raw = _read_tar_members(
            self.root / self.split / "pose" / f"{self.clip_id}.tar",
            suffix_filter=f".pose.{short}.npy")
        key = f"{self.clip_id}.{idx:06d}.pose.{short}.npy"
        if key not in raw:
            raise KeyError(f"{self.clip_id}: missing pose {key}")
        return np.load(io.BytesIO(raw[key]))

    def extrinsic(self, cam: str, ts_us: int) -> np.ndarray:
        """Static vehicle->cam (OpenCV axes) transform at the nearest keyframe.

        Both stored poses map sensor->world: ``C`` (camera, OpenCV axes:
        x right / y down / z forward) and ``V`` (vehicle FLU).  Hence
        vehicle->cam is ``inv(C) @ V`` — NOT ``inv(V) @ C`` (that composes the
        wrong direction and yields a garbage rotation).  Validated against
        lidar road points: >=89% in-bounds, corr(depth, v) <= -0.86 across
        val clips.
        """
        v = self.vehicle_poses[self.keyframe_index_at(ts_us)]
        c = self.cam_pose_at(cam, ts_us)
        return np.linalg.inv(c) @ v

    # ------------------------------------------------- collision-eval streams
    def object_info_at(self, ts_us: int) -> dict:
        import json
        raw = _read_tar_members(
            self.root / self.split / "all_object_info" / f"{self.clip_id}.tar",
            suffix_filter=".all_object_info.json")
        idx = self.keyframe_idx[self.keyframe_index_at(ts_us)]
        key = f"{self.clip_id}.{idx:06d}.all_object_info.json"
        if key not in raw:
            return {}
        return json.loads(raw[key])

    def map_features(self, kind: str) -> dict:
        """``kind`` in {3d_lanelines, 3d_lanes, 3d_crosswalks, 3d_road_boundaries}."""
        import json
        raw = _read_tar_members(self.root / self.split / kind / f"{self.clip_id}.tar")
        if not raw:
            return {}
        return json.loads(next(iter(raw.values())))

    # ------------------------------------------------------------------ samples
    def segment_t0s(self, num_segments: int = 20, segment_start_us: int = 2_000_000,
                    segment_step_us: int = 1_000_000, num_history_steps: int = 16,
                    num_future_steps: int = 64, time_step: float = 0.1
                    ) -> list[int]:
        """PAIMultiSegmentDataset schedule, filtered by the real clip span."""
        hist_us = int((num_history_steps - 1) * time_step * 1_000_000)
        fut_us = int(num_future_steps * time_step * 1_000_000)
        t0s = []
        for k in range(num_segments):
            t0 = int(self.ts_us[0]) + segment_start_us + k * segment_step_us
            if t0 - hist_us >= int(self.ts_us[0]) and t0 + fut_us <= int(self.ts_us[-1]):
                t0s.append(t0)
        return t0s

    def build_sample(self, t0_us: int, num_history_steps: int = 16,
                     num_future_steps: int = 64, time_step: float = 0.1,
                     num_frames: int = 4) -> dict:
        """One PAI-contract sample at ``t0_us`` (absolute micros)."""
        dt_us = int(time_step * 1_000_000)

        hist_ts = t0_us + np.arange(
            -(num_history_steps - 1) * dt_us, dt_us / 2, dt_us).astype(np.int64)
        fut_ts = t0_us + np.arange(
            dt_us, (num_future_steps + 0.5) * dt_us, dt_us).astype(np.int64)

        hist = self.pose_at(hist_ts)
        fut = self.pose_at(fut_ts)
        t0_pose = self.pose_at(t0_us)[0]
        t0_xyz = t0_pose[:3, 3]
        t0_rot = t0_pose[:3, :3]
        t0_rot_inv = t0_rot.T

        def to_local(poses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            xyz = (t0_rot_inv @ (poses[:, :3, 3] - t0_xyz).T).T
            rot = np.einsum("ij,njk->nik", t0_rot_inv, poses[:, :3, :3])
            return xyz, rot

        hist_xyz_l, hist_rot_l = to_local(hist)
        fut_xyz_l, fut_rot_l = to_local(fut)

        # --- images: num_frames timestamps ending at t0 ---
        img_ts = np.asarray(
            [t0_us - (num_frames - 1 - i) * dt_us for i in range(num_frames)],
            dtype=np.int64)
        img_idx = np.asarray([self.keyframe_index_at(t) for t in img_ts])

        frames_list, cam_idx_list, ts_list = [], [], []
        for cam, model_idx in CAMERA_SLOTS:
            frames = self.decode_frames(cam, img_idx)  # [T,H,W,3]
            ft = torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous()
            frames_list.append(ft)
            cam_idx_list.append(model_idx)
            ts_list.append(torch.from_numpy(img_ts))
        image_frames = torch.stack(frames_list, dim=0)   # [C, T, 3, H, W]
        camera_indices = torch.tensor(cam_idx_list, dtype=torch.int64)
        all_ts = torch.stack(ts_list, dim=0)             # [C, T]
        rel_ts = (all_ts - all_ts.min()).float() * 1e-6

        return {
            "image_frames": image_frames,
            "camera_indices": camera_indices,
            "ego_history_xyz": torch.from_numpy(hist_xyz_l).float()[None, None],
            "ego_history_rot": torch.from_numpy(hist_rot_l).float()[None, None],
            "ego_future_xyz": torch.from_numpy(fut_xyz_l).float()[None, None],
            "ego_future_rot": torch.from_numpy(fut_rot_l).float()[None, None],
            "relative_timestamps": rel_ts,
            "absolute_timestamps": all_ts,
            "t0_us": int(t0_us),
            "clip_id": self.clip_id,
        }


class RDSWaymoDataset(torch.utils.data.Dataset):
    """Flat (clip_id, t0) index over one split — PAIMultiSegmentDataset style.

    Iterating sequentially reuses one decoded-video cache; random access works
    too but re-decodes on clip switch.
    """

    def __init__(self, root: str | Path | None = None, split: str = "val",
                 clip_ids: list[str] | None = None,
                 num_segments: int = 20, segment_start_us: int = 2_000_000,
                 segment_step_us: int = 1_000_000,
                 num_history_steps: int = 16, num_future_steps: int = 64,
                 time_step: float = 0.1, num_frames: int = 4,
                 preload: bool = True, max_clips: int | None = None,
                 clip_stride: int = 1):
        self.root = Path(root) if root is not None else Path(os.environ.get(
            "RDS_WAYMO_ROOT", _REPO_ROOT / "data" / "waymo_rds_hq"))
        self.split = split
        self.num_segments = num_segments
        self.segment_start_us = segment_start_us
        self.segment_step_us = segment_step_us
        self.num_history_steps = num_history_steps
        self.num_future_steps = num_future_steps
        self.time_step = time_step
        self.num_frames = num_frames
        self.preload = preload

        clips = sorted(p.stem for p in (self.root / split / FRONT_CAM).glob("*.mp4"))
        if clip_ids is not None:
            want = set(clip_ids)
            clips = [c for c in clips if c in want]
        if clip_stride > 1:
            clips = clips[::clip_stride]
        if max_clips is not None:
            clips = clips[:max_clips]
        if not clips:
            raise RuntimeError(f"no clips under {self.root / split / FRONT_CAM}")

        self.clip_ids = clips
        self._clip_cache: dict[str, RDSWaymoClip] = {}
        self.index: list[tuple[str, int]] = []
        for cid in clips:
            try:
                clip = self._get_clip(cid)
                t0s = clip.segment_t0s(
                    num_segments, segment_start_us, segment_step_us,
                    num_history_steps, num_future_steps, time_step)
            except Exception as exc:
                print(f"  [skip] {cid}: {exc}")
                continue
            self.index.extend((cid, t0) for t0 in t0s)

    def _get_clip(self, clip_id: str) -> RDSWaymoClip:
        if clip_id not in self._clip_cache:
            if len(self._clip_cache) >= 2:  # keep at most 2 decoded clips alive
                self._clip_cache.clear()
            self._clip_cache[clip_id] = RDSWaymoClip(
                self.root, self.split, clip_id, preload=self.preload)
        return self._clip_cache[clip_id]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict:
        clip_id, t0_us = self.index[idx]
        clip = self._get_clip(clip_id)
        return clip.build_sample(
            t0_us, num_history_steps=self.num_history_steps,
            num_future_steps=self.num_future_steps, time_step=self.time_step,
            num_frames=self.num_frames)
