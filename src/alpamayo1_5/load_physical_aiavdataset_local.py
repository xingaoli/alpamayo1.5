# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load data from local PhysicalAI-AV directory (no HuggingFace, no network)."""

from typing import Any
import zipfile
import os

import numpy as np
import scipy.spatial.transform as spt
import torch
import pandas as pd
import io
from einops import rearrange
import physical_ai_av.video as video
import physical_ai_av.egomotion as egomotion_module


def load_physical_aiavdataset_local(
    clip_id: str,
    data_dir: str | None = None,
    t0_us: int = 5_100_000,
    num_history_steps: int = 16,
    num_future_steps: int = 64,
    time_step: float = 0.1,
    camera_features: list | None = None,
    num_frames: int = 4,
) -> dict[str, Any]:
    """Load data from local directory without any HuggingFace dependencies.

    This is a pure local implementation that reads files directly from disk,
    avoiding all the network checks and caching logic in physical_ai_av.

    Args:
        clip_id: The clip ID to load data from.
        data_dir: Path to the PhysicalAI-Autonomous-Vehicles-base directory.
            If None, uses ALPAMAYO_DATA_DIR environment variable.
        t0_us: The timestamp (in microseconds) at which to sample the trajectory.
        num_history_steps: Number of history trajectory steps (default: 16).
        num_future_steps: Number of future trajectory steps (default: 64).
        time_step: Time step between trajectory points in seconds (default: 0.1s).
        camera_features: List of camera features to load.
        num_frames: Number of frames per camera to load (default: 4).

    Returns:
        A dictionary with the same format as load_physical_aiavdataset.
    """
    if data_dir is None:
        data_dir = os.environ.get(
            "ALPAMAYO_DATA_DIR",
            "/home/xingao/code/Alpamayo1.5/data/PhysicalAI-Autonomous-Vehicles"
        )
    
    data_path = data_dir

    # Read clip index to get chunk info
    clip_index = pd.read_parquet(os.path.join(data_path, "clip_index.parquet"))
    chunk_id = clip_index.loc[clip_id, "chunk"]

    if camera_features is None:
        camera_features = [
            "camera_cross_left_120fov",
            "camera_front_wide_120fov",
            "camera_cross_right_120fov",
            "camera_front_tele_30fov",
        ]

    camera_name_to_index = {
        "camera_cross_left_120fov": 0,
        "camera_front_wide_120fov": 1,
        "camera_cross_right_120fov": 2,
        "camera_rear_left_70fov": 3,
        "camera_rear_tele_30fov": 4,
        "camera_rear_right_70fov": 5,
        "camera_front_tele_30fov": 6,
    }

    # Load egomotion data (use full egomotion, not egomotion.offline)
    egomotion_zip_path = os.path.join(data_path, "labels", "egomotion", f"egomotion.chunk_{chunk_id:04d}.zip")
    with zipfile.ZipFile(egomotion_zip_path, 'r') as zf:
        egomotion_df = pd.read_parquet(io.BytesIO(zf.read(f"{clip_id}.egomotion.parquet")))

    assert t0_us > num_history_steps * time_step * 1_000_000, (
        "t0_us must be greater than the history time range"
    )

    # Compute timestamps for trajectory sampling
    history_offsets_us = np.arange(
        -(num_history_steps - 1) * time_step * 1_000_000,
        time_step * 1_000_000 / 2,
        time_step * 1_000_000,
    ).astype(np.int64)
    history_timestamps = t0_us + history_offsets_us

    future_offsets_us = np.arange(
        time_step * 1_000_000,
        (num_future_steps + 0.5) * time_step * 1_000_000,
        time_step * 1_000_000,
    ).astype(np.int64)
    future_timestamps = t0_us + future_offsets_us

    # Create egomotion interpolator (same as original)
    egomotion_state = egomotion_module.EgomotionState.from_egomotion_df(egomotion_df)
    egomotion = egomotion_state.create_interpolator(egomotion_df["timestamp"].values)

    # Get egomotion at history and future timestamps (using interpolation!)
    ego_history = egomotion(history_timestamps)
    ego_history_xyz = ego_history.pose.translation
    ego_history_quat = ego_history.pose.rotation.as_quat()

    ego_future = egomotion(future_timestamps)
    ego_future_xyz = ego_future.pose.translation
    ego_future_quat = ego_future.pose.rotation.as_quat()

    # Transform to local frame
    t0_xyz = ego_history_xyz[-1].copy()
    t0_quat = ego_history_quat[-1].copy()
    t0_rot = spt.Rotation.from_quat(t0_quat)
    t0_rot_inv = t0_rot.inv()

    ego_history_xyz_local = t0_rot_inv.apply(ego_history_xyz - t0_xyz)
    ego_future_xyz_local = t0_rot_inv.apply(ego_future_xyz - t0_xyz)

    ego_history_rot_local = (t0_rot_inv * spt.Rotation.from_quat(ego_history_quat)).as_matrix()
    ego_future_rot_local = (t0_rot_inv * spt.Rotation.from_quat(ego_future_quat)).as_matrix()

    # Convert to torch tensors
    ego_history_xyz_tensor = torch.from_numpy(ego_history_xyz_local).float().unsqueeze(0).unsqueeze(0)
    ego_history_rot_tensor = torch.from_numpy(ego_history_rot_local).float().unsqueeze(0).unsqueeze(0)
    ego_future_xyz_tensor = torch.from_numpy(ego_future_xyz_local).float().unsqueeze(0).unsqueeze(0)
    ego_future_rot_tensor = torch.from_numpy(ego_future_rot_local).float().unsqueeze(0).unsqueeze(0)

    # Load camera images
    image_frames_list = []
    camera_indices_list = []
    timestamps_list = []

    image_timestamps = np.array(
        [t0_us - (num_frames - 1 - i) * int(time_step * 1_000_000) for i in range(num_frames)],
        dtype=np.int64,
    )

    for cam_feature in camera_features:
        camera_zip_path = os.path.join(data_path, "camera", cam_feature, f"{cam_feature}.chunk_{chunk_id:04d}.zip")

        with zipfile.ZipFile(camera_zip_path, 'r') as zf:
            # Read video and timestamps
            video_data = io.BytesIO(zf.read(f"{clip_id}.{cam_feature}.mp4"))
            frame_timestamps_df = pd.read_parquet(io.BytesIO(zf.read(f"{clip_id}.{cam_feature}.timestamps.parquet")))
            frame_timestamps = frame_timestamps_df["timestamp"].values

            # Use SeekVideoReader to decode video
            reader = video.SeekVideoReader(
                video_data=video_data,
                timestamps=frame_timestamps,
            )

            # Decode frames at the requested timestamps (same as original!)
            frames, frame_timestamps = reader.decode_images_from_timestamps(image_timestamps)

        frames_tensor = torch.from_numpy(frames)
        frames_tensor = rearrange(frames_tensor, "t h w c -> t c h w")

        cam_idx = camera_name_to_index.get(cam_feature, 0)
        image_frames_list.append(frames_tensor)
        camera_indices_list.append(cam_idx)
        timestamps_list.append(torch.from_numpy(image_timestamps.astype(np.int64)))

    # Stack and sort
    image_frames = torch.stack(image_frames_list, dim=0)
    camera_indices = torch.tensor(camera_indices_list, dtype=torch.int64)
    all_timestamps = torch.stack(timestamps_list, dim=0)

    sort_order = torch.argsort(camera_indices)
    image_frames = image_frames[sort_order]
    camera_indices = camera_indices[sort_order]
    all_timestamps = all_timestamps[sort_order]

    camera_tmin = all_timestamps.min()
    relative_timestamps = (all_timestamps - camera_tmin).float() * 1e-6

    return {
        "image_frames": image_frames,
        "camera_indices": camera_indices,
        "ego_history_xyz": ego_history_xyz_tensor,
        "ego_history_rot": ego_history_rot_tensor,
        "ego_future_xyz": ego_future_xyz_tensor,
        "ego_future_rot": ego_future_rot_tensor,
        "relative_timestamps": relative_timestamps,
        "absolute_timestamps": all_timestamps,
        "t0_us": t0_us,
        "clip_id": clip_id,
    }
