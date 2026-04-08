from physical_ai_av.dataset import PhysicalAIAVDatasetInterface
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='download data')
parser.add_argument('--chunk_id', type=str, help='download chunk_id')
parser.add_argument('--local_dir', type=str, help='download local_dir')

args = parser.parse_args()

if args.chunk_id == "-1":
    chunk_id_list = np.arange(50, 200)
else:
    chunk_id_list = args.chunk_id.split(',')

print(chunk_id_list)

# chunk_id_list = args.chunk_id.split(',')
local_dir = args.local_dir
# 禁用下载大小确认提示（后台运行无法交互式输入）
ds = PhysicalAIAVDatasetInterface(
    token=True,
    local_dir=local_dir,
    confirm_download_threshold_gb=float("inf"),
)

for chunk_id in chunk_id_list:

    ds.download_chunk_features(
        int(chunk_id),
        features=ds.features.CALIBRATION.ALL
    )

    # egomotion data
    ds.download_chunk_features(
        int(chunk_id),
        features=ds.features.LABELS.ALL
    )

    # ds.download_chunk_features(
    #     int(chunk_id),
    #     features=ds.features.RADAR.ALL
    # )

    ds.download_chunk_features(
        chunk_id=int(chunk_id),
        features=ds.features.CAMERA.CAMERA_CROSS_LEFT_120FOV,
    )
    ds.download_chunk_features(
        chunk_id=int(chunk_id),
        features=ds.features.CAMERA.CAMERA_FRONT_WIDE_120FOV,
    )
    ds.download_chunk_features(
        chunk_id=int(chunk_id),
        features=ds.features.CAMERA.CAMERA_CROSS_RIGHT_120FOV,
    )
    ds.download_chunk_features(
        chunk_id=int(chunk_id),
        features=ds.features.CAMERA.CAMERA_FRONT_TELE_30FOV,
    )

    # lidar data
    # ds.download_chunk_features(
    #     int(chunk_id),
    #     features=ds.features.LIDAR.LIDAR_TOP_360FOV
    # )
