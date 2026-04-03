from physical_ai_av.dataset import PhysicalAIAVDatasetInterface
import argparse

parser = argparse.ArgumentParser(description='download data')
parser.add_argument('--chunk_id', type=str, help='download chunk_id')
parser.add_argument('--local_dir', type=str, help='download local_dir')

args = parser.parse_args()

chunk_id_list = args.chunk_id.split(',')
local_dir = args.local_dir
ds = PhysicalAIAVDatasetInterface(token=True, local_dir=local_dir)

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

    ds.download_chunk_features(
        int(chunk_id),
        features=ds.features.RADAR.ALL
    )

    ds.download_chunk_features(
        chunk_id=int(chunk_id),
        features=ds.features.CAMERA.ALL,
    )

    # lidar data
    ds.download_chunk_features(
        int(chunk_id),
        features=ds.features.LIDAR.LIDAR_TOP_360FOV
    )
