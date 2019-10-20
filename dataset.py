from datasets.loader import VideoLoader
from datasets.videodataset_multiclips import (VideoDatasetMultiClips,
                                              collate_fn)
from pathlib import Path
from os.path import join
def get_validation_data(video_path,
                        annotation_path,
                        spatial_transform=None,
                        temporal_transform=None,
                        target_transform=None):
    loader = VideoLoader(lambda x: f'image_{x:05d}.jpg')
    video_path_formatter = (
        lambda root_path, label, video_id: Path(join(root_path, label, video_id)))

    validation_data = VideoDatasetMultiClips(
        video_path,
        annotation_path,
        'validation',
        spatial_transform=spatial_transform,
        temporal_transform=temporal_transform,
        target_transform=target_transform,
        video_loader=loader,
        video_path_formatter=video_path_formatter)

    return validation_data, collate_fn
