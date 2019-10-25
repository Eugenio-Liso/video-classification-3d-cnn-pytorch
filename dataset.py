from datasets.loader import VideoLoader
from datasets.videodataset_multiclips import (VideoDatasetMultiClips,
                                              collate_fn)
from pathlib import Path
from os.path import join


def get_validation_data(video_path,
                        annotation_path,
                        sample_duration,
                        spatial_transform=None,
                        temporal_transform=None,
                        target_transform=None):
    loader = VideoLoader(lambda x: f'image_{x:05d}.jpg')
    video_path_formatter = (
        lambda root_path, label, video_id: Path(join(root_path, label, video_id)))

    # TODO allow a more general way to specify validation or test
    # Right now, with validation we mean 'test set', so if in training you have
    # a 'real' validation set, this won't work, since you should load data with subset 'testing'
    validation_data = VideoDatasetMultiClips(
        video_path,
        annotation_path,
        'validation',
        sample_duration,
        spatial_transform=spatial_transform,
        temporal_transform=temporal_transform,
        target_transform=target_transform,
        video_loader=loader,
        video_path_formatter=video_path_formatter)

    return validation_data, collate_fn
