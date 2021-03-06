import json

import torch
import torch.utils.data as data

from .loader import VideoLoader

from pathlib import Path
from os.path import join


def get_class_labels(data):
    class_labels_map = {}
    index = 0
    for class_label in data['labels']:
        class_labels_map[class_label] = index
        index += 1
    return class_labels_map


def get_video_ids_and_annotations(data, subset):
    video_ids = []
    annotations = []

    for key, value in data['database'].items():
        this_subset = value['subset']
        if this_subset == subset:
            video_ids.append(key)
            annotations.append(value['annotations'])

    return video_ids, annotations


class VideoDataset(data.Dataset):

    def __init__(self,
                 root_path,
                 annotation_path,
                 subset,
                 sample_duration,
                 use_alternative_label,
                 augment_filters=None,
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 video_loader=None,
                 video_path_formatter=lambda root_path, label, video_id: Path(join(root_path, label, video_id)),
                 image_name_formatter=lambda x: f'image_{x:05d}.jpg',
                 target_type='label'):
        self.data, self.class_names = self.__make_dataset(
            root_path, annotation_path, subset, video_path_formatter, sample_duration, use_alternative_label)

        self.augment_filters = augment_filters
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform

        if video_loader is None:
            self.loader = VideoLoader(image_name_formatter)
        else:
            self.loader = video_loader

        self.target_type = target_type

    def __make_dataset(self, root_path, annotation_path, subset,
                       video_path_formatter, sample_duration, use_alternative_label):
        if use_alternative_label:
            print("Using alternative label in 'database' data in annotation file")
        with annotation_path.open('r') as f:
            data = json.load(f)
        video_ids, annotations = get_video_ids_and_annotations(data, subset)
        class_to_idx = get_class_labels(data)
        idx_to_class = {}
        for name, label in class_to_idx.items():
            idx_to_class[label] = name

        n_videos = len(video_ids)
        dataset = []
        for i in range(n_videos):
            if n_videos > 5 and i % (n_videos // 5) == 0:
                print('Dataset loading [{}/{}]'.format(i, len(video_ids)))

            if 'label' in annotations[i]:
                label = annotations[i]['label']
                label_id = class_to_idx[label]
            else:
                label = 'test'
                label_id = -1

            if use_alternative_label:
                alternative_label = annotations[i]['original_label']
                video_path = video_path_formatter(root_path, alternative_label, video_ids[i])
            else:
                video_path = video_path_formatter(root_path, label, video_ids[i])
            if not video_path.exists():
                print(f"Warning: discarding {video_path} since it does not exists. Check the annotation file")
                continue

            segment = annotations[i]['segment']
            if segment[1] == 1:
                print(f"Warning: skipping {video_path} since it has only one frame")
                continue

            if segment[1] < sample_duration:
                print(f"Warning: skipping {video_path} since it has {segment[1]} frames which do not reach the minimum "
                      f"of {sample_duration} frames for prediction")
                continue

            frame_indices = list(range(segment[0], segment[1]))
            sample = {
                'video': video_path,
                'segment': segment,
                'frame_indices': frame_indices,
                'video_id': video_ids[i],
                'label': label_id
            }
            dataset.append(sample)

        return dataset, idx_to_class

    def __loading(self, path, frame_indices):
        clip = self.loader(path, frame_indices)

        if self.augment_filters is not None:
            original_clip_size = len(clip)
            clip = self.augment_filters(clip)
            assert original_clip_size == len(clip), "The augmented clip should have the same dimension"
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

        return clip

    def __getitem__(self, index):
        path = self.data[index]['video']
        if isinstance(self.target_type, list):
            target = [self.data[index][t] for t in self.target_type]
        else:
            target = self.data[index][self.target_type]

        frame_indices = self.data[index]['frame_indices']
        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)

        clip = self.__loading(path, frame_indices)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return clip, target

    def __len__(self):
        return len(self.data)
