import argparse
import os
import shutil
from pathlib import Path


def copytree(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--frames_dir', required=True, type=Path,
                        help='Root directory containing all frames.')
    parser.add_argument('--validation_set_file', required=True, type=Path,
                        help='File that contains videos, one per line')
    parser.add_argument('--destination_val_set', required=True, type=Path, help='Destination of the validation set')

    args = parser.parse_args()

    validation_set = args.validation_set_file
    frames_dir = args.frames_dir
    destination_val_set = args.destination_val_set

    validation_videos = open(validation_set).read().splitlines()

    os.makedirs(destination_val_set, exist_ok=True)

    for target_class in os.listdir(frames_dir):
        target_class_dir = os.path.join(frames_dir, target_class)
        for video_id in os.listdir(target_class_dir):
            if video_id in validation_videos:
                source_video_dir = os.path.join(target_class_dir, video_id)
                dst_dir = os.path.join(destination_val_set, target_class, video_id)
                os.makedirs(dst_dir, exist_ok=True)
                copytree(source_video_dir, dst_dir)
