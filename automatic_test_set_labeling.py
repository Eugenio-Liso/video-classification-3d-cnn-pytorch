import argparse
import json
import os
from pathlib import Path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--labeled_videos_output', default='labeled_videos.json', type=Path,
                        help='Output json path containing labeled videos')
    parser.add_argument('--videos_dir', default='labeled_videos.json', type=Path, required=True,
                        help='Path containing the test set videos. Should point to the root folder, and it should have '
                             'subfolders, one per target class, containing the videos id')
    parser.add_argument('--sample_duration', type=int, default=16,
                        help='Sample duration. Must be the same used in the prediction phase')

    args = parser.parse_args()

    videos_dir = args.videos_dir
    labeled_videos_output = args.labeled_videos_output
    sample_duration = args.sample_duration

    videos_with_target_labels = {}

    for target_class in os.listdir(videos_dir):
        target_dir = os.path.join(videos_dir, target_class)

        for video_id in os.listdir(target_dir):
            n_frames = len(os.listdir(os.path.join(target_dir, video_id)))
            if n_frames < sample_duration:
                print(
                    f"Warning. Skipping video: {video_id} in {target_dir} because it has n_frames: {n_frames} "
                    f"that are below the minimum number of frames: {sample_duration}")
            else:
                videos_with_target_labels[video_id] = target_class

    with open(labeled_videos_output, 'w+') as out_json:
        json.dump(videos_with_target_labels, out_json, indent=4)
