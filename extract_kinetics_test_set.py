import argparse
import json
import os
import subprocess
from pathlib import Path
import shutil

from pytube import YouTube
from pytube.exceptions import VideoUnavailable


def hou_min_sec(millis):
    millis = int(millis)
    seconds = (millis / 1000) % 60
    seconds = int(seconds)
    minutes = (millis / (1000 * 60)) % 60
    minutes = int(minutes)
    hours = millis / (1000 * 60 * 60)
    hours = int(hours)

    msecs = millis % 1000
    return "%s:%s:%s.%s" % (str(hours).zfill(2), str(minutes).zfill(2), str(seconds).zfill(2), str(msecs).zfill(3))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--output_frames_dir', required=True, type=Path,
                        help='Directory where the output downloaded frames will be put')
    parser.add_argument("--filter_on_classes", nargs='*', help="Optional classes to filter on", default=[])
    parser.add_argument("--test_set_json", default='test_set/kinetics_400/kinetics_test.json',
                        type=Path, help="Path to the test set json of kinectics")
    parser.add_argument("--max_videos_for_class", default=100,
                        type=int, help="Max number of videos per class to download")
    parser.add_argument('--classes_list', default='classes_list/class_names_list_kinetics', type=str,
                        help='File containing the available Kinetics classes')

    args = parser.parse_args()

    output_frames_dir = args.output_frames_dir
    filter_on_classes = args.filter_on_classes
    test_set_json = args.test_set_json
    max_videos_for_class = args.max_videos_for_class
    classes_list = args.classes_list

    counter_for_classes = {}

    with open(classes_list) as f:
        for row in f:
            class_name = row[:-1]
            counter_for_classes[class_name] = 0

    for target_class in os.listdir(output_frames_dir):
        target_dir = os.path.join(output_frames_dir, target_class)

        count = 0
        for video in os.listdir(target_dir):
            video_dir = os.path.join(target_dir, video)
            if os.path.isdir(video_dir):
                if len(os.listdir(video_dir)) == 0:
                    print(f"Directory {video_dir} is empty. Removing it.")
                    os.rmdir(video_dir)
                elif count > max_videos_for_class:
                    # raise Exception(f"{target_class} has more videos than {max_videos_for_class}. Remove {(count - max_videos_for_class) + 1} video(s)")
                    print(f"{target_class} has more videos than {max_videos_for_class}. Removing {(count - max_videos_for_class) + 1} video(s)")
                    shutil.rmtree(video_dir)
                else:
                    count += 1
        counter_for_classes[target_class] = count

    print(f"Current number of videos per class: {counter_for_classes}")

    tmp_dir = 'tmp'
    os.makedirs(tmp_dir, exist_ok=True)
    video_extension = 'mp4'
    with open(test_set_json) as input_json:
        videos_with_labels = json.load(input_json)

        for video_id in videos_with_labels:
            video_url = videos_with_labels[video_id]['url']
            video_annotations = videos_with_labels[video_id]['annotations']
            segment = video_annotations['segment']
            target_class = video_annotations['label']

            if counter_for_classes[target_class] < max_videos_for_class or not filter_on_classes or (target_class in filter_on_classes):
                counter_for_classes[target_class] += 1
                start_seconds = hou_min_sec(float(segment[0]) * 1000)
                end_seconds = hou_min_sec(float(segment[1]) * 1000)

                output_frames_subdir = os.path.join(output_frames_dir,
                                                    target_class,
                                                    f"{video_id}_{start_seconds.replace('.', '_')}_{end_seconds.replace('.', '_')}")
                output_frames_path = os.path.join(output_frames_subdir, "image_%05d.jpg")

                if os.path.exists(output_frames_subdir):
                    print(f"{output_frames_subdir} already exists. Skipping...")
                    continue
                else:
                    os.makedirs(output_frames_subdir, mode=0o755)

                print(f"Downloading and extracting frames for video {video_id}")

                # Download video
                try:
                    yt = YouTube(video_url)
                except VideoUnavailable:
                    print(f'WARNING: Skipping video at url {video_url} because it is unavailable.')
                    continue
                except Exception:
                    print(f"The video at url {video_url} is not available in your country or a random error occurred")
                    continue

                yt.streams.filter(subtype=video_extension).first().download(output_path=tmp_dir, filename=video_id)
                print(f"Total video downloaded for class {target_class} is {counter_for_classes[target_class]}")

                input_video_path = os.path.join(tmp_dir, f"{video_id}.{video_extension}")
                ffmpeg_command = 'ffmpeg -ss %(start_timestamp)s -i "%(videopath)s" -to %(clip_length)s -copyts -loglevel error "%(outpath)s"' % {
                    'start_timestamp': start_seconds,
                    'clip_length': end_seconds,
                    'videopath': input_video_path,
                    'outpath': output_frames_path}

                subprocess.call(ffmpeg_command, shell=True)

                subprocess.call(f'rm {input_video_path}', shell=True)
            else:
                print(f"Class {target_class} has {counter_for_classes[target_class]}, that are more than {max_videos_for_class}")