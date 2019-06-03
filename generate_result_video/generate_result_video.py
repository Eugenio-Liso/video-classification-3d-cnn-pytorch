import os
import json
import subprocess
import numpy as np
from opts_predictions import parse_opts_prediction
from PIL import Image, ImageDraw, ImageFont


def get_fps(video_file_path, frames_directory_path):
    p = subprocess.Popen('ffprobe -i "{}"'.format(video_file_path),
                         shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    _, res = p.communicate()
    res = res.decode('utf-8')

    duration_index = res.find('Duration:')
    duration_str = res[(duration_index + 10):(duration_index + 21)]
    hour = float(duration_str[0:2])
    minute = float(duration_str[3:5])
    sec = float(duration_str[6:10])
    total_sec = hour * 3600 + minute * 60 + sec

    n_frames = len(os.listdir(frames_directory_path))
    fps = round(n_frames / total_sec, 2)
    return fps


if __name__ == '__main__':
    opt = parse_opts_prediction()
    output_json = opt.output_json
    input_videos_folder = opt.input_video_folder
    prediction_folder = opt.prediction_video_folder

    if not os.path.exists(prediction_folder):
        subprocess.call('mkdir -p {}'.format(prediction_folder), shell=True)
    classes_list = opt.classes_list
    temporal_unit_window = int(opt.frames_for_prediction)

    with open(output_json, 'r') as f:
        results = json.load(f)

    with open(classes_list, 'r') as f:
        class_names = []
        for row in f:
            class_names.append(row[:-1])

    for index in range(len(results)):
        video_path = os.path.join(input_videos_folder, results[index]['video'])
        print('Starting annotation on input video: {}'.format(video_path))

        clips = results[index]['clips']
        unit_classes = []
        unit_segments = []
        if temporal_unit_window == 0:
            unit = len(clips)
        else:
            unit = temporal_unit_window
        for i in range(0, len(clips), unit):
            n_elements = min(unit, len(clips) - i)
            scores = np.array(clips[i]['scores'])
            for j in range(i, min(i + unit, len(clips))):
                scores += np.array(clips[i]['scores'])
            scores /= n_elements
            unit_classes.append(class_names[np.argmax(scores)])
            unit_segments.append([clips[i]['segment'][0],
                                  clips[i + n_elements - 1]['segment'][1]])

        if os.path.exists('tmp'):
            subprocess.call('rm -rf tmp', shell=True)
        subprocess.call('mkdir tmp', shell=True)

        subprocess.call('ffmpeg -hide_banner -loglevel error -i "{}" tmp/image_%05d.jpg'.format(video_path), shell=True)

        fps = get_fps(video_path, 'tmp')

        for i in range(len(unit_classes)):
            for j in range(unit_segments[i][0], unit_segments[i][1] + 1):
                image = Image.open('tmp/image_{:05}.jpg'.format(j)).convert('RGB')
                min_length = min(image.size)
                font_size = int(min_length * 0.05)
                font = ImageFont.truetype(os.path.join(os.path.dirname(__file__),
                                                       'SourceSansPro-Regular.ttf'),
                                          font_size)
                d = ImageDraw.Draw(image)
                textsize = d.textsize(unit_classes[i], font=font)
                x = int(font_size * 0.5)
                y = int(font_size * 0.25)
                x_offset = x
                y_offset = y
                rect_position = (x, y, x + textsize[0] + x_offset * 2,
                                 y + textsize[1] + y_offset * 2)
                d.rectangle(rect_position, fill=(30, 30, 30))
                d.text((x + x_offset, y + y_offset), unit_classes[i],
                       font=font, fill=(235, 235, 235))
                image.save('tmp/image_{:05}_pred.jpg'.format(j))

        dst_file_path = os.path.join(prediction_folder, video_path.split('/')[-1])
        subprocess.call(
            'ffmpeg -hide_banner -loglevel error -y -r {} -i tmp/image_%05d_pred.jpg -b:v 1000k {}'.format(fps,
                                                                                                           dst_file_path),
            shell=True)

        if os.path.exists('tmp'):
            subprocess.call('rm -rf tmp', shell=True)
