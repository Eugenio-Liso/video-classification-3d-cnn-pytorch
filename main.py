import gc
import json
import os
import statistics
import subprocess

import cv2 as cv
import numpy as np
import torch
from PIL import Image

from classify import classify_video_offline
from classify import classify_video_online
from logging_utils import logger_factory as lf
from mean import get_mean
from model import generate_model
from opts import parse_opts
from pathlib import Path

logger = lf.getBasicLogger(os.path.basename(__file__))
from classify import retrieve_spatial_temporal_transforms


# https://discuss.pytorch.org/t/how-to-debug-causes-of-gpu-memory-leaks/6741/18

# logger.infos currently alive Tensors and Variables
def print_tensors_dump():
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                string = "object type: {} - object size: {}".format(type(obj), obj.size())
                logger.info(string)
        except:
            pass


# import psutil
# def memReport():
#     for obj in gc.get_objects():
#         if torch.is_tensor(obj):
#             logger.info(type(obj), obj.size())
#
#
# def cpuStats():
#     logger.info(sys.version)
#     logger.info(psutil.cpu_percent())
#     logger.info(psutil.virtual_memory())  # physical memory usage
#     pid = os.getpid()
#     py = psutil.Process(pid)
#     memoryUse = py.memory_info()[0] / 2. ** 30  # memory use in GB...I think
#     logger.info('memory GB:', memoryUse)
#

if __name__ == "__main__":
    opt = parse_opts()
    opt.mean = get_mean()
    opt.arch = '{}-{}'.format(opt.model_name, opt.model_depth)
    opt.sample_size = 112
    using_frames = opt.frames

    class_names_list = opt.class_names_list

    class_names = []
    n_classes = 0
    with open(class_names_list) as f:
        for row in f:
            class_name = row[:-1]
            class_names.append(class_name)
            n_classes += 1

    opt.n_classes = n_classes

    model = generate_model(opt)
    logger.info('Loading model in: {}'.format(opt.model))
    model_data = torch.load(opt.model)
    # assert opt.arch == model_data['arch']
    model.load_state_dict(model_data['state_dict'])
    model.eval()

    print(f"Input model: {model}")

    if os.path.exists('tmp'):
        subprocess.call('rm -rf tmp/*', shell=True)
    elif not os.path.exists('tmp'):
        os.makedirs('tmp')

    input_video_dir = opt.video_root

    type_of_prediction = opt.type_of_prediction

    outputs = []
    executions_times_with_video_names = []

    if using_frames:
        all_video_results, all_execution_times = classify_video_offline(class_names, model, opt)
        print(f'Prediction phase completed! Output metrics csv written in path: {opt.output_csv}')
        outputs = all_video_results
        executions_times_with_video_names = all_execution_times

    else:
        if type_of_prediction == 'offline':
            root_tmp_dir = 'tmp'

            for target_class in os.listdir(input_video_dir):
                target_dir = os.path.join(input_video_dir, target_class)

                for input_file in os.listdir(target_dir):
                    video_path = os.path.join(target_dir, input_file)

                    input_dir = f"{root_tmp_dir}/{target_class}/{input_file}"
                    os.makedirs(input_dir, exist_ok=True)
                    vidcap = cv.VideoCapture(video_path)

                    print(
                        'Width = ' + str(vidcap.get(3)) + ' Height = ' + str(vidcap.get(4)) + ' fps = ' + str(
                            vidcap.get(5)))

                    print(f"Reading frames of video {input_file}...")

                    success, image = vidcap.read()
                    count = 1
                    while success:
                        cv.imwrite(os.path.join(input_dir, "image_%05d.jpg" % count),
                                   image)  # save frame as JPEG file
                        count += 1
                        success, image = vidcap.read()

            opt.video_root = root_tmp_dir

            # Exclude label from input_dir
            all_video_results, all_execution_times = \
                classify_video_offline(class_names, model, opt)
            print(f'Prediction phase completed! Output metrics csv written in path: {opt.output_csv}')
            outputs = all_video_results
            executions_times_with_video_names = all_execution_times

        elif type_of_prediction == 'live':
            input_video_files = [f for f in os.listdir(input_video_dir) if
                                 os.path.isfile(os.path.join(input_video_dir, f))]

            for input_file in input_video_files:
                video_path = os.path.join(input_video_dir, input_file)
                logger.info('Prediction on input: {}'.format(video_path))

                thickness_text = 1
                font_size = 1

                cap = cv.VideoCapture(video_path)

                width = int(cap.get(3))
                height = int(cap.get(4))
                fps = round(cap.get(5))

                logger.info(
                    'Width = ' + str(width) + ' Height = ' + str(height) + ' fps = ' + str(fps))

                secondsToWaitBetweenFrames = int((1 / fps) * 1000)

                frame_list = []

                success, frame = cap.read()
                count = 1
                text_with_prediction = ''

                font = cv.FONT_HERSHEY_COMPLEX

                single_video_result = {
                    'video': input_file,
                    'clips': []
                }

                exec_times_with_segments = []

                while success:
                    frame_list.append(frame)

                    # Does not do all the predictions if the input video has a number of frames X such that X % opt.sample_duration != 0
                    # In other words, the last frames that do not fit into a 'opt.sample_duration' batch will be discarded
                    if count % opt.sample_duration == 0:
                        frames_as_images = [Image.fromarray(np.array(frame), 'RGB') for frame in frame_list]

                        spatial_transform, _ = retrieve_spatial_temporal_transforms(opt)

                        if spatial_transform is not None:
                            spatial_transform.randomize_parameters()
                            frames_as_images = [spatial_transform(img) for img in frames_as_images]

                        frames_as_images = torch.stack(frames_as_images, 0).permute(1, 0, 2, 3)

                        # Adding batch ID as first dimension. Automatically fills it with value 1
                        frames_as_images = frames_as_images[None, :, :, :, :]

                        text_with_prediction = \
                            classify_video_online(frames_as_images, count, class_names, model, opt,
                                                  single_video_result, exec_times_with_segments)

                        frame_list.clear()

                    # Color in BGR!
                    min_length = min(width, height)
                    textsize = cv.getTextSize(text_with_prediction, fontFace=font, fontScale=font_size,
                                              thickness=thickness_text)[0]
                    x = int(font_size * 50)
                    y = int(font_size * 25)
                    x_offset = x
                    y_offset = y
                    cv.rectangle(frame, (x, y), (x + textsize[0] + x_offset * 2, y + textsize[1] + y_offset * 2),
                                 (30, 30, 30), cv.FILLED)
                    cv.putText(frame, text_with_prediction, (x + x_offset, y + y_offset * 2), font, font_size,
                               (235, 235, 235), thickness_text, cv.LINE_AA)

                    # Disegna predizione da quel frame in poi, fino alla prossima prediction
                    # cv.namedWindow('Frame', cv.WINDOW_NORMAL)
                    # cv.resizeWindow('Frame', (width, height))
                    cv.imshow('Frame', frame)

                    cv.waitKey(secondsToWaitBetweenFrames)

                    success, frame = cap.read()
                    count += 1

                single_video_exec_time = {
                    input_file: exec_times_with_segments
                }

                outputs.append(single_video_result)
                executions_times_with_video_names.append(single_video_exec_time)

                cap.release()
                cv.destroyAllWindows()

        else:
            raise ValueError(
                'Got input parameter for prediction type: ' +
                type_of_prediction +
                ' but expected one between [offline,live]')

    if os.path.exists('tmp'):
        subprocess.call('rm -rf tmp/*', shell=True)

    with open(opt.output, 'w') as f:
        json.dump(outputs, f, indent=4)

    mean_execution_times = {}

    for prediction in executions_times_with_video_names:
        for video_name, exec_times_with_segments in prediction.items():

            mean_exec_time = []
            for segment, exec_time in exec_times_with_segments:
                mean_exec_time.append(exec_time)

            mean_execution_times.update({video_name: statistics.mean(mean_exec_time)})

    with open(opt.output_times, 'w') as f:
        json.dump(executions_times_with_video_names, f, indent=4)

    with open(opt.output_mean_times, 'w') as f:
        json.dump(mean_execution_times, f, indent=4)

    logger.info("Execution times: {}".format(executions_times_with_video_names))
    logger.info("Mean execution times: {}".format(mean_execution_times))
