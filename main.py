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

logger = lf.getBasicLogger(os.path.basename(__file__))


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
    if opt.verbose:
        logger.info(model)

    if os.path.exists('tmp'):
        subprocess.call('rm -f tmp/*', shell=True)
    elif not os.path.exists('tmp'):
        os.makedirs('tmp')

    input_video_dir = opt.video_root
    input_video_files = [f for f in os.listdir(input_video_dir) if os.path.isfile(os.path.join(input_video_dir, f))]

    prediction_input_mode = opt.prediction_input_mode
    type_of_prediction = opt.type_of_prediction

    outputs = []
    executions_times_with_video_names = []

    if using_frames:
        classify_video_offline(class_names, model, opt)
        print(f'Prediction phase completed! Output metrics csv written in path: {opt.output_csv}')
        exit(0)

    else:
        logger.info('Input video files: {}'.format(input_video_files))

        for input_file in input_video_files:
            video_path = os.path.join(input_video_dir, input_file)
            if os.path.exists(video_path):
                logger.info('Prediction on input: {}'.format(video_path))

                if prediction_input_mode == 'legacy':

                    # The "{}" are useful to expand path also with spaces
                    subprocess.call('ffmpeg -hide_banner -loglevel error -i "{}" tmp/image_%05d.jpg'.format(video_path),
                                    shell=True)

                    result, exec_times_with_video_name_on_prediction = classify_video_offline(class_names, model,
                                                                                              opt)
                elif prediction_input_mode == 'opencv':

                    if type_of_prediction == 'offline':

                        input_dir = "tmp"
                        vidcap = cv.VideoCapture(video_path)

                        logger.info(
                            'Width = ' + str(vidcap.get(3)) + ' Height = ' + str(vidcap.get(4)) + ' fps = ' + str(
                                vidcap.get(5)))

                        logger.info("Reading frames...")

                        success, image = vidcap.read()
                        count = 0
                        while success:
                            cv.imwrite(os.path.join(input_dir, "image_%05d.jpg" % count),
                                       image)  # save frame as JPEG file

                            success, image = vidcap.read()
                            count += 1

                        result, exec_times_with_video_name_on_prediction = \
                            classify_video_offline('tmp', input_file, class_names, model, opt)

                    elif type_of_prediction == 'live':
                        # Async
                        # http://blog.blitzblit.com/2017/12/24/asynchronous-video-capture-in-python-with-opencv/
                        # cap = VideoCaptureAsync(video_path)
                        #
                        # # Start a separate thread
                        # cap.start()
                        #
                        # # Stop the separate thread for opencv
                        # cap.stop()

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

                        while success:
                            frame_list.append(frame)

                            # TODO check se sample_duration può essere cambiata (il batch size riguarda il come caricare
                            #  i frame, mentre la sample_duration è la lunghezza della clip)
                            if count % opt.sample_duration == 0:
                                frames_as_images = [Image.fromarray(np.array(frame), 'RGB') for frame in frame_list]

                                result, exec_times_with_video_name_on_prediction = \
                                    classify_video_online(frames_as_images, count, class_names, model, opt)

                                text_with_prediction = result[0]
                                frame_list.clear()

                            # Color in BGR!
                            cv.putText(frame, text_with_prediction, (int(width * 0.40), int(height * 0.10)), font, 1,
                                       (0, 0, 255), 3, cv.LINE_AA)

                            # Disegna predizione da quel frame in poi, fino alla prossima prediction
                            # cv.namedWindow('Frame', cv.WINDOW_NORMAL)
                            # cv.resizeWindow('Frame', (width, height))
                            cv.imshow('Frame', frame)

                            cv.waitKey(secondsToWaitBetweenFrames)

                            success, frame = cap.read()
                            count += 1

                        cap.release()
                        cv.destroyAllWindows()

                        # TODO riempire
                        result = []
                        exec_times_with_video_name_on_prediction = []
                    else:
                        raise ValueError(
                            'Got input parameter for prediction type: ' +
                            type_of_prediction +
                            ' but expected one between [offline,live]')
                else:
                    raise ValueError(
                        'Got input parameter for prediction: ' +
                        prediction_input_mode +
                        ' but expected one between [opencv,legacy]')

                if result is not None and exec_times_with_video_name_on_prediction is not None:
                    outputs.append(result)
                    executions_times_with_video_names.append(exec_times_with_video_name_on_prediction)

                subprocess.call('rm -f tmp/*', shell=True)

                # TODO see if this helps with memory
                torch.cuda.empty_cache()

                # Does not work
                # memory_still_in_use = ctypes.cast(id(torch.cuda.memory_allocated), ctypes.py_object).value
                # logger.info('Memory GPU allocated: {}'.format(str(memory_still_in_use)))
                # logger.info_tensors_dump()
            else:
                logger.info('{} does not exist'.format(input_file))

    if os.path.exists('tmp'):
        subprocess.call('rm -f tmp/*', shell=True)

    with open(opt.output, 'w') as f:
        json.dump(outputs, f)

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
