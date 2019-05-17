import os
from os import listdir
from os.path import isfile, join
import sys
import json
import subprocess
import numpy as np
import torch
from torch import nn
import ctypes

from opts import parse_opts
from model import generate_model
from mean import get_mean
from classify import classify_video
import time
import gc
import statistics
from logging import logger_factory as lf

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
    opt.sample_duration = 16
    opt.n_classes = 400

    model = generate_model(opt)
    logger.info('Loading model in: {}'.format(opt.model))
    model_data = torch.load(opt.model)
    assert opt.arch == model_data['arch']
    model.load_state_dict(model_data['state_dict'])
    model.eval()
    if opt.verbose:
        logger.info(model)

    class_names = []
    with open('class_names_list') as f:
        for row in f:
            class_names.append(row[:-1])

    if os.path.exists('tmp'):
        subprocess.call('rm -rf tmp', shell=True)

    input_video_dir = opt.video_root
    input_video_files = [f for f in listdir(input_video_dir) if isfile(join(input_video_dir, f))]

    logger.info('Input video files: {}'.format(input_video_files))

    prediction_input_mode = opt.prediction_input_mode

    outputs = []
    executions_times_with_video_names = []
    for input_file in input_video_files:
        video_path = os.path.join(input_video_dir, input_file)
        if os.path.exists(video_path):
            logger.info('Prediction on input file: {}'.format(video_path))

            if prediction_input_mode == 'legacy':
                subprocess.call('mkdir tmp', shell=True)

                # The "{}" are useful to expand path also with spaces
                subprocess.call('ffmpeg -hide_banner -loglevel fatal -i "{}" tmp/image_%05d.jpg'.format(video_path),
                                shell=True)

                result, exec_times_with_video_name_on_prediction = classify_video('tmp', input_file, class_names, model,
                                                                                  opt)
            elif prediction_input_mode == 'opencv':
                # TODO
                logger.info("TODO")
                result = []
                exec_times_with_video_name_on_prediction = []
            else:
                raise ValueError(
                    'Got input parameter for prediction: ' +
                    prediction_input_mode +
                    ' but expected one between [opencv,legacy]')

            outputs.append(result)
            executions_times_with_video_names.append(exec_times_with_video_name_on_prediction)

            subprocess.call('rm -rf tmp', shell=True)

            # TODO see if this helps with memory
            torch.cuda.empty_cache()

            # Does not work
            # memory_still_in_use = ctypes.cast(id(torch.cuda.memory_allocated), ctypes.py_object).value
            # logger.info('Memory GPU allocated: {}'.format(str(memory_still_in_use)))
            # logger.info_tensors_dump()
        else:
            logger.info('{} does not exist'.format(input_file))

    if os.path.exists('tmp'):
        subprocess.call('rm -rf tmp', shell=True)

    with open(opt.output, 'w') as f:
        json.dump(outputs, f)

    mean_execution_times = []

    for prediction in executions_times_with_video_names:
        for video_name, exec_times_with_segments in prediction.items():

            mean_exec_time = []
            for segment, exec_time in exec_times_with_segments:
                mean_exec_time.append(exec_time)

            mean_execution_times.append((video_name, statistics.mean(mean_exec_time)))

    with open(opt.output_times, 'w') as f:
        json.dump(mean_execution_times, f)

    logger.info("Execution times: {}".format(executions_times_with_video_names))
    logger.info("Mean execution times: {}".format(mean_execution_times))
