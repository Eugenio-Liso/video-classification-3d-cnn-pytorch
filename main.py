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


# prints currently alive Tensors and Variables
def print_tensors_dump():
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(type(obj), obj.size())
        except:
            pass


# import psutil
# def memReport():
#     for obj in gc.get_objects():
#         if torch.is_tensor(obj):
#             print(type(obj), obj.size())
#
#
# def cpuStats():
#     print(sys.version)
#     print(psutil.cpu_percent())
#     print(psutil.virtual_memory())  # physical memory usage
#     pid = os.getpid()
#     py = psutil.Process(pid)
#     memoryUse = py.memory_info()[0] / 2. ** 30  # memory use in GB...I think
#     print('memory GB:', memoryUse)
#

if __name__ == "__main__":
    opt = parse_opts()
    opt.mean = get_mean()
    opt.arch = '{}-{}'.format(opt.model_name, opt.model_depth)
    opt.sample_size = 112
    opt.sample_duration = 16
    opt.n_classes = 400

    model = generate_model(opt)
    print('loading model {}'.format(opt.model))
    model_data = torch.load(opt.model)
    assert opt.arch == model_data['arch']
    model.load_state_dict(model_data['state_dict'])
    model.eval()
    if opt.verbose:
        print(model)

    class_names = []
    with open('class_names_list') as f:
        for row in f:
            class_names.append(row[:-1])

    ffmpeg_loglevel = 'quiet'
    if opt.verbose:
        ffmpeg_loglevel = 'info'

    if os.path.exists('tmp'):
        subprocess.call('rm -rf tmp', shell=True)

    input_video_dir = opt.video_root
    input_video_files = [f for f in listdir(input_video_dir) if isfile(join(input_video_dir, f))]

    print('Input video files: {}'.format(input_video_files))

    outputs = []
    executions_times = []
    for input_file in input_video_files:
        video_path = os.path.join(input_video_dir, input_file)
        if os.path.exists(video_path):
            print('Prediction on input file: {}'.format(video_path))
            subprocess.call('mkdir tmp', shell=True)

            # The "{}" are useful to expand path also with spaces
            subprocess.call('ffmpeg -hide_banner -loglevel fatal -i "{}" tmp/image_%05d.jpg'.format(video_path),
                            shell=True)

            start_time = time.time()
            result = classify_video('tmp', input_file, class_names, model, opt)
            end_time = time.time()

            execution_time = end_time - start_time
            print("--- Execution time: %s seconds ---" % execution_time)

            outputs.append(result)
            executions_times.append((video_path, execution_time))

            subprocess.call('rm -rf tmp', shell=True)

            # TODO see if this helps with memory
            torch.cuda.empty_cache()

            # Does not work
            memory_still_in_use = ctypes.cast(id(torch.cuda.memory_allocated), ctypes.py_object).value
            print('Memory GPU allocated: {}'.format(str(memory_still_in_use)))
            print_tensors_dump()
        else:
            print('{} does not exist'.format(input_file))

    if os.path.exists('tmp'):
        subprocess.call('rm -rf tmp', shell=True)

    with open(opt.output, 'w') as f:
        json.dump(outputs, f)

    print("Execution times: {}".format(executions_times))
