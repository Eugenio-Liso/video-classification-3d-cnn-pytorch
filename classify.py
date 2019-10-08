import torch
import time
from torch.autograd import Variable

from dataset import Video
from spatial_transforms import (Compose, Normalize, Scale, CenterCrop, ToTensor)
from temporal_transforms import LoopPadding
from logging_utils import logger_factory as lf

import os

logger = lf.getBasicLogger(os.path.basename(__file__))


def classify_video_offline(video_dir, video_name, class_names, model, opt):
    assert opt.mode in ['score', 'feature']

    batch_size, data_loader = create_dataset_offline(opt, video_dir)

    video_outputs = []
    video_segments = []
    executions_times = []

    with torch.no_grad():
        for i, (inputs, segments) in enumerate(data_loader):
            start_time = time.time()

            outputs = model(inputs)
            end_time = time.time()

            execution_time = end_time - start_time

            print("--- Execution time for segment {}: {} seconds ---".format(segments, execution_time))
            executions_times.append(execution_time)

            video_outputs.append(outputs.cpu().data)
            video_segments.append(segments)

    video_outputs = torch.cat(video_outputs)
    video_segments = torch.cat(video_segments)

    # for execTimeIdx in range(len(executions_times)):
    #     print('i: {}, execTime: {}'.format(execTimeIdx, executions_times[execTimeIdx]))
    #
    # for i in range(video_outputs.size(0)):
    #     print("i: {}, segments: {}".format(i, video_segments[i].tolist()))

    executions_times_with_video_name = {}
    # TODO maybe this can be generalized when lengths differs
    # Must be 1 to make sure that len(exec_times) == len(segments)
    if batch_size == 1:

        exec_times_with_segments = []

        for i in range(video_outputs.size(0)):
            segment = video_segments[i].tolist()
            exec_time = executions_times[i]

            exec_times_with_segments.append((segment, exec_time))

        executions_times_with_video_name = {
            video_name: exec_times_with_segments
        }

        print('Exec times final result with video name: {}'.format(executions_times_with_video_name))
    else:
        print('Resulting exec times cannot be calculated since length of segments and exec times may differ.')

    results = {
        'video': video_name,
        'clips': []
    }

    _, max_indices = video_outputs.max(dim=1)
    for i in range(video_outputs.size(0)):
        clip_results = {
            'segment': video_segments[i].tolist(),
        }

        if opt.mode == 'score':
            clip_results['label'] = class_names[max_indices[i]]
            clip_results['scores'] = video_outputs[i].tolist()
        elif opt.mode == 'feature':
            clip_results['features'] = video_outputs[i].tolist()

        results['clips'].append(clip_results)

    return results, executions_times_with_video_name


def create_dataset_offline(opt, video_dir):
    batch_size = opt.batch_size_multiplier

    # TODO check se ha senso Scale + Crop stessa dimensione
    spatial_transform = Compose([Scale(opt.sample_size),
                                 CenterCrop(opt.sample_size),
                                 ToTensor(),
                                 Normalize(opt.mean, [1, 1, 1])])

    # Considera opt.sample_duration numero di frames quando esegue le predictions
    temporal_transform = LoopPadding(opt.sample_duration)
    data = Video(video_dir, spatial_transform=spatial_transform,
                 temporal_transform=temporal_transform,
                 sample_duration=opt.sample_duration)
    data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                              shuffle=False, num_workers=opt.n_threads, pin_memory=True)
    return batch_size, data_loader


def classify_video_online(frames_list, current_starting_frame_index, class_names, model, opt):
    assert opt.mode in ['score', 'feature']

    input_frames, segments = extract_input_live_predictions(current_starting_frame_index, frames_list, opt)

    executions_times = []

    logger.debug('Input tensor in live prediction: {}'.format(input_frames.size()))
    inputs = Variable(input_frames, volatile=True)
    start_time = time.time()

    outputs = model(inputs)
    end_time = time.time()

    execution_time = end_time - start_time

    print("--- Execution time for segment {}: {} seconds ---".format(segments, execution_time))
    executions_times.append(execution_time)

    prediction_output = outputs.cpu().data

    _, max_index_predicted_class = prediction_output.max(dim=1)

    predicted_class = class_names[max_index_predicted_class]

    logger.info('Prediction for frames [{},{}]: {}'
                .format(current_starting_frame_index, current_starting_frame_index + opt.sample_duration - 1,
                        predicted_class))

    # TODO IF USEFUL, adapt code to return a structure that can be analyzed after (see offline)
    return [predicted_class], ""


# Codice adattato da dataset.py
# TODO ma a che serve restituire segments (i.e. tensore costruito ad hoc? Solo per visualizzazione risultati??)
def extract_input_live_predictions(current_starting_frame_index, frames_list, opt):
    batch_size = opt.batch_size_multiplier

    # Non ha senso utilizzare batch size > 1 perchè stiamo facendo predizioni su ogni batch di frame (16 by default)
    assert batch_size == 1

    # TODO check se ha senso Scale + Crop stessa dimensione
    spatial_transform = Compose([Scale(opt.sample_size),
                                 CenterCrop(opt.sample_size),
                                 ToTensor(),
                                 Normalize(opt.mean, [1, 1, 1])])
    # Non utilizzata nella predizione live
    # temporal_transform = LoopPadding(opt.sample_duration)

    clip = [spatial_transform(img) for img in frames_list]

    clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

    # Adding batch ID as first dimension. Automatically fills it with value 1
    clip = clip[None, :, :, :, :]

    target = torch.IntTensor([current_starting_frame_index, current_starting_frame_index + opt.sample_duration - 1])

    return clip, target
