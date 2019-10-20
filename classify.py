import csv
import os
import time
# Removes useless warning when precision, recall or fscore are zero
import warnings

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from torch.autograd import Variable

from dataset import get_validation_data
from logging_utils import logger_factory as lf
from spatial_transforms import (Resize, ScaleValue, Compose, Normalize, Scale, CenterCrop, ToTensor)
from temporal_transforms import (TemporalSubsampling, TemporalNonOverlappingWindow, Compose as TemporalCompose, TemporalEvenCrop)
from utils import AverageMeter, calculate_accuracy, ground_truth_and_predictions
from utils import worker_init_fn, get_mean_std

warnings.filterwarnings('ignore', message='(.*)Precision and F-score are ill-defined(.*)')
warnings.filterwarnings('ignore', message='(.*)Recall and F-score are ill-defined(.*)')

logger = lf.getBasicLogger(os.path.basename(__file__))


def create_column_metric_csv_content(contents, csv_row):
    for result in contents:
        csv_row.append(result)


def create_column_metric_csv_header(column_prefix, class_names, header):
    for target_class in class_names:
        header.append(f"{column_prefix}_{target_class}")


def classify_video_offline(class_names, model, opt):
    device = torch.device('cpu' if opt.no_cuda else 'cuda')

    data_loader = create_dataset_offline(opt)

    accuracies = AverageMeter()

    class_size = len(class_names)
    class_idx = list(range(0, class_size))

    ground_truth_labels = []
    predicted_labels = []
    executions_times = []

    print('Starting prediction phase')

    with torch.no_grad():
        for _, (inputs, targets) in enumerate(data_loader):
            targets = targets.to(device, non_blocking=True)

            start_time = time.time()
            outputs = model(inputs)
            end_time = time.time()

            execution_time = (end_time - start_time)
            # print(len(inputs))
            # print(len(targets))
            print(f'Execution time: {execution_time}')

            executions_times.append(execution_time)
            acc = calculate_accuracy(outputs, targets)

            ground_truth, predictions = ground_truth_and_predictions(outputs, targets)
            # print(ground_truth)
            # print(predictions)
            ground_truth_labels.extend(ground_truth)
            predicted_labels.extend(predictions)

            accuracies.update(acc, inputs.size(0))

        accuracies_avg = accuracies.avg

        precision, recall, fscore, _ = \
            precision_recall_fscore_support(ground_truth_labels,
                                            predicted_labels,
                                            labels=class_idx)

        mean_exec_times = np.mean(executions_times)
        std_exec_times = np.std(executions_times)

        print(f'Acc:{accuracies_avg}')
        print(f'prec: {precision}')
        print(f'rec: {recall}')
        print(f'f-score: {fscore}')
        print(mean_exec_times)
        print(std_exec_times)

        with open(opt.output_csv, 'w+') as csv_file:
            writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            header = ["VIDEO_NAME", "MEAN_PREDICTION_TIME", "STANDARD_DEV_PREDICTION_TIME", "ACCURACY"]

            create_column_metric_csv_header("PRECISION", class_names, header)
            create_column_metric_csv_header("RECALL", class_names, header)
            create_column_metric_csv_header("F-SCORE", class_names, header)

            writer.writerow(header)

            final_row = ['Metrics_overall_dataset', mean_exec_times, std_exec_times, accuracies_avg]

            create_column_metric_csv_content(precision, final_row)
            create_column_metric_csv_content(recall, final_row)
            create_column_metric_csv_content(fscore, final_row)

            writer.writerow(final_row)


def get_normalize_method(mean, std, no_mean_norm, no_std_norm):
    if no_mean_norm:
        if no_std_norm:
            return Normalize([0, 0, 0], [1, 1, 1])
        else:
            return Normalize([0, 0, 0], std)
    else:
        if no_std_norm:
            return Normalize(mean, [1, 1, 1])
        else:
            return Normalize(mean, std)


def create_dataset_offline(opt):
    opt.mean, opt.std = get_mean_std(opt.value_scale, dataset=opt.mean_dataset)

    normalize = get_normalize_method(opt.mean, opt.std, opt.no_mean_norm,
                                     opt.no_std_norm)
    spatial_transform = [Resize(opt.sample_size),
                         CenterCrop(opt.sample_size),
                         ToTensor(),
                         ScaleValue(opt.value_scale),
                         normalize]
    spatial_transform = Compose(spatial_transform)

    temporal_transform = []
    if opt.sample_t_stride > 1:
        temporal_transform.append(TemporalSubsampling(opt.sample_t_stride))
    temporal_transform.append(
        TemporalNonOverlappingWindow(opt.sample_duration))
    temporal_transform = TemporalCompose(temporal_transform)

    validation_data, collate_fn = get_validation_data(
        opt.video_root, opt.annotation_path,
        spatial_transform, temporal_transform)
    val_loader = torch.utils.data.DataLoader(validation_data,
                                             batch_size=opt.batch_size_prediction,
                                             shuffle=False,
                                             num_workers=opt.n_threads,
                                             pin_memory=True,
                                             worker_init_fn=worker_init_fn,
                                             collate_fn=collate_fn)

    return val_loader


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
