import csv
import os
import time
# Removes useless warning when precision, recall or fscore are zero
import warnings
from os.path import join
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support

from dataset import get_validation_data
from logging_utils import logger_factory as lf
from spatial_transforms import (Resize, ScaleValue, Compose, Normalize, CenterCrop, ToTensor)
from temporal_transforms import (TemporalSubsampling, TemporalNonOverlappingWindow, Compose as TemporalCompose)
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


def flatten(list_elems):
    return [item for sublist in list_elems for item in sublist]


def classify_video_offline(class_names, model, opt, video_path_formatter=lambda root_path, label, video_id: Path(
    join(root_path, label, video_id))):
    device = torch.device('cpu' if opt.no_cuda else 'cuda')

    data_loader = create_dataset_offline(opt, video_path_formatter)

    accuracies = AverageMeter()

    class_size = len(class_names)
    class_idx = list(range(0, class_size))

    ground_truth_labels = []
    predicted_labels = []
    executions_times = []

    all_video_results = []
    all_execution_times = []

    print('Starting prediction phase')

    with torch.no_grad():
        for (inputs, targets, segments, video_name) in data_loader:
            targets = targets.to(device, non_blocking=True)

            # One video at a time
            video_name = video_name[0]
            segments = segments[0]
            print(f'Giving input {video_name} to the NN...')

            start_time = time.time()
            outputs = model(inputs)
            end_time = time.time()

            execution_time = (end_time - start_time)
            # print(len(inputs))
            # print(len(targets))
            # print(f'Execution time: {execution_time}')

            executions_times.append(execution_time)
            acc = calculate_accuracy(outputs, targets)

            ground_truth, predictions = ground_truth_and_predictions(outputs, targets)
            predictions = flatten(predictions)

            # print(ground_truth)
            # print(predictions)
            ground_truth_labels.extend(ground_truth)
            predicted_labels.extend(predictions)

            accuracies.update(acc, inputs.size(0))

            video_outputs = outputs.cpu().data

            exec_times_with_segments = []

            for i in range(len(segments)):
                segment = segments[i]
                # TODO this is not totally correct, but i have no choice here
                # This is because now, the input frame batches are processed only once and together, so i do not have the
                # execution time for a single batch. So, this outputs the time the NN takes to process all the batches of
                # a single video
                # To fix this, you should adapt the code in the 'live' settings here, because there the prediction time
                # is per batch. See `classify_video_online` or just slice the input tensor
                exec_time = execution_time

                exec_times_with_segments.append((segment, exec_time))

            executions_times_with_video_name = {
                video_name: exec_times_with_segments
            }

            single_video_result = {
                'video': video_name,
                'clips': []
            }

            for i in range(len(predictions)):
                clip_results = {
                    'segment': segments[i]
                }

                if opt.mode == 'score':
                    clip_results['label'] = class_names[predictions[i]]
                    clip_results['scores'] = video_outputs[i].tolist()
                elif opt.mode == 'feature':
                    clip_results['features'] = video_outputs[i].tolist()

                single_video_result['clips'].append(clip_results)

            all_video_results.append(single_video_result)
            all_execution_times.append(executions_times_with_video_name)

        accuracies_avg = accuracies.avg

        precision, recall, fscore, _ = \
            precision_recall_fscore_support(ground_truth_labels,
                                            predicted_labels,
                                            labels=class_idx)

        mean_exec_times = np.mean(executions_times)
        std_exec_times = np.std(executions_times)

        # print(f'Acc:{accuracies_avg}')
        # print(f'prec: {precision}')
        # print(f'rec: {recall}')
        # print(f'f-score: {fscore}')
        # print(mean_exec_times)
        # print(std_exec_times)

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

        return all_video_results, all_execution_times


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


def create_dataset_offline(opt, video_path_formatter):
    spatial_transform, temporal_transform = retrieve_spatial_temporal_transforms(opt)

    validation_data, collate_fn = get_validation_data(
        opt.video_root, opt.annotation_path, opt.sample_duration, opt.use_alternative_label, video_path_formatter,
        spatial_transform, temporal_transform)
    val_loader = torch.utils.data.DataLoader(validation_data,
                                             batch_size=opt.batch_size_prediction,
                                             shuffle=False,
                                             num_workers=opt.n_threads,
                                             pin_memory=True,
                                             worker_init_fn=worker_init_fn,
                                             collate_fn=collate_fn)

    return val_loader


def retrieve_spatial_temporal_transforms(opt):
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
    return spatial_transform, temporal_transform


def classify_video_online(input_frames, current_starting_frame_index, class_names, model, opt,
                          single_video_result, exec_times_with_segments):
    assert opt.mode in ['score', 'feature']

    logger.debug('Input tensor in live prediction: {}'.format(input_frames.size()))
    segment = [current_starting_frame_index, current_starting_frame_index + opt.sample_duration - 1]

    with torch.no_grad():
        start_time = time.time()
        outputs = model(input_frames)
        end_time = time.time()

        execution_time = (end_time - start_time)

        print("--- Execution time for segment {}: {} seconds ---".format(segment, execution_time))

        prediction_scores = outputs.cpu().data
        _, max_index_predicted_class = prediction_scores.max(dim=1)
        prediction_scores = flatten(prediction_scores.tolist())
        predicted_class = class_names[max_index_predicted_class]

        logger.info('Prediction for frames [{},{}]: {}'
                    .format(current_starting_frame_index, current_starting_frame_index + opt.sample_duration - 1,
                            predicted_class))

        exec_times_with_segments.append((segment, execution_time))

        clip_results = {
            'segment': segment
        }

        if opt.mode == 'score':
            clip_results['label'] = predicted_class
            clip_results['scores'] = prediction_scores
        elif opt.mode == 'feature':
            clip_results['features'] = prediction_scores

        single_video_result['clips'].append(clip_results)

        return predicted_class
