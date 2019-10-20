import argparse
from pathlib import Path

def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_root', default='', type=str, help='Root path of input videos')
    parser.add_argument('--model', default='', type=str, help='Model file path')
    parser.add_argument('--output', default='output.json', type=str, help='Output file path with predictions')
    parser.add_argument('--class_names_list', default='./classes_list/class_names_list_kinetics', type=str,
                        help='File containing predictions classes')
    parser.add_argument('--output_times', default='output_times.json', type=str,
                        help='Output file path with execution times (taken for each clip)')
    parser.add_argument('--output_mean_times', default='output_mean_times.json', type=str,
                        help='Output file path with mean execution times')
    parser.add_argument('--mode', default='score', type=str,
                        help='Mode (score | feature). score outputs class scores. feature outputs features (after '
                             'global average pooling).')
    parser.add_argument('--batch_size_multiplier', default=1, type=int,
                        help='Batch Size multiplier. For example, if 2 it'
                             'means that the DataLoader will load sample_duration x 2 frames'
                             'at a time')
    parser.add_argument('--batch_size_prediction', default=1, type=int,
                        help='Batch Size when loading data in Dataloader')
    parser.add_argument('--sample_duration', default=16, type=int,
                        help='The sample duration, i.e. how many frames to take when predicting')
    parser.add_argument('--n_threads', default=4, type=int, help='Number of threads for multi-thread loading')
    parser.add_argument('--model_name', default='resnet', type=str, help='Currently only support resnet')
    parser.add_argument('--model_depth', default=34, type=int, help='Depth of resnet (10 | 18 | 34 | 50 | 101)')
    parser.add_argument('--resnet_shortcut', default='A', type=str, help='Shortcut type of resnet (A | B)')
    parser.add_argument('--wide_resnet_k', default=2, type=int, help='Wide resnet k')
    parser.add_argument('--resnext_cardinality', default=32, type=int, help='ResNeXt cardinality')
    parser.add_argument('--no_cuda', action='store_true', help='If true, cuda is not used.')
    parser.add_argument('--verbose', action='store_true', help='')
    parser.set_defaults(verbose=False)
    parser.add_argument('--prediction_input_mode', default='opencv', type=str,
                        help='If legacy, the input data will be processed with FFMPEG. If opencv, data will be loaded '
                             'with opencv.')
    parser.add_argument('--type_of_prediction', default='offline', type=str,
                        help='If offline, the frames will be extracted from the video and predictions will be done on '
                             'them. If live, predictions will be showed in real time. This option can be used only '
                             'when prediction_input_mode is set to opencv.')
    parser.add_argument('--frames', action='store_true',
                        help='If activated, the video frames will be used instead of a video. The video_root must contain '
                             'subfolders, each one representing a target class and containing a set of frames. '
                             'Applicable only when using opencv.')
    parser.add_argument(
        '--value_scale',
        default=1,
        type=int,
        help=
        'If 1, range of inputs is [0-1]. If 255, range of inputs is [0-255].')
    parser.add_argument('--mean_dataset',
                        default='kinetics',
                        type=str,
                        help=('dataset for mean values of mean subtraction'
                              '(activitynet | kinetics)'))
    parser.add_argument('--no_mean_norm',
                        action='store_true',
                        help='If true, inputs are not normalized by mean.')
    parser.add_argument(
        '--no_std_norm',
        action='store_true',
        help='If true, inputs are not normalized by standard deviation.')
    parser.add_argument(
        '--sample_t_stride',
        default=1,
        type=int,
        help='If larger than 1, input frames are subsampled with the stride.')
    parser.add_argument('--n_val_samples',
                        default=3,
                        type=int,
                        help='Number of validation samples for each activity')
    parser.add_argument('--output_csv', default='metrics.csv', type=str,
                        help='The output path of the csv with the calculated results')
    parser.add_argument('--annotation_path',
                        default=None,
                        type=Path,
                        help='Annotation file path')
    args = parser.parse_args()

    return args
