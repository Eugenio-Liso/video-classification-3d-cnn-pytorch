import argparse


def parse_opts_prediction():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_json', default='output.json', type=str, help='Output json from prediction phase')
    parser.add_argument('--input_video_folder', type=str, help='Root path of input videos')
    parser.add_argument('--prediction_video_folder', type=str, help='Output folder where annotated videos will be put')
    parser.add_argument('--video_format', type=str, default='mp4', help='The format of the input video to label')
    parser.add_argument('--classes_list', default='class_names_list', type=str,
                        help='File containing the available classes for prediction (i.e. class labels)')
    parser.add_argument('--frames_for_prediction', type=int, default=16,
                        help='Determines how many group of frames (i.e. clips) will be averaged in order to select '
                             'the output class (by default, a clip is 16 frames long)')
    # parser.add_argument('--video_name_frames_formatter', action='store_true', help='If used, every video name in '
    #                                                                                'output_json is considered as formatted '
    #                                                                                'as an output of a frame extraction phase, i.e. '
    #                                                                                'the real video is named X but the "video" label '
    #                                                                                'in the JSON is X_HH:MM:SS_SSS_HH:MM:SS_SSS')

    args = parser.parse_args()

    return args
