import argparse


def parse_opts_prediction():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_json', default='output.json', type=str, help='Output json from prediction phase')
    parser.add_argument('--input_video_folder', type=str, help='Root path of input videos')
    parser.add_argument('--prediction_video_folder', type=str, help='Output folder where annotated videos will be put')
    parser.add_argument('--classes_list', default='class_names_list', type=str,
                        help='File containing the available classes for prediction (i.e. class labels)')
    parser.add_argument('--frames_for_prediction', type=str,
                        help='Determines how many group of frames (i.e. clips) will be averaged in order to select the output class (by default, a clip is 16 frames long)')

    args = parser.parse_args()

    return args
