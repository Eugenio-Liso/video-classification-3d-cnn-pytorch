import argparse


def parse_opts_benchmark():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='output.json', type=str,
                        help='Output json from prediction phase')
    parser.add_argument('--labeled_videos', default='', type=str,
                        help='A JSON containing a list of tuples (video name, ground truth label)')

    args = parser.parse_args()

    return args
