import argparse
from pathlib import Path

def parse_opts_metrics_plot():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv', nargs='+', help='CSV containing metrics')
    parser.add_argument('--classes_list', default='', type=Path,
                        help='A text file containing the possible outcomes. One class per row.')
    parser.add_argument('--filter_on_class', type=str,
                        help='Specifies the single class to use when merging multiple csv metrics')
    parser.add_argument('--merge', action='store_true',
                        help='If used, it will merge the input_csv files along the class specified with "filter_on_class". '
                             'Only a subset of the metrics will be displayed')
    parser.add_argument('--colormap', default='gist_rainbow', type=str,
                        help='Colormap to use when drawing the chart')
    parser.add_argument('--output_plot', default='plot_metrics.png', type=Path, required=True,
                        help='Output path of the plot containing the calculated metrics')

    args = parser.parse_args()

    return args
