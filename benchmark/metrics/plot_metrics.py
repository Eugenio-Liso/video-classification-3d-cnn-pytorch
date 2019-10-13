import csv

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from opts_metrics_plot import parse_opts_metrics_plot

sns.set()


# from matplotlib.ticker import FormatStrFormatter


def get_metric(class_names, idx_metrics, row):
    result = []
    for _ in class_names:
        result.append(float(row[idx_metrics]))
        idx_metrics += 1
    return result, idx_metrics


def insert_values_on_bars(ax, bars):
    """Attach a text label above each bar in *bars*, displaying its height."""
    for rect in bars:
        height = rect.get_height()
        ax.annotate('{:.3f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha='center', va='bottom')


def build_plot(idx_chart, classes_metric, class_names, x_axis, title, mean_prediction_time=None, mean_accuracy=None,
               std_prediction_time=None):
    ax = plt.subplot(idx_chart)
    # ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    barlist = plt.bar(x_axis, classes_metric)
    insert_values_on_bars(ax, barlist)

    plt.xticks(x_axis, class_names)
    plt.ylim(top=1)

    # Think of a general way to actually change colors dynamically
    if len(classes_metric) == 3:
        barlist[0].set_color('b')
        barlist[1].set_color('g')
        barlist[2].set_color('r')

    if mean_prediction_time is not None and mean_accuracy is not None:
        plt.xlabel(
            f'\nMean prediction time: {mean_prediction_time} - STD prediction time: {std_prediction_time} - Accuracy: {mean_accuracy}')

    plt.title(title)


if __name__ == '__main__':
    opt = parse_opts_metrics_plot()
    plt.rc('font', family='serif')
    _ = plt.figure('Testing Metrics')

    input_csv = opt.input_csv
    classes_list = opt.classes_list

    with open(classes_list, 'r') as f:
        class_names = []
        for row in f:
            class_names.append(row[:-1])

    with open(input_csv, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        # Did not found any good way to take the last line of a CSV. I know it's inefficient
        for row in csv_reader:
            if len(row) != 0 and row[0] == 'Metrics_overall_dataset':
                mean_prediction_time = row[1]
                std_prediction_time = row[2]
                mean_accuracy = row[3]

                idx_metrics = 4

                classes_precision, idx_metrics = get_metric(class_names, idx_metrics, row)
                classes_recall, idx_metrics = get_metric(class_names, idx_metrics, row)
                classes_fscore, _ = get_metric(class_names, idx_metrics, row)

                length_chart = len(class_names)
                x_axis = np.arange(length_chart)

                mean_prediction_time = '{:.4f}'.format(float(mean_prediction_time))
                std_prediction_time = '{:.4f}'.format(float(std_prediction_time))
                mean_accuracy = '{:.4f}'.format(float(mean_accuracy))

                build_plot(131, classes_precision, class_names, x_axis, "Classes Precision")
                build_plot(132, classes_recall, class_names, x_axis, "Classes Recall", mean_prediction_time,
                           mean_accuracy, std_prediction_time)
                build_plot(133, classes_fscore, class_names, x_axis, "Classes F-Score")

                plt.subplots_adjust(wspace=0.5, hspace=1)
                plt.show()
            else:
                continue
