import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from opts_metrics_plot import parse_opts_metrics_plot
import matplotlib.cm as cm
from matplotlib.colors import Normalize
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


def build_plot(idx_chart, classes_metric, class_names, x_axis, title, cmap, mean_prediction_time=None,
               mean_accuracy_keys=None, mean_accuracy_value=None, std_prediction_time=None):
    ax = plt.subplot(idx_chart)
    # ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    cm_map = cm.get_cmap(cmap)
    norm = Normalize(vmin=0, vmax=len(classes_metric))

    barlist = plt.bar(x_axis, classes_metric, color=cm_map(norm(x_axis)))
    insert_values_on_bars(ax, barlist)
    ax.xaxis.grid()  # horizontal grid only

    plt.xticks(x_axis, class_names)
    plt.ylim(top=1)

    # Think of a general way to actually change colors dynamically
    # if len(classes_metric) == 3:
    #     barlist[0].set_color('b')
    #     barlist[1].set_color('g')
    #     barlist[2].set_color('r')

    if mean_prediction_time is not None and std_prediction_time is not None and mean_accuracy_value is not None:
        plt.xlabel(
            f'\nMean prediction time: {mean_prediction_time} - STD prediction time: {std_prediction_time} - Accuracy: {mean_accuracy_value}')
    elif mean_accuracy_keys is not None:
        printed_str = ''
        for keyName, acc in mean_accuracy_keys.items():
            printed_str += f'Accuracy on {keyName}: {"{:.4f}".format(float(acc))}\n'
        plt.xlabel(printed_str)

    plt.title(title)


if __name__ == '__main__':
    opt = parse_opts_metrics_plot()
    plt.rc('font', family='serif')
    _ = plt.figure('Testing Metrics')

    input_csv = opt.input_csv
    classes_list = opt.classes_list
    merge = opt.merge
    filter_on_class = opt.filter_on_class
    cmap = opt.colormap

    if not merge and len(input_csv) != 1:
        raise Exception("When not merging different csv metrics, you should specify only one csv in input")
    elif merge and (len(input_csv) < 2 or not filter_on_class):
        raise Exception("When merging csv files, you should specify at least two input csv and a class filter")

    if not merge:
        with open(classes_list, 'r') as f:
            class_names = []
            for row in f:
                class_names.append(row[:-1])
        length_chart = len(class_names)
        x_axis = np.arange(length_chart)
    else:
        x_axis = np.arange(len(input_csv))

    if not merge:
        with open(input_csv[0], 'r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

            # Did not found any good way to take the last line of a CSV. I know it's inefficient
            for row in csv_reader:
                if len(row) != 0 and row[0] == 'Metrics_overall_dataset':
                    mean_prediction_time = row[1]
                    std_prediction_time = row[2]
                    mean_accuracy = row[3]

                    idx_metrics = 4

                    class_precision, idx_metrics = get_metric(class_names, idx_metrics, row)
                    class_recall, idx_metrics = get_metric(class_names, idx_metrics, row)
                    class_fscore, _ = get_metric(class_names, idx_metrics, row)

                    mean_prediction_time = '{:.4f}'.format(float(mean_prediction_time))
                    std_prediction_time = '{:.4f}'.format(float(std_prediction_time))
                    mean_accuracy = '{:.4f}'.format(float(mean_accuracy))

                    build_plot(131, class_precision, class_names, x_axis, "Classes Precision", cmap)
                    build_plot(132, class_recall, class_names, x_axis, "Classes Recall", cmap,
                               mean_prediction_time=mean_prediction_time,
                               mean_accuracy_value=mean_accuracy, std_prediction_time=std_prediction_time, )
                    build_plot(133, class_fscore, class_names, x_axis, "Classes F-Score", cmap)

                    plt.subplots_adjust(wspace=0.5, hspace=1)
                    plt.show()
                else:
                    continue
    else:
        first_header = True
        class_indexes_for_metrics = []

        # Also prediction time?
        mean_accuracies = {}
        classes_precisions = []
        classes_recalls = []
        classes_fscores = []
        dummy_class_names = [filter_on_class]

        for metric_csv in input_csv:
            with open(metric_csv, 'r') as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                accuracy_key = os.path.basename(os.path.normpath(metric_csv))

                for row in csv_reader:
                    if first_header:
                        first_header = False
                        class_found = False
                        tmp_idx = 0
                        while tmp_idx < len(row):
                            row_value = row[tmp_idx]
                            if "_" in row_value and row_value.split('_')[1] == filter_on_class:
                                class_found = True
                                class_indexes_for_metrics.append(tmp_idx)
                            tmp_idx += 1
                        if not class_found:
                            raise Exception(f"The class specified: {filter_on_class} cannot be found in csv metrics")
                        assert len(class_indexes_for_metrics) == 3, "The class metrics should be three"

                    if len(row) != 0 and row[0] == 'Metrics_overall_dataset':
                        mean_accuracy = row[3]

                        if '.csv' in accuracy_key:
                            accuracy_key = accuracy_key.split('.')[0]
                        mean_accuracies[accuracy_key] = mean_accuracy

                        class_precision, _ = get_metric(dummy_class_names, class_indexes_for_metrics[0], row)
                        class_recall, _ = get_metric(dummy_class_names, class_indexes_for_metrics[1], row)
                        class_fscore, _ = get_metric(dummy_class_names, class_indexes_for_metrics[2], row)

                        classes_precisions.extend(class_precision)
                        classes_recalls.extend(class_recall)
                        classes_fscores.extend(class_fscore)

        x_labels = mean_accuracies.keys()
        build_plot(131, classes_precisions, x_labels, x_axis, f"{filter_on_class} Precision", cmap)
        build_plot(132, classes_recalls, x_labels, x_axis, f"{filter_on_class} Recall", cmap,
                   mean_accuracy_keys=mean_accuracies)
        build_plot(133, classes_fscores, x_labels, x_axis, f"{filter_on_class} F-Score", cmap)

        plt.subplots_adjust(wspace=0.5, hspace=1)
        plt.show()
