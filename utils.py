import csv
import random
from functools import partialmethod
from pathlib import Path

import numpy as np
import torch
from torch import device


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def calculate_accuracy(outputs, targets):
    with torch.no_grad():
        batch_size = targets.size(0)

        _, pred = outputs.topk(1, 1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1))
        n_correct_elems = correct.float().sum().item()

        return n_correct_elems / batch_size


def class_counts(x): return 1 if x > 0 else 0


def worker_init_fn(worker_id):
    torch_seed = torch.initial_seed()

    random.seed(torch_seed + worker_id)

    if torch_seed >= 2 ** 32:
        torch_seed = torch_seed % 2 ** 32
    np.random.seed(torch_seed + worker_id)


def ground_truth_and_predictions(outputs, targets):
    with torch.no_grad():
        _, pred = outputs.topk(1, 1, largest=True, sorted=True)

        ground_truth = targets.view(-1, 1).cpu().numpy()
        predictions = pred.cpu().numpy()

        return ground_truth, predictions


def get_mean_std(value_scale, dataset):
    assert dataset in ['activitynet', 'kinetics']

    if dataset == 'activitynet':
        mean = [0.4477, 0.4209, 0.3906]
        std = [0.2767, 0.2695, 0.2714]
    elif dataset == 'kinetics':
        mean = [0.4345, 0.4051, 0.3775]
        std = [0.2768, 0.2713, 0.2737]

    mean = [x * value_scale for x in mean]
    std = [x * value_scale for x in std]

    return mean, std
