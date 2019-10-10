import numpy as np


class AverageMetricsNumPyArray(object):

    def __init__(self, class_names):
        self.np_array = np.zeros(len(class_names))
        self.count = dict.fromkeys(class_names, 0)

    def update(self, val, target_class):
        self.np_array += val
        self.count[target_class] += 1

    def average(self):
        return np.divide(self.np_array, self.count, out=np.zeros_like(self.np_array), where=self.count != 0)


class AverageTimes(object):

    def __init__(self):
        self.value = 0
        self.count = 0

    def update(self, val):
        self.value += val
        self.count += 1

    def average(self):
        return self.value / self.count


class AverageAccuracy(object):

    def __init__(self):
        self.value = 0
        self.count = 0

    def update(self, val, counts):
        self.value += val
        self.count += counts

    def average(self):
        return self.value / self.count
