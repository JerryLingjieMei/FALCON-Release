import random
from itertools import chain

import torch
from torch.utils.data import WeightedRandomSampler


class WeightedConcatBatchSampler:
    def __init__(self, dataset, ratio, batch_size):
        self.dataset = dataset
        self.ratio = ratio
        self.batch_size = batch_size

    @property
    def segments(self):
        segments = []
        for dataset, size in zip(self.dataset.datasets, [0] + self.dataset.cumulative_sizes):
            s = list(range(size, size + len(dataset)))
            random.shuffle(s)
            segment = []
            for i in range(len(s) // self.batch_size):
                segment.append(s[i * self.batch_size:(i + 1) * self.batch_size])
            segments.append(segment)
        alpha = min(len(segment) / r for segment, r in zip(segments, self.ratio))
        segments = list(
            chain.from_iterable(segment[:int(alpha * r)] for segment, r in zip(segments, self.ratio)))
        return segments

    def __iter__(self):
        for segment in self.segments:
            yield segment

    def __len__(self):
        return len(self.segments)


class WeightedConcatDataset(torch.utils.data.ConcatDataset):

    def __init__(self, datasets, ratio):
        super().__init__(datasets)
        self.ratio = ratio

    def log_info(self):
        for dataset in self.datasets:
            dataset.log_info()

    @property
    def iteration(self):
        return self.datasets[0].iteration

    @iteration.setter
    def iteration(self, other):
        for dataset in self.datasets:
            dataset.iteration = other

    def __str__(self):
        return '+'.join(f"{d}*{r:.02f}" for r, d in zip(self.ratio, self.datasets))

    @property
    def tag(self):
        if self.datasets[0].split in ["train", "val"]:
            return self.datasets[0].split
        else:
            return str(self)

    @property
    def info(self):
        info = {}
        for dataset in self.datasets:
            info.update(dataset.info)
        return info

    def get_batch_sampler(self, batch_size):
        return WeightedConcatBatchSampler(self, self.ratio, batch_size)

    def batch_evaluate(self, inputs, outputs, evaluated):
        self.datasets[0].batch_evaluate(inputs, outputs, evaluated)

    def result_evaluate(self, evaluated):
        return self.datasets[0].result_evaluate(evaluated)

    def __getattr__(self, item):
        for dataset in self.datasets:
            if item in dir(dataset):
                return getattr(dataset, item)
        raise AttributeError(f"Attribute \"{item}\" not found in weighted dataset composed by "
                             f"{', '.join(dataset.name for dataset in self.datasets)}.")
