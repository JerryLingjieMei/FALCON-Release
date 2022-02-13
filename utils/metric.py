from collections import deque, defaultdict

import torch

from .io import join

class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20):
        self.deque = deque(maxlen=window_size)
        self.series = []
        self.total = 0.0
        self.count = 0

    def update(self, value):
        self.deque.append(value)
        self.series.append(value)
        self.count += 1
        self.total += value

    @property
    def median(self):
        d = torch.Tensor(list(self.deque))
        return d.median().item()

    @property
    def mean(self):
        d = torch.Tensor(list(self.deque))
        return d.mean().item()

    @property
    def global_mean(self):
        return self.total / self.count


class Metric:
    def __init__(self, delimiter, summary_writer):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.summary_writer = summary_writer

    def update(self, **kwargs):
        self._update(self.meters, **kwargs)

    def _update(self, x, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            x[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {meter.mean:.4f} ({meter.global_mean:.4f})")
        return self.delimiter.join(loss_str)

    @property
    def mean(self):
        out = {name: meter.mean for name, meter in self.meters.items()}
        return out

    def log_summary(self, tag, iteration):
        for name, meter in self.meters.items():
            self.summary_writer.add_scalar(join(tag, name), meter.mean, iteration)
