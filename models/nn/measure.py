import torch
from torch import nn
from torch.nn import functional as F

from utils import EPS


class Measure(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.dim = cfg.DIMENSION
        self.temperature = cfg.TEMPERATURE

    def forward(self, x):
        return torch.sum(torch.log(self.softplus(x[..., self.dim:])), dim=-1)

    def measure_along_axis(self, x):
        return self.softplus(x[..., self.dim:])

    @classmethod
    def log2logit(cls, log):
        log = torch.clamp(log, max=-EPS)
        logit = log - torch.log(1 - torch.exp(log))
        return logit

    def intersection(self, x, y):
        x_center, x_offset = x.chunk(2, -1)
        y_center, y_offset = y.chunk(2, -1)
        maxima = torch.min(x_center + x_offset, y_center + y_offset)
        minima = torch.max(x_center - x_offset, y_center - y_offset)
        intersection = torch.cat([maxima + minima, maxima - minima], -1) / 2
        return intersection

    def entailment(self, x, y):
        x_center, x_offset = x.chunk(2, -1)
        y_center, y_offset = y.chunk(2, -1)
        maxima = torch.min(x_center + x_offset, y_center + y_offset)
        minima = torch.max(x_center - x_offset, y_center - y_offset)
        intersection_volume = F.softplus((maxima - minima) / 2, 1 / self.temperature).clamp(min=EPS)
        volume = F.softplus(x_offset, 1 / self.temperature).clamp(min=EPS)
        log = torch.sum(torch.log((intersection_volume / volume)), -1).clamp(max=-EPS)
        logit = log - torch.log(1 - torch.exp(log))
        return logit

    def union(self, x, y):
        x_center, x_offset = x.chunk(2, -1)
        y_center, y_offset = y.chunk(2, -1)
        maxima = torch.max(x_center + x_offset, y_center + y_offset)
        minima = torch.min(x_center - x_offset, y_center - y_offset)
        union = torch.cat([maxima + minima, maxima - minima], -1) / 2
        return union

    def specific_boundary(self, xs):
        x_center, x_offset = xs.chunk(2, -1)
        minima, _ = torch.min(x_center - x_offset, 0)
        maxima, _ = torch.max(x_center + x_offset, 0)
        union = torch.cat([maxima + minima, maxima - minima]) / 2
        return union

    def iou(self, x, y):
        intersection = self.intersection(x, y)
        union = self.union(x, y)
        log_pr = self(intersection) - self(union)
        return log_pr
