import torch
from torch import nn as nn
from torch.nn import functional as F

from models.fewshot.program_executor import MetaLearner
from models.nn import Measure
from utils import bind, unbind


class AggregateLearner(MetaLearner):

    def __init__(self, cfg):
        super().__init__()
        self.dim = cfg.DIMENSION
        self.rep = cfg.REPRESENTATION
        if self.rep == "box":
            mid_channels = cfg.AGGREGATE.MID_CHANNELS
            self.measure = Measure(cfg)
        else:
            mid_channels = cfg.AGGREGATE.MID_CHANNELS_WEIGHT
            self.measure = None
        self.feature_fc = nn.Linear(2, mid_channels)
        self.hypernym_fc = nn.Linear(2, mid_channels)
        self.samekind_fc = nn.Linear(2, mid_channels)
        self.final_fc = nn.Linear(mid_channels, 2)

    def _unbind(self, x):
        return unbind(x) if self.rep == "box" else x

    def _bind(self, x):
        return bind(x) if self.rep == "box" else x

    def forward(self, p):
        xs = []
        if p.is_fewshot:
            specific_boundary = self.measure.specific_boundary(p.train_features)
            xs.append(F.leaky_relu(self.feature_fc(self._unbind(specific_boundary.unsqueeze(0)))))
        else:
            specific_boundary = None
        if p.is_attached:
            xs.append(F.leaky_relu(self.hypernym_fc(self._unbind(p.hypernym_embeddings))))
            xs.append(F.leaky_relu(self.samekind_fc(self._unbind(p.samekind_embeddings))))
        queried_embedding = self._bind(self.final_fc(torch.cat(xs, 0).mean(0, keepdim=True)).squeeze(0))

        if specific_boundary is not None:
            return {"queried_embedding": queried_embedding, "specific_boundary": specific_boundary.detach()}
        else:
            return {"queried_embedding": queried_embedding}
