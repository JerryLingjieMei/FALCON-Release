from torch import nn
from torch.nn import functional as F

from utils import Singleton
from .measure import Measure


class Entailment(nn.Module, metaclass=Singleton):
    rep = "box"

    def __init__(self, cfg):
        super().__init__()
        self.measure = Measure(cfg)

    def forward(self, premise, consequence):
        return self.measure.entailment(premise, consequence)


class PlaneEntailment(nn.Module, metaclass=Singleton):
    rep = "plane"

    def __init__(self, cfg):
        super().__init__()
        self.margin = .2

    def forward(self, premise, consequence):
        logit_pr = (premise * consequence - self.margin).mean(-1).clamp(-1, 1) * 8.
        return logit_pr


class ConeEntailment(nn.Module, metaclass=Singleton):
    rep = "cone"

    def __init__(self, cfg):
        super().__init__()
        self.weight = 8.
        self.margin = .8

    def forward(self, premise, consequence):
        logit_pr = self.weight / self.margin * (F.cosine_similarity(premise, consequence, -1) - 1 + self.margin)
        return logit_pr


REP2ENTAILMENT = {"box": Entailment, "plane": PlaneEntailment, "cone": ConeEntailment, }


def build_entailment(cfg):
    return REP2ENTAILMENT[cfg.REPRESENTATION](cfg)
