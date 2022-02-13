import torch
from torch import nn

from models.fewshot.program_executor import MetaLearner
from models.nn import build_entailment
from utils import create_dummy


class PrototypicalLearner(MetaLearner):
    def __init__(self, cfg):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.))
        self.bias = nn.Parameter(torch.tensor(0.))
        self.entailment = build_entailment(cfg)

    def forward(self, p):
        queried_embedding = p.train_features.squeeze() + self.dummy - self.dummy
        return {"queried_embedding": queried_embedding}


class OracleLearner(MetaLearner):
    def __init__(self, cfg):
        super().__init__()
        self.dummy = create_dummy()
        self.entailment = build_entailment(cfg)

    def forward(self, p):
        queried_embedding = p.gt_embeddings[-1] + self.dummy - self.dummy
        return {"queried_embedding": queried_embedding}


class AllTrueLearner(MetaLearner):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.REPRESENTATION == "box"
        self.dummy = create_dummy()
        self.entailment = build_entailment(cfg)
        self.dim = cfg.DIMENSION

    @property
    def embedding(self):
        return torch.cat([torch.zeros(self.dim), torch.ones(self.dim) * 10])

    def forward(self, p):
        queried_embedding = self.embedding.to(p.gt_embeddings.device) + self.dummy - self.dummy
        return {"queried_embedding": queried_embedding}


class AllFalseLearner(MetaLearner):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.REPRESENTATION == "box"
        self.dummy = create_dummy()
        self.entailment = build_entailment(cfg)
        self.dim = cfg.DIMENSION

    @property
    def embedding(self):
        return torch.cat([torch.ones(self.dim) * 10, torch.zeros(self.dim)])

    def forward(self, p):
        queried_embedding = self.embedding.to(p.gt_embeddings.device) + self.dummy - self.dummy
        return {"queried_embedding": queried_embedding}
