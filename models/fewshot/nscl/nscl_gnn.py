import torch

from models.fewshot import ConceptGraphGNN
from models.fewshot.program_executor import MetaLearner
from models.nn import build_entailment, Measure


class NsclGnnLearner(MetaLearner):

    def __init__(self, cfg):
        super().__init__()
        self.dim = cfg.DIMENSION
        self.measure = Measure(cfg)
        self.gnn = ConceptGraphGNN(cfg)
        self.rep = cfg.REPRESENTATION
        self.entailment = build_entailment(cfg)

    def forward(self, p):
        if p.is_fewshot:
            if self.rep == "box":
                specific_boundary = self.measure.specific_boundary(p.train_features)
            else:
                specific_boundary = p.train_features.mean(0)
        else:
            specific_boundary = torch.zeros(*p.train_features.shape[1:]).to(p.train_features.device)
        embeddings = torch.cat([p.support_embeddings, specific_boundary.unsqueeze(0)], 0).unsqueeze(0)
        new_embeddings = self.gnn(embeddings, p.relations)
        queried_embedding = new_embeddings[0, -1]

        return {"queried_embedding": queried_embedding, "specific_boundary": specific_boundary.detach()}
