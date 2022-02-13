import torch
import torch.nn.functional as F

from models.fewshot.falcon.bayes import BayesLearner
from models.fewshot.nn import ConceptGraphGNN, ExampleGraphGNN
from utils import log_normalize


class GraphicalLearner(BayesLearner):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.gnn_1 = ConceptGraphGNN(cfg)
        self.gnn_2 = ExampleGraphGNN(cfg)

    def forward(self, p):
        queried = self.prior.sample(self.n_particles)
        others = p.support_embeddings.unsqueeze(0).expand(self.n_particles, -1, -1)
        features = p.train_features.unsqueeze(0).expand(self.n_particles, -1, -1)

        if p.is_attached:
            embeddings = torch.cat([others, queried.unsqueeze(1)], 1)
            queried = self.gnn_1(embeddings, p.relations)[:, -1]
        if p.is_fewshot:
            embeddings = torch.cat([features, queried.unsqueeze(1)], 1)
            queried = self.gnn_2(embeddings)[:, -1]
        log_probs = log_normalize(
            F.logsigmoid(self.entailment(queried.unsqueeze(1), features)).sum(1)) if p.is_fewshot else None
        queried_embedding = self.reduction(queried, log_probs)

        if self.training:
            prior_reg = self.prior_reg(p.parent.concept_index.unsqueeze(0))
            return {"queried_embedding": queried_embedding, "prior_reg": prior_reg}
        else:
            return {"queried_embedding": queried_embedding}
