import random

import torch
import torch.nn.functional as F
from torch import nn

from models.fewshot.falcon.bayes import BayesLearner
from models.fewshot.nn import ConceptGraphGNN, ExampleGraphGNN
from utils import log_normalize


class RecurrentLearner(BayesLearner):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.rnn_1 = ConceptGraphGNN(cfg)
        self.rnn_2 = ExampleGraphGNN(cfg)
        self.gamma = nn.Parameter(torch.tensor(cfg.BAYES.GAMMA))

    def get_coefficient(self, i):
        return (i + 1) ** -self.gamma

    def forward(self, p):
        queried = self.prior.sample(self.n_particles)
        others = p.gt_embeddings.unsqueeze(0).expand(self.n_particles, -1, -1)
        features = p.train_features.unsqueeze(0).expand(self.n_particles, -1, -1)

        schedule = []
        if p.is_fewshot:
            schedule += list(("feature", _) for _ in range(len(p.train_features)))
        if p.is_attached:
            schedule += list(("metaconcept", _) for _ in range(len(p.gt_embeddings)))
        random.shuffle(schedule)

        for category, i in schedule:
            coefficient = self.get_coefficient(i)
            if category == "feature":
                embeddings = torch.stack([features[:, i], queried], 1)
                queried = self.rnn_2(embeddings, None, coefficient)[:, -1]
            else:
                subset_relations = p.relations[i:i + 1]
                embeddings = torch.stack([others[:, i], queried], 1)
                queried = self.rnn_1(embeddings, subset_relations, coefficient)[:, -1]
            queried = queried.clamp(-100, 100)

        log_probs = log_normalize(
            F.logsigmoid(self.entailment(queried.unsqueeze(1), features)).sum(1)) if p.is_fewshot else None
        queried_embedding = self.reduction(queried, log_probs)

        if self.training:
            prior_reg = self.prior_reg(p.parent.concept_index.unsqueeze(0))
            return {"queried_embedding": queried_embedding, "prior_reg": prior_reg}
        else:
            return {"queried_embedding": queried_embedding}
