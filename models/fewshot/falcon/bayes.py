import torch

from models.fewshot.falcon.prior import build_prior
from models.fewshot.program_executor import MetaLearner
from models.nn import build_entailment
from utils import log_normalize


class BayesLearner(MetaLearner):
    def __init__(self, cfg):
        super().__init__()
        self.prior = build_prior(cfg)
        bayes_cfg = cfg.BAYES
        self.n_particles = bayes_cfg.N_PARTICLES
        self.max_iter = bayes_cfg.MAX_ITER
        reduction = bayes_cfg.REDUCTION
        assert reduction in ["mean", "max"]
        if reduction == "mean":
            self.reduction = self.mean_reduce
        else:
            self.reduction = self.max_reduce
        self.entailment = build_entailment(cfg)
        self.prior_lambda = bayes_cfg.PRIOR_LAMBDA

    def sample_prior(self, gt_embeddings, n_particle):
        """
        :return: P x N x D
        """
        prior_sampled = self.prior.sample(n_particle)
        embeddings = gt_embeddings.expand(n_particle, *([-1] * len(gt_embeddings.shape)))
        if self.training:
            embeddings = self._add_noise(embeddings)
        embeddings[:, -1] = prior_sampled
        return embeddings

    def prior_reg(self, embedding, detach=False):
        dim = embedding.shape[-1] // 2
        embedding = torch.cat([embedding[..., :dim], embedding[..., dim:].clamp(min=.02, max=0.48)], -1)
        return self.prior_lambda * self.prior.reg(embedding, detach).mean()

    def max_reduce(self, queried_embeddings, log_weights=None):
        if log_weights is None:
            max_index = torch.randint(len(queried_embeddings), (1,)).item()
        else:
            max_index = torch.max(log_weights, 0).indices.item()
        return queried_embeddings[max_index]

    def mean_reduce(self, queried_embeddings, log_weights=None):
        if log_weights is None:
            log_weights = log_normalize(
                torch.ones((len(queried_embeddings),), device=queried_embeddings.device))
        return (torch.exp(log_weights.unsqueeze(1)) * queried_embeddings).sum(0)
