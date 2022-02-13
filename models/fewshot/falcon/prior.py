import torch
from torch import nn
from torch.distributions import Dirichlet
from torch.distributions import Normal


class DirichletPrior(nn.Module):
    def forward(self, *input):
        raise NotImplementedError

    def __init__(self, concentration, scale, dimension):
        super().__init__()
        # noinspection PyCallingNonCallable
        self.concentration = nn.Parameter(torch.tensor(concentration))
        self.scale = scale
        self.dim = dimension

    def dirichlet(self, detach=False):
        return Dirichlet(self.concentration.detach() if detach else self.concentration)

    def sample(self, *size):
        sampled = self.dirichlet().sample((*size, self.dim))
        result = self.scale * torch.cat([sampled[..., 1] - sampled[..., 0], sampled[..., 2]], -1)
        return result

    def reg(self, embeddings, detach=False):
        x = ((-embeddings[..., :self.dim] - embeddings[..., self.dim:]) / self.scale + 1) / 2
        y = ((embeddings[..., :self.dim] - embeddings[..., self.dim:]) / self.scale + 1) / 2
        z = 1 - x - y
        return -self.dirichlet(detach).log_prob(torch.stack([x, y, z], -1))


class NormalPrior(nn.Module):
    def forward(self, *input):
        raise NotImplementedError

    def __init__(self, loc, scale, dimension):
        super().__init__()
        # noinspection PyCallingNonCallable
        self.mu = nn.Parameter(torch.tensor(loc))
        self.sigma = nn.Parameter(torch.tensor(scale))
        self.dim = dimension

    def normal(self, detach=False):
        return Normal(self.mu.detach(), self.sigma.detach()) if detach else Normal(self.mu, self.sigma)

    def sample(self, *size):
        sampled = self.normal().sample((*size, self.dim))
        return sampled

    def reg(self, embeddings, detach=False):
        return - self.normal(detach).log_prob(embeddings)


class SpherePrior(nn.Module):
    def forward(self, *input):
        raise NotImplementedError

    def __init__(self, dimension):
        super().__init__()
        self.mu = nn.Parameter(torch.zeros(1).squeeze())
        self.sigma = nn.Parameter(torch.tensor(1.))
        self.dim = dimension

    def normal(self, detach=False):
        return Normal(self.mu.detach(), self.sigma.detach()) if detach else Normal(self.mu, self.sigma)

    def sample(self, *size):
        sampled = self.normal().sample((*size, self.dim))
        return sampled / sampled.norm(dim=-1, keepdim=True)

    def reg(self, embeddings, detach=False):
        return 0


def build_prior(cfg):
    prior_cfg = cfg.BAYES.PRIOR
    if cfg.REPRESENTATION == "box":
        return DirichletPrior(*prior_cfg.BOX_PARAMS, cfg.DIMENSION)
    else:
        return NormalPrior(*prior_cfg.PLANE_PARAMS, cfg.DIMENSION)
