import torch
from numpy import euler_gamma
from torch.distributions.dirichlet import Dirichlet

from snippets.snippet_utils import cfg2test_loader, cfg2model
from utils import ArgumentParser

eps = 10e-2
beta = .5


def digamma(x):
    return torch.where(x > .6, torch.log(x - .5), -1 / x - euler_gamma)


def inverse_digamma(x):
    # noinspection PyTypeChecker
    return torch.where(x > -2.22, torch.exp(x) + .5, -1 / (x + euler_gamma))


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    args = arg_parser.parse_args()
    config_file = args.config_file
    test_loader = cfg2test_loader(config_file, args)
    model = cfg2model(config_file, args)
    test_set = test_loader.dataset

    indices = (test_set.concept_split_specs <= 0).nonzero(as_tuple=False).squeeze(1)
    concepts = model.box_registry[indices]
    dim = model.box_registry.mid_channels
    start = concepts[:, :dim] - concepts[:, dim:]
    end = concepts[:, :dim] + concepts[:, dim:]

    x = (start / beta / 2 + .5).flatten()
    y = (.5 - end / beta / 2).flatten()
    z = 1 - x - y
    samples = torch.stack([x, y, z])
    log_pk = torch.log(samples).mean(1)
    # noinspection PyCallingNonCallable
    alphas = torch.tensor([9, 9, 5]).to("cuda")
    old_alphas = alphas
    for i in range(1000):
        alphas = inverse_digamma(digamma(alphas.sum()) + log_pk)
        if (alphas - old_alphas).abs().sum() < .001:
            break
        else:
            old_alphas = alphas
    dirichlet = Dirichlet(alphas)
    ll = dirichlet.log_prob(samples.permute(1, 0)).sum()
    print(beta, alphas, ll)
