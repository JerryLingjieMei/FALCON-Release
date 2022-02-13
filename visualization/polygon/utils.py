import torch
from math import pi


def regular_polygon_vertices(n_vertices=16):
    return torch.stack([torch.cos(torch.arange(0, n_vertices).float() / n_vertices * 2 * pi),
        torch.sin(torch.arange(0, n_vertices).float() / n_vertices * 2 * pi)], dim=0).T


def circle_projection(pca, center, offset, directions):
    center = torch.Tensor(pca.transform(torch.unsqueeze(center, 0))).squeeze()
    disentangled = torch.unsqueeze(offset, 1) * torch.Tensor(pca.components_).T
    radius = (disentangled @ directions.T).abs().sum(0).max(0).values
    return center, radius