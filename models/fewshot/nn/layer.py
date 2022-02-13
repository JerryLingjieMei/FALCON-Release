import torch
from torch import nn as nn

from models.nn.mlp import MLP
from utils.tensor import unbind, bind


class MessagePassing(nn.Module):
    def __init__(self, dimension, in_channels, mid_channels, out_channels, n_edge_types):
        super().__init__()
        self.dim = dimension
        self.width = in_channels // 2
        self.out_channels = out_channels
        self.mlp = MLP(in_channels, mid_channels, out_channels * n_edge_types, bias=False)

    def forward(self, from_embeddings, to_embeddings, edge_types):
        """
        :param from_embeddings: Ex2D
        :param to_embeddings: Ex2D
        :param edge_types: E
        :return Ex2D
        """
        # flattened: ExDx4
        flattened = torch.cat([unbind(from_embeddings, self.width), unbind(to_embeddings, self.width)], -1)
        # x: ExDx(OxK)
        x = self.mlp(flattened)
        # indices: ExDxO
        indices = edge_types[:, None, None].expand(-1, self.dim, -1) * self.out_channels + torch.arange(
            self.out_channels, device=edge_types.device).unsqueeze(0).unsqueeze(0)
        # x: ExDxO
        x = x.gather(-1, indices)
        return x


class Updater(nn.Module):
    def __init__(self, dimension, in_channels, mid_channels, out_channels):
        super().__init__()
        self.dim = dimension
        self.width = out_channels // 2
        self.mlp = MLP(in_channels, mid_channels, out_channels, bias=False)

    def forward(self, embeddings, aggregated, coefficient):
        """
        :param coefficient: 0
        :param embeddings: Fx2D
        :param aggregated: FxDxO
        :return: Ex2D
        """
        # flattened FxDx2
        flattened = unbind(embeddings, self.width)
        # x FxDx(O+2)
        x = torch.cat([flattened, aggregated], -1)
        # x FxDx4
        # displacement FxDx2
        gate, value = self.mlp(x).chunk(2, -1)
        displacement = torch.sigmoid(gate) * value * coefficient
        # x Fx2D
        x = bind(flattened + displacement)
        return x


class PointMessagePassing(nn.Module):
    def __init__(self, dimension, in_channels, mid_channels, out_channels, n_edge_types):
        super().__init__()
        self.dim = dimension
        self.out_channels = out_channels
        self.mlp = MLP(in_channels, mid_channels, out_channels * n_edge_types)

    def forward(self, from_embeddings, to_embeddings, edge_types):
        """
        :param from_embeddings: ExD
        :param to_embeddings: ExD
        :param edge_types: E
        :return: ExO
        """
        # flattened_embeddings: Ex2D
        flattened_embeddings = torch.cat([from_embeddings, to_embeddings], dim=-1)
        # x: Ex(OxK)
        x = self.mlp(flattened_embeddings)
        # indices: ExO
        indices = edge_types.unsqueeze(1) * self.out_channels + torch.arange(self.out_channels,
            device=edge_types.device).unsqueeze(0)
        # x: ExO
        x = x.gather(-1, indices)
        return x


class PointUpdater(nn.Module):
    def __init__(self, dimension, in_channels, mid_channels, out_channels):
        super().__init__()
        self.dim = dimension
        self.mlp = MLP(in_channels, mid_channels, out_channels)

    def forward(self, embeddings, aggregated, coefficient):
        """
        :param coefficient:
        :param embeddings: FxD
        :param aggregated: FxO
        :return: ExD
        """
        # x Fx(D+O)
        x = torch.cat([embeddings, aggregated], -1)
        # x Fx2D
        x = self.mlp(x)
        # displacement FxD
        displacement = torch.sigmoid(x[..., :self.dim]) * x[..., self.dim:] * coefficient
        # x FxD
        x = embeddings + displacement
        # x FxD
        return x
