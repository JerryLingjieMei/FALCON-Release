import torch
from torch import nn as nn

from .layer import MessagePassing, Updater, PointMessagePassing, PointUpdater


class GNNLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        dim = cfg.DIMENSION
        gnn_cfg = cfg.GNN
        if cfg.REPRESENTATION == "box":
            mid_channels = gnn_cfg.MID_CHANNELS
            out_channels = gnn_cfg.OUT_CHANNELS
            self.message_passing = MessagePassing(dim, 4, mid_channels, out_channels, gnn_cfg.N_EDGE_TYPES)
            self.updater = Updater(dim, out_channels + 2, mid_channels, 4)
        else:
            mid_channels = gnn_cfg.MID_CHANNELS_WEIGHT
            out_channels = gnn_cfg.OUT_CHANNELS_WEIGHT
            self.message_passing = PointMessagePassing(dim, 2 * dim, mid_channels, out_channels,
                gnn_cfg.N_EDGE_TYPES)
            self.updater = PointUpdater(dim, out_channels + dim, mid_channels, 2 * dim)

    def forward(self, embeddings, relations, coefficient):
        # Message Passing
        from_concepts = embeddings[:, :-1].contiguous().view(-1, embeddings.shape[-1])
        to_concepts = embeddings[:, -1:].expand(-1, len(relations), -1).contiguous().view(-1,
            embeddings.shape[-1])
        messages = self.message_passing(from_concepts, to_concepts, relations.repeat(embeddings.shape[0]))
        # Aggregating messages
        if len(messages) > 0:
            aggregated_messages = messages.view(len(embeddings), -1, *messages.shape[1:]).max(1).values
        else:
            aggregated_messages = torch.zeros(len(embeddings), *messages.shape[1:]).to(messages.device)
        updated_message = self.updater(embeddings[:, -1], aggregated_messages, coefficient)
        new_embeddings = embeddings.clone()
        new_embeddings[:, -1] = updated_message
        return new_embeddings


class ConceptGraphGNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        n_layers = cfg.GNN.N_LAYERS
        self.layers = nn.ModuleList([GNNLayer(cfg) for _ in range(n_layers)])
        assert n_layers > 0
        with torch.no_grad():
            if cfg.REPRESENTATION == "box":
                self.layers[-1].updater.mlp.linear_2.weight[3] += 1.

    def forward(self, embeddings, relations, coefficient=1.):
        x = embeddings
        for layer in self.layers:
            x = layer(x, relations, coefficient)
        return x


class ExampleGraphGNN(ConceptGraphGNN):

    def forward(self, embeddings, relations=None, coefficient=1.):
        if relations is None:
            relations = torch.zeros(len(embeddings[0]) - 1, device=embeddings.device, dtype=torch.long)
        return super().forward(embeddings, relations, coefficient)
