import torch
from torch import nn

from models.fewshot.program_executor import MetaLearner
from models.nn import MLP, build_entailment


class NsclLstmLearner(MetaLearner):

    def __init__(self, cfg):
        super().__init__()
        language_cfg = cfg.LANGUAGE
        self.rep = cfg.REPRESENTATION
        dim = cfg.DIMENSION * 2 if cfg.REPRESENTATION == "box" else cfg.DIMENSION
        pad_channels = language_cfg.PAD_CHANNELS
        n_layers = language_cfg.N_LAYERS
        bidirectional = language_cfg.BIDIRECTIONAL
        hidden_channels = language_cfg.HIDDEN_CHANNELS
        self.segment_emb = nn.Embedding(cfg.GNN.N_EDGE_TYPES, pad_channels)
        self.concept_lstm = nn.LSTM(dim + pad_channels, hidden_channels, n_layers, bidirectional=bidirectional)
        self.feature_lstm = nn.LSTM(dim, hidden_channels, n_layers, bidirectional=bidirectional)
        self.size = hidden_channels * (2 if bidirectional else 1)
        self.linear = nn.Linear(self.size * 2, dim)
        self.entailment =build_entailment(cfg)
        self.reset_parameters()

    def reset_parameters(self):
        if self.rep == "box":
            with torch.no_grad():
                self.linear.bias[self.linear.bias.shape[0] // 2:] += 0.05

    def forward(self, p):
        if p.is_fewshot:
            feature_out, _ = self.feature_lstm(p.train_features.unsqueeze(1))
            feature_out = feature_out[-1, :]
        else:
            feature_out = torch.zeros(1, self.size).to(p.device)
        if p.is_attached:
            concept_out, _ = self.concept_lstm(
                torch.cat([p.support_embeddings, self.segment_emb(p.relations)], -1).unsqueeze(1))
            concept_out = concept_out[-1, :]
        else:
            concept_out = torch.zeros(1, self.size).to(p.device)
        queried_embedding = self.linear(torch.cat([feature_out, concept_out], -1)).squeeze(0)
        return {"queried_embedding": queried_embedding}
