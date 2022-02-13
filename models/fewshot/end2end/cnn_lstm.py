import torch
from torch import nn
from models.fewshot.program_executor import MetaLearner
from models.nn import MLP
from utils import invert


class CnnLstmLearner(MetaLearner):

    def __init__(self, cfg):
        super().__init__()
        language_cfg = cfg.LANGUAGE
        hidden_channels = language_cfg.HIDDEN_CHANNELS
        pad_channels = language_cfg.PAD_CHANNELS
        n_layers = language_cfg.N_LAYERS
        self.embedding = nn.Embedding(language_cfg.WORD_ENTRIES, hidden_channels)
        self.segment = nn.Embedding(language_cfg.PAD_ENTRIES, pad_channels)
        self.dim = cfg.DIMENSION
        self.lstm = nn.LSTM(hidden_channels + pad_channels, hidden_channels, n_layers, batch_first=True)
        self.mlp = MLP(cfg.DIMENSION * 2 + hidden_channels, hidden_channels, language_cfg.WORD_ENTRIES)

    def compute_logits(self, p, **kwargs):
        embedded = torch.cat([self.embedding(p.composite_tokenized), self.segment(p.composite_segment)], -1)
        lengths_sorted, index_sorted = torch.tensor(p.composite_length).sort(-1, descending=True)
        embedded = nn.utils.rnn.pack_padded_sequence(embedded[index_sorted], lengths_sorted.cpu(),
            batch_first=True)
        encoded, _ = self.lstm(embedded)
        encoded, length = nn.utils.rnn.pad_packed_sequence(encoded, batch_first=True)
        encoded = encoded[torch.arange(len(length)), length - 1]
        encoded = encoded[invert(index_sorted)]

        if p.is_fewshot:
            train_features = p.train_program[0].object_collections.expand(len(p.val_program), -1)
        else:
            train_features = torch.zeros(len(p.val_program), self.dim).to(p.val_tokenized[0].device)
        val_features = torch.cat([q.object_collections for q in p.val_program])

        train_end = torch.ones(len(p.train_program)).to(p.train_target.device)
        val_end = self.mlp(torch.cat([encoded, train_features, val_features], -1))

        return {**kwargs, "train_sample": {"end": train_end, "query_object": [0] * len(p.train_program)},
            "val_sample": {"end": val_end, "query_object": [0] * len(p.val_program)},
            "queried_embedding": torch.zeros(self.dim).to(p.train_target.device)}
