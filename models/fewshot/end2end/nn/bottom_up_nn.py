import numpy as np
import torch
from torch import nn as nn
from torch.nn.utils import weight_norm


class NewAttention(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid, dropout=0.2):
        super(NewAttention, self).__init__()

        self.v_proj = FCNet([v_dim, num_hid])
        self.q_proj = FCNet([q_dim, num_hid])
        self.dropout = nn.Dropout(dropout)
        self.linear = weight_norm(nn.Linear(num_hid, 1), dim=None)

    def forward(self, v, q):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        logits = self.logits(v, q)
        w = nn.functional.softmax(logits, 1)
        return w

    def logits(self, v, q):
        batch, k, _ = v.size()
        v_proj = self.v_proj(v)  # [batch, k, qdim]
        q_proj = self.q_proj(q).unsqueeze(1).expand(-1, k, -1)
        joint_repr = v_proj * q_proj
        joint_repr = self.dropout(joint_repr)
        logits = self.linear(joint_repr)
        return logits


class Attention(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid):
        super(Attention, self).__init__()
        self.nonlinear = FCNet([v_dim + q_dim, num_hid])
        self.linear = weight_norm(nn.Linear(num_hid, 1), dim=None)

    def forward(self, v, q):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        logits = self.logits(v, q)
        w = nn.functional.softmax(logits, 1)
        return w

    def logits(self, v, q):
        num_objs = v.size(1)
        q = q.unsqueeze(1).expand(-1, num_objs, -1)
        vq = torch.cat((v, q), 2)
        joint_repr = self.nonlinear(vq)
        logits = self.linear(joint_repr)
        return logits


class SimpleClassifier(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout):
        super(SimpleClassifier, self).__init__()
        layers = [weight_norm(nn.Linear(in_dim, hid_dim), dim=None), nn.ReLU(),
            nn.Dropout(dropout, inplace=True), weight_norm(nn.Linear(hid_dim, out_dim), dim=None)]
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.main(x)
        return logits


class FCNet(nn.Module):
    """Simple class for non-linear fully connect network
    """

    def __init__(self, dims):
        super(FCNet, self).__init__()

        layers = []
        for i in range(len(dims) - 2):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
            layers.append(nn.ReLU())
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))
        layers.append(nn.ReLU())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class WordEmbedding(nn.Module):
    """Word Embedding

    The ntoken-th dim is used for padding_idx, which agrees *implicitly*
    with the definition in Dictionary.
    """

    def __init__(self, n_token, emb_dim, dropout):
        super(WordEmbedding, self).__init__()
        self.emb = nn.Embedding(n_token + 1, emb_dim, padding_idx=n_token)
        self.dropout = nn.Dropout(dropout)
        self.n_token = n_token
        self.emb_dim = emb_dim

    def init_embedding(self, np_file):
        weight_init = torch.from_numpy(np.load(np_file))
        assert weight_init.shape == (self.n_token, self.emb_dim)
        self.emb.weight.data[:self.n_token] = weight_init

    def forward(self, x):
        emb = self.emb(x)
        emb = self.dropout(emb)
        return emb


class QuestionEmbedding(nn.Module):
    def __init__(self, in_dim, num_hid, n_layers, bidirectional, dropout):
        """Module for question embedding
        """
        super(QuestionEmbedding, self).__init__()
        self.rnn = nn.LSTM(in_dim, num_hid, n_layers, bidirectional=bidirectional, dropout=dropout, batch_first=True)
        self.num_hid = num_hid
        self.n_directions = 1 + int(bidirectional)

    def forward(self, x):
        # x: [batch, sequence, in_dim]
        self.rnn.flatten_parameters()
        output, _ = self.rnn(x)

        if self.n_directions == 1:
            return output[:, -1]

        forward_ = output[:, -1, :self.num_hid]
        backward = output[:, 0, self.num_hid:]
        return torch.cat((forward_, backward), dim=1)
