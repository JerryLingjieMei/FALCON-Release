import torch
import torch.nn as nn

from models.fewshot.end2end.nn.bottom_up_nn import SimpleClassifier, FCNet, QuestionEmbedding, Attention
from models.fewshot.program_executor import MetaLearner


class BottomUpLearner(MetaLearner):

    def __init__(self, cfg):
        super().__init__()
        language_cfg = cfg.LANGUAGE
        self.dim = cfg.DIMENSION
        word_channels = language_cfg.WORD_CHANNELS
        self.w_emb = nn.Embedding(language_cfg.WORD_ENTRIES, word_channels)
        self.q_emb = QuestionEmbedding(word_channels, self.dim, 1, False, 0.0)
        self.meta_emb = QuestionEmbedding(word_channels, self.dim, 1, False, 0.0)
        self.num_objects = language_cfg.NUM_OBJECTS

        self.train_att = Attention(word_channels, self.dim, self.dim)
        self.val_att = Attention(word_channels, self.dim * 2 + cfg.DIMENSION, self.dim)

        self.q_net = FCNet([self.dim * 3, self.dim])
        self.v_net = FCNet([cfg.DIMENSION, self.dim])

        self.classifier = SimpleClassifier(self.dim, self.dim * 2, language_cfg.WORD_ENTRIES, 0.5)

    def compute_logits(self, p, **kwargs):
        device = p.val_tokenized[0].device
        if p.is_fewshot:
            v = torch.zeros(1, self.num_objects, self.dim).to(device)
            objects = p.train_program[0].object_collections
            v[:, :len(objects)] = objects
            train_emb = self.q_emb(self.w_emb(p.train_tokenized))
            train_emb = (self.train_att(v, train_emb) * v).sum(1).squeeze(0)
        else:
            train_emb = torch.zeros(self.dim).to(device)

        if p.is_attached:
            meta_emb = self.meta_emb(
                self.w_emb(p.metaconcept_tokenized[None, :p.metaconcept_length])).squeeze(0)
        else:
            meta_emb = torch.zeros(self.dim).to(device)

        v = torch.zeros(len(p.val_program), self.num_objects, self.dim).to(device)
        for i, q in enumerate(p.val_program):
            v[i, :len(q.object_collections)] = q.object_collections
        val_emb = self.q_emb(self.w_emb(p.val_tokenized))

        qe = torch.cat(
            [train_emb.unsqueeze(0).expand(len(v), -1), meta_emb.unsqueeze(0).expand(len(v), -1), val_emb], -1)
        v_emb = (self.val_att(v, qe) * v).sum(1)
        joint_repr = self.q_net(qe) * self.v_net(v_emb)

        train_end = torch.ones_like(p.train_target)
        val_end = self.classifier(joint_repr)
        return {**kwargs, "train_sample": {"end": train_end, "query_object": [0] * len(p.train_program)},
            "val_sample": {"end": val_end, "query_object": [0] * len(p.val_program)},
            "queried_embedding": torch.zeros(self.dim).to(p.train_target.device)}
