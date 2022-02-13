import itertools

import torch
from torch import nn
from torch.nn import functional as F

from models.nn import build_entailment
from utils import underscores, freeze


class FewshotProgramExecutor(nn.Module):
    NETWORK_REGISTRY = {}

    def __init__(self, cfg):
        super().__init__()
        network = self.NETWORK_REGISTRY[cfg.NAME](cfg)
        entailment = build_entailment(cfg)
        self.learner = PipelineLearner(network, entailment)

    def forward(self, q):
        return q(self.learner)


class MetaLearner(nn.Module):

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        name = MetaLearner.get_name(cls.__name__)
        FewshotProgramExecutor.NETWORK_REGISTRY[name] = cls
        cls.name = name

    @staticmethod
    def get_name(name):
        return underscores(name[:-len('Learner')])

    def forward(self, p):
        return {}

    def compute_logits(self, p, **kwargs):
        return p.evaluate_logits(self, **kwargs)


class PipelineLearner(nn.Module):
    def __init__(self, network, entailment):
        super().__init__()
        self.network = network
        self.entailment = entailment

    def forward(self, p):
        shots = []
        for q in p.train_program:
            end = q(self)["end"]
            index = end.squeeze(0).max(0).indices
            shots.append(q.object_collections[index])
        shots = torch.stack(shots)
        if not p.is_fewshot:
            shots = shots[0:0]
        fewshot = p.to_fewshot(shots)
        return fewshot(self)

    def compute_logits(self, p, **kwargs):
        return self.network.compute_logits(p, **kwargs)


class GenerativeLearner(nn.Module):
    def __init__(self, network, entailment):
        super().__init__()
        self.network = network
        self.entailment = entailment

    def nll(self, category, logit, target):
        if category == "boolean":
            return F.binary_cross_entropy_with_logits(logit, target)
        elif category == "count":
            return F.mse_loss(logit, target)
        else:
            raise NotImplementedError

    def forward(self, p, ):
        assert p.is_fewshot
        outputs = []
        for chain in itertools.product(*(enumerate(f) for f in p.train_features)):
            chain, shots = zip(*chain)
            shots = torch.stack(shots)
            output = self.network(p.to_fewshot(shots))
            logits = []
            for f in p.train_features:
                q = p.train_program.evaluate_token(output["queried_embedding"])
                q.object_collections = f
                logits.append(q(self)["logit"])
            logits = torch.cat(logits)
            train_nll = self.nll(p.train_category, logits, p.train_target)
            train_log_prob = -F.logsigmoid(
                self.entailment(shots, output["queried_embedding"].unsqueeze(0))).sum()
            output.update({"train_logit": logits, "train_log_prob": train_log_prob + train_nll, "chain": chain})
            outputs.append(output)
        train_log_prob = torch.stack([output["train_log_prob"] for output in outputs])
        index = train_log_prob.min(0).indices

        return {"shot_object": outputs[index]["chain"], **outputs[index]}
