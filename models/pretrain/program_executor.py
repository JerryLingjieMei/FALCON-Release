import torch
from torch import nn

from models.nn import build_entailment


class PretrainProgramExecutor(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.entailment = build_entailment(cfg)

    def forward(self, q):
        return q(self)

    def clean_output(self, outputs, inputs):
        queried_embeddings = []
        for queried_embedding, category, target in zip(outputs["queried_embedding"], inputs["category"],
                inputs["target"]):
            if category == "boolean" or category == "count":
                queried_embedding = queried_embedding[target.max(0).indices]
            elif category == "choice":
                queried_embedding = queried_embedding[target[0]]
            else:
                raise NotImplementedError
            queried_embeddings.append(queried_embedding)
        outputs["queried_embedding"] = torch.stack(queried_embeddings)
        outputs["feature"] = torch.stack([f[0] for f in outputs["feature"]])
