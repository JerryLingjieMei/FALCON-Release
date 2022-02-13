import torch
import torch.nn.functional as F
from torch import nn

from utils import EPS


class FewshotLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tau = cfg.LOSS.TAU

    def forward(self, outputs, inputs):
        losses = []
        for i, categories in enumerate(inputs["val_sample"]["category"]):
            ends = outputs["val_sample"]["end"][i]
            targets = inputs["val_sample"]["target"][i]
            answers = inputs["val_sample"]["answer_tokenized"][i]
            ls = []
            for j, category in enumerate(categories):
                if category == "boolean":
                    l = F.binary_cross_entropy_with_logits(ends[j].unsqueeze(0),
                        targets[j].unsqueeze(0).unsqueeze(0))
                elif category == "choice":
                    l = F.nll_loss(ends[j].clamp(min=EPS).log(), targets[j].unsqueeze(0))
                elif category == "count":
                    l = F.l1_loss(ends[j].unsqueeze(0), targets[j].unsqueeze(0).unsqueeze(0))
                elif category == "token":
                    l = F.cross_entropy(ends[j].unsqueeze(0), answers[j].unsqueeze(0))
                else:
                    raise NotImplementedError
                ls.append(l)
            losses.append(torch.stack(ls).mean())
        return {"validation_loss": torch.stack(losses).mean()}
