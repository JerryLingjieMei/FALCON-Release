import torch
import torch.nn.functional as F
from torch import nn


class PretrainLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()

    def forward(self, outputs, inputs):
        ce_losses = []
        for i, category in enumerate(inputs["category"]):
            end = outputs["end"][i]
            target = inputs["target"][i]
            if category == "boolean":
                loss = F.binary_cross_entropy_with_logits(end.unsqueeze(0), target.unsqueeze(0))
            elif category == "choice":
                loss = F.nll_loss(end.log(), target)
            elif category == "count":
                loss = F.l1_loss(end.unsqueeze(0), target.unsqueeze(0))
            else:
                raise NotImplementedError
            ce_losses.append(loss)
        return {"pretrain_loss": (torch.stack(ce_losses).mean())}
