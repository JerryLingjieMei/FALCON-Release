import re
from torch import nn


class PretrainReg(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.lam = 10

    def forward(self, named_parameters):
        linear_1 = named_parameters['feature_extractor.backbone.linear_1.weight']
        linear_2 = named_parameters['feature_extractor.backbone.linear_2.weight']
        return {"l1_reg": (linear_1.abs().mean() + linear_2.abs().mean()) * self.lam}
