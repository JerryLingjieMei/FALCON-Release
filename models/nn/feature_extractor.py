import torch
from torch import nn

from utils import EPS, freeze
from .mlp import MLP
from .resnet import make_resnet_layers


class FeatureExtractor(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.dim = cfg.DIMENSION
        self.rep = cfg.REPRESENTATION
        feature_cfg = cfg.FEATURE_EXTRACTOR
        self.from_feature_dim = feature_cfg.FROM_FEATURE_DIM
        if self.from_feature_dim:
            self.linear = nn.Identity()
            self.relations = None
            self.backbone = MLP(self.from_feature_dim, self.dim, self.dim)  # self.backbone = nn.Linear(self.from_feature_dim, self.dim, bias=False)
        else:
            resnet_layers = make_resnet_layers(feature_cfg.IS_PRETRAINED)

            weight = resnet_layers[0].weight
            placeholder = torch.zeros(weight.shape[0], feature_cfg.IN_CHANNELS, *weight.shape[2:]).to(
                weight.device)
            placeholder[:, :3] = weight
            resnet_layers[0].weight.data = placeholder
            self.backbone = nn.Sequential(*resnet_layers)

            if cfg.REPRESENTATION != "box":
                self.linear = nn.Identity()
            else:
                self.linear = nn.Linear(512, self.dim)

            if not feature_cfg.HAS_RELATIONS:
                self.relations = None
            else:
                self.relations = MLP(2 * 512, feature_cfg.MID_CHANNELS, self.dim)

        self.reset_parameters()

    def reset_parameters(self):
        if self.rep == "box":
            with torch.no_grad():
                if self.from_feature_dim:
                    self.backbone.linear_2.weight /= 10  # self.backbone.linear_2.bias /= 10
                else:
                    self.linear.weight /= 10
                    self.linear.bias /= 10

    @property
    def out_dim(self):
        if self.rep == "box":
            return self.dim * 2,
        else:
            return self.dim,

    def forward_backbone(self, x):
        # x NxCxHxW
        x = self.backbone(x)
        return x.view(len(x), -1)

    def forward_feature_relation(self, x):
        n = len(x)
        relations = torch.zeros(n, n, *self.out_dim).to(x.device)
        feature = self.linear(x)
        if self.relations is not None:
            relations = self.relations(
                torch.cat([(x.unsqueeze(0).expand(n, -1, -1)), (x.unsqueeze(1).expand(-1, n, -1))], -1))
        if self.rep == "box":
            return torch.cat([feature, torch.ones_like(feature) * EPS], -1), torch.cat(
                [relations, torch.ones_like(relations) * EPS], -1)
        return feature, relations

    def forward(self, x):
        x = self.forward_backbone(x)
        x = self.forward_feature_relation(x)
        return x


class BatchedFeatureExtractor(FeatureExtractor):

    def forward(self, xs):
        xs = self.forward_backbone(torch.cat([_ for _ in xs])).split([len(_) for _ in xs])
        return zip(*(self.forward_feature_relation(x) for x in xs))


class CachedFeatureExtractor(FeatureExtractor):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.feature_buffers = nn.Module()
        self.relation_buffers = nn.Module()
        self.has_cache = cfg.FEATURE_EXTRACTOR.HAS_CACHE
        freeze(self)

    def train(self, mode=True):
        return self

    def get(self, key, x):
        name = f"{key:05d}"
        feature = getattr(self.feature_buffers, name, None)
        relation = getattr(self.relation_buffers, name, None)
        if feature is None or relation is None or self.from_feature_dim:
            feature, relation = super().forward(x)
        return feature, relation

    def set(self, key, value):
        if self.from_feature_dim > 0 or not self.has_cache: return
        name = f"{key:05d}"
        feature, relation = value
        if not hasattr(self.feature_buffers, name):
            self.feature_buffers.register_buffer(name, feature, persistent=False)
        if not hasattr(self.relation_buffers, name):
            self.relation_buffers.register_buffer(name, relation, persistent=False)
