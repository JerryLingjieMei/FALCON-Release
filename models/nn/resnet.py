from torch import nn
from torchvision.models.resnet import resnet34


class LinearBlock(nn.Module):

    @staticmethod
    def from_basic(other):
        this = LinearBlock()
        for name in ["conv1", "bn1", "relu", "conv2", "bn2", "downsample", "stride"]:
            setattr(this, name, getattr(other, name))
        return this

    def forward(self, x):
        identity = x
        out = self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x)))))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out


def make_resnet_layers(pretrained=True, has_relu=True):
    resnet = resnet34(pretrained=pretrained)
    resnet_layers = list(resnet.children())
    resnet_layers.pop()
    if not has_relu:
        resnet_layers[-2][-1] = LinearBlock.from_basic(resnet_layers[-2][-1])
    return resnet_layers
