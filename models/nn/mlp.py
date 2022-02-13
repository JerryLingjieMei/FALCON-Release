from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, bias=True, bn=False, dropout=False):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, mid_channels, bias=bias)
        self.linear_2 = nn.Linear(mid_channels, out_channels, bias=bias)
        mid = nn.BatchNorm1d(mid_channels) if bn else nn.Dropout() if dropout else nn.Identity()
        self.mid = mid

    def forward(self, x):
        x = F.leaky_relu(self.mid(self.linear_1(x)), inplace=True)
        x = self.linear_2(x)
        return x
