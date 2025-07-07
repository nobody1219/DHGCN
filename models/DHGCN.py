import math
from torch.nn.parameter import Parameter
from torch import nn
import torch.nn.functional as F
import torch


class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv, self).__init__()

        self.weight = Parameter(torch.Tensor(in_ft, out_ft))
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, G: torch.Tensor):
        x = x.matmul(self.weight)
        if self.bias is not None:
            x = x + self.bias
        x = G.matmul(x)
        return x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, G):
        if G is not None:
            return self.fn(self.norm(x), G)
        return self.fn(self.norm(x))


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, G):
        return self.net(x)


class HGCB(nn.Module):
    def __init__(self, dim, depth, dropout=0.25):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, HGNN_conv(dim, dim)),
                PreNorm(dim, FeedForward(dim, dim, dropout=dropout))
            ]))

    def forward(self, x, G):
        for hgc, ff in self.layers:
            x = hgc(x, G) + x
            x = ff(x, G) + x
        return x


class DHGCN(nn.Module):
    def __init__(self, in_ch, n_class, time_point, dropout=0.5):
        super(DHGCN, self).__init__()
        self.dropout = dropout
        self.fc1 = nn.Linear(in_ch * 2, 16)
        self.fc2 = nn.Linear(in_ch * 2, 16)
        self.fc3 = nn.Linear(32, n_class)

        self.thgcn = HGCB(depth=2, dim=in_ch, dropout=dropout)
        self.shgcn = HGCB(depth=2, dim=time_point, dropout=dropout)
        self.avgpool1 = nn.AdaptiveAvgPool1d(1)
        self.maxpool1 = nn.AdaptiveMaxPool1d(1)
        self.avgpool2 = nn.AdaptiveAvgPool1d(1)
        self.maxpool2 = nn.AdaptiveMaxPool1d(1)

    def forward(self, x, G, G_sp):
        x2 = self.thgcn(x.transpose(1, 2), G).transpose(1, 2)
        x1 = self.shgcn(x, G_sp)

        x1 = torch.cat([self.avgpool1(x1), self.maxpool1(x1)], dim=-1)
        x2 = torch.cat([self.avgpool1(x2), self.maxpool1(x2)], dim=-1)

        x1 = self.fc1(x1.flatten(start_dim=1))
        x2 = self.fc2(x2.flatten(start_dim=1))

        x = torch.cat([x1, x2], dim=-1)
        x = F.gelu(x)
        x = self.fc3(x)
        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    block = DHGCN(64, 2, 128)
    input = torch.rand(64, 64, 128)  # B C L
    G = torch.rand(64, 128, 128)  # temporal hypergraph
    G_sp = torch.rand(64, 64, 64)  # spatial hypergraph
    output = block(input, G, G_sp)
    print(block)
    print(output.size())
    for name, param in block.named_parameters():
        print(name, param.numel())
    print(f"The model has {count_parameters(block):,} trainable parameters.")
