import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, sz=None):
        super().__init__()
        sz = sz or (1, 1)
        self.ap = nn.AdaptiveAvgPool2d(sz)
        self.mp = nn.AdaptiveMaxPool2d(sz)

    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1.0 / self.p)


class SEBlock(nn.Module):
    def __init__(self, in_ch, r=8):
        super(SEBlock, self).__init__()
        self.linear_1 = nn.Linear(in_ch, in_ch // r)
        self.linear_2 = nn.Linear(in_ch // r, in_ch)

    def forward(self, x):
        input_x = x
        x = F.relu(self.linear_1(x), inplace=True)
        x = self.linear_2(x)
        x = torch.sigmoid(x)
        x = input_x * x
        return x


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class SelfAttentionLayer(nn.Module):
    def __init__(self, in_features):
        super(SelfAttentionLayer, self).__init__()
        self.in_features = in_features

        # Linear layers for Q, K, V
        self.query = nn.Linear(in_features, in_features)
        self.key = nn.Linear(in_features, in_features)
        self.value = nn.Linear(in_features, in_features)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x shape: (batch_size, num_patches, in_features)

        # Project inputs to Q, K, V
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Calculate attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.in_features ** 0.5)

        # Apply softmax to get attention weights
        attention_weights = self.softmax(scores)

        # Multiply weights by V to get the output
        output = torch.matmul(attention_weights, V)

        return output
