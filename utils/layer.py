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
        batch_size = input.values.size(0)
        return input.values.view(batch_size, -1)


class MixUp:
    """MixUp data augmentation"""
    def __init__(self, alpha=0.2):
        self.alpha = alpha

    def __call__(self, data, targets):
        if self.alpha > 0:
            lam = torch.distributions.Beta(self.alpha, self.alpha).sample()
        else:
            lam = 1

        batch_size = data.size(0)
        index = torch.randperm(batch_size).to(data.device)

        mixed_data = lam * data + (1 - lam) * data[index, :]
        targets_a, targets_b = targets, targets[index]

        return mixed_data, targets_a, targets_b, lam


class CutMix:
    """CutMix data augmentation"""
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def __call__(self, data, targets):
        if self.alpha > 0:
            lam = torch.distributions.Beta(self.alpha, self.alpha).sample()
        else:
            lam = 1

        batch_size = data.size(0)
        index = torch.randperm(batch_size).to(data.device)

        _, _, h, w = data.size()
        cut_rat = torch.sqrt(1. - lam)
        cut_w = (w * cut_rat).long()
        cut_h = (h * cut_rat).long()

        # uniform
        cx = torch.randint(w, (1,)).item()
        cy = torch.randint(h, (1,)).item()

        bbx1 = torch.clamp(cx - cut_w // 2, 0, w)
        bby1 = torch.clamp(cy - cut_h // 2, 0, h)
        bbx2 = torch.clamp(cx + cut_w // 2, 0, w)
        bby2 = torch.clamp(cy + cut_h // 2, 0, h)

        data[:, :, bby1:bby2, bbx1:bbx2] = data[index, :, bby1:bby2, bbx1:bbx2]

        # adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))
        targets_a, targets_b = targets, targets[index]

        return data, targets_a, targets_b, lam



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
