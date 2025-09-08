import torch
import torch.nn as nn
import efficientnet_pytorch as efficientnet_model

class EfficientNet(nn.Module):
    pre_trained_model = {
        'efficientnet-b0': 'pre-trained-models/efficientnet-b0-08094119.pth'
    }
    def __init__(self, backbone, output_dimensions, pre_trained_model=None):
        super(EfficientNet, self).__init__()
        if pre_trained_model is not None:
            self.pre_trained_model = pre_trained_model

        self.efficient_net = efficientnet_model.EfficientNet.from_pretrained(backbone)
        self.efficient_net.load_state_dict(
            torch.load(self.pre_trained_model.get(backbone), weights_only=True)
        )
        self.fully_connected = nn.Linear(self.efficient_net._fc.in_features, output_dimensions)
        self.efficient_net._fc = nn.Identity()

    def extract(self, inputs):
        return self.efficient_net(inputs)

    def forward(self, inputs):
        x = self.extract(inputs)
        x = self.fully_connected(x)
        return x


class EfficientNetMultiColor(nn.Module):
    pre_trained_model = {
        'efficientnet-b0': 'pre-trained-models/efficientnet-b0-08094119.pth'
    }
    def __init__(self, backbone, output_dimensions, pre_trained_model=None):
        super(EfficientNetMultiColor, self).__init__()
        if pre_trained_model is not None:
            self.pre_trained_model = pre_trained_model

        self.efficient_net = efficientnet_model.EfficientNet.from_pretrained(backbone)
        self.efficient_net.load_state_dict(
            torch.load(self.pre_trained_model.get(backbone), weights_only=True)
        )

        old_conv = self.efficient_net._conv_stem  # conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)

        self.efficient_net._conv_stem = nn.Conv2d(
            in_channels=18,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None
        )
        nn.init.kaiming_normal_(self.efficient_net._conv_stem.weight, mode='fan_out', nonlinearity='relu')
        self.fully_connected = nn.Linear(self.efficient_net._fc.in_features, output_dimensions)
        self.efficient_net._fc = nn.Identity()

    def extract(self, inputs):
        return self.efficient_net(inputs)

    def forward(self, inputs):
        x = self.extract(inputs)
        x = self.fully_connected(x)
        return x