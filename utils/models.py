import torch
import torch.nn as nn
import efficientnet_pytorch as efficientnet_model
from torchvision.models.efficientnet import EfficientNet as EfficientNetPytorch

from utils.layer import AdaptiveConcatPool2d, GeM, Flatten, SEBlock, SelfAttentionLayer


class EfficientNet(nn.Module):
    pre_trained_model = {
        'efficientnet-b0': 'pre-trained-models/efficientnet-b0-08094119.pth'
    }
    def __init__(
        self,
        backbone,
        output_dimensions,
        pre_trained_model=None
    ):
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


class EfficientNetApi(nn.Module):
    def __init__(
        self,
        model: EfficientNetPytorch,
        output_dimensions: int
    ):
        super(EfficientNetApi, self).__init__()
        self.model = model

        for param in list(self.model.parameters())[:-100]:
            param.requires_grad = False

        self.fully_connected = nn.Linear(self.model.classifier[1].in_features, output_dimensions)
        self.dropout = nn.Dropout(0.3)
        self.model.classifier[1] = nn.Identity()

    def extract(self, inputs):
        return self.model(inputs)

    def forward(self, inputs):
        x = self.extract(inputs)
        x = self.dropout(x)
        x = self.fully_connected(x)
        return x


class EfficientNetApiGem(nn.Module):
    def __init__(
        self,
        model: EfficientNetPytorch,
        output_dimensions: int,
        pool_type: str = "avg",
        dropout_rate: float = 0.3,
        use_se_block: bool = True,
        se_reduction: int = 8,
        use_self_attention: bool = False
    ):
        super(EfficientNetApiGem, self).__init__()

        self.model = model
        self.pool_type = pool_type
        self.use_se_block = use_se_block
        self.use_self_attention = use_self_attention

        # Freeze early layers (keep last 100 parameters trainable)
        for param in list(self.model.parameters())[:-100]:
            param.requires_grad = False

        original_features = self.model.classifier[1].in_features

        # Replace the avgpool layer with custom pooling
        if pool_type == "concat":
            self.model.avgpool = AdaptiveConcatPool2d()
            final_features = original_features * 2
        elif pool_type == "gem":
            self.model.avgpool = GeM()
            final_features = original_features
        else:  # avg pooling (default)
            # Keep the original avgpool
            final_features = original_features

            # 1. Adicionar o bloco de self-attention após as features
            if self.use_self_attention:
                # Assumindo que a saída das features é 2D (batch, channels, H, W)
                # Para a atenção, precisamos de uma sequência. Vamos achatar H e W
                self.self_attention = SelfAttentionLayer(in_features=original_features)
                # A saída da atenção terá a mesma dimensão de entrada
                final_features = original_features


        classifier_layers = [Flatten()]

        if self.use_se_block:
            classifier_layers.append(SEBlock(final_features, r=se_reduction))

        classifier_layers.extend([
            nn.Dropout(dropout_rate),
            nn.Linear(final_features, output_dimensions)
        ])

        # Replace the original classifier
        self.model.classifier = nn.Sequential(*classifier_layers)

    def extract(self, inputs):
        x = self.model.features(inputs)

        if self.use_self_attention:
            x = x.permute(0, 2, 3, 1)
            x = x.reshape(x.size(0), -1, x.size(-1))
            x = self.self_attention(x)
            x = x.mean(dim=1)
        else:
            x = self.model.avgpool(x)

        return x

    def forward(self, inputs):
        x = self.extract(inputs)
        x = self.model.classifier(x)
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


class FixedScheduler:
    def __init__(self, lr):
        self.init_lr = lr
    def step(self):
        pass

    def get_last_lr(self):
        return [self.init_lr]