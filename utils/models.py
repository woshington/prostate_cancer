import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
import efficientnet_pytorch as efficientnet_model
from torchvision.models.efficientnet import EfficientNet as EfficientNetPytorch
from typing import List, Literal, Optional
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
        output_dimensions: int,
        dropout_rate=0.3
    ):
        super(EfficientNetApi, self).__init__()
        self.model = model
        self.dropout_rate = dropout_rate

        for param in list(self.model.parameters())[:-150]:
            param.requires_grad = False

        self.fully_connected = nn.Linear(self.model.classifier[1].in_features, output_dimensions)
        self.dropout = nn.Dropout(self.dropout_rate)
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

        # Freeze early layers (keep last 150 parameters trainable for better fine-tuning)
        for param in list(self.model.parameters())[:-150]:
            param.requires_grad = False

        original_features = self.model.classifier[1].in_features

        if pool_type == "concat":
            self.model.avgpool = AdaptiveConcatPool2d()
            final_features = original_features * 2
        elif pool_type == "gem":
            self.model.avgpool = GeM()
            final_features = original_features
        else:
            final_features = original_features

            if self.use_self_attention:
                self.self_attention = SelfAttentionLayer(in_features=original_features)
                final_features = original_features


        classifier_layers = []

        if self.use_se_block:
            classifier_layers.append(SEBlock(final_features, r=se_reduction))

        classifier_layers.append(nn.BatchNorm1d(final_features))
        classifier_layers.extend([
            nn.Dropout(dropout_rate),
            nn.Linear(final_features, output_dimensions)
        ])

        # Replace the original classifier
        self.model.classifier = nn.Sequential(
            Flatten(),
            *classifier_layers
        )

    def extract(self, inputs):
        x = self.model.features(inputs)

        if self.use_self_attention:
            x = x.permute(0, 2, 3, 1)
            x = x.reshape(x.size(0), -1, x.size(-1))
            x = self.self_attention(x)
            x = x.max(dim=1)
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


class EnsembleEfficientNet(nn.Module):
    def __init__(
            self,
            models: List[nn.Module],
            method: Literal['max', 'mean', 'weighted_mean', 'majority_vote', 'weighted_vote'] = 'mean',
            weights: Optional[List[float]] = None,
            temperature: float = 1.0
    ):
        """
        Ensemble de modelos com diferentes métodos de agregação.

        Args:
            models: Lista de modelos treinados
            method: Método de agregação
                - 'max': Máximo das probabilidades
                - 'mean': Média simples das probabilidades
                - 'weighted_mean': Média ponderada das probabilidades
                - 'majority_vote': Voto majoritário (hard voting)
                - 'weighted_vote': Voto ponderado (soft voting com pesos)
            weights: Pesos para cada modelo (deve somar 1.0 se fornecido)
            temperature: Temperatura para suavização das probabilidades
        """
        super().__init__()

        self.models = nn.ModuleList(models)
        self.method = method
        self.temperature = temperature

        # Validar e normalizar pesos
        if weights is not None:
            if len(weights) != len(models):
                raise ValueError(
                    f"Número de pesos ({len(weights)}) deve ser igual ao número de modelos ({len(models)})")

            weights = np.array(weights)
            if not np.isclose(weights.sum(), 1.0):
                print(f"⚠️ Normalizando pesos (soma atual: {weights.sum():.4f})")
                weights = weights / weights.sum()

            self.weights = torch.tensor(weights, dtype=torch.float32)
        else:
            # Pesos uniformes
            self.weights = torch.ones(len(models)) / len(models)

    def forward(self, x):
        """
        Forward pass pelo ensemble.

        Returns:
            torch.Tensor: Logits ou probabilidades agregadas
        """
        # Coletar outputs de todos os modelos
        outputs = []
        for model in self.models:
            model.eval()  # Garantir modo eval
            with torch.no_grad():
                out = model(x)
            outputs.append(out)

        outputs = torch.stack(outputs)  # Shape: (n_models, batch_size, n_classes)

        # Mover pesos para o device correto
        weights = self.weights.to(outputs.device).view(-1, 1, 1)

        # Aplicar método de agregação
        if self.method == 'max':
            result, _ = torch.max(outputs, dim=0)

        elif self.method == 'mean':
            result = torch.mean(outputs, dim=0)

        elif self.method == 'weighted_mean':
            result = torch.sum(outputs * weights, dim=0)

        elif self.method == 'majority_vote':
            # Hard voting: cada modelo vota na classe com maior probabilidade
            probs = F.softmax(outputs / self.temperature, dim=-1)
            votes = torch.argmax(probs, dim=-1)  # Shape: (n_models, batch_size)

            # Contar votos para cada classe
            batch_size = votes.shape[1]
            n_classes = outputs.shape[-1]
            vote_counts = torch.zeros(batch_size, n_classes, device=outputs.device)

            for i in range(votes.shape[0]):
                vote_counts.scatter_add_(1, votes[i].unsqueeze(1),
                                         torch.ones_like(votes[i].unsqueeze(1), dtype=torch.float32))

            result = vote_counts  # Retorna contagem de votos (pode usar argmax depois)

        elif self.method == 'weighted_vote':
            # Soft voting ponderado
            probs = F.softmax(outputs / self.temperature, dim=-1)
            result = torch.sum(probs * weights, dim=0)
            # Converter de volta para logits se necessário
            result = torch.log(result + 1e-8)

        else:
            raise ValueError(f"Método desconhecido: {self.method}")

        return result

    def predict_proba(self, x):
        """Retorna probabilidades em vez de logits."""
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)

    def predict(self, x):
        """Retorna as classes preditas."""
        logits = self.forward(x)
        return torch.argmax(logits, dim=-1)

    def set_method(self, method: str):
        """Altera o método de agregação dinamicamente."""
        valid_methods = ['max', 'mean', 'weighted_mean', 'majority_vote', 'weighted_vote']
        if method not in valid_methods:
            raise ValueError(f"Método deve ser um de: {valid_methods}")
        self.method = method

    def set_weights(self, weights: List[float]):
        """Atualiza os pesos dos modelos."""
        weights = np.array(weights)
        if len(weights) != len(self.models):
            raise ValueError(f"Número de pesos deve ser igual ao número de modelos")

        if not np.isclose(weights.sum(), 1.0):
            weights = weights / weights.sum()

        self.weights = torch.tensor(weights, dtype=torch.float32)