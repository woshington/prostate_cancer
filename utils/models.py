import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
import efficientnet_pytorch as efficientnet_model
from torchvision.models.efficientnet import EfficientNet as EfficientNetPytorch
from torchvision.models.convnext import ConvNeXt as ConvNeXtPytorch
from typing import List, Literal, Optional
from utils.layer import AdaptiveConcatPool2d, GeM, Flatten, SEBlock, SelfAttentionLayer, DeformableConv2d


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


class EfficientNetApiA(nn.Module):
    def __init__(
        self,
        model: EfficientNetPytorch,
        output_dimensions: int,
        fine_tune=100,
        dropout_rate=0.4
    ):
        super(EfficientNetApiA, self).__init__()
        self.model = model
        self.dropout_rate = dropout_rate

        for param in list(self.model.parameters())[:-fine_tune]:
            param.requires_grad = False

        in_features = self.model.classifier[1].in_features

        # Replace the final classification layer with Identity
        self.model.classifier[1] = nn.Identity()

        self.fully_connected = nn.Linear(in_features, output_dimensions)

    def extract(self, inputs):
        """Extract features from the model (before final FC layer)"""
        return self.model(inputs)

    def forward(self, inputs):
        """Complete forward pass including dropout and classification"""
        x = self.extract(inputs)
        x = self.fully_connected(x)
        return x

class EfficientNetApi(nn.Module):
    def __init__(
            self,
            model: EfficientNetPytorch,
            output_dimensions: int,
            dropout_rate=0.3,
            use_deformable=False
    ):
        super(EfficientNetApi, self).__init__()
        self.model = model
        self.dropout_rate = dropout_rate
        self.use_deformable = use_deformable


        for param in list(self.model.parameters())[:-150]:
            param.requires_grad = False

        # Get input features before replacing classifier
        in_features = self.model.classifier[1].in_features

        # Replace the final classification layer with Identity
        self.model.classifier[1] = nn.Identity()

        # Add deformable convolution if requested
        if use_deformable:
            # Para efficientnet_b0 do torchvision
            last_conv_channels = None

            # Iterar pelos módulos de features para encontrar a última Conv2d
            for module in reversed(list(self.model.features.modules())):
                if isinstance(module, nn.Conv2d):
                    last_conv_channels = module.out_channels
                    break

            if last_conv_channels is None:
                raise ValueError("Não foi possível encontrar camada convolucional no modelo")

            self.deformable_layer = DeformableConv2d(
                in_channels=last_conv_channels,
                out_channels=last_conv_channels,
                kernel_size=3,
                stride=1,
                padding=1
            )

            # Armazenar o forward original de features
            self.original_features_forward = self.model.features.forward

            # A função precisa aceitar self quando vinculada como método
            def new_features_forward(self_features, x):
                x = self.original_features_forward(x)
                x = self.deformable_layer(x)
                return x

            import types
            self.model.features.forward = types.MethodType(
                new_features_forward, self.model.features
            )

        # Add dropout and final fully connected layer
        self.dropout = nn.Dropout(self.dropout_rate)
        self.fully_connected = nn.Linear(in_features, output_dimensions)

    def extract(self, inputs):
        """Extract features from the model (before final FC layer)"""
        return self.model(inputs)

    def forward(self, inputs):
        """Complete forward pass including dropout and classification"""
        x = self.extract(inputs)
        x = self.dropout(x)
        x = self.fully_connected(x)
        return x


class ConvNeXtApi(nn.Module):
    def __init__(
            self,
            model: ConvNeXtPytorch,
            output_dimensions: int,
            dropout_rate: float = 0.3,
    ):
        super(ConvNeXtApi, self).__init__()
        self.model = model

        # Freeze the backbone features entirely to avoid splitting LayerNorm
        # weight/bias across the freeze boundary (which causes NativeLayerNormBackward errors)
        for param in self.model.features[-150:].parameters():
            param.requires_grad = False

        # ConvNeXt classifier: [LayerNorm, Flatten, Linear]
        in_features = self.model.classifier[2].in_features
        self.model.classifier[2] = nn.Identity()

        self.dropout = nn.Dropout(dropout_rate)
        self.fully_connected = nn.Linear(in_features, output_dimensions)

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


class OrdinalRegressionLoss(nn.Module):
    """
    Ordinal regression loss for ordered categories.

    For each sample with grade k, we create k binary labels:
    - ISUP 0: [0, 0, 0, 0, 0]
    - ISUP 1: [1, 0, 0, 0, 0]
    - ISUP 2: [1, 1, 0, 0, 0]
    - ISUP 3: [1, 1, 1, 0, 0]
    - ISUP 4: [1, 1, 1, 1, 0]
    - ISUP 5: [1, 1, 1, 1, 1]
    """

    def __init__(self):
        super(OrdinalRegressionLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        """
        Args:
            logits: (batch_size, num_classes) - raw outputs from model
            targets: (batch_size, num_classes) - ordinal encoded targets
        """
        return self.bce(logits, targets)


def encode_ordinal_labels(labels, num_classes=5):
    """
    Convert categorical labels to ordinal encoding.

    Args:
        labels: Tensor or array of ISUP grades (0-5)
        num_classes: Number of thresholds (5 for ISUP 0-5)

    Returns:
        Ordinal encoded labels: (batch_size, num_classes)
    """
    # Convert to numpy if tensor
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    batch_size = len(labels)
    ordinal = torch.zeros((batch_size, num_classes), dtype=torch.float32)

    for i, label in enumerate(labels):
        label = int(label)  # Convert to int
        if label > 0:
            ordinal[i, :label] = 1

    return ordinal


def decode_ordinal_predictions(logits):
    """
    Convert ordinal predictions back to class labels.

    Args:
        logits: (batch_size, num_classes) - raw model outputs

    Returns:
        Predicted ISUP grades (0-5)
    """
    # Apply sigmoid to get probabilities
    probs = torch.sigmoid(logits)

    # Sum probabilities > 0.5 to get predicted grade
    predictions = (probs > 0.5).sum(dim=1)

    return predictions



from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import torch.nn as nn
import torch.nn.functional as F
import torch

class AttentionMIL(nn.Module):
    def __init__(self, base_model, output_classes=5, dropout_rate=0.6):
        super(AttentionMIL, self).__init__()
        self.base_model = base_model
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])

        self.attention = nn.Sequential(
            nn.Linear(1280, 512),
            nn.Tanh(),
            nn.Linear(512, 1)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(1280, output_classes)
        )

    def forward(self, x):
        # x: (batch_size, num_instances, C, H, W)
        b, n, C, H, W = x.size()
        x = x.view(b*n, C, H, W)

        features = self.feature_extractor(x).squeeze()  # (b*n, 1280)
        features = features.view(b, n, -1)

        attn_weights = self.attention(features)  # (b, n, 1)
        attn_weights = F.softmax(attn_weights, dim=1)

        bag_representation = torch.sum(attn_weights * features, dim=1)

        logits = self.classifier(bag_representation)
        return logits
