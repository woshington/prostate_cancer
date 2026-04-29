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



class EfficientNetMIL(nn.Module):
    def __init__(
        self,
        model: EfficientNetPytorch,
        output_classes: int,
        fine_tune=100,
        dropout_rate=0.4
    ):
        super(EfficientNetMIL, self).__init__()
        self.dropout_rate = dropout_rate

        # Congela as primeiras camadas
        for param in list(model.parameters())[:-fine_tune]:
            param.requires_grad = False

        in_features = model.classifier[1].in_features

        # Remove a cabeça de classificação
        model.classifier[1] = nn.Identity()
        self.feature_extractor = model

        # Attention Network
        self.attention = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.Tanh(),
            nn.Linear(512, 1)
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, output_classes)
        )

    def extract(self, x):
        """Extrai features de todos os patches"""
        return self.feature_extractor(x)

    def forward(self, x):
        b, n, C, H, W = x.size()

        # Processa todos os patches de uma vez
        x = x.view(b * n, C, H, W)
        features = self.extract(x)           # (b*n, in_features)
        features = features.view(b, n, -1)   # (b, n, in_features)

        # Atenção sobre os patches
        attn_weights = self.attention(features)              # (b, n, 1)
        attn_weights = F.softmax(attn_weights, dim=1)

        # Agregação ponderada
        bag = torch.sum(attn_weights * features, dim=1)     # (b, in_features)

        # Classificação
        logits = self.classifier(bag)                        # (b, output_classes)
        return logits
        
import torch
import torch.nn as nn


class ConvNeXtApi(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        output_dimensions: int,
        dropout_rate: float = 0.3,
        freeze_backbone: bool = True,
        unfreeze_last_blocks: int = 1,
    ):
        super().__init__()

        self.model = model

        # ========================
        # Freeze backbone (robusto)
        # ========================
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

            # Unfreeze últimos estágios (melhor que [: -150])
            if hasattr(self.model, "features"):
                for block in self.model.features[-unfreeze_last_blocks:]:
                    for param in block.parameters():
                        param.requires_grad = True

        # ========================
        # Extrair in_features corretamente
        # ========================
        if hasattr(self.model, "classifier"):
            if isinstance(self.model.classifier, nn.Sequential):
                in_features = self.model.classifier[-1].in_features
            else:
                in_features = self.model.classifier.in_features
        else:
            raise ValueError("ConvNeXt model without classifier attribute")

        # ========================
        # Remove head original
        # ========================
        self.model.classifier = nn.Identity()

        # ========================
        # Head melhorado
        # ========================
        self.head = nn.Sequential(
            nn.LayerNorm(in_features),   # ConvNeXt gosta disso
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, output_dimensions)
        )

    def extract(self, x):
        x = self.model(x)

        # Garantir shape (B, C)
        if x.ndim == 4:
            x = x.mean(dim=[2, 3])  # global average pooling

        return x

    def forward(self, x):
        x = self.extract(x)
        x = self.head(x)
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

class ViTApi(nn.Module):
    def __init__(
        self,
        model,
        output_dimensions: int,
        dropout_rate=0.3
    ):
        super(ViTApi, self).__init__()
        self.model = model
        self.dropout_rate = dropout_rate

        for param in list(self.model.parameters())[:-150]:
            param.requires_grad = False

        # Get input features before replacing classifier
        in_features = self.model.heads.head.in_features

        # Replace the final classification layer with Identity
        self.model.heads.head = nn.Identity()

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


class SwinApi(nn.Module):
    def __init__(
        self,
        model,
        output_dimensions: int,
        dropout_rate=0.3
    ):
        super(SwinApi, self).__init__()
        self.model = model
        self.dropout_rate = dropout_rate

        for param in list(self.model.parameters())[:-150]:
            param.requires_grad = False

        # Get input features before replacing head
        in_features = self.model.head.in_features

        # Replace the final classification layer with Identity
        self.model.head = nn.Identity()

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


class SwinAttentionMIL(nn.Module):
    def __init__(self, base_model, output_classes=5, dropout_rate=0.6, freeze_backbone=True):
        super(SwinAttentionMIL, self).__init__()
        self.base_model = base_model
        
        # Freezing backbone parameter
        if freeze_backbone:
            # Let's freeze all but the last 150 parameters (similar to EfficientNet) or just freeze all features
            # swin_v2_t has many params, freezing most of them saves VRAM.
            params = list(self.base_model.parameters())
            freeze_up_to = max(0, len(params) - 100)  # leave the last 100 params unfrozen (usually last block)
            for i, param in enumerate(params):
                if i < freeze_up_to:
                    param.requires_grad = False

        layers = list(base_model.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)

        in_features = base_model.head.in_features
        
        self.attention = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.Tanh(),
            nn.Linear(512, 1)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, output_classes)
        )

    def forward(self, x):
        b, n, C, H, W = x.size()
        x = x.view(b*n, C, H, W)

        # Process features in chunks to save peak VRAM
        chunk_size = 12
        features_list = []
        for i in range(0, b*n, chunk_size):
            chunk = x[i:i+chunk_size]
            features_list.append(self.feature_extractor(chunk))
            
        features = torch.cat(features_list, dim=0)
        features = features.view(b, n, -1)

        attn_weights = self.attention(features)
        attn_weights = F.softmax(attn_weights, dim=1)

        bag_representation = torch.sum(attn_weights * features, dim=1)

        logits = self.classifier(bag_representation)
        return logits

