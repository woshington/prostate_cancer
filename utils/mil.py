import numpy as np
import torch
import skimage.io as skio
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, Callable
import logging
from torchvision.models.efficientnet import EfficientNet as EfficientNetPytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal

logger = logging.getLogger(__name__)


class PandasWithMilDataset(Dataset):
    """
    Dataset para MIL (Multiple Instance Learning) com patches histopatológicos PANDAS.

    Args:
        patches_dir: Diretório raiz contendo subpastas com patches por imagem.
        dataframe: DataFrame com colunas `image_id` e `isup_grade`.
        transforms: Transforms Albumentations (recebem dict com 'image').
        normalize: Se True, divide por 255 (só útil sem ToFloat no transform).
        num_classes: Número de classes ordinais (ordinal encoding).
        max_patches: Máximo de patches por bag; faz padding com branco se necessário.
        patch_ext: Extensões de arquivo aceitas (filtra não-imagens na pasta).
    """

    PAD_VALUE = 1.0  # após normalização (equivale a 255 antes)

    def __init__(
        self,
        patches_dir: str | Path,
        dataframe,
        transforms: Optional[Callable] = None,
        normalize: bool = False,
        num_classes: int = 5,
        max_patches: int = 36,
        patch_ext: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif", ".tiff"),
    ):
        self.patches_dir = Path(patches_dir)
        self.dataframe = dataframe.reset_index(drop=True)
        self.transforms = transforms
        self.normalize = normalize
        self.num_classes = num_classes
        self.max_patches = max_patches
        self.patch_ext = {e.lower() for e in patch_ext}

        self._validate()

    # ------------------------------------------------------------------
    # Validação antecipada
    # ------------------------------------------------------------------
    def _validate(self):
        if not self.patches_dir.exists():
            raise FileNotFoundError(f"patches_dir não encontrado: {self.patches_dir}")
        required_cols = {"image_id", "isup_grade"}
        missing = required_cols - set(self.dataframe.columns)
        if missing:
            raise ValueError(f"DataFrame está faltando colunas: {missing}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _list_patches(self, folder: Path) -> list[Path]:
        """Retorna patches ordenados, filtrando por extensão."""
        return sorted(
            p for p in folder.iterdir()
            if p.suffix.lower() in self.patch_ext
        )

    def _load_patch(self, path: Path) -> np.ndarray:
        patch = skio.imread(str(path))
        # Garante RGB (descarta alpha se existir)
        if patch.ndim == 2:
            patch = np.stack([patch] * 3, axis=-1)
        elif patch.shape[-1] == 4:
            patch = patch[..., :3]
        return patch

    def _apply_transforms(self, patch: np.ndarray) -> np.ndarray:
        if self.transforms is not None:
            patch = self.transforms(image=patch)["image"]
        return patch

    def _to_tensor(self, patch: np.ndarray) -> torch.Tensor:
        if self.normalize:
            patch = patch.astype(np.float32) / 255.0
        else:
            patch = patch.astype(np.float32)
        # HWC → CHW
        return torch.from_numpy(np.ascontiguousarray(patch.transpose(2, 0, 1)))

    def _make_padding(self, ref_tensor: torch.Tensor) -> torch.Tensor:
        """Cria um patch de padding branco com o mesmo shape do ref."""
        value = self.PAD_VALUE if self.normalize else 255.0
        return torch.full_like(ref_tensor, fill_value=value)

    def _encode_label(self, isup_grade: int) -> torch.Tensor:
        """Ordinal encoding: [1,1,0,0,0] para grade=2."""
        label = np.zeros(self.num_classes, dtype=np.float32)
        label[:isup_grade] = 1.0
        return torch.from_numpy(label)

    # ------------------------------------------------------------------
    # Interface Dataset
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int):
        row = self.dataframe.iloc[index]
        img_id = str(row.image_id).strip()
        patch_folder = self.patches_dir / img_id

        if not patch_folder.exists():
            raise FileNotFoundError(f"Pasta de patches não encontrada: {patch_folder}")

        patch_files = self._list_patches(patch_folder)

        if not patch_files:
            raise RuntimeError(f"Nenhum patch encontrado em: {patch_folder}")

        patch_files = patch_files[: self.max_patches]

        # Carrega e transforma cada patch
        patches: list[torch.Tensor] = []
        for patch_path in patch_files:
            raw = self._load_patch(patch_path)
            raw = self._apply_transforms(raw)
            patches.append(self._to_tensor(raw))

        num_real = len(patches)

        # Padding até max_patches
        if num_real < self.max_patches:
            pad_tensor = self._make_padding(patches[0])
            patches.extend(
                [pad_tensor.clone() for _ in range(self.max_patches - num_real)]
            )

        bag = torch.stack(patches)          # (max_patches, C, H, W)

        mask = torch.zeros(self.max_patches, dtype=torch.float32)
        mask[:num_real] = 1.0               # 1 = patch real, 0 = padding

        label = self._encode_label(int(row.isup_grade))

        return bag, mask, label, img_id

# ─────────────────────────────────────────────
# Módulo de atenção isolado e reutilizável
# ─────────────────────────────────────────────
class GatedAttention(nn.Module):
    """
    Gated Attention Mechanism (Ilse et al., 2018).

    A(h) = softmax( W · tanh(V·h) ⊙ sigmoid(U·h) )

    A atenção padrão usa só tanh; a versão "gated" adiciona
    um sigmoid como portão multiplicativo, permitindo que a
    rede suprima patches irrelevantes de forma mais seletiva.

    Args:
        in_features:  dimensão da feature de entrada.
        hidden_dim:   dimensão da camada interna.
        gated:        se False, usa tanh simples (ablation).
        dropout_rate: dropout aplicado às features antes da atenção.
    """

    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 512,
        gated: bool = True,
        dropout_rate: float = 0.25,
    ):
        super().__init__()
        self.gated = gated

        self.V = nn.Sequential(nn.Dropout(dropout_rate), nn.Linear(in_features, hidden_dim))
        self.U = nn.Sequential(nn.Dropout(dropout_rate), nn.Linear(in_features, hidden_dim)) if gated else None
        self.W = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, features: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            features: (B, N, D)
            mask:     (B, N)  — 1 = patch real, 0 = padding

        Returns:
            attn_weights: (B, N, 1)  softmax já aplicado
        """
        h_tanh = torch.tanh(self.V(features))  # (B, N, H)

        if self.gated and self.U is not None:
            h_sig = torch.sigmoid(self.U(features))  # (B, N, H)
            h = h_tanh * h_sig
        else:
            h = h_tanh

        logits = self.W(h)  # (B, N, 1)

        # Mascara patches de padding com -inf antes do softmax
        if mask is not None:
            pad_mask = (mask == 0).unsqueeze(-1)  # (B, N, 1)
            logits = logits.masked_fill(pad_mask, float("-inf"))

        attn_weights = F.softmax(logits, dim=1)  # (B, N, 1)
        return attn_weights


# ─────────────────────────────────────────────
# Modelo principal
# ─────────────────────────────────────────────
class EfficientNetMIL(nn.Module):
    """
    EfficientNet + Multiple Instance Learning com Gated Attention.

    Args:
        model:          backbone EfficientNet (efficientnet-pytorch).
        output_classes: número de saídas (ordinal encoding → num_classes).
        fine_tune:      quantos parâmetros (do fim) ficam treináveis.
        dropout_rate:   dropout no classificador e na atenção.
        hidden_dim:     dimensão da camada oculta da atenção.
        gated:          usa Gated Attention se True.
        pool:           'att' = attention pooling | 'mean' = mean pooling (baseline).
    """

    def __init__(
            self,
            model: EfficientNetPytorch,
            output_classes: int,
            fine_tune: int = 100,
            dropout_rate: float = 0.4,
            hidden_dim: int = 512,
            gated: bool = True,
            pool: Literal["att", "mean"] = "att",
    ):
        super().__init__()
        self.pool = pool

        # ── Congela backbone parcialmente ───────────────────────────
        all_params = list(model.parameters())
        for param in all_params[:-fine_tune]:
            param.requires_grad = False

        # ── Remove cabeça de classificação ──────────────────────────
        in_features: int = model.classifier[1].in_features
        model.classifier[1] = nn.Identity()
        self.feature_extractor = model

        # ── Normalização de features (estabiliza treino) ─────────────
        self.feat_norm = nn.LayerNorm(in_features)

        # ── Attention pooling ────────────────────────────────────────
        if pool == "att":
            self.attention = GatedAttention(
                in_features=in_features,
                hidden_dim=hidden_dim,
                gated=gated,
                dropout_rate=dropout_rate,
            )

        # ── Cabeça de classificação ──────────────────────────────────
        self.classifier = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, output_classes),
        )

    # ── helpers ─────────────────────────────────────────────────────
    def extract(self, x: torch.Tensor) -> torch.Tensor:
        """Extrai features de um batch de patches (B*N, C, H, W) → (B*N, D)."""
        return self.feature_extractor(x)

    def _pool(self, features: torch.Tensor, mask: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Agrega patches → representação da bag.

        Returns:
            bag:          (B, D)
            attn_weights: (B, N, 1) ou None se pool='mean'
        """
        if self.pool == "att":
            attn_weights = self.attention(features, mask)  # (B, N, 1)
            bag = torch.sum(attn_weights * features, dim=1)  # (B, D)
            return bag, attn_weights

        # mean pooling com máscara
        if mask is not None:
            m = mask.unsqueeze(-1).float()  # (B, N, 1)
            bag = (features * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
        else:
            bag = features.mean(dim=1)
        return bag, None

    # ── forward ─────────────────────────────────────────────────────
    def forward(
            self,
            x: torch.Tensor,
            mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            x:    (B, N, C, H, W)
            mask: (B, N)  — 1 = real, 0 = padding  ← USE SEMPRE

        Returns:
            dict com:
              'logits':  (B, output_classes)
              'attn':    (B, N, 1) ou None
              'features': (B, D)  — útil para visualização / probing
        """
        B, N, C, H, W = x.size()

        # ── Extração de features ─────────────────────────────────────
        x_flat = x.view(B * N, C, H, W)
        feats = self.extract(x_flat)  # (B*N, D)
        feats = feats.view(B, N, -1)  # (B, N, D)
        feats = self.feat_norm(feats)  # normaliza por patch

        # ── Pooling ──────────────────────────────────────────────────
        bag, attn_weights = self._pool(feats, mask)  # (B, D)

        # ── Classificação ────────────────────────────────────────────
        logits = self.classifier(bag)  # (B, output_classes)

        return {
            "logits": logits,
            "attn": attn_weights,
            "features": bag,
        }


# ─────────────────────────────────────────────
# ConvNeXt + MIL
# ─────────────────────────────────────────────
class ConvNeXtMIL(nn.Module):
    """
    ConvNeXt + Multiple Instance Learning com Gated Attention.

    Args:
        model:                  backbone ConvNeXt (torchvision).
        output_classes:         número de saídas (ordinal encoding).
        unfreeze_last_blocks:   quantos estágios finais ficam treináveis.
        dropout_rate:           dropout no classificador e na atenção.
        hidden_dim:             dimensão da camada oculta da atenção.
        gated:                  usa Gated Attention se True.
        pool:                   'att' = attention pooling | 'mean' = mean pooling.
    """

    def __init__(
            self,
            model: nn.Module,
            output_classes: int,
            unfreeze_last_blocks: int = 2,
            dropout_rate: float = 0.4,
            hidden_dim: int = 512,
            gated: bool = True,
            pool: Literal["att", "mean"] = "att",
    ):
        super().__init__()
        self.pool = pool

        # Congela todo o backbone
        for param in model.parameters():
            param.requires_grad = False

        # Descongela os últimos estágios
        if hasattr(model, "features"):
            for block in model.features[-unfreeze_last_blocks:]:
                for param in block.parameters():
                    param.requires_grad = True

        # in_features a partir do classificador original
        if isinstance(model.classifier, nn.Sequential):
            in_features: int = model.classifier[-1].in_features
        else:
            in_features = model.classifier.in_features

        model.classifier = nn.Identity()
        self.feature_extractor = model

        self.feat_norm = nn.LayerNorm(in_features)

        if pool == "att":
            self.attention = GatedAttention(
                in_features=in_features,
                hidden_dim=hidden_dim,
                gated=gated,
                dropout_rate=dropout_rate,
            )

        self.classifier = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, output_classes),
        )

    def extract(self, x: torch.Tensor) -> torch.Tensor:
        out = self.feature_extractor(x)
        # avgpool deixa (B, C, 1, 1) quando classifier é Identity
        if out.ndim == 4:
            out = out.squeeze(-1).squeeze(-1)
        return out

    def _pool(self, features: torch.Tensor, mask: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.pool == "att":
            attn_weights = self.attention(features, mask)
            bag = torch.sum(attn_weights * features, dim=1)
            return bag, attn_weights
        if mask is not None:
            m = mask.unsqueeze(-1).float()
            bag = (features * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
        else:
            bag = features.mean(dim=1)
        return bag, None

    def forward(
            self,
            x: torch.Tensor,
            mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        B, N, C, H, W = x.size()
        x_flat = x.view(B * N, C, H, W)
        feats = self.extract(x_flat)
        feats = feats.view(B, N, -1)
        feats = self.feat_norm(feats)
        bag, attn_weights = self._pool(feats, mask)
        logits = self.classifier(bag)
        return {"logits": logits, "attn": attn_weights, "features": bag}


# ─────────────────────────────────────────────
# Swin Transformer + MIL
# ─────────────────────────────────────────────
class SwinMIL(nn.Module):
    """
    Swin Transformer + Multiple Instance Learning com Gated Attention.

    Args:
        model:                backbone Swin (torchvision).
        output_classes:       número de saídas (ordinal encoding).
        unfreeze_last_blocks: quantos estágios finais de model.features ficam treináveis.
                              As camadas pós-features (norm, permute, avgpool, flatten)
                              são sempre descongeladas pois afetam a qualidade das features.
        dropout_rate:         dropout no classificador e na atenção.
        hidden_dim:           dimensão da camada oculta da atenção.
        gated:                usa Gated Attention se True.
        pool:                 'att' = attention pooling | 'mean' = mean pooling.
    """

    def __init__(
            self,
            model: nn.Module,
            output_classes: int,
            unfreeze_last_blocks: int = 2,
            dropout_rate: float = 0.4,
            hidden_dim: int = 512,
            gated: bool = True,
            pool: Literal["att", "mean"] = "att",
    ):
        super().__init__()
        self.pool = pool

        # Congela todo o backbone
        for param in model.parameters():
            param.requires_grad = False

        # Descongela os últimos estágios das features hierárquicas
        if hasattr(model, "features"):
            for block in model.features[-unfreeze_last_blocks:]:
                for param in block.parameters():
                    param.requires_grad = True

        # Camadas pós-features sempre treináveis (norm + pooling)
        for attr in ("norm", "permute", "avgpool", "flatten"):
            module = getattr(model, attr, None)
            if module is not None:
                for param in module.parameters():
                    param.requires_grad = True

        in_features: int = model.head.in_features
        model.head = nn.Identity()
        self.feature_extractor = model

        self.feat_norm = nn.LayerNorm(in_features)

        if pool == "att":
            self.attention = GatedAttention(
                in_features=in_features,
                hidden_dim=hidden_dim,
                gated=gated,
                dropout_rate=dropout_rate,
            )

        self.classifier = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, output_classes),
        )

    def extract(self, x: torch.Tensor) -> torch.Tensor:
        out = self.feature_extractor(x)
        if out.ndim > 2:
            out = out.flatten(1)
        return out

    def _pool(self, features: torch.Tensor, mask: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.pool == "att":
            attn_weights = self.attention(features, mask)
            bag = torch.sum(attn_weights * features, dim=1)
            return bag, attn_weights
        if mask is not None:
            m = mask.unsqueeze(-1).float()
            bag = (features * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
        else:
            bag = features.mean(dim=1)
        return bag, None

    def forward(
            self,
            x: torch.Tensor,
            mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        B, N, C, H, W = x.size()
        x_flat = x.view(B * N, C, H, W)
        feats = self.extract(x_flat)
        feats = feats.view(B, N, -1)
        feats = self.feat_norm(feats)
        bag, attn_weights = self._pool(feats, mask)
        logits = self.classifier(bag)
        return {"logits": logits, "attn": attn_weights, "features": bag}


# ─────────────────────────────────────────────
# ViT + MIL
# ─────────────────────────────────────────────
class ViTMIL(nn.Module):
    """
    Vision Transformer (ViT) + Multiple Instance Learning com Gated Attention.

    Args:
        model:          backbone ViT (torchvision).
        output_classes: número de saídas (ordinal encoding).
        fine_tune:      quantos parâmetros finais ficam treináveis.
        dropout_rate:   dropout no classificador e na atenção.
        hidden_dim:     dimensão da camada oculta da atenção.
        gated:          usa Gated Attention se True.
        pool:           'att' = attention pooling | 'mean' = mean pooling.
    """

    def __init__(
            self,
            model: nn.Module,
            output_classes: int,
            fine_tune: int = 100,
            dropout_rate: float = 0.4,
            hidden_dim: int = 512,
            gated: bool = True,
            pool: Literal["att", "mean"] = "att",
    ):
        super().__init__()
        self.pool = pool

        all_params = list(model.parameters())
        for param in all_params[:-fine_tune]:
            param.requires_grad = False

        in_features: int = model.heads.head.in_features
        model.heads.head = nn.Identity()
        self.feature_extractor = model

        self.feat_norm = nn.LayerNorm(in_features)

        if pool == "att":
            self.attention = GatedAttention(
                in_features=in_features,
                hidden_dim=hidden_dim,
                gated=gated,
                dropout_rate=dropout_rate,
            )

        self.classifier = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, output_classes),
        )

    def extract(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_extractor(x)

    def _pool(self, features: torch.Tensor, mask: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.pool == "att":
            attn_weights = self.attention(features, mask)
            bag = torch.sum(attn_weights * features, dim=1)
            return bag, attn_weights
        if mask is not None:
            m = mask.unsqueeze(-1).float()
            bag = (features * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
        else:
            bag = features.mean(dim=1)
        return bag, None

    def forward(
            self,
            x: torch.Tensor,
            mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        B, N, C, H, W = x.size()
        x_flat = x.view(B * N, C, H, W)
        feats = self.extract(x_flat)
        feats = feats.view(B, N, -1)
        feats = self.feat_norm(feats)
        bag, attn_weights = self._pool(feats, mask)
        logits = self.classifier(bag)
        return {"logits": logits, "attn": attn_weights, "features": bag}