import os
from typing import Any

from torch.utils.data import Dataset
from skimage import io as skio, color
import numpy as np
import torch
from albumentations.core.transforms_interface import ImageOnlyTransform

class PandasDataset(Dataset):
    def __init__(
        self,
        image_dir,
        dataframe,
        transforms=None,
        normalize=False,
        format="jpg",
        num_classes=5
    ):
        self.image_dir = image_dir
        self.dataframe = dataframe
        self.transforms = transforms
        self.normalize = normalize
        self.format = format
        self.num_classes = num_classes

    def __len__(self):
        return self.dataframe.shape[0]

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        img_id = row.image_id.strip()

        file_path = f"{self.image_dir}/{img_id}.{self.format}"
        try:
            image = skio.imread(file_path)

            if self.transforms is not None:
                image = self.transforms(image=image)['image']

            if self.normalize:
                image = image.astype(np.float32) / 255.0
            image = np.transpose(image, (2, 0, 1))

            label = np.zeros(self.num_classes).astype(np.float32)
            label[:row.isup_grade] = 1.

            return torch.tensor(image, dtype=torch.float32), torch.tensor(label, dtype=torch.float32), img_id
        except:
            pass


class PandasWithMilDataset(Dataset):
    def __init__(
        self,
        patches_dir,
        dataframe,
        transforms=None,
        normalize=False,
        num_classes=5,
        max_patches=36,  # 🔥 começa com 36, pode mudar depois
    ):
        self.patches_dir = patches_dir
        self.dataframe = dataframe
        self.transforms = transforms
        self.normalize = normalize
        self.num_classes = num_classes
        self.max_patches = max_patches

    def __len__(self):
        return self.dataframe.shape[0]

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        img_id = row.image_id.strip()

        patch_folder = os.path.join(self.patches_dir, img_id)
        try:
            patch_files = sorted(os.listdir(patch_folder))
            patch_files = [f for f in patch_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif'))]

            patch_files = patch_files[:self.max_patches]

            patches = []
            for patch_name in patch_files:
                patch_path = os.path.join(patch_folder, patch_name)
                patch = skio.imread(patch_path)

                if self.transforms is not None:
                    patch = self.transforms(image=patch)['image']

                if self.normalize:
                    patch = patch.astype(np.float32) / 255.0

                patch = np.transpose(patch, (2, 0, 1))
                patches.append(torch.tensor(patch, dtype=torch.float32))

            num_current = len(patches)

            if num_current == 0:
                raise RuntimeError(f"No patches found in {patch_folder}")

            if num_current < self.max_patches:
                c, h, w = patches[0].shape
                pad = torch.ones((c, h, w), dtype=torch.float32) * 255.0

                for _ in range(self.max_patches - num_current):
                    patches.append(pad)
            bag = torch.stack(patches)  # (max_patches, C, H, W)

            # ✅ máscara (fundamental)
            mask = torch.zeros(self.max_patches)
            mask[:num_current] = 1

            label = np.zeros(self.num_classes).astype(np.float32)
            label[:row.isup_grade] = 1.

            return bag, mask, torch.tensor(label, dtype=torch.float32), img_id

        except Exception as e:
            raise RuntimeError(f"Erro ao carregar patches de {img_id}: {e}")

class SicapDataset(Dataset):
    def __init__(
        self,
        image_dir,
        dataframe,
        transforms=None,
        normalize=False,
        format="jpg",
        num_classes=5
    ):
        self.image_dir = image_dir
        self.dataframe = dataframe
        self.transforms = transforms
        self.normalize = normalize
        self.format = format
        self.num_classes = num_classes

    def __len__(self):
        return self.dataframe.shape[0]

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        img_id = row.slide_id.strip()

        file_path = f"{self.image_dir}/{img_id}.{self.format}"
        try:
            image = skio.imread(file_path)

            if self.transforms is not None:
                image = self.transforms(image=image)['image']

            if self.normalize:
                image = image.astype(np.float32) / 255.0
            image = np.transpose(image, (2, 0, 1))

            label = np.zeros(self.num_classes).astype(np.float32)
            label[:row.isup_grade] = 1.

            return torch.tensor(image, dtype=torch.float32), torch.tensor(label, dtype=torch.float32), img_id
        except:
            pass





class PandasDatasetSimple(Dataset):
    """
    Dataset simplificado que retorna apenas imagem e label (não o image_id).
    """
    def __init__(self, image_dir, dataframe, transforms=None):
        self.image_dir = image_dir
        self.dataframe = dataframe
        self.transforms = transforms

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        img_id = row.image_id.strip()

        file_path = f"{self.image_dir}/{img_id}.jpg"
        image = skio.imread(file_path)

        if self.transforms is not None:
            transformed = self.transforms(image=image)
            image = transformed['image']

        label = row.isup_grade

        return image, label

class PatchBagDataset(Dataset):
    def __init__(
        self,
        image_dir,
        dataframe,
        transforms=None,
        normalize=False,
        format="png",
        num_classes=5,
        patch_size=256,
        grid_size=6
    ):
        self.image_dir = image_dir
        self.dataframe = dataframe
        self.transforms = transforms
        self.normalize = normalize
        self.format = format
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.grid_size = grid_size

    def __len__(self):
        return self.dataframe.shape[0]

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        img_id = row.image_id.strip()

        file_path = f"{self.image_dir}/{img_id}.{self.format}"
        try:
            image = skio.imread(file_path)

            # lista de patches
            patches = []
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    top = i * self.patch_size
                    left = j * self.patch_size
                    patch = image[top:top+self.patch_size, left:left+self.patch_size, :]

                    if self.transforms is not None:
                        patch = self.transforms(image=patch)['image']

                    if self.normalize:
                        patch = patch.astype(np.float32) / 255.0

                    patch = np.transpose(patch, (2, 0, 1))  # (C,H,W)
                    patches.append(torch.tensor(patch, dtype=torch.float32))

            # Bag de 36 patches
            bag = torch.stack(patches)  # (36, C, H, W)

            # Label ordinal
            label = np.zeros(self.num_classes).astype(np.float32)
            label[:row.isup_grade] = 1.

            return bag, torch.tensor(label, dtype=torch.float32), img_id
        except Exception as e:
            print(f"Erro ao carregar {file_path}: {e}")
            return None

class PandasOverlapDataset(Dataset):
    def __init__(
        self,
        patches_dir,
        dataframe,
        transforms=None,
        normalize=False,
        num_classes=5,
        max_patches=36,
        grid_size=6,       # 6x6 grid
        overlap=10,        # overlap em pixels
    ):
        self.patches_dir = patches_dir
        self.dataframe = dataframe
        self.transforms = transforms
        self.normalize = normalize
        self.num_classes = num_classes
        self.max_patches = max_patches  # deve ser grid_size^2 = 36
        self.grid_size = grid_size
        self.overlap = overlap

    def __len__(self):
        return self.dataframe.shape[0]

    def _build_mosaic(self, patches: tuple[float, list[np.ndarray]]) -> np.ndarray:
        """
        Monta um mosaico grid_size x grid_size com overlap entre patches.
        patches: lista de arrays (H, W, C), todos do mesmo tamanho.
        Patches faltantes são preenchidos com branco (255).
        """
        G = self.grid_size
        ov = self.overlap

        patches = [patch[1] for patch in patches]

        # Usa o primeiro patch para inferir dimensão
        h, w, c = patches[0].shape

        # Tamanho total do mosaico levando em conta o overlap
        mosaic_h = G * h - (G - 1) * ov
        mosaic_w = G * w - (G - 1) * ov

        mosaic = np.full((mosaic_h, mosaic_w, c), 255, dtype=patches[0].dtype)

        for idx in range(G * G):
            row = idx // G
            col = idx % G

            y0 = row * (h - ov)
            x0 = col * (w - ov)

            if idx < len(patches):
                patch = patches[idx]
            else:
                patch = np.full((h, w, c), 255, dtype=patches[0].dtype)

            # Na região de overlap, faz média simples com o que já está no mosaico
            region = mosaic[y0:y0 + h, x0:x0 + w]
            overlap_mask = np.zeros((h, w), dtype=bool)

            if row > 0:
                overlap_mask[:ov, :] = True   # faixa superior
            if col > 0:
                overlap_mask[:, :ov] = True   # faixa esquerda

            blended = patch.copy().astype(np.float32)
            blended[overlap_mask] = (
                region[overlap_mask].astype(np.float32) * 0.5 +
                patch[overlap_mask].astype(np.float32) * 0.5
            )
            mosaic[y0:y0 + h, x0:x0 + w] = blended.astype(patches[0].dtype)

        return mosaic

    def tissue_ratio(self, patch, threshold=240):
        """
        Retorna a proporção de tecido no patch.
        Pixels muito claros são considerados fundo.
        """
        if patch.ndim == 3:
            mask = np.mean(patch, axis=2) < threshold
        else:
            mask = patch < threshold

        return mask.mean()

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        img_id = row.image_id.strip()

        patch_folder = os.path.join(self.patches_dir, img_id)
        try:
            patch_files = sorted(os.listdir(patch_folder))
            patch_files = [
                f for f in patch_files
                if f.lower().endswith(('.png'))
            ]
            patch_files = patch_files[:self.max_patches]

            if len(patch_files) == 0:
                raise RuntimeError(f"No patches found in {patch_folder}")

            # Lê e aplica transforms em cada patch (ainda em HWC)
            raw_patches = []
            for patch_name in patch_files:
                patch_path = os.path.join(patch_folder, patch_name)
                patch = skio.imread(patch_path)

                if self.transforms is not None:
                    patch = self.transforms(image=patch)['image']
                
                score = self.tissue_ratio(patch)
                raw_patches.append((score, patch))  # (score, HWC, uint8 ou float)
                
            raw_patches.sort(key=lambda x: x[0], reverse=True)
            # Monta o mosaico (ainda HWC, antes do normalize)
            mosaic = self._build_mosaic(raw_patches)  # (mosaic_H, mosaic_W, C)

            if self.normalize:
                mosaic = mosaic.astype(np.float32) / 255.0

            # CHW para PyTorch
            mosaic = np.transpose(mosaic, (2, 0, 1))
            mosaic_tensor = torch.tensor(mosaic, dtype=torch.float32)

            label = np.zeros(self.num_classes, dtype=np.float32)
            label[:row.isup_grade] = 1.0

            return mosaic_tensor, torch.tensor(label, dtype=torch.float32), img_id

        except Exception as e:
            raise RuntimeError(f"Erro ao carregar patches de {img_id}: {e}")