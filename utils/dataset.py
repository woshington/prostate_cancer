import kornia.color as kc
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
        normalize=False
    ):
        self.image_dir = image_dir
        self.dataframe = dataframe
        self.transforms = transforms
        self.normalize = normalize

    def __len__(self):
        return self.dataframe.shape[0]

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        img_id = row.image_id.strip()

        file_path = f"{self.image_dir}/{img_id}.jpg"
        image = skio.imread(file_path)

        if self.transforms is not None:
            image = self.transforms(image=image)['image']

        if self.normalize:
            image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))

        label = np.zeros(5).astype(np.float32)
        label[:row.isup_grade] = 1.

        return torch.tensor(image, dtype=torch.float32), torch.tensor(label, dtype=torch.float32), img_id


class RemovePenMarkAlbumentations(ImageOnlyTransform):
    def __init__(self, is_white=True):
        super().__init__(p=1)
        self.is_white = is_white

    @staticmethod
    def calculate_channel_sums(image):
        red_sum = np.sum(image[:, :, 0])
        green_sum = np.sum(image[:, :, 1])
        blue_sum = np.sum(image[:, :, 2])
        return red_sum, green_sum, blue_sum

    def analyze_histogram(self, image, threshold):
        red_sum, green_sum, blue_sum = self.calculate_channel_sums(image)

        green_dominance = green_sum > threshold * red_sum
        blue_dominance = blue_sum > threshold * red_sum

        return green_dominance or blue_dominance

    def apply(self, img, **params: Any):
        chip_size = (16, 16)
        overlap = 0

        height, width = img.shape[:2]
        chip_h, chip_w = chip_size

        for y in range(0, height, chip_h - overlap):
            for x in range(0, width, chip_w - overlap):
                chip = img[y:y + chip_h, x:x + chip_w]

                if chip.shape[0] < chip_h or chip.shape[1] < chip_w:
                    padded_chip = np.zeros((chip_h, chip_w, img.shape[2]), dtype=img.dtype)
                    padded_chip[:chip.shape[0], :chip.shape[1]] = chip
                    chip = padded_chip

                if self.analyze_histogram(chip, threshold=1):
                    img[y:y + chip_h, x:x + chip_w] = 255 if self.is_white else 0

        return img


class RGB2XYZTransform(ImageOnlyTransform):
    def __init__(self, p=1.0):
        super().__init__(p=p)

    def apply(self, image, **params):
        img = image.astype(np.float32) / 255.0
        image = color.rgb2xyz(img)
        return image
        # return image.astype(np.float32)


class RGB2HedTransform(ImageOnlyTransform):
    def __init__(self, p=1.0):
        super().__init__(p=p)

    def apply(self, image, **params):
        image = color.rgb2hed(image)
        return image

class RGB2HematoxylinTransform(ImageOnlyTransform):
    def __init__(self, p=1.0):
        super().__init__(p=p)

    def apply(self, image, **params):
        img = image.astype(np.float32) / 255.0
        hed = color.rgb2hed(img)
        h_channel = hed[:, :, 0]
        return h_channel.astype(np.float32)

class RGB2LABTransform(ImageOnlyTransform):
    def __init__(self, p=1.0):
        super().__init__(p=p)

    def apply(self, image, **params):
        img = image.astype(np.float32) / 255.0
        image = color.rgb2lab(img)
        return image.astype(np.float32)


class RGB2LUVTransform(ImageOnlyTransform):
    def __init__(self, p=1.0):
        super().__init__(p=p)

    def apply(self, image, **params):
        img = image.astype(np.float32) / 255.0
        image = color.rgb2luv(img)
        return image.astype(np.float32)


class RGB2HSVTransform(ImageOnlyTransform):
    def __init__(self, p=1.0):
        super().__init__(p=p)

    def apply(self, image, **params):
        img = image.astype(np.float32) / 255.0
        image = color.rgb2hsv(img)
        return image.astype(np.float32)

class MultiColorSpaceTransform(ImageOnlyTransform):
    def __init__(self, p=1.0):
        super().__init__(p=p)

    def apply(self, image, **params):
        img = image.astype(np.float32) / 255.0
        image_xyz = color.rgb2xyz(img)
        image_hed = color.rgb2hed(img)
        image_lab = color.rgb2lab(img)
        image_luv = color.rgb2luv(img)
        image_hsv = color.rgb2hsv(img)

        image = np.concatenate((img, image_xyz, image_hed, image_lab, image_luv, image_hsv), axis=-1)
        return image

class RGB2YHUTransform(ImageOnlyTransform):
    def __init__(self, p=1.0):
        super().__init__(p=p)

    def apply(self, image, **params):
        img = image.astype(np.float32) / 255.0

        image_hed = color.rgb2hed(img)
        image_xyz = color.rgb2xyz(img)
        image_luv = color.rgb2luv(img)

        h_channel = image_hed[:, :, 0]  # H do HED
        y_channel = image_xyz[:, :, 1]  # Y do XYZ
        u_channel = image_luv[:, :, 1]  # U do LUV

        image = np.stack([h_channel, y_channel, u_channel], axis=-1)
        return image.astype(np.float32)


class RGB2YHVTransform(ImageOnlyTransform):
    def __init__(self, p=1.0):
        super().__init__(p=p)

    def apply(self, image, **params):
        img = image.astype(np.float32) / 255.0  # normaliza RGB para [0,1]

        image_xyz = color.rgb2xyz(img)  # XYZ
        image_hed = color.rgb2hed(img)  # HED
        image_luv = color.rgb2luv(img)  # LUV

        # Extrai os canais
        y_channel = image_xyz[:, :, 1]   # canal Y do XYZ
        h_channel = image_hed[:, :, 0]   # canal H do HED
        v_channel = image_luv[:, :, 2]   # canal V do LUV

        # Normaliza Y (luminância) entre 0 e 1 (min-max da imagem)
        y_min, y_max = y_channel.min(), y_channel.max()
        y_norm = (y_channel - y_min) / (y_max - y_min + 1e-8)

        # Normaliza H do HED (min-max da imagem)
        h_min, h_max = h_channel.min(), h_channel.max()
        h_norm = (h_channel - h_min) / (h_max - h_min + 1e-8)

        # Normaliza V do LUV (min-max da imagem)
        v_min, v_max = v_channel.min(), v_channel.max()
        v_norm = (v_channel - v_min) / (v_max - v_min + 1e-8)

        # Empilha na ordem Y, H, V
        image_out = np.stack([y_norm, h_norm, v_norm], axis=-1).astype(np.float32)

        return image_out



# class RGB2Fusion(ImageOnlyTransform):
#     def __init__(self, mode="sum", p=1.0, space_colors=None):
#         """
#         mode: 'sum', 'mean', or 'max'
#         """
#         super().__init__(p=p)
#         if space_colors is None:
#             space_colors = ["rgb", "xyz"]
#         self.space_colors = space_colors
#         self.mode = mode
#
#     def apply(self, image, **params):
#         img_rgb = image.astype(np.float32) #/ 255.0
#         # img_xyz = color.rgb2xyz(img_rgb)
#
#         color_map = {
#             "rgb": img_rgb,
#             "xyz": color.rgb2xyz(img_rgb),
#             "hed": color.rgb2hed(img_rgb),
#             "lab": color.rgb2lab(img_rgb),
#             "luv": color.rgb2luv(img_rgb),
#             "hsv": color.rgb2hsv(img_rgb)
#         }
#
#         selected = [color_map[space_color] for space_color in self.space_colors]
#
#         # normalized = []
#         # for img in selected:
#         #     min_val, max_val = np.min(img), np.max(img)
#         #     if max_val - min_val > 1e-6:
#         #         norm_img = (img - min_val) / (max_val - min_val)
#         #     else:
#         #         norm_img = img  # já está normalizado
#         #     normalized.append(norm_img)
#
#         if self.mode == "sum":
#             # if image.max() >= 255:
#             #     image = image / 255
#             fused = np.clip(np.sum(selected, axis=0), 0.0, 1.0)
#         elif self.mode == "mean":
#             fused = np.mean(selected, axis=0)
#         elif self.mode == "max":
#             fused = np.max(selected, axis=0)
#         else:
#             raise ValueError(f"Modo inválido: {self.mode}. Escolha 'sum', 'mean' ou 'max'.")
#
#         # Converte de volta para uint8 (se necessário)
#         fused_uint8 = (fused * 255.0).astype(np.uint8)
#
#         return fused_uint8
        # if self.mode == "sum":
        #     fused = np.clip(
        #         np.sum([color_map[space_color] for space_color in self.space_colors], axis=0),
        #         a_min=0.0,
        #         a_max=1.0
        #     )
        #     # fused = np.clip(img_rgb + img_xyz, 0.0, 1.0)
        # elif self.mode == "mean":
        #     fused = np.mean([color_map[space_color] for space_color in self.space_colors], axis=0)
        # elif self.mode == "max":
        #     fused = np.max([color_map[space_color] for space_color in self.space_colors], axis=0)
        # else:
        #     raise ValueError(f"Modo inválido: {self.mode}. Escolha 'sum', 'mean' ou 'max'.")
        #
        # return fused

class RGB2Fusion(ImageOnlyTransform):
    def __init__(self, mode="sum", p=1.0, space_colors=None, normalize_input=True):
        """
        mode: 'sum', 'mean', or 'max'
        normalize_input: Se True, normaliza a imagem de entrada para [0,1]
        """
        super().__init__(p=p)
        if space_colors is None:
            space_colors = ["rgb", "xyz"]
        self.space_colors = space_colors
        self.mode = mode
        self.normalize_input = normalize_input

    def normalize_color_space(self, img, space_name):
        """Normaliza cada espaço de cor para [0,1]"""
        if space_name == "lab":
            # LAB: L[0,100], a,b[-128,127]
            normalized = img.copy()
            normalized[:, :, 0] = img[:, :, 0] / 100.0  # L
            normalized[:, :, 1] = (img[:, :, 1] + 128) / 255.0  # a
            normalized[:, :, 2] = (img[:, :, 2] + 128) / 255.0  # b
            return normalized
        elif space_name == "hsv":
            # HSV: H[0,360], S,V[0,1]
            normalized = img.copy()
            normalized[:, :, 0] = img[:, :, 0] / 360.0  # H
            return normalized
        else:
            # Para RGB, XYZ, HED, LUV - normalização min-max
            min_val = np.min(img, axis=(0, 1), keepdims=True)
            max_val = np.max(img, axis=(0, 1), keepdims=True)

            # Evita divisão por zero
            range_val = max_val - min_val
            range_val = np.where(range_val > 1e-6, range_val, 1.0)

            return (img - min_val) / range_val

    def apply(self, image, **params):
        img_rgb = image.astype(np.float32)

        if self.normalize_input:
            img_rgb /= 255.0

        color_map = {
            "rgb": img_rgb,
            "xyz": color.rgb2xyz(img_rgb),
            "hed": color.rgb2hed(img_rgb),
            "lab": color.rgb2lab(img_rgb),
            "luv": color.rgb2luv(img_rgb),
            "hsv": color.rgb2hsv(img_rgb)
        }

        normalized_spaces = []
        for space_color in self.space_colors:
            if space_color not in color_map:
                raise ValueError(f"Espaço de cor '{space_color}' não suportado")

            space_img = color_map[space_color]
            normalized_img = self.normalize_color_space(space_img, space_color)
            normalized_spaces.append(normalized_img)

        # Fusão
        if self.mode == "sum":
            fused = np.sum(normalized_spaces, axis=0)
            fused = fused / len(normalized_spaces)
        elif self.mode == "mean":
            fused = np.mean(normalized_spaces, axis=0)
        elif self.mode == "max":
            fused = np.max(normalized_spaces, axis=0)
        else:
            raise ValueError(f"Modo inválido: {self.mode}")

        fused = np.clip(fused, 0.0, 1.0)
        fused_uint8 = (fused * 255.0).astype(np.uint8)

        return fused_uint8


class RGB2FusionTorch(ImageOnlyTransform):
    def __init__(self, mode="sum", p=1.0, space_colors=None, normalize_input=True):
        super().__init__(p=p)
        if space_colors is None:
            space_colors = ["rgb", "xyz"]
        self.space_colors = space_colors
        self.mode = mode
        self.normalize_input = normalize_input

    def normalize_color_space(self, img, space_name):
        """img: tensor [B,C,H,W]"""
        if space_name == "lab":
            # LAB: L[0,100], a,b[-128,127]
            L, a, b = img[:, 0], img[:, 1], img[:, 2]
            L = L / 100.0
            a = (a + 128) / 255.0
            b = (b + 128) / 255.0
            return torch.stack([L, a, b], dim=1)
        elif space_name == "hsv":
            H, S, V = img[:, 0], img[:, 1], img[:, 2]
            H = H / 360.0
            return torch.stack([H, S, V], dim=1)
        else:
            # min-max normalização por canal
            min_val = img.amin(dim=(2, 3), keepdim=True)
            max_val = img.amax(dim=(2, 3), keepdim=True)
            range_val = torch.where((max_val - min_val) > 1e-6, max_val - min_val, torch.tensor(1.0, device=img.device))
            return (img - min_val) / range_val

    def apply(self, image, **params):
        # numpy -> torch -> cuda
        img_rgb = torch.from_numpy(image.astype("float32")).permute(2, 0, 1).unsqueeze(0).cuda()

        if self.normalize_input:
            img_rgb = img_rgb / 255.0

        # gerar mapas de cor
        color_map = {
            "rgb": img_rgb,
            "xyz": kc.rgb_to_xyz(img_rgb),
            "lab": kc.rgb_to_lab(img_rgb),
            # "hsv": kc.rgb_to_hsv(img_rgb),  # se precisar
        }

        normalized_spaces = []
        for space_color in self.space_colors:
            if space_color not in color_map:
                raise ValueError(f"Espaço de cor '{space_color}' não suportado")
            space_img = self.normalize_color_space(color_map[space_color], space_color)
            normalized_spaces.append(space_img)

        # stack e fusão
        stacked = torch.stack(normalized_spaces, dim=0)  # [N, B, C, H, W]

        if self.mode == "sum":
            fused = stacked.sum(dim=0) / len(normalized_spaces)
        elif self.mode == "mean":
            fused = stacked.mean(dim=0)
        elif self.mode == "max":
            fused = stacked.max(dim=0).values
        else:
            raise ValueError(f"Modo inválido: {self.mode}")

        fused = torch.clamp(fused, 0.0, 1.0)  # já no GPU
        fused = fused[0].permute(1, 2, 0)  # [H,W,C]

        # volta para numpy se a lib albumentations exigir
        return fused.cpu().numpy()
