from torch.utils.data import Dataset
import albumentations as A
import torchvision.transforms as T
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import random


class LosenBaseDataset(Dataset):
    normalization_constants = [[0.8436, 0.7788, 0.6695], [0.1518, 0.1558, 0.1578]]

    def __init__(self,
                 im_size,
                 pad_text,
                 augment,
                 tokenizer,
                 normalize=True):
        self.im_size = im_size
        self.pad_text = pad_text
        self.augment = augment
        self.tokenizer = tokenizer
        self.normalize = normalize

    def transform_image(self, image, return_height_scale=False):
        # PIL Image has size (w, h)
        # transforms.Resize take size = (h, w)
        if image.mode == 'RGBA':
            image = self.alpha_to_color(image)
        if self.augment:
            random.seed()
            A_transforms = A.Compose([
                A.RandomBrightnessContrast(p=0.5, brightness_limit=0.1),
                A.RandomGamma(p=0.5),
                A.CLAHE(p=0.5),
                A.GaussNoise(p=1),
                A.RandomSnow(p=0.5,
                             snow_point_lower=0.67,
                             snow_point_upper=0.68,
                             brightness_coeff=0),
                A.ElasticTransform(p=1,
                                   alpha=150,
                                   sigma=10,
                                   alpha_affine=0.5)],
            )
            image_np = np.array(image)
            image = A_transforms(image=image_np)['image']
            image = Image.fromarray(image)

        # Resize to height and keep aspect ratio
        height_scale = self.im_size[1] / float(image.size[1])
        size = (self.im_size[1], int(image.size[0] * height_scale))
        pad_length = self.im_size[0] - size[1]
        if self.normalize:
            transform = T.Compose([
                T.Resize(size, interpolation=T.InterpolationMode.BILINEAR),
                T.ToTensor(),
                T.Normalize(self.normalization_constants[0], self.normalization_constants[1]),
                T.Pad((0, 0, pad_length, 0), fill=1)
            ])
        else:
            transform = T.Compose([
                T.Resize(size, interpolation=T.InterpolationMode.BILINEAR),
                T.ToTensor(),
                T.Pad((0, 0, pad_length, 0), fill=1)
            ])
        image = transform(image)
        if return_height_scale:
            return image, height_scale, pad_length
        else:
            return image

    def transform_text(self, text):
        text = self.tokenizer.encode(text).ids
        text = torch.tensor(text)
        text = F.pad(text, (0, self.pad_text - text.shape[0] + 2),
                     value=self.tokenizer.encode('<P>').ids[1])  # Pad to pad_text + <S> + <E>
        return text

    def __getitem__(self, index):
        pass

    def alpha_to_color(self, image, color=(255, 255, 255)):
        image.load()
        background = Image.new('RGB', image.size, color)
        background.paste(image, mask=image.split()[3])  # 3 is the alpha channel
        return background
