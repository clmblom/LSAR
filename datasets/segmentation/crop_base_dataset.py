from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random


class CropBaseDataset(Dataset):
    normalization_constants = [[0.8740, 0.8136, 0.7095], [0.1190, 0.1211, 0.1258]]

    def __init__(self,
                 augment=False,
                 deterministic=True,
                 crop_size=(0, 0)):
        self.augment = augment
        self.deterministic = deterministic
        self.crop_size = crop_size

    def transform_image(self, image, mask):
        # PIL Image has size (w, h)

        if image.mode == 'RGBA':
            image = self.alpha_to_color(image)

        if self.crop_size[0] or self.crop_size[1]:
            if image.size[0] < self.crop_size[1]:
                w_diff = self.crop_size[1] - image.size[0]
                w_diff = w_diff // 2 + 1
                image_pad = T.Pad((w_diff, 0, w_diff, 0), fill=255)
                mask_pad = T.Pad((w_diff, 0, w_diff, 0), fill=0)
                image = image_pad(image)
                mask = mask_pad(mask)
            if image.size[1] < self.crop_size[0]:
                h_diff = self.crop_size[0] - image.size[1]
                h_diff = h_diff // 2 + 1
                image_pad = T.Pad((0, h_diff, 0, h_diff), fill=255)
                mask_pad = T.Pad((0, h_diff, 0, h_diff), fill=0)
                image = image_pad(image)
                mask = mask_pad(mask)
            i, j, h, w = T.RandomCrop.get_params(image, output_size=self.crop_size)
            image = TF.crop(image, i, j, h, w)
            mask = TF.crop(mask, i, j, h, w)

        if self.augment:
            random.seed()
            A_transforms = A.Compose([
                A.CLAHE(p=1),
                A.GaussNoise(p=0.5)
            ])
            image_np = np.array(image)
            image = A_transforms(image=image_np)['image']
            image = Image.fromarray(image)

        transform_im = T.Compose([
            T.ToTensor(),
            T.Normalize(self.normalization_constants[0], self.normalization_constants[1])
        ])
        transform_ma = T.Compose([
            T.ToTensor(),
        ])

        image = transform_im(image)
        mask = transform_ma(mask)
        return image, mask

    def __getitem__(self, index):
        pass

    def alpha_to_color(self, image, color=(255, 255, 255)):
        """Alpha composite an RGBA Image with a specified color.
        Simpler, faster version than the solutions above.
        Source: http://stackoverflow.com/a/9459208/284318

        Keyword Arguments:
        image -- PIL RGBA Image object
        color -- Tuple r, g, b (default 255, 255, 255)

        """
        image.load()  # needed for split()
        background = Image.new('RGB', image.size, color)
        background.paste(image, mask=image.split()[3])  # 3 is the alpha channel
        return background