from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
import numpy as np
import torchvision.transforms as T


class PatchBaseDataset(Dataset):
    normalization_constants = [[0.8740, 0.8136, 0.7095], [0.1190, 0.1211, 0.1258]]

    def __init__(self,
                 patch_size,
                 augment=False,
                 sub_batch=1,
                 patch_overlap=(0, 0)):
        self.augment = augment
        self.patch_size = patch_size
        self.patch_overlap = patch_overlap
        self.sub_batch = sub_batch

    def transform_image(self, image):
        # PIL Image has size (w, h)

        if image.mode == 'RGBA':
            image = self.alpha_to_color(image)

        if self.augment:
            A_transforms = A.Compose([
                A.CLAHE(clip_limit=(1, 1), always_apply=True)
            ])
            image_np = np.array(image)
            image = A_transforms(image=image_np)['image']
            image = Image.fromarray(image)

        transform_im = T.Compose([
            T.ToTensor(),
            T.Normalize(self.normalization_constants[0], self.normalization_constants[1])
        ])
        image = transform_im(image)
        return image

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