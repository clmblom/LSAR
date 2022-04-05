from .patch_base_dataset import PatchBaseDataset
import os
from PIL import Image


class InferencePatchSet(PatchBaseDataset):
    def __init__(self,
                 image_folder,
                 patch_size,
                 augment=False,
                 sub_batch=1,
                 patch_overlap=(0, 0)):
        super(InferencePatchSet, self).__init__(augment=augment,
                                                sub_batch=sub_batch,
                                                patch_size=patch_size,
                                                patch_overlap=patch_overlap)
        self.image_folder = image_folder
        self.data = self.construct_data()

    def construct_data(self):
        data = dict()
        data['path'] = []
        data['name'] = []
        for root, dirs, files in os.walk(self.image_folder):
            for file in files:
                file_name, file_ext = os.path.splitext(file)
                if file_ext == '.png' or file_ext == '.jpg':
                    data['path'].append(os.path.join(root, file))
                    data['name'].append(file_name)
        return data

    def __len__(self):
        return len(self.data['path'])

    def __getitem__(self, indx):
        image = Image.open(self.data['path'][indx])
        image = self.transform_image(image)
        patch_size = self.patch_size.copy()
        if patch_size[0] == -1:
            patch_size[0] = (image.shape[1]//64)*64
        if patch_size[1] == -1:
            patch_size[1] = (image.shape[2]//64)*64
        return image, indx, patch_size, self.patch_overlap, self.sub_batch
