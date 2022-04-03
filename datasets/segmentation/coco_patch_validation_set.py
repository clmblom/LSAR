from .patch_base_dataset import PatchBaseDataset
import json
import os
from PIL import Image
import math


class CocoPatchValidationSet(PatchBaseDataset):
    def __init__(self,
                 json_path,
                 image_folder,
                 patch_size,
                 augment=False,
                 sub_batch=1,
                 patch_overlap=(0, 0)):
        super(CocoPatchValidationSet, self).__init__(augment=augment,
                                                     sub_batch=sub_batch,
                                                     patch_size=patch_size,
                                                     patch_overlap=patch_overlap)
        self.json_path = json_path
        self.image_folder = image_folder
        self.data = self.construct_data()

    def construct_data(self):
        data = dict()
        data['path'] = []
        data['gt'] = []
        with open(self.json_path) as f:
            d = json.load(f)
        for image in d['images']:
            image_name = image['file_name'].split('\\')[-1]
            image_folder = image_name.split('_')[0]
            image_id = image['id']
            segmentations = [x['segmentation'][0] for x in d['annotations'] if x['image_id'] == image_id]
            data['path'].append(os.path.join(self.image_folder, image_folder, image_name))
            data['gt'].append(segmentations)
        return data

    def __len__(self):
        return len(self.data['path'])

    def __getitem__(self, indx):
        polygons = self.get_polygon_points(indx)
        image = Image.open(self.data['path'][indx])
        image = self.transform_image(image)
        patch_size = self.patch_size.copy()
        if patch_size[0] == -1:
            patch_size[0] = (image.shape[1]//64)*64
        if patch_size[1] == -1:
            patch_size[1] = (image.shape[2]//64)*64
        return image, polygons, patch_size, self.patch_overlap, self.sub_batch

    def get_polygon_points(self, indx):
        points = []
        segmentations = self.data['gt'][indx]
        for seg in segmentations:
            poly = []
            for i in range(0, len(seg), 2):
                poly.append((math.floor(seg[i]), math.floor(seg[i + 1])))
            points.append(poly)
        return points
