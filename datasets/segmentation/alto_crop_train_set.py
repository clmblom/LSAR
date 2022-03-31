from .crop_base_dataset import CropBaseDataset
import os
from utils.polyfuncs import points_to_mask
import re
from utils.datafuncs import alto_parser
from PIL import Image


class AltoCropTrainSet(CropBaseDataset):
    def __init__(self,
                 alto_folder,
                 image_folder,
                 augment=False,
                 deterministic=True,
                 crop_size=(0, 0),
                 dataset_size=1000,
                 shrink_ratio=0.4,
                 area_thresh=float('inf')):
        super(AltoCropTrainSet, self).__init__(augment=augment,
                                               deterministic=deterministic,
                                               crop_size=crop_size)
        self.alto_folder = alto_folder
        self.image_folder = image_folder
        self.data = self.construct_data()
        self.shrink_ratio = shrink_ratio
        self.dataset_size = dataset_size
        self.area_thresh = area_thresh

    def construct_data(self):
        data = dict()
        data['path'] = []
        data['gt'] = []
        for root, dirs, files in os.walk(self.alto_folder):
            if os.path.split(root)[-1] == 'alto':
                for file in files:
                    image_name = re.findall(r'.*?_?(B.+).xml', file)
                    data['gt'].append(alto_parser(os.path.join(root, file)))
                    data['path'].append(
                        os.path.join(self.image_folder, image_name[0].split('_')[0], image_name[0] + '.jpg'))
        return data

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, indx):
        indx = indx % len(self.data['path'])
        polygons = self.get_polygon_points(indx)
        image = Image.open(self.data['path'][indx])
        mask = points_to_mask(polygons, image.size, shrink_ratio=self.shrink_ratio, area_thresh=self.area_thresh)
        image, mask = self.transform_image(image, mask)
        return image, mask

    def get_polygon_points(self, indx):
        points = []
        words = self.data['gt'][indx]
        for i, word in enumerate(words):
            add_width = 0.1 * word['width']
            add_height = 0.0 * word['height']
            width = word['width'] + add_width
            height = word['height'] + add_height
            vpos = word['vpos'] - add_height // 2
            hpos = word['hpos'] - add_width // 2
            p1 = (hpos, vpos)
            p2 = (hpos, vpos + height)
            p3 = (hpos + width, vpos + height)
            p4 = (hpos + width, vpos)
            points.append([p1, p2, p3, p4])
        return points