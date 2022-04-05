from datasets.htr.losen_base_dataset import LosenBaseDataset
import os
from PIL import Image
import utils.datafuncs as df
import torchvision.transforms as T


class InferenceSet(LosenBaseDataset):
    def __init__(self,
                 xml_folder,
                 image_folder,
                 tokenizer,
                 im_size=(128, 64),
                 pad_text=50,
                 augment=False,
                 deterministic=True,
                 normalize=True):
        super(InferenceSet, self).__init__(im_size=im_size,
                                           pad_text=pad_text,
                                           augment=augment,
                                           tokenizer=tokenizer,
                                           normalize=normalize)
        self.xml_folder = xml_folder
        self.image_folder = image_folder
        self.deterministic = deterministic
        self.im_size = im_size
        self.data = self.construct_data()

    def construct_data(self):
        data = list()
        for root, dirs, files in os.walk(self.xml_folder):
            for file in files:
                file_name, file_ext = os.path.splitext(file)
                if file_ext == '.xml':
                    segmentations, file_info = df.page_parser(os.path.join(root, file))
                    for segmentation in segmentations:
                        d = dict()
                        poly_coords = [[int(p) for p in points.split(',')] for points in
                                       segmentation['poly_coords'].split()]
                        baseline_coords = [[int(p) for p in points.split(',')] for points in
                                           segmentation['baseline_coords'].split()]
                        xs = [x for x, _ in poly_coords]
                        ys = [y for _, y in poly_coords]
                        min_x, max_x, min_y, max_y = min(xs), max(xs), min(ys), max(ys)
                        region, image_number = file_name.split('_')
                        d['path'] = os.path.join(self.image_folder, region, region + '_' + image_number + '.jpg')
                        d['poly'] = poly_coords
                        d['baseline'] = baseline_coords
                        d['crop'] = (min_x, min_y, max_x, max_y)
                        d['file_name'] = file_info['imageFilename']
                        d['width'] = file_info['imageWidth']
                        d['height'] = file_info['imageHeight']
                        data.append(d)
        return data

    def transform_image(self, image):
        # PIL Image has size (w, h)
        # transforms.Resize take size = (h, w)
        if image.mode == 'RGBA':
            image = self.alpha_to_color(image)

        height_scale = self.im_size[1] / float(image.size[1])
        size = (self.im_size[1], int(image.size[0] * height_scale))
        pad_length = self.im_size[0] - size[1]
        transform = T.Compose([
            T.Resize(size, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(self.normalization_constants[0], self.normalization_constants[1]),
            T.Pad((0, 0, pad_length, 0), fill=1)
        ])
        image = transform(image)
        return image, height_scale, pad_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, indx):
        image = Image.open(self.data[indx]['path'])
        image = image.crop(self.data[indx]['crop'])
        image, height_scale, pad_length = self.transform_image(image)
        return image, indx, height_scale, pad_length