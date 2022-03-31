from datasets.htr.losen_base_dataset import LosenBaseDataset
import os
from PIL import Image


class DeterministicSet(LosenBaseDataset):
    def __init__(self,
                 data_path,
                 tokenizer,
                 im_size=(128, 64),
                 pad_text=50,
                 images='',
                 delimiter=' ',
                 gt='gt.txt',
                 normalize=True):
        super(DeterministicSet, self).__init__(im_size=im_size,
                                               pad_text=pad_text,
                                               augment=False,
                                               tokenizer=tokenizer,
                                               normalize=normalize)
        self.data_path = data_path
        self.images = images
        self.gt_file = gt
        self.data = self.construct_data(delimiter)

    def construct_data(self, delimiter):
        data = dict()
        data['path'] = []
        data['word'] = []

        with open(os.path.join(self.data_path, self.gt_file), encoding='utf-8-sig') as f:
            for line in f:
                # Assumes   image_name word1 word2 word3...
                splitted_line = line.split(delimiter, 1)
                if len(splitted_line) == 1: # The line has no words
                    print(splitted_line[0].strip(), " does not have words")
                    continue
                else:
                    file_name = splitted_line[0]
                    word = splitted_line[1]
                word = ''.join(word).strip()
                file_path = os.path.join(self.data_path, self.images, file_name)
                data['path'].append(file_path)
                data['word'].append(word)
        return data

    def __len__(self):
        return len(self.data['path'])

    def __getitem__(self, indx):
        image_path = self.data['path'][indx]
        target = self.data['word'][indx]
        image = Image.open(image_path)
        target = self.transform_text(target)
        image = self.transform_image(image)
        return image, target, 0, 0
