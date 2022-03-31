from datasets.htr.losen_base_dataset import LosenBaseDataset
import os
from PIL import Image
import random as r


class StreamingSet(LosenBaseDataset):
    def __init__(self,
                 data_path,
                 tokenizer,
                 im_size=(128, 64),
                 pad_text=50,
                 augment=False,
                 deterministic=True,
                 images='',
                 delimiter=' ',
                 dataset_size=None,
                 gt='gt.txt',
                 normalize=True,
                 early_stop_p=0,
                 pad_between=0):
        super(StreamingSet, self).__init__(im_size=im_size,
                                           pad_text=pad_text,
                                           augment=augment,
                                           tokenizer=tokenizer,
                                           normalize=normalize)
        self.data_path = data_path
        self.images = images
        self.gt_file = gt
        self.data = self.construct_data(delimiter)
        self.deterministic = deterministic
        self.dataset_size = dataset_size
        self.im_size = im_size
        self.early_stop_p = early_stop_p
        self.pad_between = pad_between

    def construct_data(self, delimiter):
        data = dict()
        data['path'] = []
        data['word'] = []

        with open(os.path.join(self.data_path, self.gt_file), encoding='utf-8-sig') as f:
            for line in f:
                # assumes that lines are "image_path word1 word2 word3..."
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
        if self.dataset_size and not self.deterministic:
            return self.dataset_size
        else:
            return len(self.data['path'])

    def __getitem__(self, indx):
        if not self.deterministic:
            indx = r.randint(0, len(self.data['path'])-1)

        image, target = self.create_sample(indx, self.pad_text, self.pad_between)

        target = self.transform_text(target)
        image = self.transform_image(image)
        return image, target, 0, 0

    def create_sample(self, indx, txt_len_left=0, pad_between=0):
        im_width_left = self.im_size[0]
        txt_len_left = txt_len_left
        images = []
        targets = ''
        while im_width_left > 0 and txt_len_left > 0:
            im = Image.open(self.data['path'][indx])
            new_height = self.im_size[1]
            new_width = int(im.size[0] * self.im_size[1] / float(im.size[1]))
            im = im.resize((new_width, new_height))

            target = self.data['word'][indx]
            im_w, im_h = im.size

            if len(images) == 0:  # Can always fill at least one word. If it's too big it gets resized later
                images.append(im)
                targets += self.data['word'][indx]
                indx = r.randint(0, len(self.data['path'])-1)
                im_width_left -= im_w + pad_between
                txt_len_left -= len(target)
            elif im_w < im_width_left and len(target) < txt_len_left + 1:
                images.append(im)
                targets += ' ' + self.data['word'][indx]
                indx = r.randint(0, len(self.data['path'])-1)
                im_width_left -= im_w + pad_between
                txt_len_left -= (len(target) + 1)
            else:
                im_width_left = 0
                txt_len_left = 0
            if r.random() < self.early_stop_p:
                im_width_left = 0
                txt_len_left = 0
        images = self.stitch_sample(images, pad_between)
        return images, targets

    def stitch_sample(self, images, pad_between=0):
        image_sizes_w = list(map(lambda x: x.size[0], images))
        image_sizes_h = list(map(lambda x: x.size[1], images))
        max_height = max(image_sizes_h)
        total_width = sum(image_sizes_w) + (len(images)-1)*pad_between
        im = Image.new('RGB', (total_width, max_height), color=(255, 255, 255))
        curr_w = 0
        for i, (image, w) in enumerate(zip(images, image_sizes_w)):
            im.paste(image, (curr_w, 0))
            curr_w += w + pad_between
        return im
