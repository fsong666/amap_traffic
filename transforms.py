import torchvision as tv
import torch
import numpy as np
from PIL import Image


class Rescale(object):
    def __init__(self, output_size):
        assert isinstance(output_size, tuple)
        self.output_dize = output_size

    def __call__(self, x):
        # if sample['frames'][0].size[:2] != self.output_dize:
        # print('rescale!!!')
        # print('input img.size=', sample['frames'][0].size)
        new_h, new_w, = self.output_dize
        new_h, new_w = int(new_h), int(new_w)

        # frames = []
        # for img in sample['frames']:
        #     image = img.resize((new_w, new_h), Image.BILINEAR)
        #     # print('output img.size=', image.size)
        #     frames.append(image)
        # # print('output img.size=', sample['frames'][0].size)
        # sample['frames'] = frames

        return x.resize((new_w, new_h), Image.BILINEAR)


class To_Tensor(object):

    def __init__(self, test=False):
        self.test = test

    def __call__(self, x):
        totensor = tv.transforms.ToTensor()
        # frames = []
        # for img in sample['frames']:
        #     frames.append(totensor(img))
        # sample['frames'] = frames
        #
        # if not self.test:
        #     target = torch.zeros(3)
        #     target[sample['status']] = 1.0
        #     sample['status'] = target

        return totensor(x)


class To_Tuple(object):
    def __init__(self, keys):
        assert isinstance(keys, tuple)
        self.keys = keys

    def __call__(self, sample):
        output = []
        for k in self.keys:
            output.append(sample[k])

        return output


class Normalize(object):
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.normalize = tv.transforms.Normalize(mean=mean, std=std)

    def __call__(self, x):
        # frames = []
        # for img in sample['frames']:
        #     frames.append(self.normalize(img))
        #
        # sample['frames'] = frames

        return self.normalize(x)
