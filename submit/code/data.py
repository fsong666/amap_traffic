import torch
from torch.utils import data
import json
from PIL import Image
import os


class Dataset_test(data.Dataset):
    def __init__(self, json_file, dataset_dir, transform=None):
        self.json_file = json_file
        with open(self.json_file, 'r') as f:
            dict = json.load(f)
        self.dict_list = dict['annotations']
        self.transform = transform
        self.dataset_dir = dataset_dir

    def __len__(self):
        return len(self.dict_list)

    def _read_images(self, frame_list, id):
        frames = []
        i = 0
        for img_dict in frame_list:
            i += 1
            if i > 3:
                break
            frame_name = img_dict["frame_name"]
            img_name = os.path.join(self.dataset_dir, id, frame_name)
            img = Image.open(img_name)

            if self.transform is not None:
                img = self.transform(img)

            frames.append(img)

        return frames

    def __getitem__(self, index):
        dict = self.dict_list[index]

        status = dict["status"]

        id = dict["id"]

        frame_list = dict["frames"]
        frames = self._read_images(frame_list, id)
        x_3d = torch.stack(frames, dim=0)

        label = torch.tensor(status)
        return x_3d, label


