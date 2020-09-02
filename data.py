import torch
from torch.utils import data
import json
from PIL import Image
import os
import numpy as np
from torch.utils.data import DataLoader
import torchvision as tv
from sklearn.model_selection import train_test_split
from collections import Counter
from imblearn.over_sampling import SMOTE

PATH = '.'
PATH_json = PATH + '/dataset/amap_traffic_annotations_train.json'
PATH_imgs = PATH + '/dataset/amap_traffic_train_0712'


class Dataset(data.Dataset):
    def __init__(self, json_file, dataset_dir, folders, transform=None, show=False):
        self.json_file = json_file
        with open(self.json_file, 'r') as f:
            dict = json.load(f)
        self.dict_list = dict['annotations']
        self.transform = transform
        self.dataset_dir = dataset_dir
        self.show = show
        self.folders = folders

    def __len__(self):
        return len(self.folders)

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

            if self.show:
                print('img.type= ', type(img))
                print('img.shape= ', np.array(img).shape)
                img.show()

            if self.transform is not None:
                img = self.transform(img)

            frames.append(img)

        return frames

    def __getitem__(self, index):
        id = self.folders[index]
        dict = self.dict_list[int(id) - 1]
        status = dict["status"]
        # print('id = {} status= {}'.format(id, status))

        frame_list = dict["frames"]
        frames = self._read_images(frame_list, id)
        x_3d = torch.stack(frames, dim=0)

        label = torch.tensor(status)
        return x_3d, label


class Dataset_test(data.Dataset):
    def __init__(self, json_file, dataset_dir, transform=None, show=False):
        self.json_file = json_file
        with open(self.json_file, 'r') as f:
            dict = json.load(f)
        self.dict_list = dict['annotations']
        self.transform = transform
        self.dataset_dir = dataset_dir
        self.show = show

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

            if self.show:
                print('img.type= ', type(img))
                print('img.shape= ', np.array(img).shape)
                img.show()

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


def get_data(val_split=0.2, random_state=3):
    data_transforms = [
        tv.transforms.Resize((224, 224)),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize([0.4451, 0.4687, 0.4713], [0.3244, 0.3277, 0.3358]),
    ]

    all_list = os.listdir(PATH_imgs)
    all_list.remove('.DS_Store')

    train_list, val_list = train_test_split(all_list, test_size=val_split, random_state=random_state)

    dataset_train = Dataset(PATH_json, PATH_imgs, train_list,
                            transform=tv.transforms.Compose(data_transforms))

    dataset_val = Dataset(PATH_json, PATH_imgs, val_list,
                          transform=tv.transforms.Compose(data_transforms))

    return dataset_train, dataset_val


def get_random_split(val_split=0.2):
    data_transforms = [
        tv.transforms.Resize((224, 224)),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize([0.4451, 0.4687, 0.4713], [0.3244, 0.3277, 0.3358]),
    ]

    dataset_train = Dataset_test(PATH_json,
                                 PATH_imgs,
                                 transform=tv.transforms.Compose(data_transforms),
                                 show=False)

    train_size = int((1 - val_split) * len(dataset_train))
    val_size = int(val_split * len(dataset_train))
    split_train, split_val = data.random_split(dataset_train, [train_size, val_size])

    return split_train, split_val


def load_data(args):
    dataset_train, dataset_val = get_data(val_split=args.val_split)
    # dataset_train, dataset_val = get_random_split(val_split=args.val_split)

    dl_train = DataLoader(dataset_train,
                          batch_size=args.bs_mult,
                          shuffle=True,
                          num_workers=4)

    dl_val = DataLoader(dataset_val,
                        batch_size=1,
                        shuffle=False,
                        num_workers=4)

    print('Loaded data dl_train: {}, dl_val: {}'.format(len(dl_train), len(dl_val)))
    return dl_train, dl_val


def generate_smote_sample(data_set=None, x_name='smote_x.pth', y_name='smote_y.pth'):
    total = len(data_set)
    X = torch.zeros(total, 3 * 3 * 224 * 224)
    label = []
    for i, (x, y) in enumerate(data_set):
        X[i] = x.reshape(1, -1)
        label.append(y.item())

    print(X.shape)
    print(label)

    print('input dataset shape = ', Counter(label))
    sm = SMOTE(random_state=42, k_neighbors=3)
    X_res, y_res = sm.fit_resample(X, label)
    print('Resampled dataset shape = ', Counter(y_res))
    print(X_res.shape)

    savedir = os.path.join(os.getcwd(), 'smote_save')
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    tensor_x = torch.tensor(X_res)
    tensor_y = torch.tensor(y_res)
    torch.save(tensor_x, os.path.join(savedir, x_name))

    torch.save(tensor_y, os.path.join(savedir, y_name))
    print('stored new data_set')


class CRNNdatasets(data.Dataset):
    def __init__(self, X_smo, y_smo):
        self.X = X_smo
        self.Y = y_smo

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index].reshape(4, 3, 224, 224), self.Y[index]


def load_smote_data(args):
    train_x_smo = torch.load(os.path.join(args.savedir, 'new_x.pth'))
    train_y_smo = torch.load(os.path.join(args.savedir, 'new_y.pth'))

    val_x_smo = torch.load(os.path.join(args.savedir, 'new_test_x.pth'))
    val_y_smo = torch.load(os.path.join(args.savedir, 'new_test_y.pth'))

    # smote_save_dir = '/home/sf/Documents/tianchi/smote_save'
    # train_x_smo = torch.load(os.path.join(smote_save_dir, 'smote_x.pth'))
    # train_y_smo = torch.load(os.path.join(smote_save_dir, 'smote_y.pth'))
    #
    # val_x_smo = torch.load(os.path.join(smote_save_dir, 'smote_x_val.pth'))
    # val_y_smo = torch.load(os.path.join(smote_save_dir, 'smote_y_val.pth'))

    train_smodatasets = CRNNdatasets(train_x_smo, train_y_smo)
    val_smodatasets = CRNNdatasets(val_x_smo, val_y_smo)

    dl_train = DataLoader(train_smodatasets,
                          batch_size=args.bs_mult,
                          shuffle=True,
                          drop_last=True,  # 需整除，防止BN报错
                          num_workers=6)

    dl_val = DataLoader(val_smodatasets,
                        batch_size=1,
                        shuffle=True,
                        num_workers=4)

    print('Loaded smote_data dl_train: {}, dl_val: {}'.format(len(dl_train), len(dl_val)))
    return dl_train, dl_val


def test():
    data_transforms = [
        tv.transforms.Resize((720, 1280)),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize([0.4451, 0.4687, 0.4713], [0.3244, 0.3277, 0.3358]),
    ]

    dataset_train = Dataset_test(PATH_json,
                                 PATH_imgs,
                                 transform=tv.transforms.Compose(data_transforms),
                                 show=False)

    dl_train = DataLoader(dataset_train,
                          batch_size=4,  # x 是tensor的list,不能拼接合并
                          shuffle=True,
                          drop_last=True,
                          num_workers=1)

    print('len=', len(dl_train))
    for i, (inp, label) in enumerate(dl_train):
        print('{} | in.shape= {} | label= {}'.format(i, inp.shape, label))


if __name__ == '__main__':
    test()
    # generate_smote_sample(get_data()[0])
    # generate_smote_sample(get_data()[1], x_name='smote_x_val.pth', y_name='smote_y_val.pth')
