import argparse
from data import Dataset_test
from model import LSTM
from torch.utils.data import DataLoader
import torch
import torchvision as tv
import time
import os
import json

PATH = '.'
PATH_json = PATH + '/dataset/amap_traffic_annotations_test.json'
PATH_imgs = PATH + '/dataset/amap_traffic_test_0712'


def test():
    parser = argparse.ArgumentParser(
        description='Status of Traffic Training')

    parser.add_argument('--cpu', action='store_true', default=False)

    args = parser.parse_args()

    args.savedir = os.path.join(os.getcwd(), 'save')
    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)

    if args.cpu:
        args.device = 'cpu'
        args.device_count = 1
    else:
        args.device = 'cuda'
        args.device_count = torch.cuda.device_count()

    data_transforms = [
        # tv.transforms.Resize((720, 1280)),
        tv.transforms.Resize((224, 224)),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize([0.4451, 0.4687, 0.4713], [0.3244, 0.3277, 0.3358]),
    ]

    dataset_val = Dataset_test(PATH_json,
                               PATH_imgs,
                               transform=tv.transforms.Compose(data_transforms))

    dl_val = DataLoader(dataset_val,
                        batch_size=1,
                        shuffle=False,
                        num_workers=4)

    model = LSTM()
    model.load_state_dict(torch.load(os.path.join(args.savedir, 'best_model.pth')))
    model.eval()
    model.to(args.device)

    print('Started Test')
    predict_list = []
    start_time = time.time()
    with torch.no_grad():
        for i, (inp, _) in enumerate(dl_val):
            inp = inp.to(args.device)

            out = model(inp)

            predict = torch.argmax(out).item()
            predict_list.append(predict)
            print('out: {} | predict: {}'.format(out, predict))

    print('Time: {:02}:{:02}|Predict_list:\n {} '
          .format(int((time.time() - start_time) // 60),
                  int((time.time() - start_time) % 60),
                  predict_list, flush=True))

    write_json(savedir=args.savedir, 
               json_file=PATH + '/amap_traffic_annotations_test.json', pred_list=predict_list)
    return predict_list


def write_json(savedir, json_file='test_origin.json',
               file_name='submit.json', pred_list=None):
    with open(json_file, 'r') as f:
        dict = json.load(f)

    for id, (status) in enumerate(pred_list):
        dict['annotations'][id]['status'] = status

    with open(os.path.join(savedir, file_name), 'w') as f:
        json.dump(dict, f, indent=4)
        print('Wrote {}'.format(file_name))


if __name__ == '__main__':
    test()
    # savedir = '/home/sf/Documents/tianchi/save/'
    # pred_list = np.random.randint(0, 3, 600).tolist()
    # write_json(savedir=savedir, json_file=PATH + '/amap_traffic_annotations_test.json', pred_list=pred_list)