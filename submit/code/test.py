import argparse
from data import Dataset_test
from model import LSTM
from torch.utils.data import DataLoader
import torch
import torchvision as tv
import time
import os
import json

# path of project
PATH = '..'
PATH_json = PATH + '/data/amap_traffic_annotations_b_test_0828.json'
PATH_imgs = PATH + '/data/amap_traffic_b_test_0828'
PATH_model = PATH + '/user_data/model_data'
PATH_prediction_result = PATH + '/prediction_result'


def test():
    parser = argparse.ArgumentParser(
        description='Status of Traffic Training')

    parser.add_argument('--cpu', action='store_true', default=False)

    args = parser.parse_args()

    if args.cpu:
        print('Use cpu')
        args.device = 'cpu'
        args.device_count = 1
    else:
        print('Use cuda')
        args.device = 'cuda'
        args.device_count = torch.cuda.device_count()

    data_transforms = [
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
    model.load_state_dict(torch.load(os.path.join(PATH_model, 'model.pth')))
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

    print('Time: {:02}:{:02}|Predict_list:\n {} '
          .format(int((time.time() - start_time) // 60),
                  int((time.time() - start_time) % 60),
                  predict_list, flush=True))

    write_json(savedir=PATH_prediction_result,
               file_name='result.json',
               json_file=PATH_json,
               pred_list=predict_list)

    return predict_list


def write_json(savedir, json_file='test_origin.json',
               file_name='submit.json', pred_list=None):
    with open(json_file, 'r') as f:
        dict = json.load(f)

    for id, (status) in enumerate(pred_list):
        dict['annotations'][id]['status'] = status

    with open(os.path.join(savedir, file_name), 'w') as f:
        json.dump(dict, f, indent=4)
        print('Wrote {} \nEnd test'.format(file_name))


test()

