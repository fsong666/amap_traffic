from sklearn.model_selection import train_test_split
import numpy as np
import os
import json

def test():
    n_samples = 10
    n_dims = 3
    X = np.arange(n_samples * n_dims).reshape(n_samples, n_dims)
    y = np.arange(n_samples)
    print('X=\n', X)
    print('y=\n', y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=32, test_size=0.3)
    print('X_train=\n', X_train)
    print('y_train=\n', y_train)
    print('X_test=\n', X_test)
    print('y_test=\n', y_test)


def findlabels(data_path):
    train_label = []
    f = open(data_path, 'r', encoding='utf-8')
    status = json.load(f)
    c = 0
    while c < 1500:
        eg = status['annotations'][c]['status']
        train_label.append(eg)
        c += 1
    return train_label


def test2():
    data_path = '/home/sf/Documents/tianchi/dataset/amap_traffic_train_0712'
    all_list = os.listdir(data_path)

    all_list.remove('.DS_Store')
    # print('all_list=\n', all_list)
    print('all_list.len=\n', len(all_list))

    all_label = findlabels("/home/sf/Documents/tianchi/dataset/amap_traffic_annotations_train.json")
    print('all_label=\n', len(all_label))

    train_list, test_list, train_label, test_label = train_test_split(all_list, all_label, test_size=0.20,
                                                                      random_state=3,
                                                                      stratify=all_label)
    print('train_list.len=\n', len(train_list))
    print('train_label.len=\n', len(train_label))
    print('test_list.len=\n', len(test_list))
    print('test_label.len=\n', len(test_label))

    print('train_list.len=\n', train_list)
    print('train_label.len=\n', train_label)
    # print('test_list.len=\n', len(test_list))
    # print('test_label.len=\n', len(test_label))
    #


test2()