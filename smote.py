from imblearn.over_sampling import SMOTE
import torch
from collections import Counter
from sklearn.datasets import make_classification
from matplotlib import pyplot
from numpy import where
import torch

import random
from sklearn.neighbors import NearestNeighbors
import numpy as np
import os


def smote(k=1):
    """
    n_samples * Wmin >= k_neighbors + 1
    会将少数类的数目 == 最大样本类的数目
    原来的样本不会变，只是新增少数类的样本

    新增的样本会后插在原来的二维矩阵数据集的后面
    返回数据类型是numpy
    """
    X, y = make_classification(n_classes=3, class_sep=2,
                               weights=[0.7, 0.1, 0.2], n_informative=3, n_redundant=1, flip_y=0,
                               n_features=5, n_clusters_per_class=1, n_samples=30, random_state=20)
    # X, y = torch.tensor(X), torch.tensor(y)
    counter1 = Counter(y)
    print('X.type=\n', type(X))
    print('x.shape=\n', X.shape)
    print('x=\n', X)
    print('y.shape=\n', y.shape)
    print('y=\n', y)

    print('input dataset shape %s' % counter1)
    print_plot(X, y, counter1)

    sm = SMOTE(random_state=42, k_neighbors=k)
    X_res, y_res = sm.fit_resample(X, y)

    print('X_res.type=\n', type(X_res))
    print('X_res.shape=\n', X_res.shape)
    print('X_res=\n', X_res)
    print('y.shape=\n', y_res.shape)
    print('y_res =\n', y_res)

    counter2 = Counter(y_res)
    print('Resampled dataset shape %s' % counter2)
    print_plot(X_res, y_res, counter2)


def print_plot(X, y, counter):
    # scatter plot of examples by class label
    for label, _ in counter.items():
        row_ix = where(y == label)[0]
        pyplot.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
    pyplot.legend()
    pyplot.show()

# smote()


# SMOTE算法及其python实现
class Smote:
    def __init__(self, samples, N=2, k=5):
        """

        :param samples:
        :param N: 倍数, 扩大为原数据数目的N倍
        :param k: k,个近邻
        """
        self.n_samples, self.n_dims = samples.shape
        self.N = N
        self.k = k
        self.samples = samples
        self.newindex = 0
        print('n_sample=\n ', self.n_samples)
        print('n_attrs=\n ', self.n_dims)

    # self.synthetic=np.zeros((self.n_samples*N,self.n_attrs))

    def over_sampling(self):
        # N = int(self.N / 100)
        print('N= ', self.N)

        self.synthetic = np.zeros((self.N * self.n_samples, self.n_dims))
        neighbors = NearestNeighbors(n_neighbors=self.k).fit(self.samples)
        print('neighbors.index= ', neighbors)

        for i in range(len(self.samples)):
            print('sample= ', self.samples[i])
            nnarray = neighbors.kneighbors(self.samples[i].reshape((1, -1)),
                                           return_distance=False)[0]  # Finds the K-neighbors of a point.
            print('nnarray = ', nnarray)
            self._populate(i, nnarray)
        return self.synthetic

    def _populate(self, i, nnarray):
        """
        for each minority class sample i ,choose N of the k nearest neighbors
        and generate N synthetic samples.
        """
        for n in range(self.N):
            print('n= ', n)
            nn = random.randint(0, self.k - 1)  # 包括end
            dif = self.samples[nnarray[nn]] - self.samples[i]
            gap = random.random()
            self.synthetic[self.newindex] = self.samples[i] + gap * dif
            print('newindex= ', self.newindex)
            self.newindex += 1


def test_smote():
    a = np.array([[1, 2, 3], [4, 5, 6], [2, 3, 1], [2, 1, 2], [2, 3, 4], [2, 3, 4]])
    s = Smote(a, N=3)
    out = s.over_sampling()
    print('input=\n ', a)
    print('input.shape=\n ', a.shape)
    print('out.shape= ', out.shape)
    print('out=\n ', out)


def store_smote_data():
    x = torch.rand(3, 10)
    y = torch.randint(0, 3, (1, 3)).squeeze()

    print('store new data_set')
    savedir = os.path.join(os.getcwd(), 'smote_save')
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    torch.save(x, os.path.join(savedir, 'x.pth'))



if __name__ == '__main__':
    # test_smote()
    # smote()
    store_smote_data()