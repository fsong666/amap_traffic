#!/usr/bin/env python
# coding: utf-8

# In[4]:


import os
import numpy as np
from PIL import Image
from torch.utils import data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
# from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import numpy as np
import torchvision
from torchvision import datasets, transforms, models
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import time
import os
import copy
import json

# from sklearn.model_selection import train_test_split


BATCH_SIZE = 48
learning_rate = 0.001
num_epochs = 53


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


data_path = "/content/drive/My Drive/Colab Notebooks/amap_traffic_train_0712"
all_list = os.listdir(data_path)
all_label = findlabels("/content/drive/My Drive/Colab Notebooks/traffic.json")
train_list, test_list, train_label, test_label = train_test_split(all_list, all_label, test_size=0.20, random_state=42)
selected_frames = ["1.jpg", "2.jpg", "3.jpg", "4.jpg", "5.jpg"]
img_x = 224
img_y = 224
transform = transforms.Compose([transforms.Resize([img_x, img_y]),
                                transforms.ToTensor(),
                                transforms.Normalize([0.4451, 0.4687, 0.4713], [0.3244, 0.3277, 0.3358])])


class Dataset_3DCNN(data.Dataset):
    "Characterizes a dataset for PyTorch"

    def __init__(self, data_path, folders, labels, frames, transform=None):
        "Initialization"
        self.data_path = data_path
        self.labels = labels
        self.folders = folders
        self.transform = transform
        self.frames = frames

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.folders)

    def read_images(self, path, selected_folder, use_transform):
        X = []
        for i in self.frames:
            m = os.path.join(path, selected_folder, '{}'.format(i))

            try:
                image = Image.open(m)
            except:

                continue

            if use_transform is not None:
                image = use_transform(image)

            X.append(image)
            if i == "5.jpg":
                X.pop(0)

        if len(X) < 4:
            X.append(image)
        elif len(X) > 4:
            X.pop(0)
        X = torch.stack(X, dim=0)
        return X

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        folder = self.folders[index]

        # Load data
        X = self.read_images(self.data_path, folder, self.transform)  # (input) spatial images
        y = torch.LongTensor([self.labels[index]])  # (labels) LongTensor are for int64 instead of FloatTensor

        return X, y


traindatasets = Dataset_3DCNN(data_path, train_list, train_label, selected_frames, transform=transform)
testdatasets = Dataset_3DCNN(data_path, test_list, test_label, selected_frames, transform=transform)
train_data_loader = data.DataLoader(traindatasets, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
test_data_loader = data.DataLoader(testdatasets, batch_size=200, shuffle=True, num_workers=4)
all_data_loader = {"train": train_data_loader, "test": test_data_loader}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[5]:


class ResCNNEncoder(nn.Module):
    def __init__(self, fc_hidden1=512, fc_hidden2=512, drop_p=0.3, CNN_embed_dim=300):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(ResCNNEncoder, self).__init__()

        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        self.drop_p = drop_p

        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]  # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.fc1 = nn.Linear(resnet.fc.in_features, fc_hidden1)
        self.bn1 = nn.BatchNorm1d(fc_hidden1, momentum=0.01)
        self.fc2 = nn.Linear(fc_hidden1, fc_hidden2)
        self.bn2 = nn.BatchNorm1d(fc_hidden2, momentum=0.01)
        self.fc3 = nn.Linear(fc_hidden2, CNN_embed_dim)

    def forward(self, x_3d):

        cnn_embed_seq = []
        for t in range(x_3d.size(1)):
            # ResNet CNN
            with torch.no_grad():
                x = self.resnet(x_3d[:, t, :, :, :])  # ResNet
                x = x.view(x.size(0), -1)  # flatten output of conv

            # FC layers
            x = self.bn1(self.fc1(x))
            x = F.relu(x)
            x = self.bn2(self.fc2(x))
            x = F.relu(x)
            x = F.dropout(x, p=self.drop_p, training=self.training)
            x = self.fc3(x)

            cnn_embed_seq.append(x)
        print('resnet_x.shape= ', x.size())

        # swap time and sample dim such that (sample dim, time dim, CNN latent dim)
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)
        # cnn_embed_seq: shape=(batch, time_step, input_size)

        return cnn_embed_seq


class DecoderRNN(nn.Module):
    def __init__(self, CNN_embed_dim=300, h_RNN_layers=3, h_RNN=256, h_FC_dim=128, drop_p=0.3, num_classes=3):
        super(DecoderRNN, self).__init__()

        self.RNN_input_size = CNN_embed_dim
        self.h_RNN_layers = h_RNN_layers  # RNN hidden layers
        self.h_RNN = h_RNN  # RNN hidden nodes
        self.h_FC_dim = h_FC_dim
        self.drop_p = drop_p
        self.num_classes = num_classes

        self.LSTM = nn.LSTM(
            input_size=self.RNN_input_size,
            hidden_size=self.h_RNN,
            num_layers=h_RNN_layers,
            batch_first=True,  # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
            # bidirectional  = True
        )

        self.fc1 = nn.Linear(self.h_RNN, self.h_FC_dim)
        self.bn1 = nn.BatchNorm1d(self.h_FC_dim, momentum=0.01)
        self.fc2 = nn.Linear(self.h_FC_dim, self.num_classes)

    def forward(self, x_RNN):
        print('lstm_x.shape= ', x_RNN.size())

        self.LSTM.flatten_parameters()
        RNN_out, (h_n, h_c) = self.LSTM(x_RNN, None)
        """ h_n shape (n_layers, batch, hidden_size), h_c shape (n_layers, batch, hidden_size) """
        """ None represents zero initial hidden state. RNN_out has shape=(batch, time_step, output_size) """

        # FC layers
        x = self.fc1(RNN_out[:, -1, :])  # choose RNN_out at the last time step
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.fc2(x)

        return x


# In[6]:


CNN_fc_hidden1, CNN_fc_hidden2 = 1024, 768
CNN_embed_dim = 512  # latent dim extracted by 2D CNN
res_size = 224  # ResNet image size
dropout_p = 0  # dropout probability

# use same decoder RNN saved!
RNN_hidden_layers = 3
RNN_hidden_nodes = 512
RNN_FC_dim = 256

# training parameters
k = 3

cnn_encoder = ResCNNEncoder(fc_hidden1=CNN_fc_hidden1, fc_hidden2=CNN_fc_hidden2, drop_p=dropout_p,
                            CNN_embed_dim=CNN_embed_dim).to(device)
rnn_decoder = DecoderRNN(CNN_embed_dim=CNN_embed_dim, h_RNN_layers=RNN_hidden_layers, h_RNN=RNN_hidden_nodes,
                         h_FC_dim=RNN_FC_dim, drop_p=dropout_p, num_classes=k).to(device)

model_ft = [cnn_encoder, rnn_decoder]

# In[7]:


savedir = os.path.join(os.getcwd(), 'save')
if not os.path.exists(savedir):
    os.makedirs(savedir)


def train_model(model, dataloaders, loss_fn, optimizer, num_epochs=25):
    cnn_encoder = model[0]
    rnn_decoder = model[1]
    best_acc = 0.
    best_epoch = 0
    for epoch in range(num_epochs):
        # run epoch
        phase = 'train'

        running_loss = 0.
        running_corrects = 0.

        cnn_encoder.train()
        rnn_decoder.train()
        for batch_idx, (inputs, labels) in enumerate(dataloaders[phase]):
            print('inputs.shape= ', inputs.size())
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = rnn_decoder(cnn_encoder(inputs))  # bsize * 2
            loss = loss_fn(outputs, labels.squeeze(1))
            preds = outputs.argmax(dim=1)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds.view(-1) == labels.view(-1)).item()

        epoch_loss = running_loss / len(dataloaders[phase].dataset)
        epoch_acc = running_corrects / len(dataloaders[phase].dataset)

        print("Train loss: {:.8f}, acc: {}".format(epoch_loss, epoch_acc))

        # validation
        phase = 'test'
        cnn_encoder.eval()
        rnn_decoder.eval()
        acc = validation(dataloaders, phase, loss_fn)
        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch
            torch.save(rnn_decoder.state_dict(),
                       os.path.join(savedir, 'best_rnn.pth'))
            torch.save(cnn_encoder.state_dict(),
                       os.path.join(savedir, 'best_cnn.pth'))
        print('Best Val Acc: {:.2f}|Best epoch: {}'
              .format(best_acc, best_epoch))


def validation(dataloaders, phase, loss_fn):
    corrects = 0.
    with torch.no_grad():
        for i, (inputs, label) in enumerate(dataloaders[phase]):
            inputs, labels = inputs.to(device), label.to(device)

            outputs = rnn_decoder(cnn_encoder(inputs))

            loss = loss_fn(outputs, labels.squeeze(1))
            loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            corrects += torch.sum(preds.view(-1) == labels.view(-1)).item()

        loss = loss / len(dataloaders[phase].dataset)
        acc = corrects / len(dataloaders[phase].dataset)
    print("Validation loss: {:.8f}, acc: {}".format(loss, acc))

    return acc


# In[8]:


model_ft = [cnn_encoder.to(device), rnn_decoder.to(device)]

# In[9]:


if torch.cuda.device_count() == 1:
    print("Using", torch.cuda.device_count(), "GPUs!")
    # Combine all EncoderCNN + DecoderRNN parameters
    crnn_params = list(cnn_encoder.fc1.parameters()) + list(cnn_encoder.bn1.parameters()) + list(
        cnn_encoder.fc2.parameters()) + list(cnn_encoder.bn2.parameters()) + list(cnn_encoder.fc3.parameters()) + list(
        rnn_decoder.parameters())

optimizer = torch.optim.Adam(crnn_params, lr=learning_rate)
weight_CE = torch.FloatTensor([1, 10, 4])
loss_fn = nn.CrossEntropyLoss(weight=weight_CE)

# In[10]:


train_model(model_ft, all_data_loader, loss_fn.to(device), optimizer, num_epochs=num_epochs)

# In[ ]:


test_path = "/content/drive/My Drive/Colab Notebooks/amap_traffic_test_0712"
test_list = os.listdir(test_path)


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


# data_path = "/content/drive/My Drive/Colab Notebooks/amap_traffic_train_0712"
# train_list = os.listdir(data_path)
train_label = findlabels("/content/drive/My Drive/Colab Notebooks/traffic.json")
selected_frames = ["1.jpg", "2.jpg", "3.jpg", "4.jpg", "5.jpg"]
img_x = 224
img_y = 224
transform = transforms.Compose([transforms.Resize([img_x, img_y]),
                                transforms.ToTensor(),
                                transforms.Normalize([0.4451, 0.4687, 0.4713], [0.3244, 0.3277, 0.3358])])

testdatasets = Dataset_3DCNN(test_path, test_list, train_label, selected_frames, transform=transform)
test_data_loader = data.DataLoader(testdatasets, batch_size=1, shuffle=False, num_workers=4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict(model, test_data_loader):
    res = []
    cnn_encoder = model[0]
    rnn_decoder = model[1]
    cnn_encoder.load_state_dict(torch.load(os.path.join(savedir, 'best_cnn.pth')))
    rnn_decoder.load_state_dict(torch.load(os.path.join(savedir, 'best_rnn.pth')))

    cnn_encoder.eval()
    rnn_decoder.eval()
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_data_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = rnn_decoder(cnn_encoder(inputs))
            preds = outputs.argmax(dim=1)
            res.append(preds)
    return res


res = predict(_, test_data_loader)

# In[ ]:


print(len(res))
a = []
for each in res:
    a.append(int(each))
print(a)
