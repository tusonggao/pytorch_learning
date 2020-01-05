from __future__ import print_function, division
import os
import time
import sys
import argparse
import random
from pathlib import Path
from PIL import Image
from skimage import io, transform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from chufang_data_loader import data_preprocessing, ChufangDataset

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchsummary import summary
from torch.optim.lr_scheduler import StepLR
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

random.seed(7777)

print('prog starts here!')

class ToTensor(object):
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = image.astype('float32')
        image = image.transpose((2, 0, 1))
        return torch.from_numpy(image), torch.tensor(label)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        # self.conv1 = nn.Conv2d(3, 64, 3, 1)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(32, 32, 3, 1)
        # self.conv2 = nn.Conv2d(64, 32, 3, 1)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(32, 32, 3, 1)
        self.relu3 = nn.ReLU()

        self.pool = nn.MaxPool2d(2)

        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)

        #self.fc1 = nn.Linear(9216, 128)
        #self.fc1 = nn.Linear(123008, 128)
        self.fc1 = nn.Linear(508032, 128)
        self.fc2 = nn.Linear(128, 2)
        #self.fc2 = nn.Linear(-1, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        #x = F.max_pool2d(x, 2)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        #x = self.dropout1(x)
        x = torch.flatten(x, 1)

        x = self.fc1(x)
        #x = nn.Linear(x.size(0), 128)(x)
        x = F.relu(x)
        #x = self.dropout2(x)
        x = self.fc2(x)
        #x = nn.Linear(x.size(0), 2)(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        #print('in train data is ', data.shape, 'data.dtype is ', data.dtype)
        #print('in train target is ', target.shape, 'target.dtype is ', target.dtype)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        time.sleep(0.001)
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    print('in main()')

    # Training settings
    # parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    # parser.add_argument('--batch-size', type=int, default=64, metavar='N',
    #                     help='input batch size for training (default: 64)')
    # parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
    #                     help='input batch size for testing (default: 1000)')
    # parser.add_argument('--epochs', type=int, default=1, metavar='N',
    #                     help='number of epochs to train (default: 14)')
    # parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
    #                     help='learning rate (default: 1.0)')
    # parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
    #                     help='Learning rate step gamma (default: 0.7)')
    # parser.add_argument('--no-cuda', action='store_true', default=False,
    #                     help='disables CUDA training')
    # parser.add_argument('--seed', type=int, default=1, metavar='S',
    #                     help='random seed (default: 1)')
    # parser.add_argument('--log-interval', type=int, default=10, metavar='N',
    #                     help='how many batches to wait before logging training status')
    #
    # parser.add_argument('--save-model', action='store_true', default=False,
    #                     help='For Saving the current Model')
    # args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    #print('use_cuda is ', use_cuda)

    #print('get here 111')

    torch.manual_seed(42)

    device = torch.device("cuda" if use_cuda else "cpu")
    #device = 'cpu'
    #print('device is ', device)

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    train_dataset = ChufangDataset(root_dir='./data/chufang_data_processed/train/', 
                                   transform=transforms.Compose([ToTensor()]))
    val_dataset = ChufangDataset(root_dir='./data/chufang_data_processed/val/', 
                                 transform=transforms.Compose([ToTensor()]))
    test_dataset = ChufangDataset(root_dir='./data/chufang_data_processed/test/',
                                  transform=transforms.Compose([ToTensor()]))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=True)

    #print('get here 222')
    LOAD_MODEL = False

    #if LOAD_MODEL is False: 
    model = Net().to(device)

    print('show network grpah')
    summary(model, (3,256,256))
    #return 

    optimizer = optim.Adadelta(model.parameters(), lr=0.5)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.2)
    epoch_num = 10
    for epoch in range(epoch_num):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()
    torch.save(model, './chufang_cnn.pthmodel')

    LOAD_MODEL = True
    if LOAD_MODEL:
        #model = torch.load('./chufang_cnn.pthmodel')
        model = torch.load('./chufang_cnn.pthmodel', map_location='cpu')
        print('model loaded with CPU ! test with loaded model')
        for i in range(10):
            test(model, 'cpu', test_loader)

    print('get here 333')


if __name__=='__main__':
    #data_preprocessing()
    main()




