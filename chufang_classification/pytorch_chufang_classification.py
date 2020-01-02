from __future__ import print_function, division
import os
import time
import sys
import torch
import random
from pathlib import Path
import pandas as pd
from PIL import Image
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

random.seed(7777)

print('prog starts here!')

# import multiprocessing
# if __name__ == "__main__":
#     p = multiprocessing.Pool(10)
# plt.ion()   # interactive mode

def get_all_files(dir_name):   # 递归得到文件夹下的所有文件
    all_files_lst = []
    def get_all_files_worker(path):
        allfilelist = os.listdir(path)
        for file in allfilelist:
            filepath = os.path.join(path, file)
            #判断是不是文件夹
            if os.path.isdir(filepath):
                get_all_files_worker(filepath)
            else:
                all_files_lst.append(filepath)
    get_all_files_worker(dir_name)
    return all_files_lst


def SaltAndPepper(src, percetage=0.1):
    SP_NoiseImg = src.copy()
    SP_NoiseNum = int(percetage * src.shape[0] * src.shape[1])
    for i in range(SP_NoiseNum):
        randR = np.random.randint(0, src.shape[0] - 1)
        randG = np.random.randint(0, src.shape[1] - 1)
        randB = np.random.randint(0, 3)
        if np.random.randint(0, 1) == 0:
            SP_NoiseImg[randR, randG, randB] = 0
        else:
            SP_NoiseImg[randR, randG, randB] = 255
    return SP_NoiseImg


def addGaussianNoise(image, percetage=0.1):
    G_Noiseimg = image.copy()
    w = image.shape[1]
    h = image.shape[0]
    G_NoiseNum = int(percetage * image.shape[0] * image.shape[1])
    for i in range(G_NoiseNum):
        temp_x = np.random.randint(0, h)
        temp_y = np.random.randint(0, w)
        G_Noiseimg[temp_x][temp_y][np.random.randint(3)] = np.random.randn(1)[0]
    return G_Noiseimg
    # im = Image.open(data_path + '/negative/138.jpg')
    #
    # im_arr = np.array(im)  # image类 转 numpy
    # print('type im_arr ', type(im_arr), 'im_arr.shape is ', im_arr.shape)


def generateMutationFiles(file_name):
    p = Path(file_name)
    p.parent.mkdir(exist_ok=True, parents=True)  # 递归创建文件目录

    im = Image.open(file_name)
    im = im.convert('RGB')

    im_rotate_90 = im.transpose(Image.ROTATE_90)
    new_file_name = Path(p.parent, p.stem + '_rotate_90' + p.suffix)
    im_rotate_90.save(new_file_name, 'JPEG')

    im_rotate_180 = im.transpose(Image.ROTATE_180)
    new_file_name = Path(p.parent, p.stem + '_rotate_180' + p.suffix)
    im_rotate_180.save(new_file_name, 'JPEG')

    im_rotate_270 = im.transpose(Image.ROTATE_270)
    new_file_name = Path(p.parent, p.stem + '_rotate_270' + p.suffix)
    im_rotate_270.save(new_file_name, 'JPEG')
    return


def copy_resize_file(src_file_name, target_file_name):
    p = Path(target_file_name)
    p.parent.mkdir(exist_ok=True, parents=True)  # 递归创建文件目录

    im = Image.open(src_file_name)
    #im_resized = im.resize((256, 256))
    #im_resized = im.resize((128, 128))
    im_resized = im.resize((512, 512))
    #im_resized = im.resize((28, 28))
    im_resized = im_resized.convert('RGB')
    im_resized.save(target_file_name, 'JPEG')
    return

def data_preprocessing():
    print('in data_preprocessing')
    origin_root_dir = './data/chufang_data/'
    target_root_dir = './data/chufang_data_processed/'
    os.system('rm -rf ./data/chufang_data_processed/')
    positive_files_lst = get_all_files(origin_root_dir + '/positive/')
    random.shuffle(positive_files_lst)
    negative_files_lst = get_all_files(origin_root_dir + '/negative/')
    random.shuffle(positive_files_lst)
    positive_files_num, negative_files_num = len(positive_files_lst), len(negative_files_lst)

    train_ratio, val_ratio, test_ratio = 0.65, 0.05, 0.30

    for i, file_name in enumerate(positive_files_lst[:int(train_ratio*positive_files_num)]):
        print('file_name is ', file_name)
        new_file_name = target_root_dir + '/train/positive/' + str(i) + '.jpg'
        copy_resize_file(file_name, new_file_name)
        generateMutationFiles(new_file_name)

    for i, file_name in enumerate(negative_files_lst[:int(train_ratio*negative_files_num)]):
        print('file_name is ', file_name)
        new_file_name = target_root_dir + '/train/negative/' + str(i) + '.jpg'
        copy_resize_file(file_name, new_file_name)
        generateMutationFiles(new_file_name)

    ##########################################################################################

    for i, file_name in enumerate(positive_files_lst[int(train_ratio * positive_files_num):
                                  int(train_ratio * positive_files_num + val_ratio * positive_files_num)]):
        print('file_name is ', file_name)
        new_file_name = target_root_dir + '/val/positive/' + str(i) + '.jpg'
        copy_resize_file(file_name, new_file_name)
        generateMutationFiles(new_file_name)

    for i, file_name in enumerate(negative_files_lst[int(train_ratio * negative_files_num):
                                  int(train_ratio * negative_files_num + val_ratio * negative_files_num)]):
        print('file_name is ', file_name)
        new_file_name = target_root_dir + '/val/negative/' + str(i) + '.jpg'
        copy_resize_file(file_name, new_file_name)
        generateMutationFiles(new_file_name)

    ##########################################################################################

    for i, file_name in enumerate(positive_files_lst[
                                  int(train_ratio * positive_files_num + val_ratio*positive_files_num):]):
        print('file_name is ', file_name)
        new_file_name = target_root_dir + '/test/positive/' + str(i) + '.jpg'
        copy_resize_file(file_name, new_file_name)
        generateMutationFiles(new_file_name)

    for i, file_name in enumerate(negative_files_lst[
                                  int(train_ratio * negative_files_num + val_ratio*negative_files_num):]):
        print('file_name is ', file_name)
        new_file_name = target_root_dir + '/test/negative/' + str(i) + '.jpg'
        copy_resize_file(file_name, new_file_name)
        generateMutationFiles(new_file_name)

    ##########################################################################################

    print('len of positive_files_lst is ', len(positive_files_lst))
    print('len of negative_files_lst is ', len(negative_files_lst))
    print('positive_files_lst is ', positive_files_lst)

    return positive_files_lst


class ChufangDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.load(root_dir)

    def load(self, root_dir):
        self.samples = []
        self.positive_files_lst = get_all_files(root_dir + '/positive/')
        self.negative_files_lst = get_all_files(root_dir + '/negative/')
        random.shuffle(self.positive_files_lst)
        random.shuffle(self.negative_files_lst)

    def __len__(self):
        return len(self.positive_files_lst) + len(self.negative_files_lst)

    def __getitem__(self, idx):
        if idx < len(self.positive_files_lst):
            img_name, label = self.positive_files_lst[idx], 1
        else:
            img_name, label = self.negative_files_lst[idx-len(self.positive_files_lst)], 0

        image = io.imread(img_name)
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample

class ToTensor(object):
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = image.astype('float32')
        image = image.transpose((2, 0, 1))
        return torch.from_numpy(image), torch.tensor(label)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv1 = nn.Conv2d(3, 64, 3, 1)
        self.relu1 = nn.ReLU()

        #self.conv2 = nn.Conv2d(32, 32, 3, 1)
        self.conv2 = nn.Conv2d(64, 32, 3, 1)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(32, 32, 3, 1)
        self.relu3 = nn.ReLU()

        self.pool = nn.MaxPool2d(2)

        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)

        #self.fc1 = nn.Linear(9216, 128)
        #self.fc1 = nn.Linear(123008, 128)
        #self.fc1 = nn.Linear(508032, 128)
        self.fc1 = nn.Linear(2064512, 128)
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

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=0.5)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.2)
    for epoch in range(1, 10 + 1):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    if True:
        torch.save(model.state_dict(), "chufang_cnn.pt")

    print('get here 333')


if __name__=='__main__':
    data_preprocessing()
    main()

