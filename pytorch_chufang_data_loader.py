from __future__ import print_function, division
import os
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
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

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
    im_resized = im.resize((512, 512))
    im_resized = im_resized.convert('RGB')
    im_resized.save(target_file_name, 'JPEG')
    return

def data_preprocessing():
    print('in data_preprocessing')
    origin_root_dir = './data/chufang_data/'
    target_root_dir = './data/chufang_data_processed/'
    positive_files_lst = get_all_files(origin_root_dir + '/positive/')
    random.shuffle(positive_files_lst)
    negative_files_lst = get_all_files(origin_root_dir + '/negative/')
    random.shuffle(positive_files_lst)
    positive_files_num, negative_files_num = len(positive_files_lst), len(negative_files_lst)

    train_ratio, val_ratio, test_ratio = 0.65, 0.15, 0.30

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
                                  int(train_ratio * negative_files_num + val_ratio * negative_files_num)]):
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
                                  int(train_ratio * negative_files_num + val_ratio*negative_files_num):]):
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

data_preprocessing()
sys.exit(0)

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
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image), 'label': torch.tensor(label)}

transformed_dataset = FaceLandmarksDataset(root_dir='./data/chufang_data_processed/train/',
                                           transform=transforms.Compose([ToTensor()]))
dataloader = DataLoader(transformed_dataset, batch_size=4, shuffle=True, num_workers=4)

for i_batch, sample_batched in enumerate(dataloader):
    print(i_batch, sample_batched['image'].size(), sample_batched['label'].size())


print('prog ends here!')



