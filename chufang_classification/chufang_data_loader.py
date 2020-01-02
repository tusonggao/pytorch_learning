from __future__ import print_function, division
import os
import shutil
import hashlib
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

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

random_seed = 77777
random.seed(random_seed)

print('prog starts here!')

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

def filemd5(file_name):
    with open(file_name, 'rb') as fp:
        data = fp.read()
    file_md5 = hashlib.md5(data).hexdigest()
    return file_md5

print('model md5 is ', filemd5('./chufang_cnn.pthmodel'))

def remove_duplicated_files(file_names_lst):

    md5_set = set()
    unique_file_names_lst = []
    duplicated_num = 0
    for file_name in file_names_lst:
        file_md5 = filemd5(file_name)
        if file_md5 in md5_set:
            duplicated_num += 1
            continue
        unique_file_names_lst.append(file_name)
        md5_set.add(file_md5)
    print('in remove_duplicated_files() found duplicated_num is ', duplicated_num)
    return unique_file_names_lst


def SaltAndPepper(src_img, percetage=0.1):
    SP_NoiseImg = np.array(src_img).copy()
    SP_NoiseNum = int(percetage * SP_NoiseImg.shape[0] * SP_NoiseImg.shape[1])
    for i in range(SP_NoiseNum):
        randR = np.random.randint(0, SP_NoiseImg.shape[0] - 1)
        randG = np.random.randint(0, SP_NoiseImg.shape[1] - 1)
        randB = np.random.randint(0, 3)
        if np.random.randint(0, 1) == 0:
            SP_NoiseImg[randR, randG, randB] = 0
        else:
            SP_NoiseImg[randR, randG, randB] = 255
    new_im = Image.fromarray(SP_NoiseImg)
    return new_im


def addGaussianNoise(src_img, percetage=0.1):
    G_Noiseimg = np.array(src_img).copy()
    w, h = G_Noiseimg.shape[1], G_Noiseimg.shape[0]
    G_NoiseNum = int(percetage * w * h)
    for i in range(G_NoiseNum):
        temp_x = np.random.randint(0, h)
        temp_y = np.random.randint(0, w)
        G_Noiseimg[temp_x][temp_y][np.random.randint(3)] = np.random.randn(1)[0]
    new_im = Image.fromarray(G_Noiseimg)
    return new_im


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

    im_add_saltpepper_noise = SaltAndPepper(im, percetage=0.1)
    new_file_name = Path(p.parent, p.stem + '_saltpepper_noise' + p.suffix)
    im_add_saltpepper_noise.save(new_file_name, 'JPEG')

    im_add_gaussian_noise = addGaussianNoise(im, percetage=0.1)
    new_file_name = Path(p.parent, p.stem + '_gaussian_noise' + p.suffix)
    im_add_gaussian_noise.save(new_file_name, 'JPEG')

    return


def copy_resize_file(src_file_name, target_file_name, size = (256, 256)):
    if src_file_name.endswith('gif'):
        print('in copy_resize_file, src_file_name: ', src_file_name,
              'target file name ', target_file_name)

    p = Path(target_file_name)
    p.parent.mkdir(exist_ok=True, parents=True)  # 递归创建文件目录

    im = Image.open(src_file_name)
    im_resized = im.resize(size)
    im_resized = im_resized.convert('RGB')
    im_resized.save(target_file_name, 'JPEG')
    return

def data_preprocessing(img_size = (256, 256)):
    print('in data_preprocessing')
    origin_root_dir = './data/chufang_data/'
    target_root_dir = './data/chufang_data_processed/'
    shutil.rmtree(target_root_dir, True)   # 删除原有的处理后的文件夹，以防止使用老的数据

    positive_files_lst = remove_duplicated_files(get_all_files(origin_root_dir + '/positive/'))
    random.shuffle(positive_files_lst)
    negative_files_lst = remove_duplicated_files(get_all_files(origin_root_dir + '/negative/'))
    random.shuffle(positive_files_lst)
    positive_files_num, negative_files_num = len(positive_files_lst), len(negative_files_lst)

    train_ratio, val_ratio, test_ratio = 0.65, 0.05, 0.30

    for i, file_name in enumerate(positive_files_lst[:int(train_ratio*positive_files_num)]):
        print('file_name is ', file_name)
        new_file_name = target_root_dir + '/train/positive/' + str(i) + '.jpg'
        copy_resize_file(file_name, new_file_name, img_size)
        generateMutationFiles(new_file_name)

    for i, file_name in enumerate(negative_files_lst[:int(train_ratio*negative_files_num)]):
        print('file_name is ', file_name)
        new_file_name = target_root_dir + '/train/negative/' + str(i) + '.jpg'
        copy_resize_file(file_name, new_file_name, img_size)
        generateMutationFiles(new_file_name)

    ##########################################################################################

    for i, file_name in enumerate(positive_files_lst[int(train_ratio * positive_files_num):
                                  int(train_ratio * positive_files_num + val_ratio * positive_files_num)]):
        print('file_name is ', file_name)
        new_file_name = target_root_dir + '/val/positive/' + str(i) + '.jpg'
        copy_resize_file(file_name, new_file_name, img_size)
        generateMutationFiles(new_file_name)

    for i, file_name in enumerate(negative_files_lst[int(train_ratio * negative_files_num):
                                  int(train_ratio * negative_files_num + val_ratio * negative_files_num)]):
        print('file_name is ', file_name)
        new_file_name = target_root_dir + '/val/negative/' + str(i) + '.jpg'
        copy_resize_file(file_name, new_file_name, img_size)
        generateMutationFiles(new_file_name)

    ##########################################################################################

    for i, file_name in enumerate(positive_files_lst[
                                  int(train_ratio * positive_files_num + val_ratio*positive_files_num):]):
        print('file_name is ', file_name)
        new_file_name = target_root_dir + '/test/positive/' + str(i) + '.jpg'
        copy_resize_file(file_name, new_file_name, img_size)
        generateMutationFiles(new_file_name)

    for i, file_name in enumerate(negative_files_lst[
                                  int(train_ratio * negative_files_num + val_ratio*negative_files_num):]):
        print('file_name is ', file_name)
        new_file_name = target_root_dir + '/test/negative/' + str(i) + '.jpg'
        copy_resize_file(file_name, new_file_name, img_size)
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


if __name__=='__main__':
    #data_preprocessing()
    pass


