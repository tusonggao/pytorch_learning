from __future__ import print_function, division
import os
import shutil
import hashlib
import time
import sys
import torch
import random
import requests
from pathlib import Path
import pandas as pd
from PIL import Image
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

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


all_files = get_all_files('./data/test_data/')
for file_name in all_files:
# files = {'file': open('./data/chufang_data/negative/6.jpg', 'rb')}
# files = {'file': open('./data/chufang_data_processed/test/positive/8.jpg', 'rb')}
    #files = {'file': open('./data/test_data/0.jpg', 'rb')}
    files = {'file': open(file_name, 'rb')}
    #r = requests.post("http://172.17.30.118:8080/check_chufang_file", files=files)
    r = requests.post("http://127.0.0.1:8080/check_chufang_file", files=files)
    print('file_name: ', file_name, 'prediction is ', r.text)






# files = {'file': open('./data/chufang_data/positive/1007.jpg', 'rb')}  # [0.0009495963, 0.99905044]
# files = {'file': open('./data/chufang_data_processed/val/positive/3.jpg', 'rb')}  # [0.002257867, 0.9977422]
# user_info = {'name': 'tsg'}

# r = requests.post("http://127.0.0.1:8080/check_chufang_image", data=user_info, files=files)
