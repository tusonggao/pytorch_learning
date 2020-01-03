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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.optim.lr_scheduler import StepLR

import argparse
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


random_seed = 77777
random.seed(random_seed)
from flask import Flask, request
from werkzeug import secure_filename   # 获取上传文件的文件名

print('prog starts here!')

UPLOAD_FOLDER = 'F:/jianke_chufang_recognition/uploaded_data/'  #  上传路径
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])         #  允许上传的文件类型

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

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

cnn_model = torch.load('./chufang_cnn.pthmodel', map_location='cpu')

# 验证上传的文件名是否符合要求，文件名必须带点并且符合允许上传的文件类型要求，两者都满足则返回 true
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


def get_tensor_data_from_file(file_name, size=(256, 256)):
    im = Image.open(file_name)
    image = im.resize(size)
    image = image.convert('RGB')
    image = np.array(image)
    image = image.astype('float32')
    image = image.transpose((2, 0, 1))
    print('image shape before is ', image.shape)
    image = image.reshape((-1, 3, 256, 256))
    print('image shape after is ', image.shape)
    return torch.from_numpy(image)


def predict(model, data, device='cpu'):
    model.eval()
    with torch.no_grad():
        data = data.to(device)
        output = model(data)
        print('output is ', output)
        return output


@app.route('/check_chufang_file', methods=['GET', 'POST'])
def check_chufang_file():
    global cnn_model
    print('in check_chufang_file')
    # return 'get here!!!'
    if request.method == 'POST':   # 如果是 POST 请求方式
        file = request.files['file']   # 获取上传的文件
        if file and allowed_file(file.filename):   # 如果文件存在并且符合要求则为 true
            filename = secure_filename(file.filename)   # 获取上传文件的文件名
            print('filename is ', filename)
            print('{} upload successed!'.format(filename))  # 返回保存成功的信息
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))   # 保存文件
            data = get_tensor_data_from_file(os.path.join(app.config['UPLOAD_FOLDER'], filename), size=(256, 256))
            output = predict(cnn_model, data).numpy().flatten()
            v1, v2 = output[0], output[1]
            v1, v2 = np.exp(v1)/ (np.exp(v1) + np.exp(v2)), np.exp(v2)/ (np.exp(v1) + np.exp(v2))

            return 'Positive: {} Negative: {}'.format(v2, v1)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)  # 这种方法可以支持对外访问
    # app.run()