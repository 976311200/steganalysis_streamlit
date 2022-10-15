#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import argparse
import numpy as np
import cv2
from pathlib import Path
import random
from glob import glob

import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
from srm_filter_kernel import all_normalized_hpf_list
from MPNCOV import *  # MPNCOV

IMAGE_SIZE = 256
BATCH_SIZE = 32//2
num_levels = 3
EPOCHS = 200
LR = 0.005


WEIGHT_DECAY = 5e-4
TRAIN_PRINT_FREQUENCY = 100
EVAL_PRINT_FREQUENCY = 1
DECAY_EPOCH = [400]
OUTPUT_PATH = Path(__file__).stem


def conv3x3(in_planes, out_planes, stride=1, groups=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, groups=groups, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class TLU(nn.Module):
    def __init__(self, threshold):
        super(TLU, self).__init__()
        self.threshold = threshold

    def forward(self, input):
        output = torch.clamp(input, min=-self.threshold, max=self.threshold)
        return output

class ADD(nn.Module):
    def __init__(self):
        super(ADD, self).__init__()

    def forward(self, input1, input2):
        output = torch.add(input1, input2)
        return output

class HPF(nn.Module):
    def __init__(self):
        super(HPF, self).__init__()

        # Load 30 SRM Filters
        all_hpf_list_5x5 = []

        for hpf_item in all_normalized_hpf_list:
            if hpf_item.shape[0] == 3:
                hpf_item = np.pad(hpf_item, pad_width=((1, 1), (1, 1)), mode='constant')

            all_hpf_list_5x5.append(hpf_item)

        hpf_weight = nn.Parameter(torch.Tensor(all_hpf_list_5x5).view(30, 1, 5, 5), requires_grad=False)

        self.hpf = nn.Conv2d(1, 30, kernel_size=5, padding=2, bias=False)
        self.hpf.weight = hpf_weight

        # Truncation, threshold = 3
        self.tlu = TLU(3.0)

    def forward(self, input):

        output = self.hpf(input)
        output = self.tlu(output)

        return output

class Res2NetBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, downsample=None, stride=1, scales=4, groups=1, se=False,  norm_layer=None):
        super(Res2NetBottleneck, self).__init__()
        if planes % scales != 0:
            raise ValueError('Planes must be divisible by scales')
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        bottleneck_planes = groups * planes
        self.conv1 = conv1x1(inplanes, bottleneck_planes, stride)
        self.bn1 = norm_layer(bottleneck_planes)
        self.conv2 = nn.ModuleList([conv3x3(bottleneck_planes // scales, bottleneck_planes // scales, groups=groups) for _ in range(scales-1)])
        self.bn2 = nn.ModuleList([norm_layer(bottleneck_planes // scales) for _ in range(scales-1)])
        self.conv3 = conv1x1(bottleneck_planes, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.scales = scales

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        xs = torch.chunk(out, self.scales, 1)
        ys = []
        for s in range(self.scales):
            if s == 0:
                ys.append(xs[s])
            elif s == 1:
                ys.append(self.relu(self.bn2[s-1](self.conv2[s-1](xs[s]))))
            else:
                ys.append(self.relu(self.bn2[s-1](self.conv2[s-1](xs[s] + ys[-1]))))
        out = torch.cat(ys, 1)
        out = self.conv3(out)
        out = self.bn3(out)


        if self.downsample is not None:
            identity = self.downsample(identity)
        out += identity
        out = self.relu(out)
        return out

class Down_Sample(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(Down_Sample, self).__init__()
        self.conva = conv1x1(inplanes, outplanes, stride=2)
        self.bna = nn.BatchNorm2d(outplanes)
        self.convb = conv3x3(inplanes, outplanes)
        self.bnb = nn.BatchNorm2d(outplanes)
        self.avg = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        cur = x
        cur = self.conva(cur)
        cur = self.bna(cur)
        x = self.convb(x)
        x = self.bnb(x)
        x = self.avg(x)
        x = torch.add(cur, x)
        x = F.relu(x, inplace=True)
        return x


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.group1 = HPF()  # pre-processing Layer 1
        self.conv1 = nn.Conv2d(in_channels=30, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.res2net3 = Res2NetBottleneck(16,4)
        self.res2net4 = Res2NetBottleneck(16, 4)
        self.res2net5 = Res2NetBottleneck(16, 4)
        self.down1 = Down_Sample(16,64)
        self.down2 = Down_Sample(64, 128)
        self.down3 = Down_Sample(128, 256)
        self.down4 = Down_Sample(256, 512)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(int(256 * (256 + 1) / 2), 2)
        self.fc1 = nn.Linear(int(256 * (256 + 1) / 2), 2)


    def forward(self, x):
        x = self.group1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x,inplace=True)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.res2net3(x)
        x = self.res2net4(x)
        x = self.res2net5(x) # [32,16,256,256]
        x = self.down1(x) # [64, 128,128]
        x = self.down2(x)
        x = self.down3(x)  # [32, 256, 32, 32]
        x = CovpoolLayer(x)
        x = SqrtmLayer(x,5)
        x = TriuvecLayer(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return (x)


def evaluate(model, device, eval_loader, dir):
    model.eval()

    with torch.no_grad():
        for sample in eval_loader:
            data, label = sample['data'], sample['label']
            shape = list(data.size())
            data = data.reshape(shape[0] * shape[1], *shape[2:])
            label = label.reshape(-1)
            data, label = data.to(device), label.to(device)
            output = model(data)
            pred = output.max(1, keepdim=True)[1]
            show_pred = pred.cpu().numpy()
            show_str = "Pred Label" + str(np.squeeze(show_pred)) + '\n'
            cover_list = [x.split('\\')[-1].split(".jpg")[0] for x in glob(dir + '/*')]
            show_str += str(cover_list)
            show_str += '\n'
            for num in show_pred:
                if num:
                    show_str += "假"
                else:
                    show_str += "真"
                show_str += "\t"
    return show_str

# Initialization
def initWeights(module):
    if type(module) == nn.Conv2d:
        if module.weight.requires_grad:
            nn.init.kaiming_normal_(module.weight.data, mode='fan_in', nonlinearity='relu')

    if type(module) == nn.Linear:
        nn.init.normal_(module.weight.data, mean=0, std=0.01)
        nn.init.constant_(module.bias.data, val=0)

class ToTensor():
    def __call__(self, sample):
        data, label = sample['data'], sample['label']
        data = np.expand_dims(data, axis=1)
        data = data.astype(np.float32)
        new_sample = {
            'data': torch.from_numpy(data),
            'label': torch.from_numpy(label).long(),
        }
        return new_sample

class MyDataset(Dataset):
    def __init__(self, DATASET_DIR, partition, transform=None):
        random.seed(1234)
        self.transform = transform
        self.cover_dir = DATASET_DIR
        self.cover_list = [x.split('\\')[-1] for x in glob(self.cover_dir + '/*')]
        self.cover_list = self.cover_list
        assert len(self.cover_list) != 0, "cover_dir is empty"

    def __len__(self):
        return len(self.cover_list)

    def __getitem__(self, idx):
        file_index = int(idx)
        cover_path = os.path.join(self.cover_dir, self.cover_list[file_index])
        cover_data = cv2.imread(cover_path, -1)
        data = np.stack([cover_data])
        label = np.array([0], dtype='int32')
        sample = {'data': data, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample

def main(args):
    statePath = args.statePath
    # device = torch.device("cpu")  使用cpu进行预测
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True}
    eval_transform = transforms.Compose([
        ToTensor()
    ])

    TEST_DATASET_DIR = args.TEST_DIR

    test_dataset = MyDataset(TEST_DATASET_DIR, 2, eval_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, **kwargs)
    model = Net().to(device)

    all_state = torch.load(statePath)
    original_state = all_state['original_state']
    model.load_state_dict(original_state)

    string=evaluate(model, device, test_loader,args.TEST_DIR)
    print(string)
    return string

def myParseArgs(debug_bool):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-TEST_DIR',
        '--TEST_DIR',
        help='The path to load test_dataset',
        type=str,
        required=True
    )

    parser.add_argument(
        '-l',
        '--statePath',
        help='Path for loading model state',
        type=str,
        default=''
    )
    if debug_bool:
        args = parser
    else:
        args = parser.parse_args()
    return args


if __name__ == '__main__':
    debug_bool = True
    TEST_DIR = './data'
    args = myParseArgs(debug_bool=debug_bool)
    if debug_bool:
        args.statePath = './model_params.pt'
        args.TEST_DIR = TEST_DIR
    main(args)
