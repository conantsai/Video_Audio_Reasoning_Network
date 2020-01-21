import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset
# from torchvision import transforms, utils

class UCF101_train(Dataset):
    def __init__(self, file_list, root_dir, transform=None):
        '''
        Args:
            file_list (str): the file of ucf101 train list
            root_dir (str): the root path of files
            transform (list): a list of transform functions
        '''
        self.file_list = pd.read_csv(file_list)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        '''
        Return:
            video_num (int): the number of videos
        '''
        return len(self.file_list)

    def load_video(self, video_name, idx):
        '''
        load continues 16 frames
        Args:
            video_name (str) - video name
            idx (int) - video index
            start_frame - the start index of continues 16 frames
        Return:
            video (Tensor) - C(channel, 3) x L(frames, 16) x H x W
        '''
        # lower stand / Stand
        if 'HandStandPushups' in video_name:
            video_name = video_name.split('_')
            video_name[1] = 'HandstandPushups'
            video_name = '_'.join(video_name)
        
        video_path = os.path.join(self.root_dir, video_name)
        total_clips = len(os.listdir(video_path))
        start_frame = random.randint(1, total_clips - 1 - 16)
        
        for i in range(16):
            img_name = 'frame' + '{:06d}'.format(start_frame) + '.jpg'
            img_path = os.path.join(video_path, img_name)
            tmp_img = io.imread(img_path)

            if i == 0:
                h, w, c = tmp_img.shape
                video = np.zeros((16, h, w, c))

            try:
                video[i, :, :, :] = tmp_img
            except ValueError:
                pass
                # print(img_path)
                # print(tmp_img.shape)
            start_frame += 1

        return video

    def __getitem__(self, idx):
        video_name, target = self.file_list.iloc[idx, :]
        # from 0 ~ 100
        target -= 1

        # video - C x L(frames) x H x W
        video = self.load_video(video_name, idx)
        sample = {'video': video, 'target': target}

        if self.transform:
            sample = self.transform(sample)

        return sample


class UCF101_valid(Dataset):
    def __init__(self, file_list, root_dir, transform=None):
        '''
        Args:
            file_list (str): the file of ucf101 validation list
            root_dir (str): the root path of files
            transform (list): a list of transform functions
        '''
        self.file_list = pd.read_csv(file_list)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        '''
        Return:
            video_num (int): the number of videos
        '''
        return len(self.file_list)

    def load_video(self, video_name, idx):
        '''
        load continues 16 frames
        Args:
            video_name (str) - video name
            idx (int) - video index
            start_frame - the start index of continues 16 frames
        Return:
            video (Tensor) - C(channel, 3) x L(frames, 16) x H x W
        '''
        # lower stand / Stand
        if 'HandStandPushups' in video_name:
            video_name = video_name.split('_')
            video_name[1] = 'HandstandPushups'
            video_name = '_'.join(video_name)

        video_path = os.path.join(self.root_dir, video_name)
        total_clips = len(os.listdir(video_path))
        start_frame = random.randint(1, total_clips - 1 - 16)
        
        for i in range(16):
            img_name = 'frame' + '{:06d}'.format(start_frame) + '.jpg'
            img_path = os.path.join(video_path, img_name)
            tmp_img = io.imread(img_path)

            if i == 0:
                h, w, c = tmp_img.shape
                video = np.zeros((16, h, w, c))

            try:
                video[i, :, :, :] = tmp_img
            except ValueError:
                pass
                # print(img_path)
                # print(tmp_img.shape)
            start_frame += 1

        return video

    def __getitem__(self, idx):
        video_name, target = self.file_list.iloc[idx, :]
        # from 0 ~ 100
        target -= 1

        # video - C x L(frames) x H x W
        video = self.load_video(video_name, idx)
        sample = {'video': video, 'target': target}

        if self.transform:
            sample = self.transform(sample)

        return sample
