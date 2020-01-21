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


class CED(Dataset):
    def __init__(self, data_info, data_dir, transform=None):
        '''
        Args:
            data_info (str): the file of ucf101 train list
            data_dir (str): the root path of files
            transform (list): a list of transform functions
        '''
        self.data_info = pd.read_csv(data_info)
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        '''
        Return:
            video_num (int): the number of videos
        '''
        return len(self.data_info)

    def load_video(self, video_path, idx):
        '''
        load continues 16 frames
        Args:
            video_name (str) - video name
            idx (int) - video index
            start_frame - the start index of continues 16 frames
        Return:
            video (Tensor) - C(channel, 3) x L(frames, 16) x H x W
        '''
        
        video_path = "/home/uscc/USAI_Outsourcing/B-1/Jinag_thesis/backup_thesis/" + video_path
        video_name = video_path.split('/')[-1]
        print(video_name)
        # if os.path.exists(os.path.join(video_path, video_name + '_c.mp4')):
        #     total_clips = len(os.listdir(video_path)) - 4
        # else:
        #     total_clips = len(os.listdir(video_path)) - 3
        total_clips = len(os.listdir(video_path)) - 4

        # if total_clips < 16:
        #     print(total_clips)
        #     return None
        start_frame = random.randint(1, total_clips - 1 - 16)

        # plt.ion()
        for i in range(16):
            img_name = video_name + '_{:d}'.format(start_frame) + '.jpg'
            img_path = os.path.join(video_path, img_name)
            tmp_img = io.imread(img_path)
            # io.imshow(img_path)
            # plt.show()

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

        # show video clip

        return video

    def load_audio(self, audio_path, idx):
        audio_path = "/home/uscc/USAI_Outsourcing/B-1/Jinag_thesis/backup_thesis/" + audio_path
        audio_name = audio_path.split('/')[-1]
        audio_name = os.path.join(
            audio_path, audio_name + '.npy')
        return np.load(audio_name)

    def __getitem__(self, idx):
        name, target, score = self.data_info.iloc[idx, :]
        target %= 7

        # if name == 'fight_videos/ced/v0316_1_010':
        #     name = 'fight_videos/ced/v0316_4_060'
        # video - C x L(frames) x H x W
        video = self.load_video(name, idx)
        audio = self.load_audio(name, idx)
        audio = audio[:14]

        sample = {'video': video, 'target': target}

        if self.transform:
            sample = self.transform(sample)

        sample['audio'] = torch.from_numpy(audio).float().view(-1)
        score = np.array([score / 300])
        sample['score'] = torch.from_numpy(score).float()

        return sample