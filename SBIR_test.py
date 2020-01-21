import time
import gc
import resource
import shutil
import csv
import os
from random import randint
import pandas as pd
import cv2
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from ced_dataloader import CED
from va3d_model import VARN, VideoNet, AudioNet
from p3d_model import P3D199
from video_transforms import *
from soundnet.SBIR_extract_feat import soundnet_main

LABELS = {0: 'non-conflict',
          1: 'personal verbal conflict',
          2: 'personal physical conflict',
          3: 'personal physical conflict with weaposns',
          4: 'group verbal conflict',
          5: 'group physical conflict',
          6: 'group physical conflict with weapons'}

BEST_MODEL = 'varn_best.pth.tar'
CHECKPOINT_MODEL = 'varn_checkpoint.pth.tar'
LR = 0.001

        
def test(test_loader, model):
    model.eval()
    statistic = {}
    criterion_1 = nn.CrossEntropyLoss()
    
    for _, batch_data in enumerate(test_loader):
        video = Variable(batch_data['video']).cuda()
        audio = Variable(batch_data['audio']).cuda()
        target = Variable(batch_data['target']).cuda()
        score = Variable(batch_data['score']).cuda()

        output = model((video, audio))

        print(output[0], target.view(-1))
        # print(output[1], score.view(-1))
        # print(criterion_1(output[0], target.view(-1)).item())

        _, pred = torch.max(output[0], 1)
        print(_, pred)


    return 

def capture_frame(video_name, dst):
    cap = cv2.VideoCapture(dst + '/' + video_name + '.mp4')

    frame_count = 1
    success = True

    while(success):
        success, frame = cap.read()
        file_name = dst + '/' + video_name + '_%d.jpg' % frame_count
        if os.path.exists(file_name):
            break
        if success:
            cv2.imwrite(file_name, frame)
            frame_count += 1

def main():
    # clip vedio to image
    df = pd.read_csv('B-1/Jinag_thesis/backup_thesis/test/ced_testing.csv')

    for _, row in df.iterrows():
        name = row['name'].split('/')[-1]
        capture_frame(name, 'B-1/Jinag_thesis/backup_thesis/test/' + row['name'])

        soundnet_main("B-1/Jinag_thesis/backup_thesis/test/" + name + "/" + name + ".mp3", "B-1/Jinag_thesis/backup_thesis/test/" + name)

    

    # torch.cuda.empty_cache()
    # ced_test = CED('/home/uscc/USAI_Outsourcing/B-1/Jinag_thesis/backup_thesis/ucfInfo/ced_testing.csv', 'fight_videos/ced',
    #                transforms.Compose([Rescale(), RandomCrop(), ToTensor(), Normalize()]))

    # test_loader = DataLoader(ced_test, batch_size=4, shuffle=True, num_workers=1, pin_memory=True)

    # model = VARN()
    # optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)

    # checkpoint = torch.load(BEST_MODEL)
    # start_epoch = checkpoint['epoch']
    # best_prec1 = checkpoint['best_prec1']
    # model.load_state_dict(checkpoint['state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer'])
    # print("=> loaded checkpoint '{}' (epoch {}) Prec@1 {}" .format(CHECKPOINT_MODEL, checkpoint['epoch'], best_prec1))

    # model = model.cuda()

    # test(test_loader, model)

if __name__ == '__main__':
    main()