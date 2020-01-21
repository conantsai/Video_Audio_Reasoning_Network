import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial

from p3d_model import P3D199


class VARN(nn.Module):
    def __init__(self):
        super(VARN, self).__init__()
        self.video_layer = P3D199(pretrained=False, num_classes=101)
        self.audio_layer = nn.Sequential(
            nn.Conv1d(1, 16, 64, 2, padding=32),
            nn.MaxPool1d(8, 1, padding=0),
            nn.Conv1d(16, 32, 32, padding=16),
            nn.MaxPool1d(8, 1, padding=0),
            nn.Conv1d(32, 64, 16, 2, padding=8),
            nn.Conv1d(64, 128, 8, 2, padding=4),
            nn.Conv1d(128, 256, 4, 2, padding=2),
            nn.MaxPool1d(4, 1, padding=0),
            nn.Conv1d(256, 512, 4, 2, padding=2),
            nn.Conv1d(512, 1024, 4, 2, padding=2)
        )
        self.fusion = nn.Linear(16384, 4096)
        # self.reason_f1 = nn.Linear(2048, 1024)
        # self.reason_f2 = nn.Linear(1024, 512)
        # self.reason_f3 = nn.Linear(512, 7)

        # self.predict_f1 = nn.Linear(2048, 1024)
        # self.predict_f2 = nn.Linear(1024, 512)
        # self.predict_f3 = nn.Linear(512, 1)

        # Reason layer
        self.reason_1 = nn.Linear(4096, 2048)
        self.reason_2 = nn.Linear(2048, 512)
        self.reason_3 = nn.Linear(512, 7)
        # self.softmax = nn.Softmax()
        # Predict layer
        self.predict_1 = nn.Linear(4096, 2048)
        self.predict_2 = nn.Linear(2048, 512)
        self.predict_3 = nn.Linear(512, 1)
        # self.predict_4 = nn.Linear(1024, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        v, a = x
        v = self.video_layer(v)
        # a = self.audio_layer(audio)
        v = v.view(v.size(0), -1)
        a = a.view(v.size(0), -1)

        # print(v.shape, a.shape)
        x = torch.cat([v, a], 1)
        x = self.fusion(x)
        # print(x.shape)
        # x = torch.cat([v, a], 1)
        # print(x.shape)
        # x = torch.cat([v.view(-1), a.view(-1)], 0)
        # x = x.view(4, -1)

        x_1 = self.dropout(self.relu(self.reason_1(x)))
        x_1 = self.dropout(self.relu(self.reason_2(x_1)))
        x_1 = self.reason_3(x_1)

        x_2 = self.dropout(self.relu(self.predict_1(x)))
        x_2 = self.dropout(self.relu(self.predict_2(x_2)))
        x_2 = self.predict_3(x_2)

        # x_1 = self.dropout(self.relu(self.reason_f1(x)))
        # x_1 = self.dropout(self.relu(self.reason_f2(x_1)))
        # x_1 = self.reason_f3(x_1)

        # x_2 = self.dropout(self.relu(self.predict_f1(x)))
        # x_2 = self.dropout(self.relu(self.predict_f2(x_2)))
        # x_2 = self.predict_f3(x_2)

        return x_1, x_2


class VideoNet(nn.Module):
    def __init__(self, n_class=7):
        super(VideoNet, self).__init__()
        self.video_layer = P3D199(pretrained=False, num_classes=101)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

        self.fc = nn.Linear(2048, n_class)

        self.p1 = nn.Linear(2048, 1024)
        self.p2 = nn.Linear(1024, 512)
        self.p3 = nn.Linear(512, 1)

    def forward(self, x):
        v = self.video_layer(x)
        v = v.view(-1, self.fc.in_features)
        c = self.fc(self.dropout(v))

        r = self.dropout(self.relu(self.p1(v)))
        r = self.dropout(self.relu(self.p2(r)))
        r = self.p3(r)

        return c, r


class AudioNet(nn.Module):
    def __init__(self, n_class=7):
        super(AudioNet, self).__init__()
        # self.audio_layer = nn.Sequential(
        #     nn.Conv1d(1, 16, 64, 2, padding=32),
        #     nn.MaxPool1d(8, 1, padding=0),
        #     nn.Conv1d(16, 32, 32, padding=16),
        #     nn.MaxPool1d(8, 1, padding=0),
        #     nn.Conv1d(32, 64, 16, 2, padding=8),
        #     nn.Conv1d(64, 128, 8, 2, padding=4),
        #     nn.Conv1d(128, 256, 4, 2, padding=2),
        #     nn.MaxPool1d(4, 1, padding=0),
        #     nn.Conv1d(256, 512, 4, 2, padding=2),
        #     nn.Conv1d(512, 1024, 4, 2, padding=2)
        # )
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

        self.fc = nn.Linear(14336, n_class)

        self.p1 = nn.Linear(14336, 2048)
        self.p2 = nn.Linear(2048, 1024)
        self.p3 = nn.Linear(1024, 1)

    def forward(self, x):
        v = x.view(-1, self.fc.in_features)
        c = self.fc(self.dropout(v))

        r = self.dropout(self.relu(self.p1(v)))
        r = self.dropout(self.relu(self.p2(r)))
        r = self.p3(r)

        return c, r
