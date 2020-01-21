import time
import gc
import resource
import shutil
import csv
import os
from random import randint

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


LABELS = {0: 'non-conflict',
          1: 'personal verbal conflict',
          2: 'personal physical conflict',
          3: 'personal physical conflict with weaposns',
          4: 'group verbal conflict',
          5: 'group physical conflict',
          6: 'group physical conflict with weapons'}
LR = 0.001
RESUME = True
IS_TEST = False
M = 'VA'
'''
model_best.pth.tar
checkpoint.pth.tar

vn_best.pth.tar
vn_checkpoint.pth.tar

an_best.pth.tar
an_checkpoint.pth.tar

varn_best.pth.tar
varn_f_best.pth.tar
--no pretrain--
varn_f_nop_best.pth.tar
varn_nop_best.pth.tar
'''
BEST_MODEL = 'varn_best.pth.tar'
CHECKPOINT_MODEL = 'varn_checkpoint.pth.tar'
_EPOCH = 50


def save_checkpoint(state, is_best, filename=CHECKPOINT_MODEL):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, BEST_MODEL)


def adjust_learning_rate(optimizer, epoch):
    '''
    reduce learning rate
    '''
    lr = LR * (0.1 ** (epoch // 1000))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target):
    batch_size = target.size(0)

    _, pred = torch.max(output, 1)
    correct = torch.sum(pred == target.view(-1), 0).float().item()

    res = correct * 100.0 / batch_size

    return res


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def save(self, output='ucfInfo/ucf101_record.csv'):
        if os.path.exists(output):
            csv_writer = csv.writer(open(output, 'a'))
        else:
            csv_writer = csv.writer(open(output, 'w'))
            csv_writer.writerow(['name', 'average', 'sum', 'count'])

        csv_writer.writerow(['', self.avg, self.sum, self.count])


def train(train_loader, model, criterion_1, criterion_2, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()

    for i, batch_data in enumerate(train_loader):
        data_time.update(time.time() - end)
        video, audio = Variable(batch_data['video']).cuda(
        ), Variable(batch_data['audio']).cuda()
        target, score = Variable(batch_data['target']).cuda(
        ), Variable(batch_data['score']).cuda()

        # if len(video) == 0:
        #     continue
        # (video, audio)
        output = model((video, audio))
        # print(output[1].data[0], score.data[0])
        # assert ((output[1] >= 0.) & (output[1] <= 1.)).all()
        # loss = criterion_1(output[0], target.view(-1))
        print('loss1', criterion_1(output[0], target.view(-1)).item())
        print('loss2', criterion_2(output[1], score.view(-1)).item())

        loss = criterion_1(output[0], target.view(-1)) + \
            criterion_2(output[1], score.view(-1))

        prec1 = accuracy(output[0], target)
        # print(loss.data[0], video.size(0))
        losses.update(loss.item(), video.size(0))
        top1.update(prec1, video.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                      epoch, i, len(train_loader),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses, top1=top1))


def validate(valid_loader, model, criterion_1, criterion_2):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()

    for i, batch_data in enumerate(valid_loader):
        video, audio = Variable(batch_data['video']).cuda(
        ), Variable(batch_data['audio']).cuda()
        target, score = Variable(batch_data['target']).cuda(
        ), Variable(batch_data['score']).cuda()

        # (video, audio)
        output = model((video, audio))

        print('loss1', criterion_1(output[0], target.view(-1)).item())
        print('loss2', criterion_2(output[1], score.view(-1)).item())
        loss = criterion_1(output[0], target.view(-1)) + \
            criterion_2(output[1], score.view(-1))

        prec1 = accuracy(output[0], target)
        losses.update(loss.item(), video.size(0))
        top1.update(prec1, video.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            print('Valid: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                      i, len(valid_loader), batch_time=batch_time, loss=losses, top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))

    return top1.avg


def test(test_loader, model, criterion_1, criterion_2):
    model.eval()
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()
    statistic = {}

    for _, batch_data in enumerate(test_loader):
        video, audio = Variable(batch_data['video']).cuda(
        ), Variable(batch_data['audio']).cuda()
        target, score = Variable(batch_data['target']).cuda(
        ), Variable(batch_data['score']).cuda()

        # target = batch_data['target']

        # for i in range(4):
        #     key = target[i][0]
        #     if statistic.get(key) != None:
        #         statistic[key] += 1
        #     else:
        #         statistic[key] = 1

        #     continue
        # output = model((video, audio))
        bs = target.size(0)
        random_o = np.array([randint(0, 6) for _ in range(bs)])
        random_o = Variable(torch.from_numpy(random_o)).cuda()

        correct = torch.sum(random_o == target.view(-1), 0).float().item()
        res = correct * 100.0 / bs
        top1.update(res, video.size(0))

        # _, pred = torch.max(output[0], 1)

        # prec1 = accuracy(output[0], target)
        # top1.update(prec1, video.size(0))

        # # print('loss1', criterion_1(output[0], target.view(-1)).data[0])
        # # print('loss2', criterion_2(output[1], score.view(-1)).data[0])

        # loss = criterion_1(output[0], target.view(-1)) + \
        #     criterion_2(output[1], score.view(-1))
        # losses.update(loss.data[0], video.size(0))

        # target = batch_data['target'][0]
        # print(LABELS.get(pred.data[0]))
        # print(LABELS.get(target[0]))

        # batch_time.update(time.time() - end)
        # end = time.time()
        # break

    # print(batch_time.avg)
    # print(losses.avg)
    # print(top1.avg)
    # print('Test: [0/{0}]\t'
    #       'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
    #       'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
    #       'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
    #           len(test_loader), batch_time=batch_time, loss=losses, top1=top1))

    # print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
    return top1, losses


def test_display(test_loader, model):
    for _, batch_data in enumerate(test_loader):
        video, audio = batch_data['video'], batch_data['audio']
        target, score = batch_data['target'], batch_data['score']

        # print(video)
        video = video.numpy()
        for i in range(16):
            io.imshow(video[i].item())
            plt.show()

        break


def main():
    gc.collect()
    torch.cuda.empty_cache()

    ced_train = CED('Jinag_thesis/backup_thesis/fight_videos/ced_training.csv', 'fight_videos/ced',
                    transforms.Compose([
                        Rescale(),
                        RandomCrop(),
                        ToTensor(),
                        Normalize()]))
    ced_valid = CED('Jinag_thesis/backup_thesis/fight_videos/ced_valid.csv', 'fight_videos/ced',
                    transforms.Compose([
                        Rescale(),
                        RandomCrop(),
                        ToTensor(),
                        Normalize()]))
    ced_test = CED('Jinag_thesis/backup_thesis/fight_videos/ced_testing.csv', 'fight_videos/ced',
                   transforms.Compose([
                       Rescale(),
                       RandomCrop(),
                       ToTensor(),
                       Normalize()]))

    train_loader = DataLoader(ced_train, batch_size=4, shuffle=True,
                              num_workers=1, pin_memory=True)
    valid_loader = DataLoader(ced_valid, batch_size=2, shuffle=False,
                              num_workers=1)
    test_loader = DataLoader(ced_test, batch_size=4, shuffle=True,
                             num_workers=1, pin_memory=True)

    # model = VA3D()
    if M == 'VA':
        model = VARN()
        # checkpoint = torch.load('checkpoint.pth.tar')
        # model.video_layer.load_state_dict(checkpoint['state_dict'])
    elif M == 'VN':
        model = VideoNet()
        # checkpoint = torch.load('checkpoint.pth.tar')
        # model.video_layer.load_state_dict(checkpoint['state_dict'])
    elif M == 'AN':
        model = AudioNet()

    criterion_1 = nn.CrossEntropyLoss()
    criterion_2 = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)

    start_epoch = 0
    # baseline = 1 / 101
    best_prec1 = 0.009
    # pretrain model
    if RESUME:
        if os.path.isfile(CHECKPOINT_MODEL):
            checkpoint = torch.load(CHECKPOINT_MODEL)
            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {}) Prec@1 {}"
                  .format(CHECKPOINT_MODEL, checkpoint['epoch'], best_prec1))
        else:
            print("=> no checkpoint found at '{}'".format(CHECKPOINT_MODEL))

    # model = C3D().cuda()
    # model_best.pth.tar

    # model.video_layer = P3D199(pretrained=False, num_classes=101)

    # model.video_layer.cuda()

    model = model.cuda()

    start = time.time()

    if IS_TEST:
        # test_display(test_loader, model)
        # test(test_loader, model, criterion_1, criterion_2)
        avg = 0
        losses = 0
        for _ in range(100):
            top, loss = test(test_loader, model, criterion_1, criterion_2)
            avg += top.avg
            losses += loss.avg

        avg /= 100
        losses /= 100

        print(' * Prec@1 {:.3f}'.format(avg))
        # print(' * Prec@1 {:.3f}'.format(losses))
    else:
        for epoch in range(start_epoch, _EPOCH):
            adjust_learning_rate(optimizer, epoch)

            train(train_loader, model, criterion_1,
                  criterion_2, optimizer, epoch)
            prec1 = validate(valid_loader, model, criterion_1, criterion_2)

            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)

            # save model state
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': 'VARN',
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
            }, is_best)

    print('Execute Time: ', time.time() - start)
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()