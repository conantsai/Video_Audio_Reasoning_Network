import time
import gc
import resource
import shutil
import csv
import os

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from p3d_model import P3D199
from c3d_model import C3D
from video_transforms import *
from UCF101_dataloader import UCF101_train, UCF101_valid

LR = 0.001
RESUME = True
BEST_MODEL = 'model_best.pth.tar'
CHECKPOINT_MODEL = 'checkpoint.pth.tar'
_EPOCH = 1600


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


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
    correct = torch.sum(pred == target.view(-1), 0).float().data[0]

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


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()

    for i, batch_data in enumerate(train_loader):
        data_time.update(time.time() - end)
        video, target = Variable(batch_data['video']).cuda(
        ), Variable(batch_data['target']).cuda()

        if len(video) == 0:
            continue

        output = model(video)
        loss = criterion(output, target.view(-1))

        prec1 = accuracy(output, target)
        losses.update(loss.data[0], video.size(0))
        top1.update(prec1, video.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % 100 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                      epoch, i, len(train_loader),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses, top1=top1))


def validate(valid_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()

    for i, batch_data in enumerate(valid_loader):
        video, target = Variable(batch_data['video']).cuda(
        ), Variable(batch_data['target']).cuda()

        output = model(video)
        loss = criterion(output, target.view(-1))

        prec1 = accuracy(output, target)
        losses.update(loss.data[0], video.size(0))
        top1.update(prec1, video.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % 100 == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                      i, len(valid_loader), batch_time=batch_time, loss=losses, top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))

    return top1.avg


def main():
    gc.collect()
    torch.cuda.empty_cache()

    ucf101_train = UCF101_train('ucfInfo/trainlist03.csv',
                                'ucf_jpegs_256',
                                transforms.Compose([
                                    Rescale(),
                                    RandomCrop(),
                                    ToTensor(),
                                    Normalize()]))
    ucf101_valid = UCF101_valid('ucfInfo/validlist03.csv',
                                'ucf_jpegs_256',
                                transforms.Compose([
                                    Rescale(),
                                    RandomCrop(),
                                    ToTensor(),
                                    Normalize()]))

    train_loader = DataLoader(ucf101_train, batch_size=4, shuffle=True,
                              num_workers=2, pin_memory=True)
    valid_loader = DataLoader(ucf101_valid, batch_size=4, shuffle=False,
                              num_workers=2)

    model = P3D199(pretrained=False, num_classes=101)
    criterion = nn.CrossEntropyLoss()
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
    model = model.cuda()

    start = time.time()
    for epoch in range(start_epoch, _EPOCH):
        adjust_learning_rate(optimizer, epoch)

        train(train_loader, model, criterion, optimizer, epoch)
        prec1 = validate(valid_loader, model, criterion)

        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        # save model state
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': 'P3D',
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best)

    print('Training_set: 03')
    print('Execute Time: ', time.time() - start)
    gc.collect()
    torch.cuda.empty_cache()


def listinfo():
    model = P3D199(pretrained=False, num_classes=101)
    print(model)


if __name__ == '__main__':
    main()
    # listinfo()
