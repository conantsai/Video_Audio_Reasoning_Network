import torch
import numpy as np
from skimage import io, transform

# class ClipSubstractMean(object):
#     def __init__(self, r=123, g=117, b=104):
#         self.means = np.array([r, g, b])

#     def __call__(self, sample):
#         video, target = sample['video'], sample['target']

#         for frame in video:
#             video -= self.means

#         return {'video': video, 'target': target}


class Rescale(object):
    def __init__(self, output_size=(182, 242)):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        video, target = sample['video'], sample['target']

        # L x H x W x C
        # 16 x H x W x 3
        l, h, w = video.shape[0], video.shape[1], video.shape[2]

        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h // w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w // h
        else:
            new_h, new_w = self.output_size

        # new_h, new_w = int(new_h), int(new_w)
        new_video = np.zeros((l, new_h, new_w, 3))
        for i in range(l):
            image = video[i, :, :, :]
            img = transform.resize(image, (new_h, new_w), mode='constant')
            new_video[i, :, :, :] = img

        return {'video': new_video, 'target': target}


class RandomCrop(object):
    def __init__(self, output_size=(160, 160)):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        video, target = sample['video'], sample['target']
        # L x H x W x C
        # 16 x H x W x 3
        l, h, w = video.shape[0], video.shape[1], video.shape[2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        new_video = np.zeros((l, new_h, new_w, 3))
        for i in range(l):
            image = video[i, :, :, :]
            image = image[top: top + new_h, left: left + new_w]
            new_video[i, :, :, :] = image

        return {'video': new_video, 'target': target}


class ToTensor(object):
    def __call__(self, sample):
        '''
        Args:
            sample (dict): video -> numpy array, target -> int
        '''
        video, target = sample['video'], sample['target']

        l = video.shape[0]
        for i in range(l):
            video[i, :, :, :] = (video[i, :, :, :]) / 255

        # video - L(frames, 16) x H x W x C(channel, 3) -> C x L x H x W
        video = video.transpose((3, 0, 1, 2))
        target = np.array([target])

        return {'video': torch.from_numpy(video).float(), 'target': torch.from_numpy(target).long()}

class Normalize(object):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = torch.Tensor(mean)
        self.std = torch.Tensor(std)

    def __call__(self, sample):
        video, target = sample['video'], sample['target']
        
        # C x L x H x W -> L x C x H x W
        video = video.transpose(0, 1)
        l = video.shape[0]

        for i in range(l):
            video[i, :, :, :] = (video[i, :, :, :]).transpose(0, 2).sub_(self.mean).div_(self.std).transpose(0, 2)

        # L x C x H x W -> C x L x H x W
        video = video.transpose(0, 1)

        return {'video': video, 'target': target}