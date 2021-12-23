import numpy as np
import torch
import random
from PIL import Image

class ToTensor(object):
    """Convert cv2 ndarrays in sample to Tensors."""

    def __init__(self):
        pass

    def __call__(self, imgs):
        # swap color axis because
        # cv2 numpy image: H x W x C
        # torch image: C X H X W
        converted = []
        for img in imgs:
            if len(np.array(img).shape) == 3:
                img = np.array(img).astype(np.float32).transpose((2, 0, 1))
            else:
                img = np.expand_dims(np.array(img).astype(np.float32), -1).transpose((2, 0, 1))
            img = torch.from_numpy(img).float()
            converted.append(img/255)
        return converted

class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, imgs):
        if random.random() < self.prob:
            converted = []
            for img in imgs:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                converted.append(img)
            return converted
        else:
            return imgs

class FixedResize(object):
    def __init__(self, size):
        self.size = tuple(size)

    def __call__(self, imgs):
        converted = []
        for img in imgs:
            img = img.resize(self.size, Image.BILINEAR)
            converted.append(img)
        return converted