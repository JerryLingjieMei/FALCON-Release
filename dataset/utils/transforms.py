import random
from functools import wraps

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF

jitter = dict(brightness=0.4, contrast=0.4, saturation=0.4, hue=.4)
magnify_ratio = 1.15


def pop_single(func):
    @wraps(func)
    def wrapped_func(*args, **kwargs):
        result = func(*args, **kwargs)
        if isinstance(result, list) or isinstance(result, tuple):
            if len(result) == 1:
                return result[0]
        return result

    return wrapped_func


def to_tensor(x):
    if torch.is_tensor(x):
        return x
    else:
        return TF.to_tensor(x)


class RandomHorizontalFlip(T.RandomHorizontalFlip):
    def __call__(self, *args):
        if random.random() < self.p:
            return [TF.hflip(arg) for arg in args]
        return args


class ColorJitter(T.ColorJitter):
    def __call__(self, *args):
        if torch.is_tensor(args[-1]):
            assert args[-1].shape[0] == 3
        else:
            assert args[-1].mode == "RGB"
        return [super().__call__(arg) if _ == -1 else arg for _, arg in enumerate(args)]


class RandomCropTransform:
    is_fixed = False

    def __init__(self, image_size):
        self.resize = T.Resize([int(_ * magnify_ratio) for _ in image_size])
        self.center_crop = T.CenterCrop(image_size)
        self.color_jitter = ColorJitter(**jitter)
        self.random_horizontal_flip = RandomHorizontalFlip()

    @pop_single
    def __call__(self, *args, **kwargs):
        args = [self.center_crop(self.resize(arg)) for arg in args]
        for f in [self.color_jitter, self.random_horizontal_flip]:
            args = f(*args)
        args = [to_tensor(arg) for arg in args]
        return args


class FixedCropTransform:
    is_fixed = True

    def __init__(self, image_size):
        self.resize = T.Resize([int(_ * magnify_ratio) for _ in image_size])
        self.center_crop = T.CenterCrop(image_size)

    @pop_single
    def __call__(self, *args, **kwargs):
        args = [to_tensor(self.center_crop(self.resize(arg))) for arg in args]
        return args


class FixedResizeTransform:
    is_fixed = True

    def __init__(self, image_size):
        self.resize = T.Resize(image_size)

    @pop_single
    def __call__(self, *args, **kwargs):
        args = [to_tensor(self.resize(arg)) for arg in args]
        return args


class RandomTransform:
    is_fixed = False

    def __init__(self, image_size):
        self.color_jitter = ColorJitter(**jitter)
        self.random_horizontal_flip = RandomHorizontalFlip()

    @pop_single
    def __call__(self, *args, **kwargs):
        for f in [self.color_jitter, self.random_horizontal_flip]:
            args = f(*args)
        args = [to_tensor(arg) for arg in args]
        return args


class FixedTransform:
    is_fixed = True

    def __init__(self, image_size):
        pass

    @pop_single
    def __call__(self, *args, **kwargs):
        args = [to_tensor(arg) for arg in args]
        return args
