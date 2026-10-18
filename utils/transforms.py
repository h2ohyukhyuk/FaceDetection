import torch
import torchvision.transforms.functional as F
import numpy as np
import random
import cv2


class PixelNoise(object):

    def __init__(self, noise_factor = 0.4):
        self.noise_factor = noise_factor

    @staticmethod
    def get_params( noise_factor ):

        # Each channel is multiplied with a number
        # in the area [1-opt.noiseFactor,1+opt.noiseFactor]
        pn = np.random.uniform(1 - noise_factor, 1 + noise_factor, 3)

        return pn

    def __call__(self, img):

        pn = self.get_params(self.noise_factor)

        img[:, :, 0] = np.minimum(255.0, np.maximum(0.0, img[:, :, 0] * pn[0]))
        img[:, :, 1] = np.minimum(255.0, np.maximum(0.0, img[:, :, 1] * pn[1]))
        img[:, :, 2] = np.minimum(255.0, np.maximum(0.0, img[:, :, 2] * pn[2]))

        return img


class Gamma(object):

    def __init__(self, noise_factor = 0.4):
        self.noise_factor = noise_factor

    @staticmethod
    def get_params( gamma_range ):

        gamma_shift = random.uniform(-gamma_range, gamma_range)
        invGamma = 1.0 / (1.0 + gamma_shift)
        return invGamma

    def __call__(self, img):

        invGamma = self.get_params(self.noise_factor)

        table = np.array([((i / 255.0) ** invGamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")

        img = cv2.LUT(img, table)

        return img

class Blur(object):

    def __init__(self, blur = 0.5, kernel_range=(3, 11)):
        self.blur = blur
        self.kernel_range = kernel_range

    @staticmethod
    def get_params( blur ):
        prob = random.uniform(0.0, 1.0)

        if prob < blur:
            return True

        return False

    def __call__(self, img):

        boolBlur = self.get_params(self.blur)

        if boolBlur:
            size_gaussian_blur = random.randint(*self.kernel_range)
            if size_gaussian_blur % 2 == 0:
                size_gaussian_blur = size_gaussian_blur + 1
            img = cv2.GaussianBlur(img, (size_gaussian_blur, size_gaussian_blur), sigmaX=0)

        return img

class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transform.Compose([
        >>>      transform.RandomResized(),
        >>>      transform.RandomHorizontalFlip(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):

        for t in self.transforms:
            img = t(img)

        return img
