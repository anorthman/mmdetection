import mmcv
import numpy as np
from PIL import Image
from numpy import random


class MixUp(object):
    def __init__(self, alpha):
        self.alpha = alpha
        self.lambd = 1

    def __call__(self, img1, img2, box1, box2, labels1, labels2):
        lambd = np.random.beta(self.alpha, self.alpha)
        lambd = max(0, min(1, lambd))

        if lambd >= 1:
            weights1 = np.ones((labels1.shape[0], 1))
            label1 = np.hstack((labels1, weights1))
            return img1, label1

        height = max(img1.shape[0], img2.shape[0])
        width = max(img1.shape[1], img2.shape[1])
        mix_img = np.zeros(shape=(height, width, img1.shape[2]), dtype='float32')

        mix_img[:img1.shape[0], :img1.shape[1], :] = img1.astype('float32') * lambd
        mix_img[:img2.shape[0], :img2.shape[1], :] += img2.astype('float32') * (1. - lambd)

        labels1 = np.reshape(labels1, [labels1.shape[0], 1])
        labels2 = np.reshape(labels2, [labels2.shape[0], 1])
        ratio1 = np.full((len(labels1), 1), lambd)
        ratio2 = np.full((len(labels2), 1), 1. - lambd)
        y1 = np.hstack((labels1, ratio1))
        y2 = np.hstack((labels2, ratio2))
        mix_labels = np.vstack((y1, y2))

        mix_box = np.concatenate([box1, box2], axis=0)
        mix_weights = np.array(mix_labels[:, 1], dtype=np.float32)
        mix_labels = np.array(mix_labels[:, 0], dtype=np.int)

        return mix_img, mix_box, mix_labels, mix_weights
