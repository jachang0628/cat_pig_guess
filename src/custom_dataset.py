import json
import numpy as np
import pandas as pd
from collections import OrderedDict
import torch
import torchvision
from torch import nn, optim
from torchvision import transforms, models, datasets
from torchsummary import summary
from IPython.display import Image
import matplotlib.pyplot as plt
import random
from torch.utils.data import Dataset
from PIL import Image
import PIL.ImageOps
class catdataset(Dataset):
    def __init__(self, img_dir = 'img', transform = None):
        self.img_dir = img_dir
        self.transform = transform
        self.cats = np.load(img_dir + '/full_numpy_bitmap_cat.npy')
        self.pigs = np.load(img_dir + '/full_numpy_bitmap_pig.npy')
        self.full = np.concatenate([self.cats, self.pigs])
        self.labels = np.concatenate([np.zeros(self.cats.shape[0]), np.ones(self.pigs.shape[0])])
        self.length = self.full.shape[0]
    def __getitem__(self, index):
        image = self.full[index].reshape(28,28)
        image = torch.tensor(image)
        image = image.repeat(3, 1, 1)
        y_label = torch.tensor(int(self.labels[index]))
        if self.transform:
            image = self.transform(image)
        return (image, y_label)

    def __len__(self):
        return self.length