
import numpy as np

import torch
import torch.nn  as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim

from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score, accuracy_score

from sklearn.utils import shuffle

from dnn.models import AbstractModel, AbstractMultitaskModel

import matplotlib.pyplot as plt

import IPython, random


class DilatedCNNExtractor(AbstractMultitaskModel):

	def __init__(self, *args, **kwargs):
		self.input_dim = kwargs.pop('input_dim')
		super(DilatedCNNExtractor, self).__init__(*args, **kwargs)

	def build_model(self):
		super(DilatedCNNExtractor, self).build_model()
		self.vgg1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=(3, 3), padding=1, dilation=1), nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=(3, 3), padding=1, dilation=1), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=(3, 3), padding=1, dilation=1), nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=1))
		self.vgg2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(3, 3), padding=2, dilation=2), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(3, 3), padding=2, dilation=2), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=2, dilation=2), nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=1))
		self.vgg3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=4, dilation=4), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=4, dilation=4), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=4, dilation=4), nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=1))
		self.vgg4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=8, dilation=8), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=8, dilation=8), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=8, dilation=8), nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=1))
		self.vgg5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=16, dilation=16), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=16, dilation=16), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=16, dilation=16), nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=1))
		self.linear = nn.Linear(64, self.latent_dim)

	def extract(self, x):

		x = Variable(torch.Tensor(x).float().cuda())
		x = x.transpose(1, 3)

		x = self.vgg1(x)
		x = self.vgg2(x)
		x = self.vgg3(x)
		x = self.vgg4(x)
		x = self.vgg5(x)

		x = x.transpose(0, 1)
		x = x.clone()
		x = x.view(64, -1)

		x = x.mean(dim=1)
		x = x.transpose(0, 1)
		x = F.relu(self.linear(x))

		return x







