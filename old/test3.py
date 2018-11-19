
import os, sys, random, yaml, gc
from itertools import *

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score

from utils import *
from models import TrainableModel
import IPython



def get_mnist_mean_std():
	MNIST = datasets.MNIST('data', train=True, transform=transforms.ToTensor(), target_transform=None, download=True)
	_sum, _squred_sum = 0, 0
	for idx, data in enumerate(MNIST):
		img, label = data
		_batch, _h, _w = img.size()
		_sum += torch.sum(img)
		_squred_sum += torch.sum(img ** 2)
	mean = _sum / len(MNIST) / _h / _w
	var = _squred_sum / len(MNIST) / _h / _w - mean ** 2
	std = var ** 0.5
	print(f'mean : {mean}, std : {std}')
	return mean, std

def z_normalize_tensor(tensor, mean, std):
	return (tensor - mean) / std


class Network(TrainableModel):

	def __init__(self):
		super(Network, self).__init__()

		self.input_mean, self.input_std = get_mnist_mean_std()
		self.fc1 = nn.Linear(28*28, 256)
		self.fc2 = nn.Linear(256, 10)

		nn.init.normal(self.fc1.weight, mean=0, std=(28*28)**(-0.5))
		nn.init.normal(self.fc2.weight, mean=0, std=(256)**(-0.5))

	def forward(self, data, mask):

		x = data["data"]
		x = z_normalize_tensor(x, self.input_mean, self.input_std)
		x = F.selu(self.fc1(x.view(x.shape[0], -1)))
		x = F.alpha_dropout(x, 0.3, self.training)
		x = self.fc2(x)

		return {"pred": F.log_softmax(x, dim=1)}

	def loss(self, pred, target):
		return F.nll_loss(pred["pred"], target["pred"])

	def score(self, pred, target):
		classes = pred["pred"].argmax(axis=1)
		return {"Accuracy": accuracy_score(classes, target["pred"])}


"""Train model on MNIST dataset."""

model = Network()
model.compile(optim.SGD, lr=0.01, momentum=0.5, nesterov=True)
print (model)

train_loader = torch.utils.data.DataLoader(
		datasets.MNIST('../data', train=True, download=True,
					   transform=transforms.Compose([
						   transforms.ToTensor(),
						   transforms.Normalize((0.1307,), (0.3081,))
					   ])), 
				batch_size=32, shuffle=True)

val_loader = torch.utils.data.DataLoader(
		datasets.MNIST('../data', train=False, download=True,
					   transform=transforms.Compose([
						   transforms.ToTensor(),
						   transforms.Normalize((0.1307,), (0.3081,))
					   ])), 
				batch_size=32, shuffle=True)

def datagen(loader):
	for i, (data, target) in enumerate(train_loader):
		for X, Y in zip(data, target):
			# X = 1 MNIST example
			# Y = target class
			yield {"data": X}, {"pred": Y}


for epochs in range(0, 10):
	train_data = batched(datagen(train_loader), batch_size=32)
	val_data = batched(datagen(val_loader), batch_size=32)
	model.fit(train_data, validation=val_data, verbose=True)

