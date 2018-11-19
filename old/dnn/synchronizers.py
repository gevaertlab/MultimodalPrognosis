
import numpy as np

import torch
import torch.nn  as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim

from scipy.stats import pearsonr
from sklearn.utils import shuffle

from dnn.models import AbstractModel, AbstractMultitaskModel

import os, sys, random
import IPython

import sys


class NormSynchronizer(AbstractMultitaskModel):

	def __init__(self, *args, **kwargs):

		self.inputs = kwargs.pop('inputs')
		self.extractors = kwargs.pop('extractors')
		self.random_encodings = []
		self.cluster_error = []
		self.neighborhood_error = []
		self.train_error = []
		
		super(NormSynchronizer, self).__init__(*args, **kwargs)

	def build_model(self):
		super(NormSynchronizer, self).build_model()
		self.extractors = nn.ModuleList(self.extractors)

	def fit_on_batch(self, X, Y):
		self.zero_grad()
		loss = 0.0

		self.train()
		Y_pred = self(X)

		for target in Y.keys():
			if Y[target] is None: continue
			if target not in self.targets: continue
			target_idx = self.targets.index(target)
			
			Y_target = torch.Tensor([int(Y[target])])
			if target != 'regression': Y_target = Y_target.long()
			Y_target = Variable(Y_target, requires_grad=False)

			if target == 'regression':
				loss += (Y_pred[target_idx] - Y_target.cuda()) ** 2
			else:
				loss += F.cross_entropy(Y_pred[target_idx], Y_target.cuda())
		
		#print ("Noncluster error: ", loss.cpu().data.numpy().mean())
		self.train_error.append(loss.cpu().data.numpy())
		loss += self.cluster_loss(X)

		try:
			loss.backward(retain_variables=False)
			self.optimizer.step()
		except ZeroDivisionError:
			pass

		self.eval()

	def extract(self, x):

		vectors = []
		for input_type in x:
			if x[input_type] is None: continue
			extract_idx = self.inputs.index(input_type)
			data = self.extractors[extract_idx].extract(x[input_type])
			data = (data - data.mean().expand_as(data))/(data.norm().expand_as(data) + 1e-8)
			vectors.append(data)
		center = sum(vectors)/(len(vectors)*1.0)
		self.random_encodings.append(center.cpu().data.numpy())
		if len(self.random_encodings) > 10:
			del self.random_encodings[0]
		return F.dropout(center, p=0.3, training=self.training)

	def cluster_loss(self, x):

		y = None
		vectors = []
		for input_type in x:
			if x[input_type] is None: continue
			extract_idx = self.inputs.index(input_type)
			data = self.extractors[extract_idx].extract(x[input_type])
			data = (data - data.mean().expand_as(data))/(data.norm().expand_as(data) + 1e-8)
			vectors.append(data)
		center = sum(vectors)/(len(vectors)*1.0)
		cluster_error = sum([torch.norm(vector - center) for vector in vectors])*4
		
		neighborhood_error = sum([torch.norm(center - Variable(torch.Tensor(vector).cuda(), requires_grad=False))\
			for vector in self.random_encodings])

		if random.randint(0, 500) == 0:
			print ("Cluster error: ", cluster_error.cpu().data.numpy().mean())
			print ("Neighborhood error: ", neighborhood_error.cpu().data.numpy().mean())
		
		self.cluster_error.append(cluster_error.cpu().data.numpy().mean())
		self.neighborhood_error.append(neighborhood_error.cpu().data.numpy().mean())

		return cluster_error - neighborhood_error


