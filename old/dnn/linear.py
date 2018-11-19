
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

import IPython, random



class LinearExtractor(AbstractMultitaskModel):

	def __init__(self, *args, **kwargs):
		self.input_dim = kwargs.pop('input_dim')
		super(LinearExtractor, self).__init__(*args, **kwargs)

	def build_model(self):
		super(LinearExtractor, self).build_model()
		self.layer = nn.Linear(self.input_dim, self.latent_dim)

	def extract(self, x):
		#x = x
		#if random.randint(0, 1000) == 0: IPython.embed()
		x = super(LinearExtractor, self).extract(x)
		return F.dropout(F.relu(self.layer(x.cuda())), p=0.45, training=self.training)


