
import numpy as np

import torch
import torch.nn  as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim

from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score, accuracy_score

from utils import to_categorical
from sklearn.utils import shuffle

import IPython


class AbstractModel(nn.Module):

	def __init__(self, targets, optimizer=optim.Adadelta, lr=0.01):

		super(AbstractModel, self).__init__()

		self.targets = targets
		self.build_model()
		self.cuda()
		self.optimizer = optimizer(self.parameters(), lr=lr)

	def build_model(self):
		raise NotImplementedError()
	
	def fit(self, data_generator, epochs=20):
		
		for i in range(0, epochs):
			print ("ERA {}: ".format(i))

			train_gen = data_generator.data(mode='train')
			val_gen = data_generator.data(mode='val')

			train_scores = self.evaluate(train_gen, training=True)
			val_scores = self.evaluate(val_gen, training=False)
			print ("Training scores:", train_scores)
			print ("Testing scores:", val_scores)

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
			#print (Y_pred[target_idx], Y_target)

			if target == 'regression':
				loss += (Y_pred[target_idx] - Y_target.cuda()) ** 2
			else:
				loss += F.cross_entropy(Y_pred[target_idx], Y_target.cuda())

		self.eval()

		loss.backward(retain_variables=True)
		self.optimizer.step()

	def evaluate(self, data_gen, training=False):

		preds_history = {}
		targets_history = {}
		scores = {}

		for X, Y in data_gen:
			
			if training: self.fit_on_batch(X, Y)

			self.train(training)
			Y_pred = self(X)

			for target in Y.keys():
				if Y[target] is None: continue
				if target not in self.targets: continue
				target_idx = self.targets.index(target)

				if target not in preds_history:
					preds_history[target] = []
					targets_history[target] = []

				ypred = Y_pred[target_idx][0].cpu().data.numpy()

				if target != 'regression':
					ytarget = to_categorical([Y[target]], num_classes=self.classes[target_idx])[0]
				else:
					ytarget = Y[target]
				preds_history[target].append(ypred)
				targets_history[target].append(ytarget)
				
		for target in preds_history.keys(): 
			preds_history[target] = np.array(preds_history[target])
			targets_history[target] = np.array(targets_history[target])

			if target != 'regression':
				mask = targets_history[target].sum(axis=0) > 2
				if mask.sum() < 2: continue
				print (target, targets_history[target].shape[0])

			score, baseline = 0.0, 0.0
			if target == 'regression':
				#score = pearsonr (targets_history[target], preds_history[target][:, 0])
				#baseline = pearsonr (targets_history[target], shuffle(preds_history[target][:, 0]))
				#scores[target] = "{0:0.3f}(p={1:0.4f})/{2:0.3f}(p={3:0.4f})".format(score[0], score[1], baseline[0], baseline[1])
				score = concordance_index(targets_history[target], preds_history[target][:, 0],
					event_observed=targets_history[target]<1.0)
				baseline = concordance_index(targets_history[target], shuffle(preds_history[target][:, 0]),
					event_observed=targets_history[target]<1.0)
				scores[target] = "{0:0.3f}/{1:0.3f}".format(score, baseline)

			else:
				score = roc_auc_score (targets_history[target][:, mask], preds_history[target][:, mask])
				baseline = roc_auc_score (targets_history[target][:, mask], shuffle(preds_history[target][:, mask]))
				scores[target] = "{0:0.3f}/{1:0.3f}".format(score, baseline)

		return scores

	def save(self, file_name="results/model.pth"):
		torch.save(self.cpu().state_dict(), open(file_name, 'w'))

	def predict(self, datagen):
		predictions = []
		for X, Y in datagen:
			prediction = self(X)
			regression_pred = prediction[self.targets.index('regression')]
			regression_pred = regression_pred.cpu().data.numpy().mean()
			features = self.extract(X)
			Y['regression_pred'] = regression_pred
			Y['features'] = features.cpu().data.numpy()
			predictions.append(Y)
		return predictions

	def load(self, file_name="results/model.pth"):
		state = torch.load(open(file_name))
		self.load_state_dict(state)
		self.cuda()

	def forward(self, x):
		raise NotImplementedError()




class AbstractMultitaskModel(AbstractModel):

	def __init__(self, targets, classes=[2, 2, 10], latent_dim=128,
			optimizer=optim.Adadelta, lr=0.01):

		self.classes = classes
		self.latent_dim = latent_dim
		
		super(AbstractMultitaskModel, self).__init__(targets, optimizer=optimizer, lr=lr)

	def build_model(self):
		self.output_layers = nn.ModuleList([nn.Linear(self.latent_dim, class_num) \
			for class_num in self.classes])
		self.regression = nn.Linear(self.latent_dim, 1)

	def forward(self, x):
		x = self.extract(x)
		y = self.regression(x)
		x = [layer(x) for layer in self.output_layers]
		x = [F.softmax(pred) for pred in x] + [y]
		return x

	def extract(self, x):
		x = Variable(torch.Tensor(x).float().unsqueeze(0), requires_grad=False)
		return x




if __name__ == "__main__":
	IPython.embed()