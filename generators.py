
#code for data generation from heatmaps
import os, sys, random, yaml, itertools
from tqdm import tqdm

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import glob, sys, os, random, time, logging, threading, subprocess

from sklearn.model_selection import train_test_split

from data import fetch

import IPython


class AbstractPatientGenerator(object):

	def __init__(self, cases=fetch.cases, samples=500, val_samples=100, verbose=False):
		super(AbstractPatientGenerator, self).__init__()
		self.train_cases, self.val_cases = train_test_split(list(cases), test_size=0.15)
		self.samples, self.val_samples = samples, val_samples
		self.verbose = verbose

	def data(self, mode='train', cases=None):

		case_list = self.train_cases if mode == 'train' else self.val_cases
		num_samples = self.samples if mode == 'train' else self.val_samples

		cases = cases or (random.choice(case_list) for i in itertools.repeat(0))
		samples = (self.sample(case, mode=mode) for case in cases)
		samples = itertools.islice((x for x in samples if x is not None), num_samples)

		return samples

	def sample(self, case, mode='train'):
		raise NotImplementedError()

if __name__ == "__main__":
	x = AbstractPatientGenerator()
	data, target = x.data(mode='train')
	IPython.embed()








