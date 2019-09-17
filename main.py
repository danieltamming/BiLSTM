import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm
from time import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim import Adam
import random

import argparse
import json

from graphs.models.bilstm import BiLSTM
from graphs.losses.loss import CrossEntropyLoss
from datasets.procon import ProConDataLoader

random.seed(314)

class ProConAgent:
	def __init__(self, config):
		self.input_length = config.input_length
		self.word2vec_dim = config.word2vec_dim
		self.num_classes = config.num_classes
		self.num_epochs = config.num_epochs
		self.embed_filename = config.embed_filename
		self.train_path = config.train_path
		self.test_path = config.test_path
		self.loaders = ProConDataLoader(self.train_path, self.test_path, self.embed_filename)
		self.model = BiLSTM(self.word2vec_dim, self.num_classes)
		if torch.cuda.is_available(): self.model = self.model.cuda()
		self.loss = CrossEntropyLoss()
		self.optimizer = Adam(self.model.parameters())

	def run(self):
		for fold_count in range(5):
			dictionary = self.loaders.getFold(fold_count)
			self.train_loader = dictionary['train_loader']
			self.val_loader = dictionary['val_loader']
			self.model.reset_parameters()
			self.train()

	def train(self):
		for i in range(1, self.num_epochs+1):
			self.train_one_epoch()
			self.validate()

	def train_one_epoch(self):
		train_loss = 0
		correct = 0
		self.model.train()

		for x, y in self.train_loader:
			x = x.float()
			if torch.cuda.is_available(): 
				x = x.cuda()
				y = y.cuda()
			pred = self.model(x)
			current_loss = self.loss(pred, y)
			self.optimizer.zero_grad()
			current_loss.backward()
			self.optimizer.step()

			train_loss += current_loss.item()
			_, pred = torch.max(pred, 1)
			correct += (pred==y).sum().item()

		print('Train Loss: ' + str(round(train_loss/len(self.train_loader),4)))
		print('Train Accuracy: '+str(round(float(correct)/29316,3)))

	def validate(self):
		self.model.eval()
		val_loss = 0
		correct = 0
		for x, y in self.val_loader:
			x = x.float()
			if torch.cuda.is_available():
				x = x.cuda()
				y = y.cuda()
			pred = self.model(x)
			current_loss = self.loss(pred, y)

			val_loss += current_loss.item()
			_, pred = torch.max(pred, 1)
			correct += (pred==y).sum().item()
		print('Val Loss: ' + str(round(val_loss/len(self.val_loader),4)))
		print('Val Accuracy: ' + str(round(float(correct)/7330,3)))

if __name__ == "__main__":
	with open('configs/bilstm.json') as cf:
		config_dict = json.load(f)
	agent = ProConAgent(config)
	agent.run()