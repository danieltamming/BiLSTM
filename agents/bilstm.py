import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm
import logging
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim import Adam
import random

import argparse
import json
from easydict import EasyDict

from graphs.models.bilstm import BiLSTM
from graphs.losses.loss import CrossEntropyLoss
from datasets.procon import ProConDataLoader
from utils.metrics import AverageMeter, get_accuracy

random.seed(314)

class BiLSTMAgent:
	def __init__(self, config):
		self.config = config
		self.logger = logging.getLogger('BiLSTMAgent')
		self.cur_epoch = 0
		self.model = BiLSTM(self.config)
		if torch.cuda.is_available(): self.model = self.model.cuda()
		self.loaders = ProConDataLoader(self.config)
		self.loss = CrossEntropyLoss()
		self.optimizer = Adam(self.model.parameters())

	def run(self):
		for fold_count in range(5):
			# Initialize model here?
			dictionary = self.loaders.getFold(fold_count)
			self.train_loader = dictionary['train_loader']
			self.val_loader = dictionary['val_loader']
			self.validate()
			self.train()

	def train(self):
		for self.cur_epoch in range(1, self.config.num_epochs+1):
			# self.train_one_epoch()
			self.validate()

	def train_one_epoch(self):
		self.model.train()

		loss = AverageMeter()
		acc = AverageMeter()
		for x, y in self.train_loader:
			x = x.float()
			if torch.cuda.is_available(): 
				x = x.cuda()
				y = y.cuda()
			output = self.model(x)
			current_loss = self.loss(output, y)
			self.optimizer.zero_grad()
			current_loss.backward()
			self.optimizer.step()

			loss.update(current_loss.item())
			accuracy = get_accuracy(output, y)
			acc.update(accuracy, y.shape[0])

		print('Train Loss: ' + str(round(loss.val,4)))
		print('Train Accuracy: '+str(round(acc.val,4)))
		self.logger.info('Training epoch number '+str(self.cur_epoch)+' | loss: '
			+str(loss.val)+' - accuracy: '+str(acc.val))

	def validate(self):
		self.model.eval()
		
		loss = AverageMeter()
		acc = AverageMeter()
		for x, y in self.val_loader:
			x = x.float()
			if torch.cuda.is_available():
				x = x.cuda()
				y = y.cuda()
			output = self.model(x)
			current_loss = self.loss(output, y)

			loss.update(current_loss.item())
			accuracy = get_accuracy(output, y)
			acc.update(accuracy, y.shape[0])

		print('Val Loss: ' + str(round(loss.val,4)))
		print('Val Accuracy: ' + str(round(acc.val,4)))
		self.logger.info('Validating epoch number '+str(self.cur_epoch)+' | loss: '
			+str(round(loss.val,5))+' - accuracy: '+str(round(acc.val,5)))