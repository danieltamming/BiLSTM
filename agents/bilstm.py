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

import argparse
import json
from easydict import EasyDict

from graphs.models.bilstm import BiLSTM
from graphs.losses.loss import CrossEntropyLoss
from datasets.procon import ProConDataLoader
from utils.metrics import AverageMeter, get_accuracy

class BiLSTMAgent:
	def __init__(self, config, pct_usage=1):
		self.config = config
		self.pct_usage = pct_usage
		self.logger = logging.getLogger('BiLSTMAgent')
		self.cur_epoch = 0
		self.loss = CrossEntropyLoss()
		print('Using '+str(self.pct_usage)+' of the dataset.')
		self.logger.info('Using '+str(self.pct_usage)+' of the dataset.')
		self.loaders = ProConDataLoader(self.config, self.pct_usage)

	def initialize_model(self):
		self.model = BiLSTM(self.config)
		if torch.cuda.device_count() > 1: self.model = nn.DataParallel(self.model)
		if torch.cuda.is_available(): self.model = self.model.cuda()
		self.optimizer = Adam(self.model.parameters())

	def run(self):
		if self.config.mode == 'crossval':
			for fold_count in range(self.config.num_folds):
				self.initialize_model()
				print('Fold number '+str(fold_count))
				self.logger.info('Fold number '+str(fold_count))
				self.train_loader, self.val_loader = self.loaders.getFold(fold_count)
				self.train()
				# self.validate()

		elif self.config.mode == 'test':
			self.train_loader = self.loaders.getTrainLoader()
			self.val_loader = self.loaders.getTestLoader()
			self.initialize_model()
			self.train()
			self.validate()

	def train(self):
		for self.cur_epoch in range(self.config.num_epochs):
			self.train_one_epoch()
			if self.config.mode == 'crossval': self.validate()

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

		print('Training epoch '+str(self.cur_epoch)+' | loss: '
			+str(round(loss.val,5))+' - accuracy: '+str(round(acc.val,5)))
		self.logger.info('Training epoch '+str(self.cur_epoch)+' | loss: '
			+str(round(loss.val,5))+' - accuracy: '+str(round(acc.val,5)))

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

		print('Validating epoch '+str(self.cur_epoch)+' | loss: '
			+str(round(loss.val,5))+' - accuracy: '+str(round(acc.val,5)))
		self.logger.info('Validating epoch '+str(self.cur_epoch)+' | loss: '
			+str(round(loss.val,5))+' - accuracy: '+str(round(acc.val,5)))