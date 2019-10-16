import numpy as np
import logging
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from graphs.models.bilstm import BiLSTM
from graphs.losses.loss import CrossEntropyLoss
from datasets.procon import ProConDataLoader
from datasets.procon_sr import ProConSRDataLoader
from utils.metrics import AverageMeter, get_accuracy, EarlyStopper

class BiLSTMAgent:
	def __init__(self, config, pct_usage=1, frac=0.5, geo=0.5):
		self.config = config
		self.pct_usage = pct_usage
		self.frac = frac
		self.geo = geo
		self.logger = logging.getLogger('BiLSTMAgent')
		self.cur_epoch = 0
		self.loss = CrossEntropyLoss()
		if self.config.aug_mode == 'sr': self.loaders = ProConSRDataLoader(self.config, self.pct_usage, self.frac, self.geo)
		else: self.loaders = ProConDataLoader(self.config, self.pct_usage)
		self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
		print('Using '+str(int(100*self.pct_usage))+'% of the dataset.')
		self.logger.info('Using '+str(self.pct_usage)+' of the dataset.')
		print(str(int(100*self.frac))+'% of the training data will be original, the rest augmented.')
		self.logger.info(str(int(100*self.frac))+'% of the training data will be original, the rest augmented.')
		print('The geometric parameter for synonym selection is '+str(geo)+'.')
		self.logger.info('The geometric parameter for synonym selection is '+str(geo)+'.')

	def initialize_model(self):
		self.model = BiLSTM(self.config)
		self.model = self.model.to(self.device)
		self.optimizer = Adam(self.model.parameters())
		self.model.train()

	def run(self):
		if self.config.mode == 'crossval':
			for fold_count in range(self.config.num_folds):
				self.initialize_model()
				print('Fold number '+str(fold_count))
				self.logger.info('Fold number '+str(fold_count))
				self.train_loader, self.val_loader = self.loaders.getFold(fold_count)
				self.train()
				# acc,_ = self.validate()

		elif self.config.mode == 'test':
			self.train_loader = self.loaders.getTrainLoader()
			self.val_loader = self.loaders.getTestLoader()
			self.initialize_model()
			self.train()
			self.validate()

	def train(self):
		if self.config.mode == 'crossval':
			# stopper = EarlyStopper(self.config.patience, self.config.min_epochs)
			for self.cur_epoch in tqdm(range(self.config.num_epochs)):
				self.train_one_epoch()
				acc,_ = self.validate()
				# if stopper.update_and_check(acc): 
				# 	print('Stopped early with patience '+str(self.config.patience))
				# 	self.logger.info('Stopped early with patience '+str(self.config.patience))
				# 	break
			self.logger.info('Stopped after '+str(self.config.num_epochs)+' epochs')

		elif self.config.mode == 'test':
			for self.cur_epoch in tqdm(range(self.config.num_epochs)):
				self.train_one_epoch()

	def train_one_epoch(self):
		self.model.train()
		loss = AverageMeter()
		acc = AverageMeter()

		for x, y in self.train_loader:
			x = x.float()

			x = x.to(self.device)
			y = y.to(self.device)

			output = self.model(x)
			current_loss = self.loss(output, y)
			self.optimizer.zero_grad()
			current_loss.backward()
			self.optimizer.step()

			loss.update(current_loss.item())
			accuracy = get_accuracy(output, y)
			acc.update(accuracy, y.shape[0])

		if self.config.mode == 'crossval':
			# print('Training epoch '+str(self.cur_epoch)+' | loss: '
				+str(round(loss.val,5))+' - accuracy: '+str(round(acc.val,5)))
			self.logger.info('Training epoch '+str(self.cur_epoch)+' | loss: '
				+str(round(loss.val,5))+' - accuracy: '+str(round(acc.val,5)))

	def validate(self):
		self.model.eval()
		
		loss = AverageMeter()
		acc = AverageMeter()
		for x, y in self.val_loader:
			x = x.float()

			x = x.to(self.device)
			y = y.to(self.device)

			output = self.model(x)
			current_loss = self.loss(output, y)

			loss.update(current_loss.item())
			accuracy = get_accuracy(output, y)
			acc.update(accuracy, y.shape[0])

		# print('Validating epoch '+str(self.cur_epoch)+' | loss: '
			+str(round(loss.val,5))+' - accuracy: '+str(round(acc.val,5)))
		self.logger.info('Validating epoch '+str(self.cur_epoch)+' | loss: '
			+str(round(loss.val,5))+' - accuracy: '+str(round(acc.val,5)))
		return acc.val, loss.val