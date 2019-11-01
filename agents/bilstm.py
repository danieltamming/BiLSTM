import numpy as np
import logging
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from graphs.models.bilstm import BiLSTM
from graphs.losses.loss import CrossEntropyLoss
from datasets.procon import ProConDataManager
from utils.metrics import AverageMeter, get_accuracy, EarlyStopper
from utils.logger import print_and_log

class BiLSTMAgent:
	def __init__(self, config, pct_usage=1, frac=0.5, geo=0.5):
		self.config = config
		self.pct_usage = pct_usage
		self.frac = frac
		self.geo = geo
		self.logger = logging.getLogger('BiLSTMAgent')
		self.cur_epoch = 0
		self.loss = CrossEntropyLoss()
		self.mngr = ProConDataManager(
			self.config, self.pct_usage, frac, geo)

		self.device = (torch.device('cuda:1' if torch.cuda.is_available() 
					   else 'cpu'))
		print('Using '+str(int(100*self.pct_usage))+'% of the dataset.')
		self.logger.info('Using '+str(self.pct_usage)+' of the dataset.')

		if self.config.aug_mode == 'sr':
			s = (str(int(100*self.frac))+'% of the training data will be ' 
				 'original, the rest augmented.')
			print_and_log(self.logger, s)
			s = 'The geometric parameter is '+str(geo)+'.'
			print_and_log(self.logger, s)

	def initialize_model(self):
		self.model = BiLSTM(self.config)
		self.model = self.model.to(self.device)
		self.optimizer = Adam(self.model.parameters())
		self.model.train()

	def run(self):
		if self.config.mode == 'crosstest':
			for fold in range(self.config.num_folds):
				self.initialize_model()
				s = 'Fold number '+str(fold)
				print_and_log(self.logger, s)
				(self.train_loader, 
				self.val_loader) = self.mngr.crosstest_ldrs(fold)
				self.train()
				self.validate()

		elif self.config.mode == 'val':
			self.train_loader, self.val_loader = self.mngr.val_ldrs()
			self.initialize_model()
			self.train()
			# self.validate()

	def train(self):
		if self.config.mode == 'crosstest':
			for self.cur_epoch in range(self.config.num_epochs):
				self.train_one_epoch()
			s = 'Stopped after ' + str(self.config.num_epochs) + ' epochs'
			print_and_log(self.logger, s)

		elif self.config.mode == 'val':
			stopper = EarlyStopper(self.config.patience, 
								   self.config.min_epochs)

			start_time = time.time()

			for self.cur_epoch in range(self.config.max_epochs):
				self.train_one_epoch()
				acc,_ = self.validate()

				if start_time is not None:
					print('{} s/it'.format(round(time.time()-start_time,3)))
					start_time = None

				if stopper.update_and_check(acc, printing=True): 
					s = ('Stopped early with patience '
						'{}'.format(self.config.patience))
					print_and_log(self.logger, s)
					break

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

		# if self.config.mode == 'crossval':
		s = ('Training epoch {} | loss: {} - accuracy: ' 
		'{}'.format(self.cur_epoch, 
					round(loss.val, 5), 
					round(acc.val, 5)))
		# print_and_log(self.logger, s)
		self.logger.info(s)


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

		s = ('Validating epoch {} | loss: {} - accuracy: ' 
			'{}'.format(self.cur_epoch, 
						round(loss.val, 5), 
						round(acc.val, 5)))
		# print_and_log(self.logger, s)
		self.logger.info(s)

		return acc.val, loss.val