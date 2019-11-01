import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from utils.data import (get_split_indices, get_embeddings, 
							  get_sequences, sr_augment, 
							  partition_within_classes)

class ProConDataset(Dataset):
	def __init__(self, data, embeddings, input_length, frac=None, geo=None):
		self.data = data
		self.embeddings = embeddings
		self.input_length = input_length
		self.frac = frac # fraction of data that is original, not augmented
		self.geo = geo # geometric parameter for number of syns and syn order
		# both of the above are None if not using augmentation


	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		p = q = self.geo
		label, orig_seq, aug = self.data[idx]
		if self.geo and aug and random.random()>self.frac: 
			seq = sr_augment(org_seq, aug, p, q, self.config.input_length)
		else: 
			seq = orig_seq
		seq_embd = self.embeddings.take(seq, axis=0)
		return seq_embd, label

class ProConDataManager:
	def __init__(self, config, pct_usage, frac, geo):
		self.config = config
		self.pct_usage = pct_usage
		self.frac = frac
		self.geo = geo

		self.data = get_sequences(self.config.data_path, 
								  self.config.input_length)
		self.embeddings = get_embeddings(self.config.data_path)
		if self.config.mode == 'crosstest':
			self.folds = get_split_indices(
				self.data, self.config.seed, 
				self.config.num_classes, self.config.num_folds)
		elif self.config.mode == 'val':
			self.val_data, all_train_data = partition_within_classes(
				self.data, self.config.val_pct, self.config.num_classes)
			self.train_data, _ = partition_within_classes(
				all_train_data, self.pct_usage, self.config.num_classes)

	def val_ldrs(self):
		train_dataset = ProConDataset(
			self.train_data, self.embeddings,
			self.config.input_length, self.frac, self.geo)
		train_loader =  DataLoader(
			train_dataset, self.config.batch_size, 
			num_workers=self.config.num_workers, 
			pin_memory=True, shuffle=True)
		val_dataset = ProConDataset(
			self.val_data, self.embeddings,
			self.config.input_length)
		val_loader = DataLoader(
			val_dataset, self.config.batch_size,
			num_workers=self.config.num_workers, pin_memory=True)
		return train_loader, val_loader

	def crosstest_ldrs(self, fold_num):
		test_idxs = self.folds[fold_num]
		test_data = [tup for idx, tup in enumerate(self.data)
					  if idx in test_idxs]
		test_dataset = ProConDataset(
			test_data, self.embeddings, self.config.input_length)
		test_loader = DataLoader(
			test_dataset, self.config.batch_size,
			num_workers=self.config.num_workers, pin_memory=True)
		train_data = [tup for idx, tup in enumerate(self.data) 
					  if idx not in test_idxs]
		train_dataset = ProConDataset(
			train_data, self.embeddings,
			self.config.input_length, self.frac, self.geo)
		train_loader = DataLoader(
			train_dataset, self.config.batch_size,
			num_workers=self.config.num_workers, 
			pin_memory=True, shuffle=True)
		return train_loader, test_loader