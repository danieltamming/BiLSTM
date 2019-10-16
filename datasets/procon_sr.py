import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from utils.data import get_split_indices, get_embeddings, get_sr_sequences, sr_augment

class ProConSRDataset(Dataset):
	def __init__(self, reviews, labels, indices, embeddings, input_length, frac=None, geo=None):
		self.reviews = [reviews[idx] for idx in indices]
		self.labels = [labels[idx] for idx in indices]
		self.embeddings = embeddings
		self.input_length = input_length
		self.frac = frac # fraction of data that is original, not augmented
		self.geo = geo # geometric parameter for number of syns and syn order
		# both of the above are zero if not using augmentation


	def __len__(self):
		return len(self.reviews)

	def __getitem__(self, idx):
		p = q = self.geo

		orig, syns = self.reviews[idx]
		if self.geo and syns and random.random()>self.frac: rev = sr_augment(orig, syns, p, q, self.input_length)
		else: rev = orig
		review_embd = self.embeddings.take(rev, axis=0)
		return review_embd, self.labels[idx]

class ProConSRDataLoader:
	def __init__(self, config, pct_usage, frac, geo):
		self.config = config
		self.pct_usage = pct_usage
		self.geo = geo
		self.frac = frac
		self.reviews, self.labels = get_sr_sequences(self.config, self.config.train_path, pct_usage)
		self.embeddings = get_embeddings(self.config.embed_filename)
		self.folds = get_split_indices(self.config.seed, self.config.num_classes, self.config.num_folds, self.labels)

	def getFold(self, fold_num=0):
		val_idxs = self.folds[fold_num]
		val_dataset = ProConSRDataset(self.reviews, self.labels, val_idxs, self.embeddings, self.config.input_length)
		val_loader = DataLoader(val_dataset, self.config.batch_size,
						num_workers=self.config.num_workers, pin_memory=True)
		train_idxs = [idx for i,fold in enumerate(self.folds) if i != fold_num for idx in fold]
		train_dataset = ProConSRDataset(self.reviews, self.labels, train_idxs, self.embeddings, self.config.input_length, self.frac, self.geo)
		train_loader = DataLoader(train_dataset, self.config.batch_size,
						num_workers=self.config.num_workers, pin_memory=True)
		return train_loader, val_loader

	def getTrainLoader(self):
		train_dataset = ProConSRDataset(self.reviews, self.labels, 
						list(range(len(self.labels))), self.embeddings, self.config.input_length, self.frac, self.geo)
		return DataLoader(train_dataset, self.config.batch_size, 
				num_workers=self.config.num_workers, pin_memory=True)

	def getTestLoader(self):
		reviews, labels = get_sr_sequences(self.config, self.config.test_path, 1)
		test_dataset = ProConSRDataset(reviews, labels, list(range(len(labels))),
						self.embeddings, self.config.input_length)
		return DataLoader(test_dataset, self.config.batch_size, 
				num_workers=self.config.num_workers, pin_memory=True)