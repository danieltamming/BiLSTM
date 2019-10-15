import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from utils.data import get_split_indices, get_embeddings, get_sr_sequences, sr_augment

class ProConSRDataset(Dataset):
	def __init__(self, reviews, labels, indices, embeddings, input_length, augment):
		self.reviews = [reviews[idx] for idx in indices]
		self.labels = [labels[idx] for idx in indices]
		self.augment = augment
		self.embeddings = embeddings
		self.input_length = input_length

	def __len__(self):
		return len(self.reviews)

	def __getitem__(self, idx):
		ratio = float(1)/2
		p = 0.5
		q = 0.5

		orig, syns = self.reviews[idx]
		if self.augment and syns and random.random()>ratio: rev = sr_augment(orig, syns, p, q, self.input_length)
		else: rev = orig
		review_embd = self.embeddings.take(rev, axis=0)
		return review_embd, self.labels[idx]

class ProConSRDataLoader:
	def __init__(self, config, pct_usage):
		self.config = config
		self.pct_usage = pct_usage
		self.reviews, self.labels = get_sr_sequences(self.config, self.config.train_path, pct_usage)
		self.embeddings = get_embeddings(self.config.embed_filename)
		self.folds = get_split_indices(self.config.seed, self.config.num_classes, self.config.num_folds, self.labels)

	def getFold(self, fold_num=0):
		val_idxs = self.folds[fold_num]
		val_dataset = ProConSRDataset(self.reviews, self.labels, val_idxs, self.embeddings, self.config.input_length, False)
		val_loader = DataLoader(val_dataset, self.config.batch_size,
						num_workers=self.config.num_workers, pin_memory=True)
		train_idxs = [idx for i,fold in enumerate(self.folds) if i != fold_num for idx in fold]
		train_dataset = ProConSRDataset(self.reviews, self.labels, train_idxs, self.embeddings, self.config.input_length, True)
		train_loader = DataLoader(train_dataset, self.config.batch_size,
						num_workers=self.config.num_workers, pin_memory=True)
		return train_loader, val_loader

	def getTrainLoader(self):
		train_dataset = ProConSRDataset(self.reviews, self.labels, 
						list(range(len(self.labels))), self.embeddings, self.config.input_length, True)
		return DataLoader(train_dataset, self.config.batch_size, 
				num_workers=self.config.num_workers, pin_memory=True)

	def getTestLoader(self):
		reviews, labels = get_sr_sequences(self.config, self.config.test_path, 1)
		test_dataset = ProConSRDataset(reviews, labels, list(range(len(labels))),
						self.embeddings, self.config.input_length, False)
		return DataLoader(test_dataset, self.config.batch_size, 
				num_workers=self.config.num_workers, pin_memory=True)