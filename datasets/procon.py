import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from utils.data import get_split_indices, get_embeddings, get_sequences

class ProConDataset(Dataset):
	def __init__(self, config, data_path, embedding_file, pct_usage=1):
		self.reviews, self.labels = get_sequences(config, data_path, pct_usage)
		self.embeddings = get_embeddings(embedding_file)

	def __len__(self):
		return len(self.reviews)

	def __getitem__(self, idx):
		review_embd = self.embeddings.take(self.reviews[idx], axis=0)
		return review_embd, self.labels[idx]

class ProConDataLoader:
	def __init__(self, config, pct_usage=1):
		self.config = config
		self.pct_usage = pct_usage
		self.train_set = ProConDataset(self.config, self.config.train_path, self.config.embed_filename)
		self.test_set = ProConDataset(self.config, self.config.test_path, self.config.embed_filename)
		self.folds = get_split_indices(self.config.seed, self.config.num_classes, self.config.num_folds, self.train_set.labels)

	def getFold(self, fold_num=0):
		val_idxs = self.folds[fold_num]
		val_sampler = SubsetRandomSampler(val_idxs)
		val_loader = DataLoader(self.train_set, self.config.batch_size, 
						num_workers=self.config.num_workers, sampler=val_sampler, pin_memory=True)
		train_idxs = [idx for i,fold in enumerate(self.folds) if i!=fold_num for idx in fold]
		train_sampler = SubsetRandomSampler(train_idxs)
		train_loader = DataLoader(self.train_set, self.config.batch_size, 
						num_workers=self.config.num_workers, sampler=train_sampler, pin_memory=True)
		return train_loader, val_loader

	def getTrainLoader(self):
		return DataLoader(self.train_set, self.config.batch_size, 
				num_workers=self.config.num_workers, pin_memory=True)

	def getTestLoader(self):
		return DataLoader(self.test_set, self.config.batch_size, 
				num_workers=self.config.num_workers, pin_memory=True)