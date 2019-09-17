import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import random

random.seed(314)

class ProConDataset(Dataset):
	# Pros Cons Dataset
	def __init__(self, data_path, embedding_file, num_classes=2, input_length=25):
		# some reviews are nonexisistent
		self.labels, self.reviews, embed_list = [], [], []
		for label in range(num_classes):
			data_file = data_path + str(label) + '.txt'
			with open(data_file, 'r') as f:
				for line in f.read().splitlines():
					self.labels.append(label)
					review = [int(num) for num in line.split()]
					if len(review) < input_length: review += (input_length-len(review))*[0]
					self.reviews.append(review[:input_length])
		with open(embedding_file, 'r') as f:
			self.embedding_dim = len(f.readline().split())-1
		embed_list.append(self.embedding_dim*[0])
		with open(embedding_file, 'r') as f:
			for line in f.read().splitlines():
				arr = line.rstrip('\n').split()
				embed_list.append([float(num) for num in arr[1:]])
		self.embeddings = np.array(embed_list)

	def __len__(self):
		return len(self.reviews)

	def __getitem__(self, idx):
		review_embd = self.embeddings.take(self.reviews[idx], axis=0)
		return review_embd, self.labels[idx]

class ProConDataLoader:
	def __init__(self, train_path, test_path, embed_path, batch_size=128, num_folds=5, num_classes=2, input_length=25):
		self.batch_size = batch_size
		self.num_folds = num_folds
		self.num_classes = num_classes
		self.train_set = ProConDataset(train_path, embed_path, input_length=input_length)
		self.test_set = ProConDataset(test_path, embed_path, input_length=input_length)
		self.folds = self.__getSplitIndices()

	def getFold(self, fold_num=0):
		val_idxs = self.folds[fold_num]
		val_sampler = SubsetRandomSampler(val_idxs)
		val_loader = DataLoader(self.train_set, self.batch_size, sampler=val_sampler)
		train_idxs = [idx for i,fold in enumerate(self.folds) if i!=fold_num for idx in fold]
		train_sampler = SubsetRandomSampler(train_idxs)
		train_loader = DataLoader(self.train_set, self.batch_size, sampler=train_sampler)
		return {'val_loader':val_loader, 'train_loader':train_loader}

	def getTestSet(self):
		return DataLoader(self.test_set, self.batch_size)

	def __getSplitIndices(self):
		class_indices = [[] for _ in range(self.num_classes)]
		for i in range(self.num_classes):
			class_indices[i] = [j for j, label in enumerate(self.train_set.labels) if label == i]
			random.shuffle(class_indices[i])
		folds = [[] for _ in range(self.num_folds)]
		for i in range(self.num_folds):
			for j in range(self.num_classes):
				folds[i] += class_indices[j][i::self.num_folds]
			random.shuffle(folds[i])
		return folds


# train_path = 'data/train/'
# train_set = PCDataset(train_path, word2vec_filename, num_classes=2, input_length=input_length)
# folds = getSplitIndices(train_set)

# val_idxs = folds[fold_count]
# train_idxs = [idx for i,fold in enumerate(folds) if i!=fold_count for idx in fold]

# val_sampler = SubsetRandomSampler(val_idxs)
# train_sampler = SubsetRandomSampler(train_idxs)

# val_loader = DataLoader(train_set, batch_size=128, shuffle=False, sampler=val_sampler)
# train_loader = DataLoader(train_set, batch_size=128, shuffle=False, sampler=train_sampler)