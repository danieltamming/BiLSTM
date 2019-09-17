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

from graphs.models.bilstm import BiLSTM
from graphs.losses.loss import CrossEntropyLoss

random.seed(314)

class PCDataset(Dataset):
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

def getSplitIndices(dataset, num_folds=5):
	num_classes = len(set(dataset.labels))
	class_indices = [[] for _ in range(num_classes)]
	for i in range(num_classes):
		class_indices[i] = [j for j, label in enumerate(dataset.labels) if label == i]
		random.shuffle(class_indices[i])
	folds = [[] for _ in range(num_folds)]
	for i in range(num_folds):
		for j in range(num_classes):
			folds[i] += class_indices[j][i::num_folds]
		random.shuffle(folds[i])
	return folds

def validate(model, val_loader, loss):
	model.eval()
	val_loss = 0
	correct = 0
	for x, y in val_loader:
		x = x.float()
		if torch.cuda.is_available():
			x = x.cuda()
			y = y.cuda()
		pred = model(x)
		current_loss = loss(pred, y)

		val_loss += current_loss.item()
		_, pred = torch.max(pred, 1)
		correct += (pred==y).sum().item()
	print('Val Loss: ' + str(round(val_loss/len(val_loader),4)))
	print('Val Accuracy: ' + str(round(float(correct)/7330,3)))

def train_one_epoch(model, train_loader, loss, optimizer):
	train_loss = 0
	correct = 0
	model.train()

	for x, y in train_loader:
		x = x.float()
		if torch.cuda.is_available(): 
			x = x.cuda()
			y = y.cuda()
		pred = model(x)
		current_loss = loss(pred, y)
		optimizer.zero_grad()
		current_loss.backward()
		optimizer.step()

		train_loss += current_loss.item()
		_, pred = torch.max(pred, 1)
		correct += (pred==y).sum().item()

	print('Train Loss: ' + str(round(train_loss/len(train_loader),4)))
	print('Train Accuracy: '+str(round(float(correct)/29316,3)))

def train(model, train_loader, loss, optimizer):
	for i in range(1, num_epochs+1):
		print('Epoch number '+str(i)+'.')
		train_one_epoch(model, train_loader, loss, optimizer)
		validate(model, val_loader, loss)


word2vec_filename = 'data/embeddings.txt'
train_path = 'data/train/'
input_length = 25
word2vec_dim = 300
num_classes = 2
num_epochs = 250

train_set = PCDataset(train_path, word2vec_filename, num_classes=2, input_length=input_length)
folds = getSplitIndices(train_set)

for fold_count in range(len(folds)):
	val_idxs = folds[fold_count]
	train_idxs = [idx for i,fold in enumerate(folds) if i!=fold_count for idx in fold]

	val_sampler = SubsetRandomSampler(val_idxs)
	train_sampler = SubsetRandomSampler(train_idxs)

	val_loader = DataLoader(val_set, batch_size=128, shuffle=False, sampler=val_sampler)
	train_loader = DataLoader(train_set, batch_size=128, shuffle=False, sampler=train_sampler)

	model = BiLSTM(word2vec_dim, num_classes)
	if torch.cuda.is_available(): model = model.cuda()

	loss = CrossEntropyLoss()
	optimizer = Adam(model.parameters())

	validate(model, val_loader, loss)

	train(model, train_loader, loss, optimizer)
	break