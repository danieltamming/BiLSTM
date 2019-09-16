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

class BiLSTM(nn.Module):
	def __init__(self, embed_dim, num_classes):
		super(BiLSTM, self).__init__()
		self.lstm1 = nn.LSTM(input_size=embed_dim, hidden_size=64, batch_first=True, num_layers=1, bidirectional=True)
		self.drop1 = nn.Dropout(p=0.5)
		self.lstm2 = nn.LSTM(input_size=2*64, hidden_size=32, batch_first=True, num_layers=1, bidirectional=True)
		self.drop2 = nn.Dropout(p=0.5)
		self.dense1 = nn.Linear(2*32, 20)
		self.relu = nn.ReLU()
		self.dense2 = nn.Linear(20, num_classes)
		self.sm = nn.Softmax(dim=1)

	def forward(self, x):
		output1, _ = self.lstm1(x)
		_, (h2, _) = self.lstm2(output1)
		h2 = h2.permute(1,0,2)
		h2 = h2.contiguous().view(h2.shape[0],-1)
		d1 = self.dense1(h2)
		r1 = self.relu(d1)
		d2 = self.dense2(r1)
		output = self.sm(d2)
		return output

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


def train(model, train_loader, loss, optimizer):
	for i in range(1, num_epochs+1):

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
			correct += (pred==y).sum().float().item()

		print('Loss: ' + str(round(train_loss/len(train_loader),4)))
		print('Accuracy: '+str(round(correct/len(train_loader),3)))
		print()


word2vec_filename = 'data/embeddings.txt'
train_path = 'data/train/'
input_length = 25
word2vec_dim = 300
num_classes = 2
num_epochs = 100

train_set = PCDataset(train_path, word2vec_filename, num_classes=2, input_length=input_length)
folds = getSplitIndices(train_set)

for fold_count in range(len(folds)):
	valid_idxs = folds[fold_count]
	train_idxs = [idx for i,fold in enumerate(folds) if i!=fold_count for idx in fold]

	valid_sampler = SubsetRandomSampler(valid_idxs)
	train_sampler = SubsetRandomSampler(train_idxs)

	valid_loader = DataLoader(train_set, batch_size=128, shuffle=False, sampler=valid_sampler)
	train_loader = DataLoader(train_set, batch_size=128, shuffle=False, sampler=train_sampler)

	model = BiLSTM(word2vec_dim, num_classes)
	if torch.cuda.is_available(): model = model.cuda()

	loss = nn.CrossEntropyLoss()
	optimizer = Adam(model.parameters())

	train(model, train_loader, loss, optimizer)
	break