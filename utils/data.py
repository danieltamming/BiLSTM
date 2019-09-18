import numpy as np
import random

def get_split_indices(seed, num_classes, num_folds, labels):
	random.seed(seed)
	class_indices = [[] for _ in range(num_classes)]
	for i in range(num_classes):
		class_indices[i] = [j for j, label in enumerate(labels) if label == i]
		random.shuffle(class_indices[i])
	folds = [[] for _ in range(num_folds)]
	for i in range(num_folds):
		for j in range(num_classes):
			folds[i] += class_indices[j][i::num_folds]
		random.shuffle(folds[i])
	return folds

def get_embeddings(embedding_file):
	embed_list = []
	with open(embedding_file) as f:
		line = f.readline()
		embedding_dim = len(line.split())-1
		embed_list.append(embedding_dim*[0])
		while line:
			arr = line.rstrip('\n').split()
			embed_list.append([float(num) for num in arr[1:]])
			line = f.readline()
	return np.array(embed_list)

def get_sequences(config, data_path, pct_usage):
	seqs, labels = [], []
	for label in range(config.num_classes):
		data_file = data_path + str(label) + '.txt'
		with open(data_file) as f:
			num_seqs = f.readline()
			num_to_read = int(pct_usage*num_seqs)
			f.readline()
			for _ in range(num_to_read):
				labels.append(label)
				line = f.readline()
				seq = [int(tok) for tok in line.split()]
				if len(seq) < config.input_length: seq += (config.input_length-len(seq))*[0]
				seqs.append(seq[:config.input_length])
	return seqs, labels