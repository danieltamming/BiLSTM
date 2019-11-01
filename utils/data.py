import os
import random
import pickle

import numpy as np

def get_split_indices(data, seed, num_classes, num_folds):
	random.seed(seed)
	class_indices = [[] for _ in range(num_classes)]
	for i in range(num_classes):
		class_indices[i] = [j for j, (label,_,_) in enumerate(data) if label == i]
		random.shuffle(class_indices[i])
	folds = [[] for _ in range(num_folds)]
	for i in range(num_folds):
		for j in range(num_classes):
			folds[i] += class_indices[j][i::num_folds]
		random.shuffle(folds[i])
	return folds

def get_embeddings(data_path):
	embed_filename = os.path.join(data_path, 'embeddings.pickle')
	with open(embed_filename, 'rb') as f:
		embeddings = pickle.load(f)
	return embeddings

def get_sequences(data_path, input_length):
	data_filename = os.path.join(data_path, 'data.pickle')
	with open(data_filename, 'rb') as f:
		data = pickle.load(f)
	formatted_data = []
	for label, seq, aug in data:
		if len(seq) < input_length:
			seq += (input_length - len(seq))*[0]
		seq = seq[:input_length]
		formatted_data.append((label, seq, aug))
	return formatted_data

def sr_augment(orig, syns, p, q, input_length):
	num_to_replace = min(np.random.geometric(p), len(syns))
	idxs = random.sample(syns.keys(), num_to_replace)
	rev = []
	for i, tok in enumerate(orig):
		if i not in idxs:
			rev.append(tok)
		else:
			chosen_syn_idx = min(np.random.geometric(q), len(syns[i])) - 1
			rev.extend(syns[i][chosen_syn_idx])
	rev = rev[:input_length]
	return rev

def partition_within_classes(data, pct_in_A, num_classes):
	'''
	Partitions data into A and B, where A pct_in_A of the dataset and B is
	the rest. A will have perfectly even distribution among classes
	'''
	if pct_in_A == 1:
		return data, []
	A_size_per_class = int(pct_in_A*len(data)/num_classes)
	A_class_deficits = {i:-A_size_per_class for i in range(num_classes)}
	A, B = [], []
	for i, (label, seq, aug) in enumerate(data):
		if A_class_deficits[label] < 0:
			A.append((label, seq, aug))
			A_class_deficits[label] += 1
		else:
			B.append((label, seq, aug))
	return A, B

if __name__ == "__main__":
	config = None
	data_path = 'data/procon_sr/'
	pct_usage = 1
	reviews, labels = get_sr_sequences(config, data_path, pct_usage)
	print(reviews[0])
	print(labels[0])