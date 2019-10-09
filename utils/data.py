import numpy as np
import random
from bs4 import BeautifulSoup

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
			num_seqs = int(f.readline())
			num_to_read = int(pct_usage*num_seqs)
			f.readline()
			for _ in range(num_to_read):
				labels.append(label)
				line = f.readline()
				seq = [int(tok) for tok in line.split()]
				if len(seq) < config.input_length: seq += (config.input_length-len(seq))*[0]
				seqs.append(seq[:config.input_length])
	return seqs, labels

def get_sr_sequences(config, data_path, pct_usage):
	reviews, labels = [], []
	for label in range(config.num_classes):
		data_file = data_path + str(label) + '.txt'
		with open(data_file) as f: soup = BeautifulSoup(f, 'html.parser')
		revs = soup.find_all('rev')
		num_to_read = int(pct_usage*len(revs))
		revs = revs[:num_to_read]
		for rev in revs:
			labels.append(label)
			orig = [int(tok) for tok in rev.find('orig').string.split()]
			if len(orig) < config.input_length: orig += (config.input_length-len(orig))*[0]
			orig = orig[:config.input_length]
			synonyms = {}
			for repl in rev.find_all('repl'):
				idx = int(repl.find('idx').string)
				synonym_set = []
				for syns in repl.find_all('syns'):
					synonym_set.append([int(tok) for tok in syns.string.split()])
				synonyms[idx] = synonym_set
			reviews.append((orig, synonyms))
	return reviews, labels

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

# from tqdm import tqdm

# file = 'data/procon_sr/train/0.txt'
# with open(file) as f: soup = BeautifulSoup(f, 'html.parser')

# reviews = []
# for rev in soup.find_all('rev'):
# 	orig = [int(tok) for tok in rev.find('orig').string.split()]
# 	synonyms = {}
# 	for repl in rev.find_all('repl'):
# 		idx = int(repl.find('idx').string)
# 		synonym_set = []
# 		for syns in repl.find_all('syns'):
# 			synonym_set.append([int(tok) for tok in syns.string.split()])
# 		synonyms[idx] = synonym_set
# 	reviews.append((orig, synonyms))

# for orig, syns in tqdm(reviews):
# 	rev = sr_augment(orig, syns, 0.5, 0.5)