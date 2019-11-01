import os
import string
import re
import pickle
from collections import Counter

import numpy as np
from tqdm import tqdm
from nltk.corpus import wordnet

def get_paths(aug_mode):
	script_path = os.path.dirname(os.path.realpath(__file__))
	repo_path = os.path.join(script_path, os.pardir)
	data_parent = os.path.join(repo_path, os.pardir, 'DownloadedData')
	target_parent = os.path.join(repo_path, 'data')
	data_path = os.path.join(data_parent,'pros-cons/')
	embed_filename = os.path.join(data_parent, 'enwiki_20180420_300d.txt')
	if aug_mode == 'sr': 
		target_path = os.path.join(target_parent,'procon_sr')
	else: 
		target_path = os.path.join(target_parent,'procon')
	if not os.path.exists(target_path): 
		os.mkdir(target_path)
	return data_path, embed_filename, target_path

def flatten(arr, levels):
	if not arr: return []
	for _ in range(levels-1):
		arr = [e for subarr in arr for e in subarr]
	return arr

def get_data_from_file(review_path):
	reviews = []
	with open(review_path, 'r') as f:
		reviews_raw = f.read()
	reviews_raw = [tup[1] for tup in re.findall(r'(<Cons>|<Pros>)(.*?)(</Cons>|</Pros>)', reviews_raw)]
	for i, review in enumerate(reviews_raw):
		review = review.lower()
		review = review.replace('&amp', ' ')
		to_remove = string.punctuation + string.digits
		trans = str.maketrans(to_remove, len(to_remove)*' ')
		review = review.translate(trans)
		review = review.split()
		if len(review) > 0: reviews.append(review)
	return reviews

def get_data(data_path, pro_file, con_file):
	cons = get_data_from_file(data_path + con_file)
	pros = get_data_from_file(data_path + pro_file)
	data = ([(0, seq, {}) for seq in cons] 
			+ [(1, seq, {}) for seq in pros])
	return data

def get_embed_vocab(embed_filename, target_path, word_counter):
	embed_vocab = set()
	with open(embed_filename, 'r') as f:
		num_lines, embed_dims = [int(num) for num in f.readline().split()]
		included_embeddings = [embed_dims*[0]]
		for _ in tqdm(range(num_lines)):
			line = f.readline()
			word, vec = line.split(' ',1)
			if word in word_counter:
				embed_vocab.add(word)
				included_embeddings.append([float(num) for num in vec.split()])
	included_embeddings = np.array(included_embeddings)
	embed_target = os.path.join(target_path, 'embeddings.pickle')
	with open(embed_target, 'wb') as f:
		pickle.dump(included_embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)
	return embed_vocab

def get_tokenizer(word_counter, embed_vocab):
	valid_word_counter_list = [word for (word,_) in word_counter.most_common() if word in embed_vocab]
	word_to_num = {word:i+1 for i, word in enumerate(valid_word_counter_list)}
	num_to_word = {val:key for key, val in word_to_num.items()}
	return word_to_num, num_to_word

def tknz(data, word_to_num):
	tknzd_data = []
	for label, seq, aug in data:
		tknzd_seq = [word_to_num[word] for word in seq if word in word_to_num]
		if not tknzd_seq: 
			continue
		tknzd_aug = {}
		if aug:
			for i, synonym_set in aug.items():
				tknzd_synonym_set = []
				for synonym in synonym_set:
					tknzd_synonym = [word_to_num[word] for word in synonym if word in word_to_num]
					if len(tknzd_synonym) == len(synonym): 
						tknzd_synonym_set.append(tknzd_synonym)
				if tknzd_synonym_set:
					tknzd_aug[i] = tknzd_synonym_set
		tknzd_data.append((label, tknzd_seq, tknzd_aug))
	return tknzd_data

def get_word_counter(data):
	word_counter = Counter()
	for label, seq, aug in data:
		word_counter.update(seq)
		if aug:
			word_counter.update(flatten(aug.values(),3))
	return word_counter

def meets_requirements(synonym):
	return all(c.isalpha() or c.isspace() for c in synonym)

def get_synonyms(word, min_reputation):
	if len(word) <= 1: return []
	synonyms = Counter()
	for syn in wordnet.synsets(word):
		for lemma in syn.lemmas():
			synonym = lemma.name().lower().replace('_',' ').replace('-',' ')
			if meets_requirements(synonym) and synonym != word and synonym not in synonyms: 
				synonyms.update({synonym:lemma.count()})
	synonyms = Counter({synonym:synonyms[synonym] for synonym in synonyms if synonyms[synonym] >= min_reputation})
	return [word.split() for word,_ in synonyms.most_common()]

def get_synonym_dicts(seq, min_reputation):
	idx_to_syns = {}
	for i, word in enumerate(seq):
		synonyms = get_synonyms(word, min_reputation)
		if synonyms: idx_to_syns[i] = synonyms
	return idx_to_syns

def add_synonym_dicts(data, min_reputation):
	new_data = []
	for label, seq, _ in tqdm(data):
		aug = get_synonym_dicts(seq, min_reputation)
		new_data.append((label, seq, aug))
	return new_data

def process(aug_mode):
	data_path, embed_filename, target_path = get_paths(aug_mode)
	data = get_data(data_path, 'IntegratedPros.txt', 'IntegratedCons.txt')
	if aug_mode == 'sr': 
		data = add_synonym_dicts(data, 2)
	word_counter = get_word_counter(data)
	embed_vocab = get_embed_vocab(embed_filename, target_path, word_counter)
	word_to_num, num_to_word = get_tokenizer(word_counter, embed_vocab)
	tknzd_data = tknz(data, word_to_num)

	embed_target = os.path.join(target_path,'data.pickle')
	with open(embed_target, 'wb') as f:
		pickle.dump(tknzd_data, f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
	aug_mode = 'sr'
	# aug_mode = None
	process(aug_mode)