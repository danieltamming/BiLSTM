import os
import string
import re
import argparse
import random
import numpy as np
import math
import time
import itertools
from collections import Counter
from tqdm import tqdm
from nltk.corpus import wordnet
from nltk.probability import FreqDist, MLEProbDist
from bs4 import BeautifulSoup

random.seed(314)

def flatten(arr, levels):
	for _ in range(levels-1):
		arr = [e for subarr in arr for e in subarr]
	return arr

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('-pth', '--datapath', default='./../DownloadedData/',
		type=str, help='Folder containing embeddings and raw data?')
	return parser.parse_args()

def get_data(review_path):
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

def get_reviews(reviews_path, pro_file, con_file, test_pct=0.10):
	reviews = {'test':{}, 'train':{}}
	pro = get_data(reviews_path + pro_file)
	con = get_data(reviews_path + con_file)
	first_train_idx = int(test_pct*(len(pro) + len(con))/2)
	reviews['test'][0] = con[:first_train_idx]
	reviews['test'][1] = pro[:first_train_idx]
	reviews['train'][0] = con[first_train_idx:]
	reviews['train'][1] = pro[first_train_idx:]
	return reviews

def get_embed_vocab(embed_orig, embed_target, word_counter):
	temp_count = 1

	data_path = 'data/'
	if not os.path.exists(data_path): os.mkdir(data_path)
	data_path = 'data/procon/'
	if not os.path.exists(data_path): os.mkdir(data_path)
	embed_vocab = set()
	with open(embed_orig, 'r') as f1, open(embed_target, 'w+') as f2:
		num_lines, embed_dims = [int(num) for num in f1.readline().split()]
		for _ in tqdm(range(num_lines)):
			line = f1.readline()
			word = line.split(' ',1)[0]
			if word in word_counter:
				embed_vocab.add(word)
				f2.write(line)

				temp_count += 1
	print(temp_count)
	return embed_vocab

def get_tokenizer(word_counter, embed_vocab):
	valid_word_counter_list = [word for (word,_) in word_counter.most_common() if word in embed_vocab]
	word_to_num = {word:i+1 for i, word in enumerate(valid_word_counter_list)}
	num_to_word = {val:key for key, val in word_to_num.items()}
	return word_to_num, num_to_word

def tknz(reviews_list, word_to_num):
	tknzd_reviews_list = []
	for review, syns in reviews_list:
		tknzd_review = []
		for word in review:
			if word in word_to_num: tknzd_review.append(word_to_num[word])
		if len(tknzd_review) == 0: continue
		tknzd_syns = {}
		for i, synonym_set in syns.items():
			tknzd_synonym_set = []
			for synonym in synonym_set:
				tknzd_synonym = [word_to_num[word] for word in synonym if word in word_to_num]
				if len(tknzd_synonym) == len(synonym): tknzd_synonym_set.append(tknzd_synonym)
			if len(tknzd_synonym_set) > 0: tknzd_syns[i] = tknzd_synonym_set
		tknzd_reviews_list.append((tknzd_review, tknzd_syns))
	return tknzd_reviews_list

def tknz_dict(reviews, word_to_num):
	tknzd_reviews = {}
	for datatype, dictionary in reviews.items():
		tknzd_reviews[datatype] = {}
		for label, reviews_list in dictionary.items():
			tknzd_reviews[datatype][label] = tknz(reviews_list, word_to_num)
	return tknzd_reviews


def write_tknzd_splits(tknzd_reviews):
	data_path = 'data/procon_sr/'
	if not os.path.exists(data_path): os.mkdir(data_path)
	for datatype, dictionary in tknzd_reviews.items():
		datatype_path = data_path + datatype + '/'
		if not os.path.exists(datatype_path): os.mkdir(datatype_path)
		for label, reviews_list in dictionary.items():
			filepath = datatype_path + str(label) + '.txt'
			with open(filepath, 'w+') as f:
				# f.write(str(len(reviews_list))+'\n\n')
				# for review in reviews_list:
				# 	f.write(' '.join([str(num) for num in review])+'\n')
				for review, syns in reviews_list:
					f.write('<rev>')
					f.write('<orig>'+' '.join([str(i) for i in review])+'</orig>')
					if syns:
						for idx, synonym_list in syns.items():
							f.write('<repl>')
							f.write('<idx>'+str(idx)+'</idx>')
							for synonym in synonym_list:
								f.write('<syns>'+' '.join([str(i) for i in synonym])+'</syns>')
							f.write('</repl>')
					f.write('</rev>')

def get_word_counter(reviews):
	word_counter = Counter()
	for dtype, dictionary in reviews.items():
		for reviews_list in dictionary.values():
			for review, syns in tqdm(reviews_list):
				word_counter.update(review)
				if syns: 
					word_counter.update(flatten(syns.values(),3))
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

def get_synonym_dicts(review, dtype, min_reputation):
	if dtype == 'test': return {}
	idx_to_syns = {}
	for i, word in enumerate(review):
		synonyms = get_synonyms(word, min_reputation)
		if synonyms: idx_to_syns[i] = synonyms
	return idx_to_syns

def add_synonym_dicts(reviews, min_reputation):
	for dtype, dictionary in reviews.items():
		for reviews_list in dictionary.values():
			for i in tqdm(range(len(reviews_list))):
				review = reviews_list[i]
				reviews_list[i] = (review, get_synonym_dicts(review, dtype, min_reputation))
	return reviews

def process(aug_mode=None):
	args = get_args()
	target_path = 'data/procon_sr/'
	if not os.path.exists(target_path): os.mkdir(target_path)
	word2vec_orig = args.datapath + 'enwiki_20180420_300d.txt'
	reviews_path = args.datapath + 'pros-cons/'
	word2vec_target = 'data/procon_sr/embeddings.txt'

	reviews = get_reviews(reviews_path, 'IntegratedPros.txt', 'IntegratedCons.txt')

	# -----------------
	# reviews['train'][0] = reviews['train'][0][:5]
	# reviews['train'][1] = reviews['train'][1][:5]
	# -----------------

	if aug_mode: reviews = add_synonym_dicts(reviews, 2)
	word_counter = get_word_counter(reviews)

	embed_vocab = get_embed_vocab(word2vec_orig, word2vec_target, word_counter)
	word_to_num, num_to_word = get_tokenizer(word_counter, embed_vocab)
	tknzd_reviews = tknz_dict(reviews, word_to_num)

	tok_counter = get_word_counter(tknzd_reviews)
	print(max(list(tok_counter.keys())))

	write_tknzd_splits(tknzd_reviews)

process(aug_mode='sr')

reviews = [
['horrible', 'customer', 'service', 'delivery', 'problems'],
['doesn', 't', 'come', 'with', 'enough', 'accessories'],
['none'],
['like', 'all', 'digital', 'cameras', 'dynamic', 'range', 'of', 'contrast', 'is', 'poor', 'compared', 'to', 'film'],
['none'],
['need', 'more', 'zoom', 'and', 'ibm', 'mb', 'microdrive'],
['mediocre', 'low', 'light', 'high', 'speed', 'performance'],
['can', 'be', 'costly', 'to', 'get', 'the', 'most', 'out', 'of', 'it'],
['need', 'a', 'lot', 'of', 'light'],
['no', 'rechargeable', 'battery', 'came', 'with', 'it'],
['short', 'battery', 'life', 'weak', 'flash', 'and', 'urrrrr', 'needs', 'more', 'megapixels'],
['still', 'comes', 'with', 'only', 'a', 'mb', 'cf', 'card'],
['wimpy', 'mb', 'compact', 'flash', 'card'],
['mb', 'flash', 'card', 'included', 'semi', 'weak', 'flash', 'short', 'battery', 'life', 'no', 'zoom', 'with', 'movies'],
['battery', 'life', 'lack', 'of', 'memory'],
['soft', 'photos', 'proprietary', 'battery', 'with', 'mediocre', 'life'],
['canon', 'specific', 'a', 'c', 'adapter', 'not', 'included', 'and', 'the', 'measly', 'mb', 'flash', 'card'],
['bad', 'picture', 'quality', 'very', 'inadequate', 'flash'],
['software', 'is', 'bulky']
]

# start_time = time.time()
# file = 'data/procon_sr/train/0.txt'
# with open(file) as f: soup = BeautifulSoup(f, 'html.parser')
# # reviews = [rev for rev in soup.find_all('rev')]
# # origs = [rev.find('orig').string for rev in reviews]
# print(time.time()-start_time)

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